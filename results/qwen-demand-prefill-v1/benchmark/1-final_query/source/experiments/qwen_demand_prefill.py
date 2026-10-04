"""One preregistered last-position prefill study using existing offline Qwen assets.

No model downloads, parameter export, candidate search, or production default changes.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

SPEC = Path('configs/qwen-demand-prefill/study.json')
SOURCES = ['experiments/qwen_demand_prefill.py', 'experiments/__init__.py',
           'lab/qwen_demand_prefill.py', 'lab/qa_metrics.py', 'lab/__init__.py']


def save(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(pending, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def reserve(path):
    Path(path).mkdir(parents=True, exist_ok=False)
    return Path(path)


def seal(path):
    save(path / 'checksums.json', {str(p.relative_to(path)): sha(p)
         for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'checksums.json'})


def checked_receipt(path):
    expected = json.loads((path / 'checksums.json').read_text())
    actual = {str(p.relative_to(path)): sha(p) for p in path.rglob('*')
              if p.is_file() and p.name != 'checksums.json'}
    if actual != expected:
        raise ValueError('Receipt file identities changed')
    run = json.loads((path / 'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Incomplete audit')
    if run['spec_sha256'] != sha(SPEC):
        raise ValueError('Protocol drift')
    if run['source_sha256'] != {p: sha(p) for p in SOURCES}:
        raise ValueError('Source drift since audit')
    return run


def setup(model_path, out):
    spec = json.loads(SPEC.read_text())
    if not model_path.is_dir():
        raise ValueError('Existing offline model directory required')
    files = {p.name: sha(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    if files != spec['model_files_sha256']:
        raise ValueError('Frozen model/tokenizer/config identity mismatch')
    versions = {p: importlib.metadata.version(p) for p in spec['required_versions']}
    if versions != spec['required_versions']:
        raise ValueError('Dependency version mismatch')
    for name in ['path', 'system_path']:
        key = 'sha256' if name == 'path' else 'system_sha256'
        if sha(spec['quality'][name]) != spec['quality'][key]:
            raise ValueError('Frozen input drift')
    import mlx.core as mx
    from mlx_lm import load
    if not mx.metal.is_available():
        raise RuntimeError('Metal required; CPU fallback forbidden')
    mx.set_default_device(mx.gpu)
    model, tok = load(str(model_path))
    mx.eval(model.parameters())
    mx.synchronize()
    config = json.loads((model_path / 'config.json').read_text())
    if config.get('use_sliding_window') or config['model_type'] != 'qwen2':
        raise ValueError('Only frozen full-causal Qwen2 supported')
    upstream = {}
    for name in ['mlx_lm.models.qwen2', 'mlx_lm.models.base', 'mlx_lm.models.cache',
                 'mlx_lm.models.rope_utils', 'mlx_lm.generate', 'mlx.nn.layers.quantized',
                 'mlx.nn.layers.linear', 'mlx.nn.layers.embedding', 'mlx.nn.layers.normalization']:
        module = importlib.import_module(name)
        upstream[name] = sha(module.__file__)
    if upstream != spec['upstream_source_sha256']:
        raise ValueError('Reviewed upstream source identity mismatch')
    source = {p: sha(p) for p in SOURCES}
    for path in SOURCES:
        target = out / 'source' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
    save(out / 'protocol.json', spec)
    manifest = {'status': 'running', 'spec_sha256': sha(SPEC), 'source_sha256': source,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                'model_files_sha256': files, 'model_config': config,
                'upstream_source_sha256': upstream, 'versions': versions,
                'platform': platform.platform(), 'python': sys.version,
                'device': mx.metal.device_info(), 'default_device': str(mx.default_device()),
                'pid': os.getpid()}
    save(out / 'run.json', manifest)
    return spec, model, tok, manifest


def workloads(tok, spec):
    bases = [tok.encode(body * 600, add_special_tokens=False)
             for body in spec['benchmark']['synthetic_bodies']]
    if min(map(len, bases)) < max(spec['benchmark']['lengths']):
        raise ValueError('Synthetic input shorter than frozen shape')
    return bases, [{'case_id': f'{n}-{i}', 'length': n, 'prompt_index': i,
                    'token_ids': base[:n], 'input_sha256': digest(base[:n])}
                   for n in spec['benchmark']['lengths'] for i, base in enumerate(bases)]


def generate(model, tok, ids, mode, count, *, stop, timed=False):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from lab.qwen_demand_prefill import prefill
    cache = make_prompt_cache(model)
    x = mx.array([ids])
    mx.eval(x)
    mx.synchronize()
    if timed:
        mx.reset_peak_memory()
        active_before = mx.get_active_memory()
        start = time.perf_counter()
    output, timestamps = [], []
    for step in range(count):
        logits = prefill(model, x, cache, mode) if step == 0 else model(x, cache=cache)[:, -1:, :]
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        mx.synchronize()
        if timed:
            timestamps.append(time.perf_counter())
        output.append(int(token.item()))
        if stop and output[-1] in tok.eos_token_ids:
            break
        x = token.reshape(1, 1)
    result = {'token_ids': output, 'stop_reason': 'eos' if stop and output[-1] in tok.eos_token_ids else 'token_limit'}
    if timed:
        result.update(ttft_seconds=timestamps[0]-start,
                      decode_seconds=timestamps[-1]-timestamps[0],
                      total_seconds=timestamps[-1]-start,
                      token_elapsed_seconds=[t-start for t in timestamps],
                      active_before_bytes=active_before, peak_active_bytes=mx.get_peak_memory())
    else:
        result['prediction'] = tok.decode(output, skip_special_tokens=True)
    return result


def cache_record(cache):
    import mlx.core as mx
    import numpy as np
    mx.eval([c.state for c in cache])
    return [{'offset': c.offset, 'tensors': [
        {'shape': list(a.shape), 'dtype': str(a.dtype),
         'sha256': hashlib.sha256(np.asarray(a).tobytes()).hexdigest()}
        for a in c.state]} for c in cache]


def logits_record(logits, reference):
    import mlx.core as mx
    import numpy as np
    a = np.asarray(logits.astype(mx.float32)).reshape(-1).astype(np.float64)
    b = np.asarray(reference.astype(mx.float32)).reshape(-1).astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite logits')
    top = np.argsort(-b, kind='stable')[:2]
    return {'top1': int(a.argmax()), 'reference_top1': int(top[0]),
            'reference_top2': int(top[1]), 'reference_margin': float(b[top[0]]-b[top[1]]),
            'max_abs_error': float(np.max(np.abs(a-b))),
            'rms_error': float(np.sqrt(np.mean((a-b)**2))),
            'relative_l2_error': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)),
            'sha256_float64': hashlib.sha256(a.tobytes()).hexdigest()}


def audit(args):
    out = reserve(args.output_dir)
    spec, model, tok, manifest = setup(args.model, out)
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from lab.qwen_demand_prefill import prefill
    from lab.qa_metrics import pair_score
    bases, work = workloads(tok, spec)
    save(out / 'workloads.json', work)
    records = []
    for n in spec['audit']['shape_lengths']:
        for prior in spec['audit']['prior_cache_tokens']:
            caches = {mode: make_prompt_cache(model) for mode in spec['arms']}
            if prior:
                initial = make_prompt_cache(model)
                model(mx.array([bases[1][:prior]]), cache=initial)
                mx.eval([c.state for c in initial])
                for cache in caches.values():
                    for target, source in zip(cache, initial):
                        target.state = tuple(copy.deepcopy(a) for a in source.state)
                mx.eval([c.state for cache in caches.values() for c in cache])
            x = mx.array([bases[0][:n]])
            for step in range(spec['audit']['continuation_steps'] + 1):
                row = {'length': n, 'prior_tokens': prior, 'step': step,
                       'input_sha256': digest(bases[0][:n]) if step == 0 else digest([next_token]),
                       'arms': {}}
                ref = None
                ref_cache = None
                for mode in spec['arms']:
                    trace = {}
                    logits = prefill(model, x, caches[mode], mode, trace=trace) if step == 0 else model(x, cache=caches[mode])[:, -1:, :]
                    mx.eval(logits, [c.state for c in caches[mode]])
                    if mode == 'full':
                        ref = logits
                        ref_cache = cache_record(caches[mode])
                    state = cache_record(caches[mode])
                    row['arms'][mode] = {'logits': logits_record(logits, ref), 'cache': state,
                                         'cache_equal_full': state == ref_cache, 'trace': trace}
                next_token = row['arms']['full']['logits']['top1']
                x = mx.array([[next_token]])
                records.append(row)
            save(out / 'mechanism.json', records)
            print('audit shape', n, 'prior', prior, flush=True)
    rows = [json.loads(line) for line in Path(spec['quality']['path']).read_text().splitlines()]
    assert len(rows) == spec['quality']['count'] and len({r['id'] for r in rows}) == len(rows)
    system = json.loads(Path(spec['quality']['system_path']).read_text())['system_prompt']
    quality = []
    for index, row in enumerate(rows):
        ids = tok.apply_chat_template([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"}],
            tokenize=True, add_generation_prompt=True)
        if len(ids) > spec['quality']['max_input_tokens']:
            raise ValueError('Frozen prompt over limit; truncation forbidden')
        variants = {}
        for mode in spec['arms']:
            prediction = generate(model, tok, ids, mode, spec['quality']['max_new_tokens'], stop=True)
            answer = prediction['prediction'].strip()
            if row['is_impossible']:
                em = f1 = float(answer == 'NO_ANSWER')
            elif not answer or answer == 'NO_ANSWER':
                em = f1 = 0.0
            else:
                scores = [pair_score(answer, gold) for gold in row['answers']]
                em = max(s[0] for s in scores)
                f1 = max(s[1] for s in scores)
            variants[mode] = {**prediction, 'em': em, 'f1': f1}
        quality.append({'id': row['id'], 'input_token_ids': ids, 'input_sha256': digest(ids),
                        'arms': variants})
        save(out / 'quality.json', quality)
        if (index + 1) % 10 == 0:
            print('quality', index + 1, '/', len(rows), flush=True)
    manifest.update(status='complete', scope='Consumed public functional/numerical reproduction; no timing or new quality claim')
    save(out / 'run.json', manifest)
    seal(out)
    print('audit complete', flush=True)


def worker(args):
    out = reserve(args.output_dir)
    spec, model, tok, manifest = setup(args.model, out)
    audit_manifest = checked_receipt(args.audit_root)
    if audit_manifest['upstream_source_sha256'] != manifest['upstream_source_sha256']:
        raise ValueError('Upstream source drift since audit')
    _, work = workloads(tok, spec)
    if work != json.loads((args.audit_root / 'workloads.json').read_text()):
        raise ValueError('Token workload drift')
    records = []
    manifest.update(mode=args.mode, round=args.round, records=records)
    save(out / 'run.json', manifest)
    for n in spec['benchmark']['lengths']:
        group = [r for r in work if r['length'] == n]
        for _ in range(spec['benchmark']['warmups_per_shape']):
            generate(model, tok, group[0]['token_ids'], args.mode,
                     spec['benchmark']['forced_new_tokens'], stop=False)
        for row in group:
            manifest['current_case'] = row['case_id']
            save(out / 'run.json', manifest)
            result = generate(model, tok, row['token_ids'], args.mode,
                              spec['benchmark']['forced_new_tokens'], stop=False, timed=True)
            records.append({k: v for k, v in row.items() if k != 'token_ids'} | result)
            save(out / 'run.json', manifest)
        print('bench', args.round, args.mode, n, flush=True)
    manifest.update(status='complete', mode=args.mode, round=args.round, records=records)
    save(out / 'run.json', manifest)
    seal(out)


def benchmark(args):
    out = reserve(args.output_dir)
    spec = json.loads(SPEC.read_text())
    audit_manifest = checked_receipt(args.audit_root)
    root = {'status': 'running', 'protocol': spec, 'spec_sha256': sha(SPEC),
            'audit_checksums_sha256': sha(args.audit_root / 'checksums.json'),
            'source_sha256': audit_manifest['source_sha256'], 'workers': []}
    save(out / 'run.json', root)
    start = time.monotonic()
    try:
        for round_id, order in enumerate(spec['benchmark']['order']):
            for mode in order:
                remaining = spec['benchmark']['budget_seconds'] - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Frozen compute budget exhausted')
                folder = f'{round_id}-{mode}'
                receipt = {'round': round_id, 'mode': mode, 'folder': folder, 'status': 'running'}
                root['workers'].append(receipt)
                save(out / 'run.json', root)
                command = [sys.executable, '-B', '-m', 'experiments.qwen_demand_prefill', 'worker',
                           '--model', str(args.model.resolve()), '--audit-root', str(args.audit_root.resolve()),
                           '--output-dir', str((out / folder).resolve()), '--mode', mode, '--round', str(round_id)]
                try:
                    with (out / (folder + '.log')).open('w') as log:
                        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=remaining, check=False)
                except subprocess.TimeoutExpired:
                    receipt.update(status='timeout', exit_code=None)
                    save(out / 'run.json', root)
                    raise
                receipt.update(status='complete' if done.returncode == 0 else 'failed', exit_code=done.returncode)
                save(out / 'run.json', root)
                if done.returncode:
                    raise RuntimeError(f'Worker failed: {folder}; preserve log; no retries')
                print('completed', folder, flush=True)
        root['status'] = 'complete'
    except Exception as exc:
        root.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        root['wall_seconds'] = time.monotonic() - start
        save(out / 'run.json', root)
        seal(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'benchmark', 'worker'])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--mode', choices=['full', 'head_only', 'split_last', 'final_query'])
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    if args.command in ('worker', 'benchmark') and args.audit_root is None:
        parser.error('--audit-root required')
    if args.command == 'worker' and (args.mode is None or args.round not in range(4)):
        parser.error('worker requires frozen mode and round')
    existed = args.output_dir.exists()
    try:
        globals()[args.command](args)
    except Exception as exc:
        if not existed and args.output_dir.is_dir():
            path = args.output_dir / 'run.json'
            run = json.loads(path.read_text()) if path.exists() else {}
            run.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
            save(path, run)
            seal(args.output_dir)
        raise


if __name__ == '__main__':
    main()
