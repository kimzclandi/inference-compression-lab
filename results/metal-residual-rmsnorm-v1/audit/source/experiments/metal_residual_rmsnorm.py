"""One frozen study of a real Metal residual-add + RMSNorm fusion.

Existing offline assets only; all modes share the same full head-only decoder.
Raw receipts survive failure, and output directories cannot be reused.
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

from experiments.qwen_demand_prefill import save, sha, digest, reserve, seal, workloads, cache_record, logits_record

SPEC = Path('configs/metal-residual-rmsnorm/study.json')
SOURCES = ['experiments/metal_residual_rmsnorm.py', 'experiments/qwen_demand_prefill.py',
           'experiments/__init__.py', 'lab/metal_residual_rmsnorm.py',
           'lab/kernels/residual_rmsnorm.metal', 'lab/qwen_demand_prefill.py',
           'lab/qa_metrics.py', 'lab/__init__.py', 'third_party/MLX-MIT.txt']


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
    if manifest['git_status']:
        raise ValueError('Freeze a clean implementation commit before audit/timing')
    manifest['runtime_binary_sha256'] = {str(p.relative_to(Path(mx.__file__).parent)): sha(p)
        for p in Path(mx.__file__).parent.rglob('*')
        if p.is_file() and p.suffix in ('.so', '.dylib', '.metallib')}
    save(out / 'run.json', manifest)
    return spec, model, tok, manifest


def generate(model, tok, ids, mode, count, *, stop, timed=False):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from lab.metal_residual_rmsnorm import qwen_forward
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
        logits = qwen_forward(model, x, cache, mode)
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


def array_hash(a):
    import numpy as np
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def numerical_audit(spec, out):
    import mlx.core as mx
    import numpy as np
    from lab.metal_residual_rmsnorm import residual_rmsnorm
    rng = np.random.default_rng(spec['audit']['seed'])
    records = []
    for dtype in spec['audit']['dtypes']:
        dt = getattr(mx, dtype)
        npdt = getattr(np, dtype)
        atol = spec['audit']['fp16_atol' if dtype == 'float16' else 'fp32_atol']
        rtol = spec['audit']['fp16_rtol' if dtype == 'float16' else 'fp32_rtol']
        for width in spec['audit']['widths']:
            for rows in spec['audit']['rows']:
                for pattern in spec['audit']['patterns']:
                    size = width * (2 if pattern == 'strided' else 1)
                    x = rng.standard_normal((rows, size))
                    r = rng.standard_normal((rows, size))
                    w = rng.uniform(.5, 1.5, size)
                    if pattern == 'zero':
                        x[:] = 0; r[:] = 0
                    elif pattern == 'cancellation':
                        r = -x
                    elif pattern == 'scaled':
                        x *= 100; r *= 100
                    inputs = [mx.array(a.astype(npdt), dtype=dt) for a in (x,r,w)]
                    if pattern == 'strided':
                        inputs = [a[..., ::2] for a in inputs]
                    mx.eval(inputs)
                    row = {'dtype': dtype, 'width': width, 'rows': rows, 'pattern': pattern,
                           'input_sha256': [array_hash(a) for a in inputs], 'arms': {}}
                    reference = None
                    for mode in spec['arms']:
                        h, y = residual_rmsnorm(*inputs, 1e-6, mode=mode)
                        mx.eval(h,y)
                        ha, ya = np.asarray(h), np.asarray(y)
                        if mode == 'native':
                            reference = (ha.copy(), ya.copy())
                        rh, ry = reference
                        err = np.abs(ya.astype(np.float64)-ry.astype(np.float64))
                        limit = atol + rtol*np.abs(ry.astype(np.float64))
                        finite = bool(np.isfinite(ha).all() and np.isfinite(ya).all())
                        if not finite:
                            raise ValueError(f'Nonfinite primitive output: {dtype}/{width}/{pattern}/{mode}')
                        def ordered(a):
                            bits = a.view(np.uint16 if dtype == 'float16' else np.uint32).astype(np.int64)
                            sign = 1 << (15 if dtype == 'float16' else 31)
                            return np.where(bits & sign, sign-(bits & (sign-1)), sign+bits)
                        ratio = float(np.max(err/limit))
                        row['arms'][mode] = {'finite': finite,
                            'residual_exact': ha.tobytes() == rh.tobytes(),
                            'within_tolerance': ratio <= 1., 'max_tolerance_ratio': ratio,
                            'max_abs_error': float(err.max()),
                            'max_ulp_error': int(np.max(np.abs(ordered(ya)-ordered(ry)))),
                            'residual_sha256': array_hash(h), 'output_sha256': array_hash(y)}
                    records.append(row)
        save(out/'numerical.json', records)
        print('numerical', dtype, len(records), flush=True)
    # DOT describes MLX primitives, not hardware dispatch timings/counters.
    x, r, w = mx.ones((1,896), dtype=mx.float16), mx.full((1,896), .01, dtype=mx.float16), mx.ones((896,), dtype=mx.float16)
    mx.eval(x,r,w)
    for mode in spec['arms']:
        result = residual_rmsnorm(x,r,w,1e-6,mode=mode)
        mx.export_to_dot(str(out/(mode+'.dot')), *result)
        mx.eval(result)


def audit(args):
    out = reserve(args.output_dir)
    spec, model, tok, manifest = setup(args.model, out)
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from lab.metal_residual_rmsnorm import qwen_forward
    from lab.qwen_demand_prefill import prefill
    from lab.qa_metrics import pair_score
    numerical_audit(spec,out)
    bases, work = workloads(tok,spec)
    save(out/'workloads.json',work)
    records=[]
    for n in spec['audit']['model_lengths']:
        for prior in spec['audit']['prior_cache_tokens']:
            caches={mode:make_prompt_cache(model) for mode in spec['arms']+['upstream']}
            if prior:
                initial=make_prompt_cache(model)
                model(mx.array([bases[1][:prior]]),cache=initial)
                mx.eval([c.state for c in initial])
                for cache in caches.values():
                    for target,source in zip(cache,initial):
                        target.state=tuple(copy.deepcopy(a) for a in source.state)
                mx.eval([c.state for cache in caches.values() for c in cache])
            ids=bases[0][:n]
            for step in range(spec['audit']['continuation_steps']+1):
                x=mx.array([ids])
                row={'length':n,'prior_tokens':prior,'step':step,
                     'input_token_ids':ids,'input_sha256':digest(ids),'arms':{}}
                ref=None; ref_cache=None
                for mode in spec['arms']:
                    trace={}
                    logits=qwen_forward(model,x,caches[mode],mode,trace=trace)
                    mx.eval(logits,[c.state for c in caches[mode]])
                    state=cache_record(caches[mode])
                    if mode=='native':
                        ref=logits; ref_cache=state
                    row['arms'][mode]={'logits':logits_record(logits,ref),'cache':state,
                                     'cache_equal_native':state==ref_cache,'trace':trace}
                original=prefill(model,x,caches['upstream'],'head_only')
                mx.eval(original,[c.state for c in caches['upstream']])
                row['upstream']={'logits':logits_record(original,ref),'cache':cache_record(caches['upstream'])}
                row['upstream_native_equal']=(array_hash(original)==array_hash(ref)
                    and cache_record(caches['upstream'])==ref_cache)
                ids=[row['arms']['native']['logits']['top1']]
                records.append(row)
            save(out/'mechanism.json',records)
            print('mechanism',n,prior,flush=True)
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

def micro_benchmark(spec, mode, out, manifest):
    import mlx.core as mx
    import numpy as np
    from lab.metal_residual_rmsnorm import residual_rmsnorm
    cfg=spec['benchmark']; records=[]
    manifest['micro']=records
    for rows in cfg['micro_rows']:
        rng=np.random.default_rng(cfg['micro_inputs']['seed']+rows)
        x0=mx.array(rng.normal(0,1,(rows,cfg['micro_width'])).astype(np.float16))
        r=mx.array(rng.normal(0,.01,x0.shape).astype(np.float16))
        w=mx.array(rng.uniform(.95,1.05,cfg['micro_width']).astype(np.float16))
        mx.eval(x0,r,w); mx.synchronize()
        row={'rows':rows,'width':cfg['micro_width'],'input_sha256':[array_hash(a) for a in (x0,r,w)],
             'batch_calls':cfg['micro_batch_calls'],'samples':[]}
        records.append(row)
        save(out/'run.json',manifest)
        def chain():
            x=x0
            for _ in range(cfg['micro_batch_calls']):
                h,x=residual_rmsnorm(x,r,w,cfg['micro_inputs']['epsilon'],mode=mode)
            mx.eval(h,x); mx.synchronize()
        for _ in range(cfg['micro_warmups']):
            chain()
        for _ in range(cfg['micro_samples']):
            start=time.perf_counter(); chain(); elapsed=time.perf_counter()-start
            row['samples'].append({'total_seconds':elapsed,'per_call_seconds':elapsed/cfg['micro_batch_calls']})
            save(out/'run.json',manifest)
    return records


def worker(args):
    out = reserve(args.output_dir)
    spec, model, tok, manifest = setup(args.model, out)
    audit_manifest = checked_receipt(args.audit_root)
    if audit_manifest['upstream_source_sha256'] != manifest['upstream_source_sha256']:
        raise ValueError('Upstream source drift since audit')
    if audit_manifest['runtime_binary_sha256'] != manifest['runtime_binary_sha256']:
        raise ValueError('Runtime binary drift')
    _, work = workloads(tok, spec)
    if work != json.loads((args.audit_root / 'workloads.json').read_text()):
        raise ValueError('Token workload drift')
    records = []
    manifest.update(mode=args.mode, round=args.round)
    micro_benchmark(spec, args.mode, out, manifest)
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
                command = [sys.executable, '-B', '-m', 'experiments.metal_residual_rmsnorm', 'worker',
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
    parser.add_argument('--mode', choices=['native', 'compiled', 'metal'])
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    if args.command in ('worker', 'benchmark') and args.audit_root is None:
        parser.error('--audit-root required')
    if args.command == 'worker' and (args.mode is None or args.round not in range(3)):
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
