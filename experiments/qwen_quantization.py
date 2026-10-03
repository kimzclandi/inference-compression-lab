"""Offline Qwen first-token diagnosis, true-FP16 block restoration and matched benchmarks.

Use the existing local FP16/Q4/Q8 exports. Never downloads or mutates input models.
The orchestration reserves outputs, records failures and uses fresh benchmark processes.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

from lab.evidence import reserve_directory
from lab.qa_metrics import evaluate
from lab.quantization_diagnostics import (read, rows, write, sha, index_by_id,
                                          rank_blocks, performance)

SPEC = Path('configs/qwen-quantization/study.json')


def utc():
    return datetime.now(timezone.utc).isoformat()


def prompt_ids(row, tok, cfg):
    messages = [{'role': 'system', 'content': cfg['system_prompt']},
                {'role': 'user', 'content': f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > cfg['max_input_tokens']:
        raise ValueError('Empty prompt or forbidden truncation')
    return ids, hashlib.sha256(prompt.encode()).hexdigest()


def generate(model, tok, ids, *, count=48, fixed=False):
    """[1,S] tokens -> [1,S,V] logits -> greedy token; subsequent inputs [1,1]."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, KVCache
    cache = make_prompt_cache(model)
    if not all(type(c) is KVCache for c in cache):
        raise ValueError('Only ordinary floating KV is covered')
    x = mx.array([ids]); mx.eval(x); mx.synchronize()
    out, times = [], []
    start = time.perf_counter()
    for _ in range(count):
        logits = model(x, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token); mx.synchronize()
        times.append(time.perf_counter())
        out.append(int(token.item()))
        if not fixed and out[-1] in tok.eos_token_ids:
            break
        x = token.reshape(1, 1)
    return {'prediction': tok.decode(out, skip_special_tokens=True), 'token_ids': out,
            'input_tokens': len(ids), 'generated_tokens': len(out),
            'input_token_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'ttft_seconds': times[0] - start, 'decode_seconds': times[-1] - times[0],
            'total_seconds': times[-1] - start,
            'stop_reason': 'fixed_length' if fixed else
                ('eos' if out[-1] in tok.eos_token_ids else 'max_new_tokens')}


def first_logits(model, ids):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    result = model(mx.array([ids]), cache=make_prompt_cache(model))[0, -1, :]
    mx.eval(result); mx.synchronize()
    return np.array(result.astype(mx.float32))


def diagnose(logits, tok, reference=None):
    """Diagnostic pass only: no host logits conversion in the performance path."""
    import numpy as np
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError('Require a finite vocabulary logit vector')
    top = np.argsort(-logits, kind='stable')[:10].tolist()
    result = {'vocab_size': len(logits), 'top1': top[0], 'top2': top[1],
              'margin': float(logits[top[0]] - logits[top[1]]),
              'top10': [{'token': i, 'text': tok.decode([i]), 'logit': float(logits[i])} for i in top]}
    ref = reference or result
    a, b = ref['top1'], ref['top2']
    # Tie-aware rank agrees with greedy's first-index argmax.
    rank = 1 + int((logits > logits[a]).sum()) + int((logits[:a] == logits[a]).sum())
    result.update(reference_top1=a, reference_top2=b, reference_top1_rank=rank,
                  reference_pair_logits=[float(logits[a]), float(logits[b])],
                  reference_pair_margin=float(logits[a] - logits[b]), fp16_margin=ref['margin'])
    return result


def model_files(path):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(path.iterdir()) if p.is_file()}


def block_bytes(model, index):
    from mlx.utils import tree_flatten
    return sum(int(v.nbytes) for _, v in tree_flatten(model.model.layers[index].parameters()))


def assert_model_layout(model, restored=None, bits=None):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    counts = {}
    for i, block in enumerate(model.model.layers):
        leaves = [(p, m) for p, m in tree_flatten(block.leaf_modules())
                  if isinstance(m, (nn.Linear, nn.QuantizedLinear))]
        if len(leaves) != 7:
            raise ValueError('Expected 7 projections per Qwen2 block')
        expected_quantized = bits is not None and i != restored
        for _, m in leaves:
            if isinstance(m, nn.QuantizedLinear) != expected_quantized:
                raise ValueError('Unexpected fallback layout')
            if expected_quantized and (m.bits != bits or m.group_size != 64):
                raise ValueError('Unexpected quantizer')
        counts[str(i)] = {'quantized': 7 if expected_quantized else 0,
                          'fp16': 0 if expected_quantized else 7,
                          'tensor_bytes': block_bytes(model, i)}
    return counts


def quality(model_path, variant, out, spec, reference=None, restored=None):
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(str(model_path), tokenizer_config={'local_files_only': True})
    cfg = read(spec['prompt']); data = rows(spec['data'])
    layout = assert_model_layout(model, restored, None if variant == 'fp16' else 8 if variant == 'q8' else 4)
    diagnostics, predictions = [], []
    for row in data:
        ids, prompt_sha = prompt_ids(row, tok, cfg)
        d = diagnose(first_logits(model, ids), tok, reference[row['id']] if reference else None)
        p = generate(model, tok, ids, count=spec['quality']['max_new_tokens'])
        if d['top1'] != p['token_ids'][0]:
            raise ValueError('Diagnostic/generation first token differs')
        p.update(id=row['id'], prompt_sha256=prompt_sha)
        d.update(id=row['id'], input_token_ids_sha256=p['input_token_ids_sha256'])
        diagnostics.append(d); predictions.append(p)
    scores, scored = evaluate(data, predictions)
    write(out / f'{variant}-quality.json', {'predictions': predictions, 'metrics': scores, 'scored': scored,
                                           'layout': layout, 'model_files': model_files(model_path)})
    write(out / f'{variant}-logits.json', diagnostics)
    print(f'QUALITY {variant}: EM={sum(r["em"] for r in scored)}/{len(scored)}', flush=True)
    del model, tok; gc.collect(); mx.clear_cache()
    return index_by_id(diagnostics)


def screen(fp16_path, q4_path, out, spec, reference):
    import mlx.core as mx
    from mlx_lm import load
    fp16, tok = load(str(fp16_path), tokenizer_config={'local_files_only': True})
    q4, _ = load(str(q4_path), tokenizer_config={'local_files_only': True})
    cfg = read(spec['prompt']); data = index_by_id(rows(spec['data'])); records = []
    costs = []
    for i in spec['screen_blocks']:
        old = q4.model.layers[i]
        costs.append({'block': i, 'q4_tensor_bytes': block_bytes(q4, i),
                      'fp16_tensor_bytes': block_bytes(fp16, i)})
        try:
            q4.model.layers[i] = fp16.model.layers[i]
            for sample_id in spec['diagnosis_ids']:
                ids, _ = prompt_ids(data[sample_id], tok, cfg)
                d = diagnose(first_logits(q4, ids), tok, reference[sample_id])
                d.update(id=sample_id, block=i); records.append(d)
        finally:
            q4.model.layers[i] = old
    ranking = rank_blocks(records, spec)
    selected = ranking[0]['block']; control = (selected + 12) % 24
    if len({c['fp16_tensor_bytes'] - c['q4_tensor_bytes'] for c in costs}) != 1:
        raise ValueError('Block restoration costs differ')
    selection = {'selected': selected, 'control': control, 'ranking': ranking, 'costs': costs,
                 'rule': spec['selection_rule'], 'scope': spec['dev_role']}
    write(out / 'screen.json', records); write(out / 'selection.json', selection)
    print(f'SELECTED block {selected}, CONTROL block {control}', flush=True)
    del q4, fp16, old, tok; gc.collect(); mx.clear_cache()
    return selection


def export_block(fp16_path, q4_path, index, destination):
    """Copy ORIGINAL FP16 tensors; dequantizing Q4 would not restore lost precision."""
    import mlx.core as mx
    import numpy as np
    out = reserve_directory(destination)
    f = mx.load(str(fp16_path / 'model.safetensors'))
    q = mx.load(str(q4_path / 'model.safetensors'))
    prefix = f'model.layers.{index}.'
    mixed = {k: v for k, v in q.items() if not k.startswith(prefix)}
    mixed.update({k: v for k, v in f.items() if k.startswith(prefix)})
    if not any(k.startswith(prefix) for k in mixed):
        raise ValueError('Block not found')
    mx.save_safetensors(str(out / 'model.safetensors'), mixed, metadata={'format': 'mlx'})
    # Use upstream per-module quantization overrides for the ordinary MLX loader.
    config = read(q4_path / 'config.json')
    for key in f:
        if key.startswith(prefix) and key.endswith('.weight') and f[key].ndim == 2:
            config['quantization'][key.removesuffix('.weight')] = False
    config['quantization_config'] = config['quantization']
    write(out / 'config.json', config)
    for p in q4_path.iterdir():
        if p.is_file() and p.name not in {'config.json', 'conversion-provenance.json'} \
                and not p.name.startswith('model.'):
            shutil.copy2(p, out / p.name)
    loaded = mx.load(str(out / 'model.safetensors'))
    if set(loaded) != set(mixed):
        raise ValueError('Export tensor keys differ')
    sources = {}
    for k, value in loaded.items():
        original = f[k] if k.startswith(prefix) else q[k]
        if value.dtype != original.dtype or value.shape != original.shape or not bool(mx.array_equal(value, original).item()):
            raise ValueError('Export changed source tensor: ' + k)
        sources[k] = {'source': 'fp16' if k.startswith(prefix) else 'q4',
                      'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': int(value.nbytes),
                      'sha256': hashlib.sha256(np.array(value).tobytes()).hexdigest()}
    result = {'block': index, 'source_tensors_verified': True, 'tensors': sources,
              'model_files': model_files(out), 'weight_bytes': (out / 'model.safetensors').stat().st_size}
    del loaded, mixed, f, q; gc.collect(); mx.clear_cache()
    return result


def benchmark(model_path, variant, out, spec, restored):
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    folder = reserve_directory(out)
    stop = threading.Event(); peak = [0]; proc = psutil.Process()
    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            stop.wait(.01)
    thread = threading.Thread(target=poll, daemon=True); thread.start()
    run = {'status': 'running', 'variant': variant, 'started': utc(), 'pid': os.getpid()}
    try:
        model, tok = load(str(model_path), tokenizer_config={'local_files_only': True})
        assert_model_layout(model, restored, None if variant == 'fp16' else 8 if variant == 'q8' else 4)
        cfg = read(spec['prompt']); prepared = []
        for row in read(spec['bench_inputs']):
            ids, ph = prompt_ids(row, tok, cfg); prepared.append((row['id'], ids, ph))
        warm = {'context': 'A queue follows first in, first out.', 'question': 'What order does a queue follow?'}
        warm_ids, _ = prompt_ids(warm, tok, cfg)
        for _ in range(spec['benchmark']['warmups']):
            generate(model, tok, warm_ids, count=32, fixed=True)
        timings = []
        for sample_id, ids, ph in prepared:
            p = generate(model, tok, ids, count=32, fixed=True)
            p.update(id=sample_id, prompt_sha256=ph); timings.append(p)
        run.update(status='complete', timings=timings, performance=performance(timings),
                   model_weight_sha256=sha(model_path / 'model.safetensors'),
                   memory={'rss_peak_sampled_bytes': peak[0], 'mlx_peak_active_bytes': mx.get_peak_memory(),
                           'mlx_cache_bytes': mx.get_cache_memory()},
                   timing=spec['benchmark']['timing'])
    except BaseException as e:
        run.update(status='failed', error=repr(e)); raise
    finally:
        stop.set(); thread.join(); run['finished'] = utc(); write(folder / 'run.json', run)


def run_study(args):
    out = reserve_directory(args.output_dir); local = reserve_directory(args.work_dir)
    spec = read(args.spec); write(out / 'protocol.json', spec)
    source_files = [Path(__file__).relative_to(Path.cwd()), Path('lab/qa_metrics.py'),
                    Path('lab/quantization_diagnostics.py'), Path('lab/evidence.py')]
    for p in source_files:
        dest = out / 'source' / p; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    for label, path in [('data.jsonl', spec['data']), ('prompt.json', spec['prompt']),
                        ('bench-inputs.json', spec['bench_inputs']), ('model-identities.json', spec['identities'])]:
        shutil.copy2(path, out / label)
    status = {'status': 'running', 'started': utc(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
              'source_sha256': {str(p): sha(p) for p in source_files},
              'protocol_sha256': sha(args.spec), 'argv': sys.argv,
              'packages': {n: importlib.metadata.version(n) for n in ['mlx', 'mlx-lm', 'numpy', 'transformers', 'psutil']},
              'python': platform.python_version(), 'os': platform.platform()}
    write(out / 'started.json', status)
    try:
        import mlx.core as mx
        mx.random.seed(spec['seed']); status['device'] = mx.metal.device_info()
        paths = {v: args.model_root / f'student-{v}' for v in ['fp16', 'q4', 'q8']}
        identities = read(spec['identities'])
        for variant, path in paths.items():
            if not path.is_dir():
                raise ValueError('Missing local model: ' + str(path))
            for name, meta in identities[variant].items():
                if sha(path / name) != meta['sha256'] or (path / name).stat().st_size != meta['bytes']:
                    raise ValueError('Model identity mismatch: ' + variant + '/' + name)
        ref = quality(paths['fp16'], 'fp16', out, spec)
        for variant in ['q4', 'q8']:
            quality(paths[variant], variant, out, spec, ref)
        # Exact historical replay is an acceptance check, not a quality target.
        replay = {}
        for variant in ['fp16', 'q4', 'q8']:
            old = index_by_id(rows(Path(spec['historical']) / f'{variant}-dev.predictions.jsonl'))
            new = read(out / f'{variant}-quality.json')['predictions']
            replay[variant] = {'n': len(new), 'token_mismatches': [r['id'] for r in new if r['token_ids'] != old[r['id']]['token_ids']],
                               'prompt_mismatches': [r['id'] for r in new if r['input_token_ids_sha256'] != old[r['id']]['input_token_ids_sha256']]}
        write(out / 'historical-replay.json', replay)
        if any(r['token_mismatches'] or r['prompt_mismatches'] for r in replay.values()):
            raise ValueError('Historical replay drift: stop before selecting blocks')
        selection = screen(paths['fp16'], paths['q4'], out, spec, ref)
        for variant in ['selected', 'control']:
            paths[variant] = local / variant
            export = export_block(paths['fp16'], paths['q4'], selection[variant], paths[variant])
            write(out / f'{variant}-export.json', export)
            quality(paths[variant], variant, out, spec, ref, selection[variant])
        commands = []
        variants = spec['variants']
        for r in range(spec['benchmark']['rounds']):
            order = variants[r % len(variants):] + variants[:r % len(variants)]
            for variant in order:
                cell = f'bench-{r}-{variant}'
                argv = [sys.executable, '-m', 'experiments.qwen_quantization', 'bench', '--spec', str(args.spec),
                        '--model', str(paths[variant]), '--variant', variant, '--output-dir', str(out / cell)]
                if variant in selection:
                    argv += ['--restored', str(selection[variant])]
                entry = {'round': r, 'variant': variant, 'argv': argv, 'started': utc()}
                commands.append(entry)
                with (out / f'{cell}.log').open('x') as log:
                    completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
                entry.update(returncode=completed.returncode, finished=utc())
                with (out / 'commands.jsonl').open('a') as f:
                    f.write(json.dumps(entry) + '\n')
                if completed.returncode:
                    raise RuntimeError('Benchmark failed: ' + cell)
                print(cell + ' complete', flush=True)
        status['status'] = 'complete'
    except BaseException as e:
        status.update(status='failed', error=repr(e)); raise
    finally:
        status['finished'] = utc(); write(out / 'run.json', status)
        write(out / 'checksums.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                                      if p.is_file() and p.name != 'checksums.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['study', 'bench'])
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--variant')
    parser.add_argument('--restored', type=int)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if args.mode == 'study':
        if args.model_root is None or args.work_dir is None:
            parser.error('study requires --model-root and --work-dir')
        run_study(args)
    else:
        benchmark(args.model, args.variant, args.output_dir, read(args.spec), args.restored)


if __name__ == '__main__':
    main()
