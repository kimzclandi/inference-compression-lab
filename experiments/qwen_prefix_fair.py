"""Fair segmented baseline and separate synchronized cost diagnostics (Apple GPU)."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

from lab.evidence import reserve_directory, sha256, source_record
from lab.qwen_prefix import QwenPrefixRuntime
from experiments.qwen_prefix_study import identity, save
from experiments.verify_qwen_prefix_fair import verify


def run(rt, tokens, prefix, mode, count, profile):
    return rt.generate(tokens, prefix=() if mode == 'full' else prefix,
                       max_new_tokens=count, stop_at_eos=False,
                       reuse_prefix=mode not in ('legacy_segmented', 'direct_segmented'),
                       segmented_snapshot=mode != 'direct_segmented', profile=profile)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=Path('configs/qwen-prefix/fair-baseline.json'))
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    out = reserve_directory(args.output_dir)
    import mlx.core as mx
    from mlx_lm import load
    if not mx.metal.is_available():
        raise RuntimeError('Apple GPU required; CPU fallback forbidden')
    files, fingerprint = identity(args.model)
    config = json.loads((args.model/'config.json').read_text())
    if config.get('model_type') != 'qwen2' or config.get('quantization', {}).get('bits') != 8:
        raise ValueError('Prepared Qwen2 Q8 required')
    model, tok = load(str(args.model))
    mx.eval(model.parameters()); mx.synchronize()
    rt = QwenPrefixRuntime(model, fingerprint, max_entries=spec['max_entries'], max_bytes=spec['max_bytes'])
    historical = Path('results/qwen-prefix-v2/workloads.json')
    old = json.loads(historical.read_text())
    workloads = [{'dataset': 'historical', **w} for w in old if w['prefix_tokens'] in spec['prefix_lengths']]
    body = 'The observatory records weather, telescope calibration, star coordinates and exposure schedules. '*600
    base = tok.apply_chat_template([{'role': 'user', 'content': body}], tokenize=True, add_generation_prompt=True)
    for length in spec['prefix_lengths']:
        prefix = base[:length]
        requests = [prefix + tok.encode(f'\nTask {i}: List the calibration steps and explain each step.\nResponse:',
                                       add_special_tokens=False) for i in range(spec['requests'])]
        workloads.append({'dataset': 'confirmation', 'prefix_tokens': length, 'prefix': prefix, 'requests': requests})
    workloads = [w for w in workloads if w['dataset'] in spec['datasets']]
    save(out/'workloads.json', workloads)
    backend = Path(importlib.import_module('mlx_lm.models.cache').__file__)
    save(out/'manifest.json', {'spec': spec, 'spec_sha256': sha256(args.spec), 'source': source_record(),
        'historical_workloads_sha256': sha256(historical), 'model_files_sha256': files,
        'model_fingerprint': fingerprint, 'model_config': config,
        'environment': {'python': sys.version, 'platform': platform.platform(), 'device': mx.metal.device_info(),
            'versions': {p: importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','tokenizers']},
            'backend_cache_sha256': sha256(backend)}})
    rng = random.Random(spec['seed'])
    records, builds = [], []
    for work in workloads:
        prefix = work['prefix']
        for profile in spec['profile']:
            for mode in spec['modes']:
                for _ in range(spec['warmup_per_mode']):
                    rt.store.clear()
                    if mode == 'warm_hit': rt.prepare(prefix)
                    run(rt, work['requests'][0], prefix, mode, spec['generated_tokens'], profile)
            for round_id in range(spec['rounds']):
                order = list(spec['modes']); rng.shuffle(order)
                for mode in order:
                    rt.store.clear()
                    tag = {'dataset': work['dataset'], 'prefix_tokens': len(prefix),
                           'profile': profile, 'round': round_id, 'mode': mode}
                    if mode == 'warm_hit': builds.append({**tag, **rt.prepare(prefix)})
                    mx.synchronize(); start = time.perf_counter()
                    outputs = [run(rt, tokens, prefix, mode, spec['generated_tokens'], profile)
                               for tokens in work['requests']]
                    mx.synchronize()
                    records.append({**tag, 'seconds': time.perf_counter()-start, 'outputs': outputs})
                save(out/'timings.json', records)
                save(out/'hot-preparations.json', builds)
            print(work['dataset'], len(prefix), 'profile', profile, 'done', flush=True)
    save(out/'summary.json', verify(out))
    save(out/'complete.json', {'sha256': {p.name: sha256(p) for p in sorted(out.glob('*.json'))}})


if __name__ == '__main__': main()
