"""Controlled locality versus cache-thrashing trace; not production traffic."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time
from lab.evidence import reserve_directory, source_record
from lab.qwen_prefix import QwenPrefixRuntime
from experiments.qwen_prefix_study import identity, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.model.is_dir(): raise ValueError('Local model required')
    out = reserve_directory(args.output_dir)
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(str(args.model))
    mx.eval(model.parameters()); mx.synchronize()
    files, fingerprint = identity(args.model)
    rt = QwenPrefixRuntime(model, fingerprint, max_entries=2)
    base = tok.encode('The robot inspection log contains maintenance and sensor records. '*300,
                      add_special_tokens=False)
    suffix = tok.encode('\nSummarize:', add_special_tokens=False)
    spec = {'prefix_lengths': [64, 1024], 'rounds': 5, 'requests': 12, 'generated_tokens': 8,
            'max_entries': 2, 'seed': 20261003,
            'traces': {'hot': [0]*12, 'cyclic_three': [0,1,2]*4}}
    save(out/'manifest.json', {'source': source_record(), 'spec': spec,
                              'model_fingerprint': fingerprint, 'model_files_sha256': files,
                              'device': mx.metal.device_info()})
    records, workloads = [], []
    rng = random.Random(spec['seed'])
    for length in spec['prefix_lengths']:
        prefixes = [base[:length-1]+[token] for token in (20,21,22)]
        workloads.append({'prefix_tokens': length, 'prefixes': prefixes, 'suffix': suffix})
        for prefix in prefixes:
            for _ in range(2):
                rt.generate(prefix+suffix, prefix=prefix, max_new_tokens=8, stop_at_eos=False)
        for scenario, trace in spec['traces'].items():
            for round_id in range(spec['rounds']):
                modes = ['segmented_no_reuse', 'cached']
                rng.shuffle(modes)
                for mode in modes:
                    rt.store.clear()
                    before = rt.store.stats()
                    mx.synchronize(); start = time.perf_counter()
                    outputs = [rt.generate(prefixes[i]+suffix, prefix=prefixes[i],
                                           max_new_tokens=8, stop_at_eos=False,
                                           reuse_prefix=mode=='cached') for i in trace]
                    mx.synchronize(); elapsed = time.perf_counter()-start
                    after = rt.store.stats()
                    records.append({'prefix_tokens': length, 'scenario': scenario, 'round': round_id,
                                    'mode': mode, 'seconds': elapsed, 'outputs': outputs,
                                    'cache_delta': {k: after[k]-before[k] for k in
                                                    ['hits','misses','evictions','bypasses']}})
                save(out/'timings.json', records)
            print('trace', length, scenario, flush=True)
    summary = []
    for length in spec['prefix_lengths']:
        for scenario in spec['traces']:
            match = [r for r in records if r['prefix_tokens']==length and r['scenario']==scenario]
            base = [r for r in match if r['mode']=='segmented_no_reuse']
            cached = [r for r in match if r['mode']=='cached']
            baseline_time = statistics.median(r['seconds'] for r in base)
            cached_time = statistics.median(r['seconds'] for r in cached)
            summary.append({'prefix_tokens': length, 'scenario': scenario,
                            'segmented_median_seconds': baseline_time, 'cached_median_seconds': cached_time,
                            'time_reduction': 1-cached_time/baseline_time,
                            'hits_per_round': [r['cache_delta']['hits'] for r in cached],
                            'evictions_per_round': [r['cache_delta']['evictions'] for r in cached],
                            'token_parity': all(a['token_ids']==b['token_ids'] for r,s in zip(base,cached)
                                                for a,b in zip(r['outputs'],s['outputs']))})
    save(out/'workloads.json', workloads)
    save(out/'summary.json', summary)
    from lab.evidence import sha256
    save(out/'complete.json', {'sha256': {p.name: sha256(p) for p in sorted(out.glob('*.json'))}})


if __name__ == '__main__': main()
