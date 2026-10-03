"""Recompute coverage, token parity, phase accounting and medians without MLX/Git."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(path.read_text())


def verify(folder):
    folder = Path(folder)
    manifest = read(folder/'manifest.json'); spec = manifest['spec']
    if (folder/'complete.json').exists():
        for name, digest in read(folder/'complete.json')['sha256'].items():
            require(hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, f'Checksum: {name}')
    fingerprint = hashlib.sha256(json.dumps(manifest['model_files_sha256'], sort_keys=True).encode()).hexdigest()
    require(fingerprint == manifest['model_fingerprint'], 'Model fingerprint')
    records = read(folder/'timings.json')
    keys = ('dataset', 'prefix_tokens', 'profile', 'round', 'mode')
    keyed = {tuple(r[k] for k in keys): r for r in records}
    expected = set(itertools.product(spec['datasets'], spec['prefix_lengths'], spec['profile'],
                                    range(spec['rounds']), spec['modes']))
    require(len(keyed) == len(records) and set(keyed) == expected, 'Missing/duplicate cells')
    works = read(folder/'workloads.json')
    workloads = {(w['dataset'], w['prefix_tokens']): w for w in works}
    require(len(workloads) == len(works) == len(spec['datasets'])*len(spec['prefix_lengths']), 'Workload coverage')
    preparations = read(folder/'hot-preparations.json')
    require(len(preparations) == len(records)//len(spec['modes']), 'Hot preparation coverage')
    require({tuple(b[k] for k in keys) for b in preparations} == {k for k in expected if k[-1]=='warm_hit'}, 'Hot preparation cells')
    for b in preparations:
        require(b['status']=='miss' and b['seconds'] > 0, 'Preparation must include miss')
    statuses = {'full': ['disabled']*spec['requests'], 'legacy_segmented': ['rebuilt']*spec['requests'],
                'direct_segmented': ['direct']*spec['requests'], 'cached_workload': ['miss']+['hit']*(spec['requests']-1),
                'warm_hit': ['hit']*spec['requests']}
    for r in records:
        outputs = r['outputs']; w = workloads[r['dataset'], r['prefix_tokens']]
        require(len(outputs) == len(w['requests']) == spec['requests'], 'Request count')
        require(len(w['prefix']) == r['prefix_tokens'], 'Prefix length')
        require([o['status'] for o in outputs] == statuses[r['mode']], 'Statuses')
        reference = keyed[r['dataset'], r['prefix_tokens'], False, r['round'], 'full']['outputs']
        require(r['seconds'] >= sum(o['total_seconds'] for o in outputs), 'Wall timer coverage')
        for i, o in enumerate(outputs):
            require(w['requests'][i][:len(w['prefix'])] == w['prefix'], 'Exact prefix')
            require(o['input_tokens'] == len(w['requests'][i]), 'Input length')
            require(o['prefix_tokens'] == (0 if r['mode']=='full' else r['prefix_tokens']), 'Mode prefix')
            require(o['suffix_tokens'] == o['input_tokens']-o['prefix_tokens'], 'Suffix length')
            require(o['token_ids'] == reference[i]['token_ids'], 'Greedy token mismatch')
            require(len(o['token_ids']) == spec['generated_tokens'] and o['stop_reason']=='token_limit', 'Forced output length')
            require(0 < o['ttft_seconds'] <= o['total_seconds'], 'Positive TTFT')
            require(math.isclose(o['total_seconds'], o['ttft_seconds']+o['decode_seconds']), 'Decode accounting')
            require(math.isclose(o['decode_tokens_per_second'], (spec['generated_tokens']-1)/o['decode_seconds']), 'Decode denominator')
            expected_entries = int(r['mode'] in ('cached_workload','warm_hit'))
            require(o['cache']['entries'] == expected_entries, 'Cache residency')
            logical_bytes = 2*manifest['model_config']['num_hidden_layers']*manifest['model_config']['num_key_value_heads']*r['prefix_tokens']*64*2
            require(o['cache']['stored_tensor_bytes'] == expected_entries*logical_bytes, 'Logical KV bytes')
            require(o['cache']['stored_tensor_bytes'] <= spec['max_bytes'], 'Byte budget')
            phases = o['phases']
            require((phases is not None) == r['profile'], 'Profiling flag')
            if phases is not None:
                require(all(math.isfinite(v) and v >= 0 for v in phases.values()), 'Invalid phase')
                require(math.isclose(sum(phases.values()), o['total_seconds'], abs_tol=1e-9), 'Phase reconciliation')
                if r['mode'] in ('full', 'direct_segmented'):
                    require(phases['snapshot_copy_seconds'] == phases['request_clone_seconds'] == phases['lookup_seconds'] == 0, 'Direct path copied/queried')
                if o['status']=='hit':
                    require(phases['prefix_prefill_seconds'] == phases['snapshot_copy_seconds'] == 0, 'Hit recomputed')
    summary = []
    for dataset, length, profile in itertools.product(spec['datasets'],spec['prefix_lengths'],spec['profile']):
        cell = [r for r in records if (r['dataset'],r['prefix_tokens'],r['profile'])==(dataset,length,profile)]
        med = {mode: statistics.median(r['seconds'] for r in cell if r['mode']==mode) for mode in spec['modes']}
        phases = {}
        for mode in spec['modes']:
            outputs = [o for r in cell if r['mode']==mode for o in r['outputs']]
            phases[mode] = {'ttft_median_seconds': statistics.median(o['ttft_seconds'] for o in outputs),
                            'decode_tps_median': statistics.median(o['decode_tokens_per_second'] for o in outputs)}
            if profile:
                phases[mode]['by_status'] = {status: {k: statistics.median(o['phases'][k] for o in outputs if o['status']==status)
                    for k in outputs[0]['phases']} for status in {o['status'] for o in outputs}}
        summary.append({'dataset':dataset,'prefix_tokens':length,'profile':profile,'four_request_median_seconds':med,
                        'cached_reduction_vs_direct':1-med['cached_workload']/med['direct_segmented'],
                        'cached_reduction_vs_legacy':1-med['cached_workload']/med['legacy_segmented'],
                        'direct_reduction_vs_legacy':1-med['direct_segmented']/med['legacy_segmented'],
                        'measurements':phases})
    return {'acceptance': {'all_cells_present':True,'all_tokens_match_full':True,'phases_reconcile':True},
            'record_count':len(records),'request_count':len(records)*spec['requests'],'performance':summary}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder',type=Path); args=p.parse_args()
    summary=verify(args.folder)
    if (args.folder/'summary.json').exists(): require(summary == read(args.folder/'summary.json'), 'Saved summary differs')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
