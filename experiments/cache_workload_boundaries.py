"""Fixed synthetic real-model workload matrix, separate from frozen lifecycle study."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

from experiments.metal_mechanism_diagnostic import sha, save
from lab.prefix_cache import PrefixCache

SPEC = Path('configs/cache-workload-boundaries-v1.json')
SOURCES = ['experiments/cache_workload_boundaries.py', 'experiments/metal_mechanism_diagnostic.py',
           'lab/qwen_prefix.py', 'lab/prefix_cache.py', str(SPEC)]


def simulate(trace, capacity, byte_budget):
    cache = PrefixCache('symbolic', lambda x: x, max_entries=capacity, max_bytes=byte_budget)
    statuses = [cache.acquire([token], lambda t: (t, 1))[1] for token in trace]
    return statuses, cache.stats()


def summarize(records, spec):
    expected = {(length, name, n, arm) for length in spec['prefix_lengths']
                for name in spec['traces'] for n in range(spec['rounds']) for arm in spec['arms']}
    cells = {}
    for row in records:
        key = (row['length'], row['trace'], row['round'], row['arm'])
        if key not in expected or key in cells:
            raise ValueError('Duplicate or unexpected cell')
        outputs = row['outputs']
        trace = spec['traces'][row['trace']]
        statuses, stats = simulate(trace, spec['max_entries'], spec['budget_prefixes'][row['trace']])
        if stats['hits'] != spec['expected_hits'][row['trace']]:
            raise ValueError('Protocol hit prediction differs from simulation')
        if len(outputs) != len(trace):
            raise ValueError('Incomplete request coverage')
        for index, output in enumerate(outputs):
            wanted = statuses[index] if row['arm'] == 'cache' else 'direct'
            if output['status'] != wanted:
                raise ValueError('Unexpected cache classification')
            if len(output['token_ids']) != spec['generated_tokens'] or any(
                    type(t) is not int or t < 0 for t in output['token_ids']):
                raise ValueError('Invalid generated tokens')
            for field in ('total_seconds', 'ttft_seconds'):
                value = output[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid timing')
            if output['total_seconds'] < output['ttft_seconds']:
                raise ValueError('Invalid phase ordering')
        if row['arm'] == 'cache':
            for field in ('hits', 'misses', 'evictions', 'bypasses', 'failures'):
                if row['delta'][field] != stats[field]:
                    raise ValueError('Cache counter differs from simulation')
        cells[key] = row
    if set(cells) != expected:
        raise ValueError('Missing timing cells')
    summaries = []
    for length in spec['prefix_lengths']:
        for name in spec['traces']:
            totals = {arm: [] for arm in spec['arms']}
            for n in range(spec['rounds']):
                pair = [cells[length, name, n, arm] for arm in spec['arms']]
                if [o['token_ids'] for o in pair[0]['outputs']] != [o['token_ids'] for o in pair[1]['outputs']]:
                    raise ValueError('Cache/direct token mismatch')
                for arm in spec['arms']:
                    totals[arm].append(sum(o['total_seconds'] for o in cells[length, name, n, arm]['outputs']))
            medians = {a: statistics.median(v) for a, v in totals.items()}
            speedup = medians['direct'] / medians['cache']
            faster = sum(c < d for c, d in zip(totals['cache'], totals['direct']))
            summaries.append(dict(prefix_tokens=length, trace=name,
                hits=spec['expected_hits'][name], requests=len(spec['traces'][name]),
                median_sum_request_seconds=medians, direct_over_cache=speedup,
                faster_rounds=faster, speed_gate=speedup >= spec['minimum_speedup'] and
                faster >= spec['minimum_faster_rounds']))
    return dict(scope=spec['scope'], cells=summaries,
                compared_request_pairs=len(spec['prefix_lengths']) * len(spec['traces']) * spec['rounds'] * 12,
                token_parity=True, service_latency_claim=False)


def run(output, model_path):
    from lab.qwen_prefix import QwenPrefixRuntime
    from experiments.qwen_prefix_study import identity
    spec = json.loads(SPEC.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Require clean committed implementation')
    files, fingerprint = identity(model_path)
    if files != spec['model_files_sha256']:
        raise ValueError('Model identity mismatch')
    output.mkdir(parents=True, exist_ok=False)
    source = {}
    for path in SOURCES:
        dest = output / 'source' / path; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes()); source[path] = sha(path)
    save(output / 'protocol.json', spec)
    manifest = dict(status='running', git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=source, model_fingerprint=fingerprint, model_files_sha256=files)
    save(output / 'run.json', manifest)
    try:
        import mlx.core as mx
        from mlx_lm import load
        versions = {p: importlib.metadata.version(p) for p in spec['versions']}
        if versions != spec['versions'] or not mx.metal.is_available():
            raise ValueError('Require fixed versions and Metal')
        mx.set_default_device(mx.gpu)
        manifest.update(device=mx.metal.device_info(), versions=versions)
        model, tok = load(str(model_path)); mx.eval(model.parameters()); mx.synchronize()
        base = tok.encode('The robot inspection log contains maintenance and sensor records. ' * 300, add_special_tokens=False)
        suffix = tok.encode('\nSummarize:', add_special_tokens=False)
        records, workloads, probes = [], [], []
        rng = random.Random(spec['seed'])
        for length in spec['prefix_lengths']:
            prefixes = [base[:length-1] + [20+i] for i in range(12)]
            if len(base) < length or not suffix:
                raise ValueError('Insufficient tokenized workload')
            workloads.append(dict(length=length, prefixes=prefixes, suffix=suffix))
            for name, trace in spec['traces'].items():
                rt = QwenPrefixRuntime(model, fingerprint, max_entries=spec['max_entries'],
                    max_bytes=length * spec['bytes_per_prefix_token'] * spec['budget_prefixes'][name])
                def generate(index, arm, profile=False):
                    p = prefixes[index]
                    return rt.generate(p+suffix, prefix=p, max_new_tokens=spec['generated_tokens'],
                        stop_at_eos=False, reuse_prefix=arm=='cache', segmented_snapshot=False, profile=profile)
                for arm in spec['arms']:
                    for _ in range(spec['warmups']):
                        for index in sorted(set(trace)):
                            generate(index, arm)
                for n in range(spec['rounds']):
                    order = list(spec['arms']); rng.shuffle(order)
                    for arm in order:
                        rt.store.clear(); before = rt.store.stats()
                        outputs = [generate(i, arm) for i in trace]
                        after = rt.store.stats()
                        records.append(dict(length=length, trace=name, round=n, arm=arm, arm_order=order,
                            outputs=outputs, delta={k: after[k]-before[k] for k in ('hits','misses','evictions','bypasses','failures')}))
                        rt.store.clear()
                        save(output / 'timings.json', records)
                print(length, name, 'complete', flush=True)
            # Separate instrumented cold/hit/direct probes; excluded from the matrix.
            rt = QwenPrefixRuntime(model, fingerprint, max_entries=3, max_bytes=length*spec['bytes_per_prefix_token']*3)
            probe = {'length': length}
            for label, arm in [('miss', 'cache'), ('hit', 'cache'), ('direct', 'direct')]:
                probe[label] = rt.generate(prefixes[0]+suffix, prefix=prefixes[0],
                    max_new_tokens=spec['generated_tokens'], stop_at_eos=False,
                    reuse_prefix=arm=='cache', segmented_snapshot=False, profile=True)
            probes.append(probe); rt.store.clear()
        save(output / 'workloads.json', workloads); save(output / 'phase-probes.json', probes)
        save(output / 'summary.json', summarize(records, spec)); manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc)); raise
    finally:
        save(output / 'run.json', manifest)
        save(output / 'checksums.json', {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name != 'checksums.json'})


def verify(output):
    actual = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    if actual != json.loads((output / 'checksums.json').read_text()):
        raise ValueError('Evidence hash mismatch')
    spec = json.loads((output / 'protocol.json').read_text())
    if spec != json.loads(SPEC.read_text()):
        raise ValueError('Protocol drift')
    manifest = json.loads((output / 'run.json').read_text())
    if manifest['status'] != 'complete' or manifest['model_files_sha256'] != spec['model_files_sha256']:
        raise ValueError('Incomplete or wrong-model evidence')
    for path, expected in manifest['source_sha256'].items():
        if sha(output / 'source' / path) != expected:
            raise ValueError('Source archive drift')
    result = summarize(json.loads((output / 'timings.json').read_text()), spec)
    if result != json.loads((output / 'summary.json').read_text()):
        raise ValueError('Summary does not reproduce')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        if args.model is None:
            parser.error('--model is required to run')
        run(args.output, args.model)
    else:
        print(json.dumps(verify(args.output), indent=2))
