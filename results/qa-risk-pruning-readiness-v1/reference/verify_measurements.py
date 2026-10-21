"""Independently recompute saved timing arithmetic, coverage and artifact bindings.

No model execution or latency measurement; does not establish another host's speed.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.artifact_integrity import verify_hashes

PROTOCOL_SHA = '4efa6b19d945b475c3244a1e456b06207c00dab7241f595b7baf9904849b34ef'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify():
    folder = ROOT / 'results/qa-risk-pruning-v1'
    protocol = ROOT / 'configs/qa-risk-pruning/study.json'
    require(sha(protocol) == PROTOCOL_SHA, 'Frozen experiment protocol changed')
    spec = read(protocol)
    for name, expected in spec['inputs_sha256'].items():
        require(sha(ROOT / name) == expected, 'Reference/input changed: ' + name)
    audit = folder / 'audit-final'
    benchmark = folder / 'benchmark'
    for item in (audit, benchmark):
        verify_hashes(item, read(item / 'checksums.json'), exclude=('checksums.json',))
        state = read(item / 'run.json')
        require(state['status'] == 'pass', 'Incomplete evidence')
        require(state['protocol_sha256'] == PROTOCOL_SHA, 'Run protocol identity differs')
        for name, expected in state['source_sha256'].items():
            require(sha(ROOT / name) == expected == sha(item / 'source' / name), 'Current source differs: ' + name)
    replay = read(audit / 'summary.json')
    require(replay['counts'] == {'development': 768, 'historical_evaluation': 128}, 'Incomplete replay')
    require(replay['exact_feature_score_decision_differences'] == 0, 'Replay mismatch')
    records = read(audit / 'records.json')
    require(len(records) == len({(r['variant'], r['id']) for r in records}) == 896, 'Duplicate/missing replay records')
    samples = spec['performance']['sample_ids']
    expected_keys = {(key, rep) for key in samples for rep in range(5)}
    expected_outputs = {(r['id'], field): r[field] for r in records if r['variant'] == 'int8'
                        for field in ('feature_sha256', 'decision_sha256')}
    runs = []
    for i, arm in enumerate(spec['performance']['process_arms']):
        worker = benchmark / str(i)
        verify_hashes(worker, read(worker / 'checksums.json'), exclude=('checksums.json',))
        state = read(worker / 'run.json')
        require(state['status'] == 'pass' and state['protocol_sha256'] == PROTOCOL_SHA, 'Worker incomplete')
        require(state['source_sha256'] == read(benchmark / 'run.json')['source_sha256'], 'Worker source drift')
        run = read(worker / 'measurements.json')
        require(run['arm'] == arm and len(run['records']) == 80, 'Arm/count differs')
        computed = {}
        for kind in ('feature_only', 'full_request'):
            rows = [r for r in run['records'] if r['kind'] == kind]
            require(len(rows) == 40 and {(r['id'], r['repetition']) for r in rows} == expected_keys, 'Sample coverage')
            for row in rows:
                require(0 < row['seconds'] < 900, 'Invalid timing')
                fields = ('feature_sha256', 'decision_sha256') if kind == 'full_request' else ('feature_sha256',)
                for field in fields:
                    require(row[field] == expected_outputs[row['id'], field], 'Timed output mismatch')
            computed[kind] = statistics.median(r['seconds'] for r in rows)
        require(computed == run['median_seconds'], 'Process medians do not reproduce')
        runs.append(run)
    summary = read(benchmark / 'summary.json')
    for kind, gate in (('feature_only', 1.25), ('full_request', 1.05)):
        medians = {arm: [r['median_seconds'][kind] for r in runs if r['arm'] == arm] for arm in ('reference', 'pruned')}
        paired = []
        for i in range(0, 6, 2):
            pair = {r['arm']: r['median_seconds'][kind] for r in runs[i:i+2]}
            paired.append(pair['reference'] / pair['pruned'])
        computed = dict(process_medians_seconds=medians,
            median_seconds={k: statistics.median(v) for k, v in medians.items()},
            speedup=statistics.median(medians['reference']) / statistics.median(medians['pruned']),
            paired_speedups=paired, pairs_faster=sum(x > 1 for x in paired))
        require(summary['metrics'][kind] == computed, 'Aggregate does not reproduce')
        require(computed['speedup'] >= gate and computed['pairs_faster'] >= 2, 'Performance gate failed')
    return dict(status='pass', complete_requests=240, feature_measurements=240,
                replay_rows=896, scope='Archived arithmetic and identity verification; no new speed measurement.')


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
