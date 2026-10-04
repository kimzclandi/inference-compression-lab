"""Independently recompute saved timing arithmetic, coverage and artifact bindings.

No model execution or latency measurement; does not establish another host's speed.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.artifact_integrity import verify_hashes, safe_path

PROTOCOL_SHA = '4efa6b19d945b475c3244a1e456b06207c00dab7241f595b7baf9904849b34ef'

# Reviewed-code anchors, not signatures against someone who can edit this verifier.
# Frozen experiment files remain unchanged. The current runner only adds a
# pre-run protocol-hash guard; its timed worker and request logic are unchanged.
FROZEN_SOURCES = {'experiments/qa_risk_pruning.py': '7023bc99eb100e5583b57b1e66a3d9508e88686962c253609d86b0cd1e2ab76e',
 'lab/qa_risk_pruning.py': '788cfb3f5ed6b4c63dc5167ec7eb3398bfb911924ee3cf7ee1172c49092643a5',
 'lab/qa_risk_calibration.py': '182138f513c311a5e14dece7afc60ca28c1cbf22b1dbbf56a6db0cf3513b3bbd',
 'lab/qa_metrics.py': '78c4af7f06ba49b3acddd2610586336ecf585c06d8ad89b46ff9fb6358aaaa1f',
 'lab/extractive_qa.py': 'd859b363f2cfcca578f0358595a11bdfdcc9b31d0dda19dcbc9def3e9104de8f',
 'lab/qa_specialist_runtime.py': '40e8a0482b3ddb7d1aca5424c090571e27182c363f15e6b3da0ea72c537b1e54',
 'lab/selective_qa.py': 'd4e4c5cf1a6951316a6b5589236e8198113bc2adcab93dce25754dd8e4d16ffd',
 'experiments/qa_risk.py': 'b384e54e4e396b06ca80394187e6798633eb542acb86edf213a815aff65d7457'}
CURRENT_RUNNER_SHA = '2bd8b9d6318df251bfd611773a685b09e279b2610997969a50a2b0def6547b9d'
EVIDENCE_MANIFESTS = {'audit-final': '61a43acdb4c9f8fd41a32eb9d4c1f9e545cebf8a9d1b9aa85dcdb3cf25b496cd',
 'benchmark': '7d18a15b0c4dd695b027c5fe48797d5b9aebc60a50d0a86783860e2967ff3604'}


def check_source_snapshot(root, folder, state):
    require(state['source_sha256'] == FROZEN_SOURCES, 'Incomplete or changed source binding')
    for name, expected in FROZEN_SOURCES.items():
        require(sha(safe_path(folder / 'source', name)) == expected, 'Archived source differs: ' + name)
        current = CURRENT_RUNNER_SHA if name == 'experiments/qa_risk_pruning.py' else expected
        require(sha(safe_path(root, name)) == current, 'Current implementation differs: ' + name)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(root=ROOT):
    root = Path(root)
    folder = root / 'results/qa-risk-pruning-v1'
    protocol = safe_path(root, 'configs/qa-risk-pruning/study.json')
    require(sha(protocol) == PROTOCOL_SHA, 'Frozen experiment protocol changed')
    spec = read(protocol)
    for name, expected in spec['inputs_sha256'].items():
        require(sha(safe_path(root, name)) == expected, 'Reference/input changed: ' + name)
    audit = folder / 'audit-final'
    benchmark = folder / 'benchmark'
    for item in (audit, benchmark):
        require(sha(item / 'checksums.json') == EVIDENCE_MANIFESTS[item.name],
                'Accepted evidence manifest changed: ' + item.name)
        verify_hashes(item, read(item / 'checksums.json'), exclude=('checksums.json',))
        state = read(item / 'run.json')
        require(state['status'] == 'pass', 'Incomplete evidence')
        require(state['protocol_sha256'] == PROTOCOL_SHA, 'Run protocol identity differs')
        check_source_snapshot(root, item, state)
    replay = read(audit / 'summary.json')
    require(replay['counts'] == {'development': 768, 'historical_evaluation': 128}, 'Incomplete replay')
    require(replay['exact_feature_score_decision_differences'] == 0, 'Replay mismatch')
    records = read(audit / 'records.json')
    require(len(records) == len({(r['variant'], r['id']) for r in records}) == 896, 'Duplicate/missing replay records')
    expected_rows = set()
    for role in ('train', 'calibration', 'evaluation'):
        data = [json.loads(line) for line in (root / f'configs/qa-risk/dataset/{role}/data.jsonl').read_text().splitlines()]
        variants = ('int8',) if role == 'evaluation' else ('fp32', 'int8')
        public_role = 'historical_evaluation' if role == 'evaluation' else 'development'
        expected_rows.update((variant, row['id'], public_role) for row in data for variant in variants)
    require({(r['variant'], r['id'], r['role']) for r in records} == expected_rows, 'Replay identity/role coverage')
    counts = replay['historical_quality_replayed']['selective']
    require(tuple(counts[k] for k in ('n', 'accepted', 'accepted_correct', 'answerable',
                                     'unanswerable', 'accepted_unanswerable')) == (128, 27, 27, 64, 64, 0),
            'Historical quality counts changed')
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
        check_source_snapshot(root, worker, state)
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
    require(summary['status'] == 'pass' and summary['measured_requests'] == 240 and
            summary['feature_measurements'] == 240 and summary['warmup_requests'] == 48 and
            summary['scope'] == spec['performance']['timing'], 'Timing summary contract changed')
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
                replay_rows=896, protocol_sha256=PROTOCOL_SHA,
                extractor_sha256=FROZEN_SOURCES['lab/qa_risk_pruning.py'], scope='Archived arithmetic and identity verification; no new speed measurement.')


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
