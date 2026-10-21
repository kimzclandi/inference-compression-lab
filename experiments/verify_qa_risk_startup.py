"""Verify archived startup identities and arithmetic without loading a model.

These reviewed-code anchors bind one accepted local study. They are not digital
signatures against an actor who can replace this verifier. No latency is measured
and no new quality confirmation is performed here.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

from lab.artifact_integrity import safe_path, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = 'b804da9d77dd4280cae6abdf28758a452abd190bbfe0e5fcbf8811bb8d99a0bf'
# Reviewed source and evidence anchors for the one completed fixed study.
FROZEN_SOURCES = {'experiments/qa_risk_startup.py': '47b7ab09fcb430efea9fb303e070f63551607641cdb5a2c98e1a33c408bd51b0',
 'experiments/verify_qa_risk.py': '8424b12e3ba6704d1c9101f4bff7fe1e7c8d1df672d4a902181a227eb9018852',
 'experiments/serve_qa_specialist.py': 'e2cb97b84679d589987237465e28767bd6d40890b0da8fcec42be8732672327b',
 'experiments/verify_qa_risk_pruning.py': '4588676d7cf2c9e01a2e9f0ae4adba99df5a0c0ef15159ed197c2909f9570288',
 'experiments/qa_risk.py': 'b384e54e4e396b06ca80394187e6798633eb542acb86edf213a815aff65d7457',
 'experiments/verify_qa_specialist.py': 'e09ed0770670f2428076307251c7d37e71896640605673ba08bc1ef471a12fbc',
 'lab/qa_risk_calibration.py': '182138f513c311a5e14dece7afc60ca28c1cbf22b1dbbf56a6db0cf3513b3bbd',
 'lab/qa_risk_pruning.py': '788cfb3f5ed6b4c63dc5167ec7eb3398bfb911924ee3cf7ee1172c49092643a5',
 'lab/qa_specialist_runtime.py': '40e8a0482b3ddb7d1aca5424c090571e27182c363f15e6b3da0ea72c537b1e54',
 'lab/extractive_qa.py': 'd859b363f2cfcca578f0358595a11bdfdcc9b31d0dda19dcbc9def3e9104de8f',
 'lab/qa_metrics.py': '78c4af7f06ba49b3acddd2610586336ecf585c06d8ad89b46ff9fb6358aaaa1f',
 'lab/selective_qa.py': 'd4e4c5cf1a6951316a6b5589236e8198113bc2adcab93dce25754dd8e4d16ffd',
 'lab/qa_gate.py': 'ae1d91829e9c0039f903bc8306fb5a72256998433ef551f09fd3c6c42db5512a',
 'lab/artifact_integrity.py': '8f10b1899e83258baf3a39a8c0d2652c20cab8708d576fc3485cff7654278187',
 'lab/evidence.py': '314c7a009fb2b740db1cb3925d538f7abcd8a8eb719c0d05efd0be71c0abb91d',
 'lab/quantization_diagnostics.py': '947d5bf29b9c238c7330b75c1719044122e1f75c5fc8ad36cf5df1a05d95075f',
 'experiments/qa_risk_pruning.py': '2bd8b9d6318df251bfd611773a685b09e279b2610997969a50a2b0def6547b9d',
 'experiments/verify_qa_remediation.py': '6c2236ae2debb7c1e4ff5e6e1fae173d84a34887abf15aabed372f677126099d',
 'experiments/qa_specialist.py': '47ccb951b8e54c18b71738360900adeb819ed7f7008f0eb4b07ea2e5f744506c',
 'experiments/__init__.py': '7c54e0db53064338fa4534e5e27fadb47c6632b81ba3b249f2d251bc0c0321ba',
 'lab/__init__.py': '0925f6ce613cb6c7f7eb3feba5cea6f47b47f653ea5b4de1d358915d43ea9a79'}
EVIDENCE_MANIFESTS = {'audit': 'a3edcbcb2b1dbee2682db3d7761074a116931455e07966b89ac383a0d38d3bd3',
 'benchmark': 'eea700091cbe7e02767b78e01bd9a09399495b3134bcb3de2c87f52f7a894825'}
ARMS = ('reference', 'pruned', 'pruned', 'reference', 'reference', 'pruned')
RUNTIME_PACKAGES = ('onnxruntime', 'numpy', 'transformers', 'tokenizers')
TIMING_FIELDS = ('evidence_verification_seconds', 'training_head_rebuild_seconds',
                 'local_model_load_seconds', 'total_startup_seconds')
STARTUP_SCOPE = 'One-time initialization; excluded from request-only benchmark timings.'
AUDIT_SCOPE = 'Full fixed evidence recomputation, no new inference or quality confirmation.'


def require(value, message):
    if not value:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, 'Duplicate JSON key: ' + name)
        result[name] = value
    return result


def _constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                      parse_constant=_constant)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False).encode()).hexdigest()


def check_source_snapshot(root, folder, state):
    require(bool(FROZEN_SOURCES) and state['source_sha256'] == FROZEN_SOURCES,
            'Incomplete or changed source binding')
    verify_hashes(folder / 'source', FROZEN_SOURCES, exclude=())
    for name, expected in FROZEN_SOURCES.items():
        require(sha(safe_path(root, name)) == expected, 'Current implementation differs: ' + name)


def check_state(root, folder, mode):
    state = read(folder / 'run.json')
    require(state['status'] == 'pass' and state['mode'] == mode, 'Incomplete or wrong run mode')
    require(state['protocol_sha256'] == PROTOCOL_SHA and sha(folder / 'protocol.json') == PROTOCOL_SHA,
            'Archived protocol identity differs')
    require(state['new_quality_confirmation'] is False and state['parameters_distributed'] is False,
            'Study scope or parameter disclosure changed')
    check_source_snapshot(root, folder, state)
    return state


def check_timing(timing):
    require(set(timing) == set(TIMING_FIELDS) | {'scope'} and timing['scope'] == STARTUP_SCOPE,
            'Startup timing components or scope changed')
    for name in TIMING_FIELDS:
        value = timing[name]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value < 900,
                'Invalid startup timing: ' + name)
    require(timing['total_startup_seconds'] > 0 and timing['evidence_verification_seconds'] > 0,
            'Nonpositive startup measurement')
    require(math.isclose(sum(timing[name] for name in TIMING_FIELDS[:3]),
                        timing['total_startup_seconds'], rel_tol=1e-12, abs_tol=1e-9),
            'Startup components do not sum to total')


def recompute_metrics(records):
    require(len(records) == 6 and tuple(r['arm'] for r in records) == ARMS,
            'Incomplete or changed process sequence')
    metrics = {}
    for field in ('total_startup_seconds', 'evidence_verification_seconds'):
        values = {arm: [r['timing'][field] for r in records if r['arm'] == arm]
                  for arm in ('reference', 'pruned')}
        medians = {arm: statistics.median(values[arm]) for arm in values}
        paired = []
        for index in range(0, 6, 2):
            pair = {r['arm']: r['timing'][field] for r in records[index:index + 2]}
            paired.append(pair['reference'] / pair['pruned'])
        metrics[field] = dict(seconds=values, medians=medians,
            speedup=medians['reference'] / medians['pruned'],
            paired_speedups=paired, pairs_faster=sum(value > 1 for value in paired))
    return metrics


def check_responses(responses, cases, historical):
    require(isinstance(responses, list) and len(responses) == 3 and
            [row['id'] for row in responses] == [case['id'] for case in cases],
            'Disclosed functional case coverage differs')
    common = {'status', 'reason', 'score', 'score_kind', 'threshold', 'scope'}
    for row, case in zip(responses, cases):
        require(set(row) == {'id', 'response'}, 'Unexpected functional record fields')
        actual, old = row['response'], historical[row['id']]
        accepted = case['expected_status'] == 'answer'
        fields = common | ({'answer', 'start', 'end', 'context_sha256'} if accepted else set())
        require(set(actual) == fields, 'Functional response fields differ')
        require(actual['status'] == case['expected_status'] and
                actual['reason'] == ('fixed_threshold_met' if accepted else 'below_fixed_threshold') and
                actual['score_kind'] == 'fitted_ranking_score_not_guaranteed_correctness_probability' and
                actual['scope'] == 'english_supplied_passage_bounded_local_prototype' and
                type(actual['threshold']) is float and actual['threshold'] == .7,
                'Functional response contract changed')
        score = actual['score']
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1 and
                abs(score - old['confidence']) <= 1e-9 and (score >= .7) == accepted,
                'Disclosed functional score/decision changed')
        if accepted:
            require(all(actual[key] == old[source] for key, source in
                    (('answer', 'prediction'), ('start', 'start'), ('end', 'end'),
                     ('context_sha256', 'context_sha256'))), 'Disclosed accepted answer/span changed')


def verify(root=ROOT):
    root = Path(root)
    require(bool(FROZEN_SOURCES) and set(EVIDENCE_MANIFESTS) == {'audit', 'benchmark'},
            'Completed startup study anchors are required')
    protocol = safe_path(root, 'configs/qa-risk-startup/study.json')
    require(sha(protocol) == PROTOCOL_SHA, 'Frozen startup protocol changed')
    spec = read(protocol)
    require(tuple(spec['arms']) == ARMS and spec['quality']['feature_rows'] == 896 and
            spec['performance']['minimum_startup_speedup'] == 1.5 and
            spec['performance']['minimum_pairs_faster'] == 2 and
            spec['limits']['fresh_processes'] == 6 and spec['limits']['timing_runs'] == 1,
            'Frozen startup experiment contract differs')
    for name, expected in spec['fixed_inputs_sha256'].items():
        require(sha(safe_path(root, name)) == expected, 'Fixed startup input changed: ' + name)
    # The functional checks read saved prediction values, so validate the entire
    # fixed evaluation evidence against the protocol-bound manifest first.
    evaluation = root / 'results/qa-risk-v2/evaluation-int8'
    verify_hashes(evaluation, read(evaluation / 'checksums.json'), exclude=('checksums.json',))
    assets = read(evaluation / 'assets.json')
    packages = {name: assets['packages'][name] for name in RUNTIME_PACKAGES}
    cases = read(root / 'configs/qa-risk/demo.json')['cases']
    historical = {}
    for line in (evaluation / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
        require(row['id'] not in historical, 'Duplicate historical prediction')
        historical[row['id']] = row
    folder = root / 'results/qa-risk-startup-v1'
    audit, benchmark = folder / 'audit', folder / 'benchmark'
    states = {}
    for name, item in (('audit', audit), ('benchmark', benchmark)):
        require(sha(item / 'checksums.json') == EVIDENCE_MANIFESTS[name],
                'Accepted startup evidence manifest changed: ' + name)
        verify_hashes(item, read(item / 'checksums.json'), exclude=('checksums.json',))
        states[name] = check_state(root, item, name)
    reference, pruned = read(audit / 'reference.json'), read(audit / 'pruned.json')
    verification_sha = digest(reference)
    require(verification_sha == digest(pruned), 'Complete audit verification outputs differ')
    expected_audit = dict(status='pass', verification_sha256=verification_sha,
        calls={'reference': {'reference': 896, 'pruned': 0},
               'pruned': {'reference': 0, 'pruned': 896}}, scope=AUDIT_SCOPE)
    require(digest(read(audit / 'summary.json')) == digest(expected_audit), 'Audit calls or digest do not reproduce')
    records, response_sha = [], None
    for index, arm in enumerate(ARMS):
        worker = benchmark / str(index)
        verify_hashes(worker, read(worker / 'checksums.json'), exclude=('checksums.json',))
        state = check_state(root, worker, 'worker')
        require(all(state[name] == states['benchmark'][name] for name in ('python', 'platform')),
                'Worker Python/platform changed')
        record = read(worker / 'measurement.json')
        require(set(record) == {'status', 'arm', 'timing', 'verification_sha256', 'responses_sha256', 'packages'} and
                record['status'] == 'pass' and record['arm'] == arm, 'Worker arm or completion differs')
        require(record['packages'] == packages, 'Worker dependency versions differ from pinned assets')
        require(digest(read(worker / 'verification.json')) == record['verification_sha256'] == verification_sha,
                'Worker complete verification digest differs')
        responses = read(worker / 'responses.json')
        require(digest(responses) == record['responses_sha256'], 'Worker response digest does not reproduce')
        check_responses(responses, cases, historical)
        if response_sha is None:
            response_sha = record['responses_sha256']
        require(record['responses_sha256'] == response_sha, 'Functional outputs differ between processes')
        check_timing(record['timing'])
        records.append(record)
    metrics = recompute_metrics(records)
    primary = metrics['total_startup_seconds']
    passed = primary['speedup'] >= 1.5 and primary['pairs_faster'] >= 2
    expected_summary = dict(status='pass' if passed else 'gate_failed', processes=6, metrics=metrics,
        verification_sha256=verification_sha, responses_sha256=response_sha,
        scope=spec['performance']['timing'], new_quality_confirmation=False)
    require(digest(read(benchmark / 'summary.json')) == digest(expected_summary),
            'Startup timing aggregate does not reproduce')
    require(passed, 'Startup performance gate failed')
    return dict(status='pass', processes=6, feature_rows_per_audit_arm=896,
        protocol_sha256=PROTOCOL_SHA, verification_sha256=verification_sha,
        responses_sha256=response_sha, metrics=metrics, new_quality_confirmation=False,
        scope='Archived startup arithmetic and identity verification; no new timing or quality measurement.')


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2, allow_nan=False))
