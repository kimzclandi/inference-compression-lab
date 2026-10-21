"""One fixed startup comparison; complete evidence is recomputed in both arms."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from lab.artifact_integrity import file_hashes, git_identity, verify_hashes
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, rows, sha, write
from experiments.qa_risk_pruning import digest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/qa-risk-startup/study.json'
PROTOCOL_SHA = 'b804da9d77dd4280cae6abdf28758a452abd190bbfe0e5fcbf8811bb8d99a0bf'
SOURCES = ('experiments/qa_risk_startup.py', 'experiments/verify_qa_risk.py',
    'experiments/serve_qa_specialist.py', 'experiments/verify_qa_risk_pruning.py',
    'experiments/qa_risk.py', 'experiments/verify_qa_specialist.py',
    'lab/qa_risk_calibration.py', 'lab/qa_risk_pruning.py', 'lab/qa_specialist_runtime.py',
    'lab/extractive_qa.py', 'lab/qa_metrics.py', 'lab/selective_qa.py', 'lab/qa_gate.py',
    'lab/artifact_integrity.py', 'lab/evidence.py', 'lab/quantization_diagnostics.py',
    'experiments/qa_risk_pruning.py', 'experiments/verify_qa_remediation.py',
    'experiments/qa_specialist.py', 'experiments/__init__.py', 'lab/__init__.py')


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def protocol():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Fixed startup protocol changed')
    spec = read(PROTOCOL)
    for name, expected in spec['fixed_inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Fixed startup input changed: ' + name)
    return spec


def audit(spec, out):
    from experiments import verify_qa_risk as engine
    from lab import qa_risk_pruning as fast
    results, calls = {}, {}
    with patch.object(engine, 'extract_features', wraps=engine.extract_features) as reference, \
         patch.object(fast, 'extract_features', wraps=fast.extract_features) as pruned:
        for mode in ('reference', 'pruned'):
            reference.reset_mock(); pruned.reset_mock()
            result = engine.verify(ROOT / 'results/qa-risk-v2', ROOT / 'configs/qa-risk/study.json', feature_mode=mode)
            results[mode] = result
            calls[mode] = {'reference': reference.call_count, 'pruned': pruned.call_count}
            write(out / (mode + '.json'), result)
    expected_calls = {mode: {key: spec['quality']['feature_rows'] if key == mode else 0
                            for key in ('reference', 'pruned')} for mode in ('reference', 'pruned')}
    if calls != expected_calls or digest(results['reference']) != digest(results['pruned']):
        write(out / 'difference.json', {'calls': calls, 'expected_calls': expected_calls})
        raise ValueError('Complete verification or executed feature path differs')
    result = dict(status='pass', verification_sha256=digest(results['reference']), calls=calls,
                  scope='Full fixed evidence recomputation, no new inference or quality confirmation.')
    write(out / 'summary.json', result)
    return result


def worker(out, mode, asset_root):
    from experiments import serve_qa_specialist as serving
    from lab.qa_specialist_runtime import RUNTIME_PACKAGES
    original = serving._verify_evidence
    captured = {}
    def capture(*args, **kwargs):
        # Capture only the public result, after actual computation. Nothing is
        # cached, and no computation is bypassed. Same instrumentation in both arms.
        result = original(*args, **kwargs)
        captured['verification'] = result
        return result
    with patch.object(serving, '_verify_evidence', side_effect=capture):
        service = serving.load_service(ROOT / 'configs/qa-risk/policy.json', ROOT / 'results/qa-risk-v2',
            ROOT / 'configs/qa-risk/study.json', asset_root, verification_mode=mode)
    write(out / 'verification.json', captured['verification'])
    data = {r['id']: r for r in rows(ROOT / 'configs/qa-risk/dataset/evaluation/data.jsonl')}
    old = {r['id']: r for r in rows(ROOT / 'results/qa-risk-v2/evaluation-int8/predictions.jsonl')}
    responses = []
    for case in read(ROOT / 'configs/qa-risk/demo.json')['cases']:
        row = data[case['id']]
        response = service.answer({key: row[key] for key in ('context', 'question')})
        response.pop('timing')  # Requests are functional checks outside the startup timer.
        if response['status'] != case['expected_status'] or abs(response['score'] - old[row['id']]['confidence']) > 1e-9:
            raise ValueError('Disclosed functional decision/score changed')
        if response['status'] == 'answer' and response['answer'] != old[row['id']]['prediction']:
            raise ValueError('Disclosed accepted answer changed')
        responses.append(dict(id=row['id'], response=response))
    write(out / 'responses.json', responses)
    result = dict(status='pass', arm=mode, timing=service.startup_timing,
        verification_sha256=digest(captured['verification']), responses_sha256=digest(responses),
        packages={name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES})
    write(out / 'measurement.json', result)
    return result


def summarize(records, spec):
    if len(records) != 6 or [r['arm'] for r in records] != spec['arms'] or any(r['status'] != 'pass' for r in records):
        raise ValueError('Incomplete or changed process sequence')
    if len({r['verification_sha256'] for r in records}) != 1 or len({r['responses_sha256'] for r in records}) != 1:
        raise ValueError('Verification/functional outputs differ between processes')
    if any(r['packages'] != records[0]['packages'] for r in records):
        raise ValueError('Dependency versions changed between processes')
    metrics = {}
    for field in ('total_startup_seconds', 'evidence_verification_seconds'):
        values = {arm: [r['timing'][field] for r in records if r['arm'] == arm] for arm in ('reference', 'pruned')}
        if any(not 0 < x < 900 for v in values.values() for x in v):
            raise ValueError('Invalid startup measurement')
        medians = {arm: statistics.median(v) for arm, v in values.items()}
        paired = []
        for i in range(0, 6, 2):
            pair = {r['arm']: r['timing'][field] for r in records[i:i + 2]}
            paired.append(pair['reference'] / pair['pruned'])
        metrics[field] = dict(seconds=values, medians=medians, speedup=medians['reference'] / medians['pruned'],
                              paired_speedups=paired, pairs_faster=sum(v > 1 for v in paired))
    primary = metrics['total_startup_seconds']
    passed = (primary['speedup'] >= spec['performance']['minimum_startup_speedup'] and
              primary['pairs_faster'] >= spec['performance']['minimum_pairs_faster'])
    return dict(status='pass' if passed else 'gate_failed', processes=6, metrics=metrics,
        verification_sha256=records[0]['verification_sha256'], responses_sha256=records[0]['responses_sha256'],
        scope=spec['performance']['timing'], new_quality_confirmation=False)


def benchmark(spec, out, asset_root, audit_root):
    verify_hashes(audit_root, read(audit_root / 'checksums.json'), exclude=('checksums.json',))
    checked = read(audit_root / 'run.json')
    if checked['status'] != 'pass' or checked['source_sha256'] != source_hashes() or checked['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Current implementation needs a completed full equality audit')
    expected = read(audit_root / 'summary.json')['verification_sha256']
    before = source_hashes(); started = time.monotonic(); records = []
    for i, arm in enumerate(spec['arms']):
        remaining = spec['limits']['compute_budget_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Fixed startup budget exhausted')
        command = [sys.executable, '-B', '-m', 'experiments.qa_risk_startup', 'worker', '--arm', arm,
                   '--asset-root', str(asset_root), '--output-dir', str(out / str(i))]
        with (out / f'{i}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, timeout=remaining, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', PYTHONPATH=''))
        if result.returncode:
            raise ValueError('Startup process failed; preserve worker log: ' + str(i))
        state = read(out / str(i) / 'run.json')
        record = read(out / str(i) / 'measurement.json')
        if state['source_sha256'] != before or state['protocol_sha256'] != PROTOCOL_SHA or record['verification_sha256'] != expected:
            raise ValueError('Worker source, protocol or complete result changed')
        records.append(record)
        print('completed startup process', i, arm, flush=True)
    result = summarize(records, spec)
    write(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'benchmark', 'worker'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--arm', choices=('reference', 'pruned'))
    args = parser.parse_args()
    out = reserve_directory(args.output_dir).resolve()
    state = dict(status='running', mode=args.mode, **git_identity(ROOT), python=platform.python_version(),
        platform=platform.platform(), source_sha256=source_hashes(), protocol_sha256=sha(PROTOCOL),
        new_quality_confirmation=False, parameters_distributed=False)
    try:
        spec = protocol()
        shutil.copy2(PROTOCOL, out / 'protocol.json')
        for name in SOURCES:
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if args.mode == 'audit':
            result = audit(spec, out)
        elif args.mode == 'worker':
            if args.arm is None or args.asset_root is None:
                raise ValueError('Worker requires fixed arm and existing local assets')
            result = worker(out, args.arm, args.asset_root.resolve())
        else:
            if args.asset_root is None or args.audit_root is None:
                raise ValueError('Benchmark requires local assets and equality audit')
            result = benchmark(spec, out, args.asset_root.resolve(), args.audit_root.resolve())
        protocol()
        if source_hashes() != state['source_sha256']:
            raise ValueError('Source changed during the fixed study')
        state['status'] = result['status']
        return 0 if result['status'] == 'pass' else 1
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(out / 'run.json', state)
        write(out / 'checksums.json', file_hashes(out, exclude=('checksums.json',)))


if __name__ == '__main__':
    raise SystemExit(main())
