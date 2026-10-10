"""Real CLI checks on a disposable source copy; never mutate supplied assets.

Use the pinned full runtime Python. These are functional checks, not timing runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'cases': [], 'scope':
               'Actual CLI status/exit-code checks; diagnostic startup times are not a latency study.',
               'source_sha256': {name: sha(root / name) for name in
                   ('experiments/serve_qa_specialist.py', 'experiments/verify_qa_risk.py',
                    'lab/qa_risk_pruning.py')}}
    env = dict(os.environ, PYTHONPATH='', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'source'
            for name in ('experiments', 'lab', 'configs', 'results'):
                shutil.copytree(root / name, copy / name, ignore=shutil.ignore_patterns('__pycache__'))
            data = {row['id']: row for row in map(json.loads,
                (copy / 'configs/qa-risk/dataset/evaluation/data.jsonl').read_text().splitlines())}
            case = json.loads((copy / 'configs/qa-risk/demo.json').read_text())['cases'][0]
            payload = {name: data[case['id']][name] for name in ('context', 'question')}
            request = Path(tmp) / 'request.json'
            request.write_text(json.dumps(payload))

            def invoke(label, expected_code, expected_status, extra=(), expected_reason=None):
                command = [sys.executable, '-B', '-m', 'experiments.serve_qa_specialist',
                           '--asset-root', str(args.asset_root.resolve()), '--input-json', str(request), *extra]
                result = subprocess.run(command, cwd=copy, env=env, text=True,
                                        capture_output=True, timeout=120)
                (out / (label + '.stdout.json')).write_text(result.stdout)
                (out / (label + '.stderr.log')).write_text(result.stderr)
                response = json.loads(result.stdout)
                assert (result.returncode, response['status']) == (expected_code, expected_status), label
                if expected_reason is not None:
                    assert expected_reason in response['reason'], (label, response)
                receipt['cases'].append({'case': label, 'exit_code': result.returncode,
                    'status': response['status'], 'reason': response.get('reason'),
                    'stdout_sha256': sha(out / (label + '.stdout.json'))})

            invoke('default_accepted', 0, 'answer')
            invoke('missing_assets', 3, 'unavailable_model',
                   ['--asset-root', str(Path(tmp) / 'missing-assets')])
            changed = copy / 'lab/qa_risk_pruning.py'
            original = changed.read_bytes()
            changed.write_bytes(original + b'\n# Deliberate source drift in disposable copy.\n')
            invoke('source_drift', 2, 'unavailable_quality', expected_reason='Current implementation differs')
            changed.write_bytes(original)
            training = copy / 'results/qa-risk-v2/training'
            features = training / 'int8-training-features.json'
            values = json.loads(features.read_text())
            values[0]['features'][0] += 1
            features.write_text(json.dumps(values, indent=2) + '\n')
            manifest = {str(p.relative_to(training)): sha(p) for p in sorted(training.rglob('*'))
                        if p.is_file() and p.name != 'checksums.json'}
            (training / 'checksums.json').write_text(json.dumps(manifest, indent=2) + '\n')
            invoke('rehashed_training_features', 2, 'unavailable_quality',
                   expected_reason='Training features/targets do not reproduce')
        receipt['status'] = 'pass'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        (out / 'summary.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
