"""CPU fault injection through the actual CUDA harness; no CUDA device is used."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments import attention_backend_study as study
from lab.attention_failure import AttentionValidationFailure, validate_case, verify_failure


class HostTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cuda(self):
        return self

    def float(self):
        return HostTensor(self.value.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class AttentionFailureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'failed'
        self.spec = json.loads(study.SPEC.read_text())
        # Small synthetic CPU fixtures, not new protocol or GPU experiment cells.
        self.spec.update(batches=[1], shapes=[[1, 3]], heads=1, head_dim=2)
        self.reference = np.zeros((1, 1, 1, 2), dtype=np.float32)

    def prepare(self, output, mode):
        output.mkdir()
        hashes = {}
        for name in ('configs/attention-backend-v1.json', 'lab/attention_reference.py',
                     'lab/attention_failure.py', 'experiments/attention_backend_study.py'):
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith('configs/'):
                target.write_text(json.dumps(self.spec))
            else:
                target.write_bytes(Path(name).read_bytes())
            hashes[name] = study.digest(target)
        info = dict(mode=mode, status='running', source_sha256=hashes,
                    cuda_performance_measured=False)
        study.save(output / 'run.json', info)
        # Preserve partial measurements from an earlier successful shape.
        study.save(output / 'timings.json', [dict(cpu_fixture='earlier partial timing')])
        return info

    def execute_failure(self, kind):
        fake = ModuleType('torch')
        fake.__version__ = self.spec['torch_version']
        fake.version = SimpleNamespace(cuda='CPU-test-facade')
        fake.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
        fake.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
            get_device_properties=lambda _: SimpleNamespace(name='CPU-test-facade', total_memory=1),
            get_device_capability=lambda: (0, 0),
            synchronize=Mock(side_effect=AssertionError('Failure must stop before warmup/memory/timing')),
            Event=Mock(side_effect=AssertionError('No timing after failure')))
        fake.inference_mode = nullcontext
        fake.set_num_threads = lambda _: None
        fake.from_numpy = HostTensor
        fake.testing = SimpleNamespace(assert_close=lambda a, b, **kw:
            np.testing.assert_allclose(a.value, b.value, equal_nan=False, **kw))
        attention = ModuleType('torch.nn.attention')
        attention.SDPBackend = SimpleNamespace(MATH='math', FLASH_ATTENTION='flash')
        attention.sdpa_kernel = lambda _: nullcontext()
        calls = []

        def backend(_torch, q, k, v, arm):
            calls.append(arm)
            if kind == 'reference_exception' and len(calls) == 1:
                raise RuntimeError('reference injected')
            if kind == 'reference_nan' and len(calls) == 1:
                return HostTensor(self.reference + np.nan)
            if arm == 'math':
                if kind == 'backend_exception':
                    raise LookupError('backend injected')
                if kind == 'shape':
                    return HostTensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
                if kind == 'nan':
                    return HostTensor(self.reference + np.nan)
                return HostTensor(self.reference + 1)
            return HostTensor(self.reference)

        protocol = Path(self.folder.name) / 'test-protocol.json'
        protocol.write_text(json.dumps(self.spec))
        with patch.dict('sys.modules', {'torch': fake, 'torch.nn': ModuleType('torch.nn'),
                                       'torch.nn.attention': attention}), \
                patch.object(study, 'SPEC', protocol), patch.object(study, 'begin', self.prepare), \
                patch.object(study, 'torch_attention', backend):
            with self.assertRaises(AttentionValidationFailure):
                study.run(self.output, 'cuda')
        fake.cuda.synchronize.assert_not_called()
        fake.cuda.Event.assert_not_called()
        self.assertEqual(calls, ['eager_fp32'] if kind.startswith('reference') else
                         ['eager_fp32', 'eager_fp32', 'math'])
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertFalse((self.output / 'memory.json').exists())
        self.assertEqual(json.loads((self.output / 'timings.json').read_text()),
                         [dict(cpu_fixture='earlier partial timing')])
        with self.assertRaisesRegex(ValueError, 'Study did not complete'):
            study.verify(self.output)
        return json.loads((self.output / 'failure.json').read_text())

    def test_tolerance_failure_replays_all_arrays_and_keeps_passed_arm(self):
        receipt = self.execute_failure('tolerance')
        self.assertEqual(receipt['stage'], 'comparison')
        self.assertFalse(receipt['metrics']['array_tolerance_pass'])
        checks = json.loads((self.output / 'correctness.json').read_text())
        self.assertEqual([(x['arm'], x['status']) for x in checks],
                         [('eager_fp32', 'passed'), ('math', 'failed')])
        with np.load(self.output / 'failure-case.npz', allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {'q', 'k', 'v', 'reference', 'actual'})
            np.testing.assert_array_equal(arrays['actual'], self.reference + 1)
        result = verify_failure(self.output)
        self.assertTrue(result['failure_evidence_valid'])
        self.assertFalse(result['study_accepted'])
        self.assertFalse(result['cuda_reexecuted'])
        self.assertFalse(result['performance_conclusion_available'])

    def test_shape_failure_is_archived_before_stopping(self):
        receipt = self.execute_failure('shape')
        self.assertFalse(receipt['metrics']['shapes_equal'])
        self.assertIsNone(receipt['metrics']['max_abs_error'])
        self.assertTrue(verify_failure(self.output)['failure_evidence_valid'])

    def test_nan_output_has_safe_json_and_complete_nan_array(self):
        receipt = self.execute_failure('nan')
        self.assertFalse(receipt['metrics']['actual_finite'])
        self.assertIsNone(receipt['metrics']['normalized_error'])
        json.dumps(receipt, allow_nan=False)
        with np.load(self.output / 'failure-case.npz') as arrays:
            self.assertTrue(np.isnan(arrays['actual']).all())
        self.assertTrue(verify_failure(self.output)['failure_evidence_valid'])

    def test_backend_exception_retains_reference_and_inputs(self):
        receipt = self.execute_failure('backend_exception')
        self.assertEqual(receipt['stage'], 'backend_execution')
        self.assertEqual(receipt['error_type'], 'LookupError')
        self.assertEqual(set(receipt['arrays']), {'q', 'k', 'v', 'reference'})
        self.assertTrue(verify_failure(self.output)['failure_evidence_valid'])

    def test_reference_exception_retains_inputs(self):
        receipt = self.execute_failure('reference_exception')
        self.assertEqual(receipt['stage'], 'reference_execution')
        self.assertEqual(set(receipt['arrays']), {'q', 'k', 'v'})
        self.assertTrue(verify_failure(self.output)['failure_evidence_valid'])

    def test_reference_nonfinite_retains_invalid_reference(self):
        receipt = self.execute_failure('reference_nan')
        self.assertEqual(receipt['stage'], 'reference_validation')
        self.assertFalse(receipt['metrics']['reference_finite'])
        self.assertTrue(verify_failure(self.output)['failure_evidence_valid'])

    def rehash(self):
        run_path = self.output / 'run.json'
        run = json.loads(run_path.read_text())
        for name in ('failure.json', 'failure-case.npz', 'correctness.json'):
            run['artifact_sha256'][name] = study.digest(self.output / name)
        run_path.write_text(json.dumps(run))

    def test_array_tamper_is_rejected(self):
        self.execute_failure('tolerance')
        with (self.output / 'failure-case.npz').open('ab') as file:
            file.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            verify_failure(self.output)

    def test_rehashed_tolerance_drift_is_rejected_against_archived_protocol(self):
        receipt = self.execute_failure('tolerance')
        receipt['atol'] = 10
        (self.output / 'failure.json').write_text(json.dumps(receipt))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'tolerance differs'):
            verify_failure(self.output)

    def test_rehashed_numeric_edit_is_rejected_by_array_replay(self):
        receipt = self.execute_failure('tolerance')
        receipt['metrics']['max_abs_error'] = 0
        (self.output / 'failure.json').write_text(json.dumps(receipt))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'numerical replay mismatch'):
            verify_failure(self.output)

    def test_archived_source_tamper_is_rejected(self):
        self.execute_failure('tolerance')
        path = self.output / 'source/lab/attention_failure.py'
        path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            verify_failure(self.output)

    def test_failed_run_cannot_attach_a_success_summary(self):
        self.execute_failure('tolerance')
        study.save(self.output / 'summary.json', {'accepted': True})
        with self.assertRaisesRegex(ValueError, 'unexpected failure artifacts'):
            verify_failure(self.output)

    def test_missing_source_identity_is_rejected(self):
        self.execute_failure('tolerance')
        path = self.output / 'run.json'
        run = json.loads(path.read_text())
        run['source_sha256'] = {}
        path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, 'archived source identity'):
            verify_failure(self.output)

    def test_rehashed_invalid_reference_cannot_masquerade_as_backend_failure(self):
        receipt = self.execute_failure('backend_exception')
        file = self.output / 'failure-case.npz'
        with np.load(file) as values:
            arrays = {key: values[key] for key in values.files}
        arrays['reference'].fill(np.nan)
        np.savez_compressed(file, **arrays)
        receipt['arrays_sha256'] = study.digest(file)
        (self.output / 'failure.json').write_text(json.dumps(receipt))
        self.rehash()
        with self.assertRaisesRegex(ValueError, 'Invalid reference cannot precede'):
            verify_failure(self.output)

    def test_each_success_is_visible_before_next_arm_and_no_success_arrays_saved(self):
        self.output.mkdir()
        inputs = [np.zeros((1, 1, n, 2), dtype=np.float16) for n in (1, 3, 3)]
        calls = []
        def evaluate(arm):
            calls.append(arm)
            if len(calls) == 3:
                saved = json.loads((self.output / 'correctness.json').read_text())
                self.assertEqual([(x['arm'], x['status']) for x in saved], [('eager_fp32', 'passed')])
            return self.reference
        validate_case(self.output, [], dict(batch=1, query_length=1, key_length=3), inputs,
                      ['eager_fp32', 'math'], evaluate, np.testing.assert_array_equal, .01, .01)
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'correctness.json'})

    def test_failure_replay_cli_requires_no_torch_or_cuda_execution(self):
        self.execute_failure('tolerance')
        result = subprocess.run([sys.executable, '-m', 'experiments.attention_backend_study',
                                 'verify-failure', '--output', str(self.output)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['study_accepted'])
