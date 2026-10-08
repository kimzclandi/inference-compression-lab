"""CPU contracts for diagnostic metrics, isolation and archival verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from lab.gqa_diagnostic import diagnostic_hooks, error_metrics
from experiments.verify_gqa_shared_diagnostic import metrics, reference, safe_path, verify


class NumericalMetricContracts(unittest.TestCase):
    def test_direction_coordinates_and_complete_independent_reconstruction(self):
        left = np.array([[0, 2], [.0001, -4]], dtype=np.float16)
        right = np.zeros_like(left)
        row = error_metrics(left, right, .1, 0)
        self.assertEqual(row, metrics(left, right, .1, 0))
        self.assertEqual(row['failed_elements'], 2)
        self.assertEqual(row['first_failure'], [0, 1])
        self.assertEqual(row['max_coordinate'], [1, 1])
        self.assertEqual((row['left_at_max'], row['right_at_max'], row['max_abs']), (-4, 0, 4))
        one, two = np.array([1], dtype=np.float16), np.array([2], dtype=np.float16)
        self.assertTrue(error_metrics(one, two, 0, .5)['allclose'])
        self.assertFalse(error_metrics(two, one, 0, .5)['allclose'])

    def test_original_half_gate_is_not_silently_replaced_by_float64(self):
        left = np.array([1], dtype=np.float16)
        right = np.nextafter(left, np.float16(2))
        # The Python scalar rounds to an FP16 threshold in the legacy gate.
        # This fixture tests format semantics, not a model acceptance threshold.
        row = error_metrics(left, right, .0009764, 0)
        self.assertTrue(row['allclose'])
        self.assertFalse(row['float64_gate_allclose'])
        self.assertEqual(row['failed_elements'], 0)
        self.assertEqual(row['float64_first_failure'], [0])
        self.assertEqual(row, metrics(left, right, .0009764, 0))

    def test_ulp_signed_zero_adjacent_values_and_wide_signed_range(self):
        zero = error_metrics(np.array([-0.], dtype=np.float16), np.array([0.], dtype=np.float16), 0, 0)
        self.assertTrue(zero['equal_values'])
        self.assertFalse(zero['bitwise_equal'])
        self.assertEqual(zero['ulp_max'], 0)
        for value, direction in [(1, 2), (-1, -2), (0, 1), (0, -1)]:
            a = np.array([value], dtype=np.float16)
            b = np.nextafter(a, np.float16(direction))
            row = error_metrics(a, b, 0, 0)
            self.assertEqual(row['ulp_max'], 1)
            self.assertEqual(row, metrics(a, b, 0, 0))
        with np.errstate(over='ignore'):
            row = error_metrics(np.array([-65504], dtype=np.float16), np.array([65504], dtype=np.float16), 0, 0)
        self.assertEqual(row['max_abs'], 131008)
        self.assertEqual(row['ulp_max'], 63486)

    def test_nonfinite_and_shape_mismatch_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite'):
                error_metrics(np.array([value], dtype=np.float16), np.array([value], dtype=np.float16), .01, .01)
        with self.assertRaisesRegex(ValueError, 'shape'):
            error_metrics(np.zeros(1), np.zeros(2), .01, .01)

    def test_independent_reference_maps_heads_and_ignores_unused_storage(self):
        q = np.zeros((1, 14, 1, 64), dtype=np.float16)
        storage = np.full((1, 2, 256, 64), 60000, dtype=np.float16)
        values = np.full_like(storage, -60000)
        storage[:, :, :129] = 0
        values[:, 0, :129] = 2
        values[:, 1, :129] = 5
        actual = reference(q, storage[:, :, :129], values[:, :, :129])
        np.testing.assert_allclose(actual[:, :7], 2, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual[:, 7:], 5, atol=1e-12, rtol=0)


class HookIsolationContracts(unittest.TestCase):
    def make_modules(self):
        calls = []
        def native(q, k, v, **kwargs):
            calls.append(('upstream', kwargs))
            return 'native-output'
        def candidate(q, k, v, mode='native'):
            calls.append(('adapter', mode))
            return mode + '-output'
        return SimpleNamespace(scaled_dot_product_attention=native), SimpleNamespace(attention=candidate), calls

    def test_adapter_native_and_candidate_route_only_selected_decode(self):
        for mode in ('native', 'shared_compiled'):
            with self.subTest(mode=mode):
                qwen, project, calls = self.make_modules()
                before_native, before_candidate = qwen.scaled_dot_product_attention, project.attention
                records = []
                with diagnostic_hooks(qwen, project, lambda *x: records.append(x), mode):
                    self.assertEqual(qwen.scaled_dot_product_attention('q', 'k', 'v', cache='cache', mask=None, scale=.125), 'native-output')
                    self.assertEqual(project.attention('q', 'k', 'v', mode='shared_compiled'), mode + '-output')
                    self.assertEqual(records[0], (True, 'q', 'k', 'v', 'native-output', 'cache'))
                    self.assertEqual(records[1], (False, 'q', 'k', 'v', mode + '-output', None))
                self.assertIs(qwen.scaled_dot_product_attention, before_native)
                self.assertIs(project.attention, before_candidate)
                self.assertEqual(calls[-1], ('adapter', mode))

    def test_callback_failure_restores_identity_and_allows_future_use(self):
        qwen, project, _ = self.make_modules()
        old_native, old_candidate = qwen.scaled_dot_product_attention, project.attention
        def fail(*args):
            raise RuntimeError('injected callback failure')
        with self.assertRaisesRegex(RuntimeError, 'injected callback'):
            with diagnostic_hooks(qwen, project, fail, 'shared_compiled'):
                project.attention('q', 'k', 'v')
        self.assertIs(qwen.scaled_dot_product_attention, old_native)
        self.assertIs(project.attention, old_candidate)
        with diagnostic_hooks(qwen, project, lambda *x: None, 'native'):
            self.assertEqual(project.attention('q', 'k', 'v'), 'native-output')

    def test_nested_and_unexpected_route_rejected_without_losing_outer_hook(self):
        qwen, project, calls = self.make_modules()
        old_native, old_candidate = qwen.scaled_dot_product_attention, project.attention
        with diagnostic_hooks(qwen, project, lambda *x: None, 'shared_compiled'):
            outer = project.attention
            with self.assertRaisesRegex(RuntimeError, 'nested'):
                with diagnostic_hooks(qwen, project, lambda *x: None, 'native'):
                    self.fail('nested hook entered')
            self.assertIs(project.attention, outer)
            with self.assertRaisesRegex(ValueError, 'route'):
                project.attention('q', 'k', 'v', mode='native')
            self.assertEqual(calls, [])
        self.assertIs(qwen.scaled_dot_product_attention, old_native)
        self.assertIs(project.attention, old_candidate)


class ArchivalContracts(unittest.TestCase):
    def test_optimized_runner_and_verifier_rejected_without_gpu(self):
        for module in ['experiments.gqa_shared_diagnostic', 'experiments.verify_gqa_shared_diagnostic']:
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, '-O', '-m', module, '--help'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('optimized mode unsupported', result.stderr)

    def test_tampered_receipt_rejected_before_any_numeric_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'receipt.json').write_text('{"gate": false}\n')
            run = dict(status='complete', protocol_commit='a' * 40, spec={},
                       artifacts={'receipt.json': hashlib.sha256(b'{"gate": true}\n').hexdigest()})
            (root / 'run.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(AssertionError, 'artifact modified'):
                verify(root)

    def test_escape_paths_are_not_accepted_as_archive_members(self):
        for name in ('../outside', '/absolute/path'):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                safe_path(Path.cwd(), name)

    def test_unlisted_nested_run_manifest_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'extra').mkdir()
            (root / 'extra/run.json').write_text('{"unlisted": true}\n')
            run = dict(status='complete', protocol_commit='a' * 40, spec={}, artifacts={})
            (root / 'run.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(AssertionError, 'artifact modified'):
                verify(root)

    @unittest.skipUnless(Path('results/gqa-shared-diagnostic-v1/run.json').exists(), 'single diagnostic has not run yet')
    def test_archived_diagnostic_replays_all_arrays_and_keeps_v1_failed(self):
        result = verify('results/gqa-shared-diagnostic-v1')
        self.assertTrue(result['evidence_valid'])
        self.assertTrue(result['diagnostic_complete'])
        self.assertEqual(result['shadow_probes_checked'], 384)
        self.assertEqual(result['final_kv_arrays_replayed'], 192)
        self.assertFalse(result['original_v1_model_gate_pass'])
        self.assertEqual(result['performance_trials'], 0)


if __name__ == '__main__':
    unittest.main()
