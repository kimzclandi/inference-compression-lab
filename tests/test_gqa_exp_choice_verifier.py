"""CPU-only tests of exp-choice generation, metric partitions and replay gates."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.verify_gqa_exp_choice import (check_generated, digest, expected_sources, members,
    npz, read, recompute_metrics, recompute_summary, verify, verify_probe, verify_receipt)


def example():
    shape = (1, 14, 1, 64)
    native = np.ones(shape, dtype=np.float16)
    standard = native.copy(); fast = native.copy()
    standard.flat[0] = np.nextafter(np.float16(1), np.float16(2))
    standard.flat[1] = np.nextafter(np.float16(1), np.float16(0))
    fast.flat[1] = standard.flat[1]  # persistent old disagreement
    fast.flat[2] = standard.flat[0]  # introduced disagreement; flat0 is resolved
    arrays = dict(native=native, original=standard.copy(), standard_half=standard,
        standard_float=standard.astype(np.float32), fast_half=fast, fast_float=fast.astype(np.float32))
    return arrays, np.ones(shape, dtype=np.float64)


def archived_probe_fixture():
    root = Path('results/gqa-shared-diagnostic-v1')
    old = next(x for x in read(root / 'shadow.json') if (x['step'], x['layer']) == (1, 5))
    archived = npz(root / 'native_a/s01-l05.npz')
    standard = archived['candidate']
    arrays = dict(native=archived['native'].copy(), original=standard.copy(), standard_half=standard.copy(),
        standard_float=standard.astype(np.float32), fast_half=standard.copy(), fast_float=standard.astype(np.float32))
    hashes = [digest(a.tobytes()) for a in [archived['q'], archived['k_storage'][:, :, :129], archived['v_storage'][:, :, :129]]]
    row = dict(key='s01-l05', step=1, layer=5, layout_before=old['layout'], layout_after=old['layout'],
               input_sha256_before=hashes, input_sha256_after=hashes,
               metrics=recompute_metrics(arrays, archived['reference']))
    return row, arrays, archived, old


class ExpMetricContracts(unittest.TestCase):
    def test_resolved_persistent_introduced_partition_and_reference_tradeoff(self):
        from lab.gqa_exp_choice import metrics
        arrays, ref = example()
        expected = recompute_metrics(arrays, ref)
        self.assertEqual(metrics(arrays, ref), expected)
        self.assertEqual((expected['resolved'], expected['persistent'], expected['introduced']), (1, 1, 1))
        self.assertEqual((expected['standard_native_unequal'], expected['fast_native_unequal']), (2, 2))
        self.assertEqual(expected['half_reference_change'], dict(improved=1, worsened=1, equal=894))
        self.assertEqual(expected['reference_rounding_disagreement'], dict(native=0, standard_half=2, fast_half=2))
        self.assertEqual(len(expected['original_disagreement_boundaries']), 2)

    def test_midpoints_use_gap_units_and_preserve_ties_to_even(self):
        from lab.gqa_exp_choice import metrics
        arrays, ref = example()
        row = recompute_metrics(arrays, ref)
        boundary = row['original_disagreement_boundaries'][0]
        self.assertTrue(boundary['adjacent'])
        self.assertEqual(boundary['midpoint_ties_to_even'], 1)
        self.assertEqual(boundary['standard_distance_gap_units'], .5)
        self.assertEqual(boundary['reference_distance_gap_units'], -.5)
        # A two-spacing endpoint gap must not be mislabeled as one ULP.
        arrays['standard_half'].flat[0] = np.nextafter(arrays['standard_half'].flat[0], np.float16(2))
        arrays['standard_float'] = arrays['standard_half'].astype(np.float32)
        arrays['original'] = arrays['standard_half'].copy()
        result = recompute_metrics(arrays, ref)
        self.assertFalse(result['original_disagreement_boundaries'][0]['adjacent'])
        self.assertEqual(result['original_disagreement_boundaries'][0]['standard_distance_gap_units'], .5)
        self.assertEqual(result, metrics(arrays, ref))

    def test_nonfinite_and_dtype_errors_cannot_be_counted_as_zero_failures(self):
        from lab.gqa_exp_choice import metrics
        arrays, ref = example()
        bad_ref = ref.copy(); bad_ref.flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, 'reference'):
            metrics(arrays, bad_ref)
        with self.assertRaises(AssertionError):
            recompute_metrics(arrays, bad_ref)
        for kind in ('half', 'float'):
            bad = copy.deepcopy(arrays); bad['fast_' + kind].flat[0] = np.inf
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                metrics(bad, ref)
            with self.assertRaisesRegex(AssertionError, 'invalid output'):
                recompute_metrics(bad, ref)
        bad = dict(arrays); bad['standard_float'] = bad['standard_float'].astype(np.float64)
        with self.assertRaises(ValueError):
            metrics(bad, ref)

    def test_aggregate_uses_every_row_and_cannot_turn_native_match_into_model_repair(self):
        from lab.gqa_exp_choice import aggregate
        arrays, ref = example(); row = recompute_metrics(arrays, ref)
        expected = recompute_summary([row, row])
        self.assertEqual(expected, aggregate([dict(metrics=row), dict(metrics=row)]))
        self.assertEqual(expected['elements'], 1792)
        self.assertEqual((expected['resolved'], expected['persistent'], expected['introduced']), (2, 2, 2))
        self.assertFalse(expected['hypothesis_supported'])
        self.assertIn('not native arithmetic identity', expected['scope'])


class GenerationAndFidelityContracts(unittest.TestCase):
    def test_generated_code_only_adds_taps_and_substitutes_three_exp_sites(self):
        from lab.gqa_exp_choice import sources
        expected = {k.replace('_metal', '.metal'): v for k, v in expected_sources().items()}
        self.assertEqual(sources(), expected)
        self.assertEqual(expected['fast_partial.metal'].count('metal::fast::exp('), 2)
        self.assertEqual(expected['fast_merge.metal'].count('metal::fast::exp('), 1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'generated'; folder.mkdir()
            for name, text in expected.items():
                (folder / name).write_text(text)
            check_generated(tmp)
            path = folder / 'fast_merge.metal'
            path.write_text(path.read_text().replace('numerator0 / denominator', 'numerator0 / (denominator + 1)'))
            with self.assertRaisesRegex(AssertionError, 'generated arithmetic changed'):
                check_generated(tmp)

    def test_recomputed_hash_and_metrics_cannot_hide_changed_original_control(self):
        row, arrays, archived, old = archived_probe_fixture()
        verify_probe(row, arrays, archived, old)
        arrays['original'].flat[0] = np.nextafter(arrays['original'].flat[0], np.float16(np.inf))
        arrays['standard_half'] = arrays['original'].copy()
        arrays['standard_float'] = arrays['original'].astype(np.float32)
        row['metrics'] = recompute_metrics(arrays, archived['reference'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'probe.npz'; np.savez_compressed(path, **arrays)
            fresh_manifest = {'probe.npz': digest(path.read_bytes())}
            self.assertEqual(fresh_manifest['probe.npz'], digest(path.read_bytes()))
            with self.assertRaisesRegex(AssertionError, 'original fidelity changed'):
                verify_probe(row, npz(path), archived, old)

    def test_tap_cast_metric_and_layout_mutations_are_rejected_independently(self):
        row, arrays, archived, old = archived_probe_fixture()
        bad = copy.deepcopy(arrays); bad['fast_float'].flat[0] = np.float32(99)
        with self.assertRaisesRegex(AssertionError, 'tap cast'):
            verify_probe(row, bad, archived, old)
        bad_row = copy.deepcopy(row); bad_row['metrics']['resolved'] += 1
        with self.assertRaisesRegex(AssertionError, 'mechanism metrics'):
            verify_probe(bad_row, arrays, archived, old)
        bad_row = copy.deepcopy(row); bad_row['layout_after'] = dict(row['layout_after'], q_shape=[1, 14, 2, 64])
        with self.assertRaises(AssertionError):
            verify_probe(bad_row, arrays, archived, old)

    def test_probe_receipt_requires_all_nine_exact_gates_and_record_identity(self):
        row, _, _, _ = archived_probe_fixture()
        keys = ['layout_reconstructed', 'values_reconstructed', 'finite', 'inputs_unchanged',
                'native_reproduced', 'original_reproduced', 'tap_reproduced', 'standard_cast', 'fast_cast']
        receipt = dict(key=row['key'], stage='after', expected_layout=row['layout_before'],
            expected_input_sha256=row['input_sha256_before'], layout_before=row['layout_before'],
            layout_after=row['layout_after'], input_sha256_before=row['input_sha256_before'],
            input_sha256_after=row['input_sha256_after'], gates={key: True for key in keys})
        verify_receipt(receipt, row)
        receipt['gates']['tap_reproduced'] = False
        with self.assertRaisesRegex(AssertionError, 'probe receipt'):
            verify_receipt(receipt, row)


class ArchivalContracts(unittest.TestCase):
    def test_optimized_runner_and_verifier_rejected_without_gpu(self):
        for module in ['experiments.gqa_exp_choice', 'experiments.verify_gqa_exp_choice']:
            result = subprocess.run([sys.executable, '-O', '-m', module, '--help'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('optimized mode unsupported', result.stderr)

    def test_nested_run_file_is_not_omitted_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'nested').mkdir()
            (root / 'nested/run.json').write_text('{}')
            (root / 'run.json').write_text(json.dumps(dict(status='complete', protocol_commit='a' * 40, spec={}, artifacts={})))
            with self.assertRaisesRegex(AssertionError, 'artifact modified'):
                verify(root)

    @unittest.skipUnless(Path('results/gqa-exp-choice-v1/run.json').exists(), 'single exp-choice diagnostic has not run')
    def test_frozen_complete_replay_and_fresh_hash_record_tampering(self):
        root = Path('results/gqa-exp-choice-v1')
        result = verify(root)
        if read(root / 'run.json')['status'] == 'failed':
            # A faithfully saved failure is a valid outcome of the sole run.
            # It must never force a rerun merely to obtain passing CI.
            self.assertTrue(result['archival_integrity_valid'])
            self.assertFalse(result['evidence_valid'])
            self.assertFalse(result['diagnostic_complete'])
            self.assertFalse(result['numerical_replay_complete'])
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / 'evidence'; shutil.copytree(root, target)
                first = next(iter(read(target / 'run.json')['artifacts']))
                path = target / first; path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaisesRegex(AssertionError, 'artifact modified'):
                    verify(target)
            return
        self.assertTrue(result['evidence_valid'])
        self.assertEqual(result['probes_replayed'], 384)
        self.assertEqual(result['full_output_elements'], 344064)
        self.assertFalse(result['original_model_gate_pass'])
        self.assertEqual(result['model_runs'], 0)
        self.assertEqual(result['performance_trials'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'evidence'; shutil.copytree(root, target)
            records = read(target / 'records.json'); records[0]['metrics']['resolved'] += 1
            (target / 'records.json').write_text(json.dumps(records))
            run = read(target / 'run.json'); run['artifacts'] = members(target)
            (target / 'run.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(AssertionError, 'mechanism metrics changed'):
                verify(target)


if __name__ == '__main__':
    unittest.main()
