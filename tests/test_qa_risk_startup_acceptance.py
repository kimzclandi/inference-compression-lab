"""Startup acceptance rejects provenance, output and timing corruption offline."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import verify_qa_risk_startup as acceptance
from lab.artifact_integrity import file_hashes

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def resign(folder):
    write(folder / 'checksums.json', file_hashes(folder, exclude=('checksums.json',)))


class TimingArithmeticTests(unittest.TestCase):
    def timing(self):
        return dict(evidence_verification_seconds=2.0, training_head_rebuild_seconds=.1,
            local_model_load_seconds=.9, total_startup_seconds=3.0, scope=acceptance.STARTUP_SCOPE)

    def test_components_must_be_finite_nonnegative_and_sum_to_total(self):
        acceptance.check_timing(self.timing())
        for field, value in [('training_head_rebuild_seconds', math.nan),
                             ('local_model_load_seconds', math.inf),
                             ('local_model_load_seconds', -.1),
                             ('total_startup_seconds', True),
                             ('total_startup_seconds', 4.0),
                             ('evidence_verification_seconds', 0.0)]:
            with self.subTest(field=field, value=value):
                timing = self.timing(); timing[field] = value
                with self.assertRaises(ValueError):
                    acceptance.check_timing(timing)

    def test_arithmetic_uses_arm_medians_and_adjacent_pairs(self):
        # Synthetic numbers test arithmetic only; they are not measured evidence.
        values = (12.0, 4.0, 5.0, 10.0, 14.0, 7.0)
        records = [dict(arm=arm, timing={name: value for name in
                   ('total_startup_seconds', 'evidence_verification_seconds')})
                   for arm, value in zip(acceptance.ARMS, values)]
        result = acceptance.recompute_metrics(records)['total_startup_seconds']
        self.assertEqual(result['medians'], {'reference': 12.0, 'pruned': 5.0})
        self.assertEqual(result['speedup'], 2.4)
        self.assertEqual(result['paired_speedups'], [3.0, 2.0, 2.0])
        self.assertEqual(result['pairs_faster'], 3)
        records[0]['arm'] = 'pruned'
        with self.assertRaisesRegex(ValueError, 'process sequence'):
            acceptance.recompute_metrics(records)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.json'
            for text in ('{"score": 1, "score": 2}', '{"seconds": NaN}', '{"seconds": Infinity}'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        acceptance.read(path)


@unittest.skipUnless((ROOT / 'results/qa-risk-startup-v1/benchmark/summary.json').is_file(),
                     'Accepted startup archive has not been installed')
class StartupAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = acceptance.read(ROOT / 'configs/qa-risk-startup/study.json')
        names = set(spec['fixed_inputs_sha256']) | set(acceptance.FROZEN_SOURCES)
        names.add('configs/qa-risk-startup/study.json')
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        evaluation = 'results/qa-risk-v2/evaluation-int8'
        shutil.copytree(ROOT / evaluation, self.root / evaluation, dirs_exist_ok=True)
        self.folder = self.root / 'results/qa-risk-startup-v1'
        shutil.copytree(ROOT / 'results/qa-risk-startup-v1', self.folder)

    def trust_mutated_manifest(self, name):
        """Bypass only the outer anchor to exercise deeper semantic validation."""
        folder = self.folder / name
        resign(folder)
        anchors = {**acceptance.EVIDENCE_MANIFESTS, name: acceptance.sha(folder / 'checksums.json')}
        return patch.object(acceptance, 'EVIDENCE_MANIFESTS', anchors)

    def test_complete_archived_study_passes(self):
        result = acceptance.verify(self.root)
        self.assertEqual((result['status'], result['processes']), ('pass', 6))
        self.assertFalse(result['new_quality_confirmation'])

    def test_resigned_claim_cannot_replace_accepted_manifest(self):
        audit = self.folder / 'audit'
        state = acceptance.read(audit / 'run.json'); state['source_sha256'] = {}
        write(audit / 'run.json', state); resign(audit)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            acceptance.verify(self.root)

    def test_complete_source_coverage_is_checked_beneath_anchor(self):
        audit = self.folder / 'audit'
        state = acceptance.read(audit / 'run.json'); state['source_sha256'].pop('experiments/verify_qa_remediation.py')
        write(audit / 'run.json', state)
        with self.trust_mutated_manifest('audit'), self.assertRaisesRegex(ValueError, 'source binding'):
            acceptance.verify(self.root)

    def test_current_source_drift_is_rejected(self):
        path = self.root / 'experiments/serve_qa_specialist.py'
        path.write_text(path.read_text() + '\n# source drift\n')
        with self.assertRaisesRegex(ValueError, 'Current implementation differs'):
            acceptance.verify(self.root)

    def test_audit_actual_json_and_call_counts_are_checked(self):
        audit = self.folder / 'audit'
        original = (audit / 'pruned.json').read_bytes()
        value = acceptance.read(audit / 'pruned.json'); value['evidence_valid'] = False
        write(audit / 'pruned.json', value)
        with self.trust_mutated_manifest('audit'), self.assertRaisesRegex(ValueError, 'outputs differ'):
            acceptance.verify(self.root)
        (audit / 'pruned.json').write_bytes(original)
        value = acceptance.read(audit / 'summary.json'); value['calls']['pruned']['pruned'] = 895
        write(audit / 'summary.json', value)
        with self.trust_mutated_manifest('audit'), self.assertRaisesRegex(ValueError, 'calls or digest'):
            acceptance.verify(self.root)

    def test_worker_digest_must_match_actual_response_json(self):
        worker = self.folder / 'benchmark/0'
        responses = acceptance.read(worker / 'responses.json')
        responses[0]['response']['score'] = .01
        write(worker / 'responses.json', responses); resign(worker)
        with self.trust_mutated_manifest('benchmark'), self.assertRaisesRegex(ValueError, 'response digest'):
            acceptance.verify(self.root)

    def test_rehashed_worker_dependencies_and_components_are_checked(self):
        worker = self.folder / 'benchmark/0'
        original = acceptance.read(worker / 'measurement.json')
        for kind in ('dependency', 'components'):
            with self.subTest(kind=kind):
                value = deepcopy(original)
                if kind == 'dependency':
                    value['packages']['numpy'] = '0.0'
                else:
                    value['timing']['training_head_rebuild_seconds'] += .1
                write(worker / 'measurement.json', value); resign(worker)
                expected = 'dependency versions' if kind == 'dependency' else 'sum to total'
                with self.trust_mutated_manifest('benchmark'), self.assertRaisesRegex(ValueError, expected):
                    acceptance.verify(self.root)

    def test_rehashed_aggregate_must_reproduce_from_worker_measurements(self):
        benchmark = self.folder / 'benchmark'
        summary = acceptance.read(benchmark / 'summary.json')
        summary['metrics']['total_startup_seconds']['speedup'] += 1
        write(benchmark / 'summary.json', summary)
        with self.trust_mutated_manifest('benchmark'), self.assertRaisesRegex(ValueError, 'aggregate'):
            acceptance.verify(self.root)


if __name__ == '__main__':
    unittest.main()
