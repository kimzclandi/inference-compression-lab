"""Optimized verification must retain rejection and full recomputation."""
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import verify_qa_risk as verification
from experiments import qa_risk_startup as startup
from lab import qa_risk_pruning
import test_verify_qa_risk as original


class PrunedTrainingVerificationTests(original.RiskTrainingVerificationTests):
    # Reuse all training corruption/serialized-matrix tests on the new path.
    feature_extractor = staticmethod(qa_risk_pruning.extract_features)


class PrunedEvaluationMutationTests(original.RecordedEvaluationMutationTests):
    feature_extractor = staticmethod(qa_risk_pruning.extract_features)


class VerificationModeTests(unittest.TestCase):
    def test_imported_verification_dependencies_are_bound(self):
        required = {'experiments/verify_qa_remediation.py', 'experiments/qa_specialist.py',
                    'experiments/__init__.py', 'lab/__init__.py'}
        self.assertTrue(required <= set(startup.SOURCES))
        before = startup.source_hashes()
        original_sha = startup.sha
        for name in required:
            def changed(path):
                return '0' * 64 if path == startup.ROOT / name else original_sha(path)
            with self.subTest(name=name), patch.object(startup, 'sha', side_effect=changed):
                self.assertNotEqual(startup.source_hashes(), before)

    def test_reference_default_remains_independent(self):
        with patch('experiments.verify_qa_risk_pruning.verify') as guard:
            self.assertIs(verification._feature_extractor('reference'), verification.extract_features)
        guard.assert_not_called()

    def test_pruned_source_guard_runs_before_any_evidence_is_read(self):
        with patch('experiments.verify_qa_risk_pruning.verify', side_effect=ValueError('source drift')) as guard, \
             patch.object(verification, 'read') as read, \
             patch.object(verification, 'verify_training') as training:
            with self.assertRaisesRegex(ValueError, 'source drift'):
                verification.verify(Path('not-read'), feature_mode='pruned')
        guard.assert_called_once_with(verification.REPO)
        read.assert_not_called()
        training.assert_not_called()

    def test_pruned_mode_selects_reviewed_extractor_after_guard(self):
        with patch('experiments.verify_qa_risk_pruning.verify') as guard:
            self.assertIs(verification._feature_extractor('pruned'), qa_risk_pruning.extract_features)
        guard.assert_called_once_with(verification.REPO)

    def test_unknown_mode_is_not_a_silent_fallback(self):
        for mode in ('cached', '', None, False):
            with self.subTest(mode=mode), patch.object(verification, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'mode'):
                    verification.verify(Path('not-read'), feature_mode=mode)
                read.assert_not_called()

    def test_matrix_extractor_is_applied_to_every_row(self):
        data = [{'id': str(i), 'context': 'Alpha Beta', 'answers': ['Alpha']} for i in range(3)]
        raw = {r['id']: original.raw_prediction(r['id'], True) for r in data}
        with patch.object(qa_risk_pruning, 'extract_features', wraps=qa_risk_pruning.extract_features) as extract:
            actual = verification.reconstruct_matrix(data, raw, feature_extractor=extract)
        self.assertEqual(extract.call_count, len(data))
        self.assertEqual(actual, verification.reconstruct_matrix(data, raw))


if __name__ == '__main__':
    unittest.main()
