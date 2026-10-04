"""Release claim regressions must fail even if lower-level records are valid."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.verify_release_rc4 import validate_results, verify

ROOT=Path(__file__).resolve().parents[1]


class RC4ClaimTests(unittest.TestCase):
    def setUp(self):
        self.historical=dict(technical_acceptance='pass',confirmation_gate_passed=False,
                             qa_remediation_quality_passed=False,qa_answer_entrypoint='unavailable_quality')
        self.specialist=json.loads((ROOT/'results/qa-specialist-v1/verification.json').read_text())
        self.risk=json.loads((ROOT/'results/qa-risk-review-v1/verification.json').read_text())

    def test_real_release_claims_distinguish_failed_optimization_and_task_pass(self):
        validate_results(self.historical,self.specialist,self.risk)

    def test_relabelled_historical_or_specialist_failure_rejected(self):
        history=deepcopy(self.historical);history['confirmation_gate_passed']=True
        with self.assertRaises(ValueError):validate_results(history,self.specialist,self.risk)
        specialist=deepcopy(self.specialist);specialist['overall_success']=True
        with self.assertRaises(ValueError):validate_results(self.historical,specialist,self.risk)
        specialist=deepcopy(self.specialist);specialist['variants']['fp32']['quality_gate']['all_pass']=True
        with self.assertRaises(ValueError):validate_results(self.historical,specialist,self.risk)

    def test_task_and_compression_claims_cannot_be_conflated(self):
        for mutate in [lambda r:r['compression_gate'].update(passed=True),
                       lambda r:r['variants']['fp32'].update(evaluation_evaluated=True),
                       lambda r:r['variants']['int8']['summary']['selective'].update(accepted=28),
                       lambda r:r.update(parameters_distributed=True),
                       lambda r:r['variants']['int8'].update(threshold=.6)]:
            risk=deepcopy(self.risk);mutate(risk)
            with self.assertRaises(ValueError):validate_results(self.historical,self.specialist,risk)

    def test_target_archive_cannot_silently_read_another_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'target source archive'):verify(tmp)

    def test_changed_protocol_rejected_before_new_study_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'configs/qa-specialist/study.json';study.parent.mkdir(parents=True);study.write_text('{}')
            with patch('experiments.verify_release_rc4.REPO',root), \
                 patch('experiments.verify_release_rc4.importlib.metadata.version',return_value='2.2.6'), \
                 patch('experiments.verify_release_rc4.verify_history',return_value=self.historical), \
                 patch('experiments.verify_release_rc4.verify_specialist') as specialist:
                with self.assertRaisesRegex(ValueError,'Frozen specialist study'):verify(root)
                specialist.assert_not_called()


if __name__=='__main__':unittest.main()
