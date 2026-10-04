"""Synthetic serving gates; these tests do not establish real QA quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import serve_qa_specialist as cli
from lab.extractive_qa import decode
from lab.qa_risk_calibration import FEATURE_NAMES, FIT_CONFIG


def verified():
    return dict(evidence_valid=True, protocol_sha256=cli.STUDY_SHA256,
                selection_sha256=cli.SELECTION_SHA256,
                variants={'int8': dict(bounded_task_eligible=True, evaluation_evaluated=True,
                    task_pass=True, threshold=.7, gate=dict(all_pass=True))})


def head(intercept=2.):
    return dict(schema='qa-risk-logistic-v1', feature_names=list(FEATURE_NAMES), config=dict(FIT_CONFIG),
                scaler=dict(mean=[0.] * 5, scale=[1.] * 5), weights=[0.] * 5,
                intercept=intercept, convergence=dict(converged=True), n_train=256)


def candidate():
    raw = [dict(start_logits=[0., 4., 1.], end_logits=[0., 1., 3.],
                offsets=[[0, 0], [0, 3], [4, 7]], context_mask=[False, True, True], cls_index=0)]
    result = decode('cat dog', raw)
    result['raw_windows'] = raw
    return result


class ServingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy = self.root / 'policy.json'
        self.policy.write_text(json.dumps(cli.EXPECTED_POLICY))
        self.study = self.root / 'study.json'
        self.study.write_text(json.dumps(dict(asset_manifest_sha256=cli.ASSET_MANIFEST_SHA256,
                                             dataset_sizes=dict(train=256))))
        self.evidence = self.root / 'evidence'
        (self.evidence / 'training').mkdir(parents=True)
        (self.evidence / 'training' / 'selection.json').write_text(json.dumps(dict(
            variants={'int8': dict(eligible=True, threshold=.7, model_sha256='fixture-training-head')})))
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        (self.assets / 'manifest.json').write_text('{}')
        self.request = self.root / 'request.json'
        self.request.write_text(json.dumps(dict(context='cat dog', question='Which animals?')))
        self.args = SimpleNamespace(policy=self.policy, evidence_root=self.evidence,
            study=self.study, asset_root=self.assets, input_json=self.request)

    def fake_digest(self, path):
        return {'study.json': cli.STUDY_SHA256, 'selection.json': cli.SELECTION_SHA256,
                'manifest.json': cli.ASSET_MANIFEST_SHA256}[Path(path).name]

    def service(self, intercept=2.):
        runtime = Mock()
        runtime.predict.return_value = candidate()
        return cli.LocalQAService(runtime, head(intercept), {'total_startup_seconds': 1.})

    def test_empty_forged_partial_untyped_or_changed_policies_fail_closed(self):
        invalid = [{}, {'enabled': True}, {**cli.EXPECTED_POLICY, 'enabled': 1},
                   {**cli.EXPECTED_POLICY, 'threshold': .6},
                   {**cli.EXPECTED_POLICY, 'variant': 'fp32'},
                   {**cli.EXPECTED_POLICY, 'study_sha256': 'a' * 64},
                   {**cli.EXPECTED_POLICY, 'selection_sha256': 'a' * 64},
                   {**cli.EXPECTED_POLICY, 'asset_manifest_sha256': 'a' * 64},
                   {**cli.EXPECTED_POLICY, 'scope': 'general_deployment'},
                   {**cli.EXPECTED_POLICY, 'schema_version': True},
                   {**cli.EXPECTED_POLICY, 'extra': 1}]
        changed = deepcopy(cli.EXPECTED_POLICY)
        changed['input_limits']['max_windows'] = 9
        invalid.append(changed)
        for record in invalid:
            self.policy.write_text(json.dumps(record))
            with self.subTest(policy=record), self.assertRaises(cli.QualityUnavailable):
                cli.load_policy(self.policy)

    def test_disabled_policy_does_not_read_request_verify_or_load(self):
        self.policy.write_text(json.dumps({**cli.EXPECTED_POLICY, 'enabled': False}))
        with patch.object(cli, 'read_request') as read_request, patch.object(cli, '_verify_evidence') as verify, \
             patch.object(cli, '_load_runtime') as runtime, patch.object(cli, '_rebuild_head') as rebuild:
            result, code = cli.serve(self.args)
        self.assertEqual((result['status'], code), ('unavailable_quality', 2))
        for action in (read_request, verify, runtime, rebuild):
            action.assert_not_called()

    def test_bad_input_or_gold_labels_rejected_before_model_load(self):
        cases = ['{}', '[]', '{"context":"cat","question":"Q","answers":["cat"]}',
                 '{"context":"cat","question":"Q","context":"dog"}',
                 '{"context":NaN,"question":"Q"}',
                 json.dumps({'context': ' ', 'question': 'Q'}),
                 json.dumps({'context': 'cat', 'question': ['Q']}),
                 json.dumps({'context': 'x' * 12001, 'question': 'Q'}),
                 json.dumps({'context': 'cat', 'question': 'x' * 1001})]
        for text in cases:
            self.request.write_text(text)
            with self.subTest(text=text[:80]), patch.object(cli, 'load_service') as load:
                result, code = cli.serve(self.args)
            self.assertEqual((result['status'], code), ('invalid_input', 1))
            load.assert_not_called()

    def test_parser_size_limit_and_invalid_utf8_fail_closed(self):
        for content in (b'x' * 131073, b'\xff'):
            self.request.write_bytes(content)
            with self.assertRaises(cli.InvalidInput):
                cli.read_request(self.request)

    def test_quality_result_must_pass_all_bound_fields(self):
        for key, value in (('bounded_task_eligible', False), ('evaluation_evaluated', False),
                           ('task_pass', False), ('threshold', .6)):
            report = verified()
            report['variants']['int8'][key] = value
            with self.subTest(key=key), self.assertRaises(cli.QualityUnavailable):
                cli.validate_verified_quality(deepcopy(cli.EXPECTED_POLICY), report)
        for key in ('evidence_valid', 'protocol_sha256', 'selection_sha256'):
            report = verified()
            report[key] = False
            with self.subTest(key=key), self.assertRaises(cli.QualityUnavailable):
                cli.validate_verified_quality(deepcopy(cli.EXPECTED_POLICY), report)
        report = verified()
        report['variants']['int8']['gate']['all_pass'] = False
        with self.assertRaises(cli.QualityUnavailable):
            cli.validate_verified_quality(deepcopy(cli.EXPECTED_POLICY), report)

    def test_wrong_study_or_selection_hash_stops_before_evidence_or_model(self):
        for name in ('study.json', 'selection.json'):
            def changed(path):
                return 'bad' if Path(path).name == name else self.fake_digest(path)
            with self.subTest(name=name), patch.object(cli, 'sha', side_effect=changed), \
                 patch.object(cli, '_verify_evidence') as verify, patch.object(cli, '_load_runtime') as runtime:
                with self.assertRaises(cli.QualityUnavailable):
                    cli.load_service(self.policy, self.evidence, self.study, self.assets)
            verify.assert_not_called()
            runtime.assert_not_called()

    def test_real_failed_quality_recomputation_cannot_be_enabled_by_policy(self):
        with patch.object(cli, 'sha', side_effect=self.fake_digest), \
             patch.object(cli, '_verify_evidence', side_effect=ValueError('tampered or failed evidence')), \
             patch.object(cli, '_load_runtime') as runtime, patch.object(cli, '_rebuild_head') as rebuild:
            with self.assertRaises(cli.QualityUnavailable):
                cli.load_service(self.policy, self.evidence, self.study, self.assets)
        runtime.assert_not_called()
        rebuild.assert_not_called()

    def test_training_only_head_is_rebuilt_before_cpu_asset_load(self):
        fake_runtime = Mock()
        with patch.object(cli, 'sha', side_effect=self.fake_digest), \
             patch.object(cli, '_verify_evidence', return_value=verified()) as verify, \
             patch.object(cli, '_rebuild_head', return_value=head()) as rebuild, \
             patch.object(cli, '_load_runtime', return_value=fake_runtime) as runtime:
            service = cli.load_service(self.policy, self.evidence, self.study, self.assets)
        verify.assert_called_once_with(self.evidence, self.study, feature_mode='pruned')
        rebuild.assert_called_once_with(self.evidence / 'training', 'fixture-training-head')
        runtime.assert_called_once_with(self.assets)
        self.assertIs(service.runtime, fake_runtime)
        for key in ('evidence_verification_seconds', 'training_head_rebuild_seconds',
                    'local_model_load_seconds', 'total_startup_seconds'):
            self.assertGreaterEqual(service.startup_timing[key], 0)

    def test_pruning_evidence_failure_stops_before_head_or_model(self):
        with patch.object(cli, 'sha', side_effect=self.fake_digest), \
             patch.object(cli, '_verify_evidence', return_value=verified()), \
             patch.object(cli, '_verify_pruning', side_effect=ValueError('pruning source drift'), create=True), \
             patch.object(cli, '_rebuild_head', return_value=head()) as rebuild, \
             patch.object(cli, '_load_runtime') as runtime:
            with self.assertRaisesRegex(cli.QualityUnavailable, 'pruning source drift'):
                cli.load_service(self.policy, self.evidence, self.study, self.assets)
        rebuild.assert_not_called()
        runtime.assert_not_called()

    def test_nontraining_rebuild_or_wrong_assets_never_load_model(self):
        wrong = head()
        wrong['n_train'] = 384
        with patch.object(cli, 'sha', side_effect=self.fake_digest), \
             patch.object(cli, '_verify_evidence', return_value=verified()), \
             patch.object(cli, '_rebuild_head', return_value=wrong), patch.object(cli, '_load_runtime') as runtime:
            with self.assertRaises(cli.QualityUnavailable):
                cli.load_service(self.policy, self.evidence, self.study, self.assets)
        runtime.assert_not_called()
        def changed_assets(path):
            return 'bad' if Path(path).name == 'manifest.json' else self.fake_digest(path)
        with patch.object(cli, 'sha', side_effect=changed_assets), \
             patch.object(cli, '_verify_evidence', return_value=verified()), \
             patch.object(cli, '_rebuild_head', return_value=head()), patch.object(cli, '_load_runtime') as runtime:
            with self.assertRaises(cli.ModelUnavailable):
                cli.load_service(self.policy, self.evidence, self.study, self.assets)
        runtime.assert_not_called()

    def test_answer_has_exact_offsets_hash_and_ranking_score_notice(self):
        service = self.service()
        response = service.answer(dict(context='cat dog', question='Which animals?'))
        self.assertEqual(response['status'], 'answer')
        self.assertEqual((response['answer'], response['start'], response['end']), ('cat dog', 0, 7))
        self.assertEqual(response['context_sha256'], hashlib.sha256(b'cat dog').hexdigest())
        self.assertEqual(response['score_kind'], cli.SCORE_KIND)
        self.assertNotIn('startup_timing', response)
        service.runtime.predict.assert_called_once_with('cat dog', 'Which animals?')

    def test_abstention_contains_no_answer_offsets_or_context_hash(self):
        response = self.service(intercept=-2.).answer(dict(context='cat dog', question='Q'))
        self.assertEqual(response['status'], 'abstain')
        for key in ('answer', 'start', 'end', 'context_sha256'):
            self.assertNotIn(key, response)

    def test_token_window_limit_errors_are_input_errors_not_refusals(self):
        for message in ('Question exceeds the frozen token limit; no silent truncation',
                        'Context exceeds the bounded window count; no silent dropping'):
            service = self.service()
            service.runtime.predict.side_effect = ValueError(message)
            with self.subTest(message=message), self.assertRaises(cli.InvalidInput):
                service.answer(dict(context='cat dog', question='Q'))

    def test_model_or_nonfinite_feature_errors_cannot_be_scored_as_refusal(self):
        service = self.service()
        service.runtime.predict.side_effect = ValueError('QA output tensor shape differs from input tokens')
        with self.assertRaises(cli.ModelUnavailable):
            service.answer(dict(context='cat dog', question='Q'))
        service = self.service()
        service.runtime.predict.return_value['raw_windows'][0]['start_logits'][1] = float('nan')
        with self.assertRaises(cli.ModelUnavailable):
            service.answer(dict(context='cat dog', question='Q'))

    def test_cli_status_mapping_and_startup_timing(self):
        for error, expected, code in ((cli.QualityUnavailable('quality'), 'unavailable_quality', 2),
                                      (cli.ModelUnavailable('assets'), 'unavailable_model', 3)):
            with patch.object(cli, 'load_service', side_effect=error):
                response, actual = cli.serve(self.args)
            self.assertEqual((response['status'], actual), (expected, code))
        with patch.object(cli, 'load_service', return_value=self.service()):
            response, code = cli.serve(self.args)
        self.assertEqual((response['status'], code), ('answer', 0))
        self.assertIn('startup_timing', response)


if __name__ == '__main__':
    unittest.main()
