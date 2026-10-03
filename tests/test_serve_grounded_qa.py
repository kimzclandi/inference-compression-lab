import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import serve_grounded_qa as cli
from lab.qa_gate import DEFAULT_CONSTRAINTS
from lab.quantization_diagnostics import sha
from lab.selective_qa import evaluate_selective


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


class ResearchCliTests(unittest.TestCase):
    def setUp(self):
        # resolve() avoids OS-owned /tmp -> /private/tmp directory aliases.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.evidence = self.root / 'evidence'
        self.study_dir = self.evidence / 'study'
        self.run_dir = self.study_dir / 'confirmation-fp16'
        # Synthetic two-row contract fixtures do not claim to pass the real
        # 128/256-example study. Mock only the outer full-study verifier, while
        # testing that it is called and its result/selection binding is obeyed.
        self.full_verify_patch = patch('experiments.verify_qa_remediation.verify', return_value={
            'overall_success': True, 'variants': {'fp16': {'threshold': .5}, 'q8': {'threshold': .5}}})
        self.full_verify = self.full_verify_patch.start()
        self.addCleanup(self.full_verify_patch.stop)
        # Synthetic installed-version metadata: no MLX import/model execution.
        self.runtime_versions = {name: 'cpu-fixture-' + name for name in cli.RUNTIME_PACKAGES}
        self.metadata_patch = patch.object(cli.importlib.metadata, 'version', side_effect=self.runtime_versions.__getitem__)
        self.metadata = self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)
        self.model = self.root / 'model'
        self.model.mkdir()
        write_json(self.model / 'config.json', {'model_type': 'qwen2'})
        (self.model / 'model.safetensors').write_bytes(b'CPU fixture, never loaded as model weights')
        model_hashes = {p.name: sha(p) for p in self.model.iterdir()}
        self.data = [dict(id='yes', context='A cat waits.', question='What waits?',
                          answers=['cat'], is_impossible=False, family_id='one', split='confirmation'),
                     dict(id='no', context='A cat waits.', question='How old is the cat?',
                          answers=[], is_impossible=True, family_id='one', split='confirmation')]
        self.predictions = [dict(id='yes', prediction='cat', confidence=.9, stop_reason='eos'),
                            dict(id='no', prediction='NO_ANSWER', confidence=0, stop_reason='eos')]
        self.run_dir.mkdir(parents=True)
        self.write_rows('data.jsonl', self.data)
        self.write_rows('predictions.jsonl', self.predictions)
        self.protocol = dict(locked=True, seed=7, max_input_tokens=2048, max_new_tokens=48,
                             allowed_runs=[dict(label='fixture-fp16', mode='grounded', bits=None,
                                                model_files_sha256=model_hashes,
                                                data_sha256=sha(self.run_dir / 'data.jsonl'))])
        write_json(self.run_dir / 'protocol.json', self.protocol)
        write_json(self.study_dir / 'protocol.json', self.protocol)
        write_json(self.study_dir / 'selection.json', {'variants': {'fp16': {'threshold': .5}, 'q8': {'threshold': .5}}})
        write_json(self.study_dir / 'confirmation-protocol.json', {'thresholds': {'fp16': .5, 'q8': .5}})
        source_hashes = {}
        for name in cli.BOUND_CODE:
            destination = self.run_dir / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cli.REPO_ROOT / name, destination)
            source_hashes[name] = sha(destination)
        self.run = dict(status='complete', mode='grounded', bits=None, model_label='fixture-fp16',
                        python=cli.platform.python_version(), packages=dict(self.runtime_versions),
                        dataset_sha256=sha(self.run_dir / 'data.jsonl'),
                        spec_sha256=sha(self.run_dir / 'protocol.json'), predictions=self.predictions,
                        source_model_files={name: {'sha256': digest} for name, digest in model_hashes.items()},
                        source_sha256=source_hashes)
        write_json(self.run_dir / 'run.json', self.run)
        summary, _ = evaluate_selective(self.data, self.predictions, .5)
        self.quality = dict(split='confirmation', threshold=.5, constraints=DEFAULT_CONSTRAINTS,
                            selective=summary['selective'])
        write_json(self.evidence / 'quality.json', self.quality)
        self.policy = dict(enabled=True, mode='grounded', bits=None, threshold=.5,
                           model_files_sha256=model_hashes, study_dir='study', run_dir='study/confirmation-fp16',
                           source_evidence_sha256={}, grounded_sha256=source_hashes['lab/grounded_qa.py'])
        self.policy_path = self.root / 'policy.json'
        self.refresh_policy()
        self.args = argparse.Namespace(policy=self.policy_path, evidence_dir=self.evidence,
                                       model=self.model, context='A cat waits.', context_file=None,
                                       question='What waits?', question_file=None)

    def write_rows(self, name, items):
        (self.run_dir / name).write_text(''.join(json.dumps(item) + '\n' for item in items))

    def refresh_policy(self):
        self.policy['source_evidence_sha256'] = {
            p.relative_to(self.evidence).as_posix(): sha(p)
            for p in self.evidence.rglob('*') if p.is_file()}
        write_json(self.policy_path, self.policy)

    def assert_preflight_failure(self):
        with patch.object(cli, '_load_and_predict') as runtime:
            result, code = cli.serve(self.args)
        runtime.assert_not_called()
        self.assertEqual(code, 1)
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('answer', result)
        return result

    def test_disabled_policy_never_reads_inputs_evidence_or_runtime(self):
        write_json(self.policy_path, {'enabled': False, 'reason': 'calibration had no feasible candidate'})
        with patch.object(cli, 'verify_enabled_evidence') as evidence, \
                patch.object(cli, '_input_text') as inputs, \
                patch.object(cli, '_load_and_predict') as runtime:
            result, code = cli.serve(self.args)
        evidence.assert_not_called()
        inputs.assert_not_called()
        runtime.assert_not_called()
        self.assertEqual((result['status'], code), ('unavailable_quality', 2))
        self.assertNotIn('answer', result)
        self.assertNotIn('NO_ANSWER', json.dumps(result))

    def test_disabled_main_needs_only_policy_and_emits_single_json(self):
        write_json(self.policy_path, {'enabled': False})
        capture = io.StringIO()
        with redirect_stdout(capture), patch.object(cli, '_load_and_predict') as runtime:
            code = cli.main(['--policy', str(self.policy_path)])
        runtime.assert_not_called()
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(capture.getvalue())['status'], 'unavailable_quality')

    def test_enabled_only_calls_runtime_after_complete_valid_evidence(self):
        with patch.object(cli, '_load_and_predict', return_value=dict(prediction='cat', confidence=.9, stop_reason='eos')) as runtime:
            result, code = cli.serve(self.args)
        runtime.assert_called_once_with(self.model, self.policy, 'A cat waits.', 'What waits?', 7)
        self.full_verify.assert_called_once_with(self.study_dir, cli.REPO_ROOT / 'configs/qa-remediation/study.json')
        self.assertEqual((result['status'], code), ('answer', 0))
        self.assertEqual(result['answer'], self.args.context[result['start']:result['end']])
        self.assertEqual(len(result['context_sha256']), 64)

    def test_low_confidence_invalid_and_truncated_outputs_never_emit_answers(self):
        for output, confidence, stop, status in (
                ('cat', .2, 'eos', 'abstain'), ('invented', .9, 'eos', 'invalid'),
                ('cat', .9, 'max_new_tokens', 'invalid'), ('NO_ANSWER', 0, 'eos', 'abstain')):
            with self.subTest(output=output, stop=stop), patch.object(cli, '_load_and_predict',
                    return_value=dict(prediction=output, confidence=confidence, stop_reason=stop)):
                result, code = cli.serve(self.args)
            self.assertEqual((result['status'], code), (status, 0))
            self.assertEqual(result['action'], 'abstain')
            self.assertNotIn('answer', result)
            self.assertNotIn('start', result)

    def test_runtime_exception_is_error_not_model_refusal(self):
        with patch.object(cli, '_load_and_predict', side_effect=RuntimeError('fixture runtime failure')):
            result, code = cli.serve(self.args)
        self.assertEqual((result['status'], code), ('error', 1))
        self.assertEqual(result['error_type'], 'RuntimeError')
        self.assertNotIn('answer', result)

    def test_unbound_or_tampered_evidence_rejected_before_runtime(self):
        (self.run_dir / 'predictions.jsonl').write_text('{}\n')
        self.assert_preflight_failure()

    def test_rehashed_forged_quality_rejected_by_recomputation(self):
        self.quality['selective']['accepted_precision'] = .999
        write_json(self.evidence / 'quality.json', self.quality)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('recomputation', result['message'])

    def test_rehashed_calibration_cannot_be_confirmation(self):
        self.quality['split'] = 'calibration'
        write_json(self.evidence / 'quality.json', self.quality)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('confirmation', result['message'])

    def test_rehashed_lowered_limits_cannot_enable(self):
        self.quality['constraints'] = dict(DEFAULT_CONSTRAINTS, min_accepted_precision=0)
        write_json(self.evidence / 'quality.json', self.quality)
        self.refresh_policy()
        self.assert_preflight_failure()

    def test_complete_study_failure_cannot_be_bypassed_by_local_quality(self):
        self.full_verify.return_value = {'overall_success': False, 'status': 'calibration_failed'}
        result = self.assert_preflight_failure()
        self.assertIn('complete frozen study', result['message'])

    def test_threshold_and_confirmation_directory_bound_to_selection(self):
        self.policy['threshold'] = .6
        self.quality['threshold'] = .6
        write_json(self.evidence / 'quality.json', self.quality)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('frozen calibration selection', result['message'])
        self.policy['threshold'] = .5
        other = self.study_dir / 'lookalike'
        shutil.copytree(self.run_dir, other)
        self.policy['run_dir'] = 'study/lookalike'
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('selected study confirmation precision', result['message'])

    def test_missing_selection_hash_cannot_enable(self):
        del self.policy['source_evidence_sha256']['study/selection.json']
        write_json(self.policy_path, self.policy)
        self.assert_preflight_failure()
        self.full_verify.assert_not_called()

    def test_raw_predictions_cannot_disagree_with_run(self):
        self.predictions[0]['prediction'] = 'waits'
        self.write_rows('predictions.jsonl', self.predictions)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('Run predictions differ', result['message'])

    def test_unsafe_path_and_evidence_symlink_rejected(self):
        self.policy['source_evidence_sha256']['../policy.json'] = sha(self.policy_path)
        write_json(self.policy_path, self.policy)
        self.assert_preflight_failure()
        self.refresh_policy()
        data = self.run_dir / 'data.jsonl'
        outside = self.root / 'outside-data.jsonl'
        data.rename(outside)
        data.symlink_to(outside)
        self.assert_preflight_failure()

    def test_model_file_symlink_to_hf_blob_is_allowed(self):
        weights = self.model / 'model.safetensors'
        blob = self.root / 'hf-blob'
        weights.rename(blob)
        weights.symlink_to(blob)
        verified = cli.verify_enabled_evidence(self.policy, self.evidence, self.model)
        self.assertTrue(verified['file_hashes_verified'])

    def test_extra_nested_or_modified_model_files_rejected(self):
        extra = self.model / 'unexpected.txt'
        extra.write_text('not in manifest')
        self.assert_preflight_failure()
        extra.unlink()
        nested = self.model / 'nested'
        nested.mkdir()
        self.assert_preflight_failure()
        nested.rmdir()
        (self.model / 'model.safetensors').write_bytes(b'changed weights')
        self.assert_preflight_failure()

    def test_changed_source_code_binding_rejected(self):
        source = self.run_dir / 'source/lab/grounded_qa.py'
        source.write_text(source.read_text() + '\n# changed\n')
        self.run['source_sha256']['lab/grounded_qa.py'] = sha(source)
        write_json(self.run_dir / 'run.json', self.run)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('Current inference/scoring source differs', result['message'])

    def test_changed_loader_source_binding_rejected(self):
        source = self.run_dir / 'source/experiments/qa_remediation.py'
        source.write_text(source.read_text() + '\n# changed loader\n')
        self.run['source_sha256']['experiments/qa_remediation.py'] = sha(source)
        write_json(self.run_dir / 'run.json', self.run)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('Current inference/scoring source differs', result['message'])

    def test_installed_runtime_version_drift_fails_before_model_load(self):
        self.runtime_versions['mlx'] = 'changed-version'
        result = self.assert_preflight_failure()
        self.assertIn('dependency versions differ', result['message'])

    def test_python_version_drift_fails_before_model_load(self):
        with patch.object(cli.platform, 'python_version', return_value='different-python'):
            result = self.assert_preflight_failure()
        self.assertIn('Python version differs', result['message'])

    def test_missing_recorded_runtime_version_fails_before_model_load(self):
        del self.run['packages']['mlx-lm']
        write_json(self.run_dir / 'run.json', self.run)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('complete runtime package version set', result['message'])

    def test_unlocked_or_nonmatching_registration_rejected(self):
        self.protocol['allowed_runs'][0]['mode'] = 'legacy'
        write_json(self.run_dir / 'protocol.json', self.protocol)
        self.run['spec_sha256'] = sha(self.run_dir / 'protocol.json')
        write_json(self.run_dir / 'run.json', self.run)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('exactly one entry', result['message'])

    def test_fake_completed_run_with_truncation_rejected(self):
        self.predictions[0]['stop_reason'] = 'max_new_tokens'
        self.write_rows('predictions.jsonl', self.predictions)
        self.run['predictions'] = self.predictions
        write_json(self.run_dir / 'run.json', self.run)
        self.refresh_policy()
        result = self.assert_preflight_failure()
        self.assertIn('complete generations', result['message'])


if __name__ == '__main__':
    unittest.main()
