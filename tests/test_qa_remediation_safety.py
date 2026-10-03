"""Dependency-free fault injection; these tests do not establish model quality."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import qa_remediation as runner
from lab.grounded_qa import binary_audit_score, encode, predict


def spec():
    return dict(seed=1, max_new_tokens=48, max_input_tokens=2048)


class AuditScoreTests(unittest.TestCase):
    def test_finite_score_and_extreme_logit_differences(self):
        self.assertEqual(binary_audit_score(3, 3), .5)
        self.assertAlmostEqual(binary_audit_score(2, 1), 1 / (1 + math.exp(-1)))
        self.assertEqual(binary_audit_score(1e300, -1e300), 1)
        self.assertEqual(binary_audit_score(-1e300, 1e300), 0)

    def test_nonfinite_or_invalid_logits_never_become_a_confident_answer(self):
        for value in (math.inf, -math.inf, math.nan, True, '1', None):
            for values in ((value, 1), (1, value)):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    binary_audit_score(*values)

    def test_input_token_limit_is_exact_and_never_truncates(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: 'test',
                                    encode=lambda *a, **k: [1, 2, 3])
        self.assertEqual(encode(tokenizer, [], 3)[0], [1, 2, 3])
        for limit in (2, 0, -1, True, 3.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                encode(tokenizer, [], limit)

    def test_bad_requests_rejected_before_importing_model_runtime(self):
        for context, question, count in (('', 'Q', 48), ('P', None, 48), ('P', 'Q', 0), ('P', 'Q', True)):
            with self.subTest(context=context, question=question, count=count), self.assertRaises(ValueError):
                predict(None, None, context, question, max_new_tokens=count)


class LockedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(label='1.5b-q8', mode='grounded', bits=8,
                             model_files_sha256={'config.json': 'a' * 64, 'model.safetensors': 'b' * 64},
                             data_sha256='c' * 64)
        self.spec = dict(**spec(), locked=True, allowed_runs=[deepcopy(self.identity)])

    def test_registered_identity_matches_exactly(self):
        self.assertTrue(runner.validate_locked_run(self.spec, **self.identity)['locked'])

    def test_wrong_model_data_label_mode_or_bits_is_rejected(self):
        mutations = dict(label='another', mode='legacy', bits=None, data_sha256='d' * 64,
                         model_files_sha256={'config.json': 'a' * 64})
        for key, value in mutations.items():
            actual = {**self.identity, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_locked_run(self.spec, **actual)
        actual = deepcopy(self.identity)
        actual['model_files_sha256']['unregistered.json'] = 'e' * 64
        with self.assertRaises(ValueError):
            runner.validate_locked_run(self.spec, **actual)

    def test_empty_duplicate_or_malformed_registration_rejected(self):
        registrations = [[], [self.identity, self.identity], [{**self.identity, 'extra': 1}],
                         [{**self.identity, 'model_files_sha256': {}}],
                         [{**self.identity, 'data_sha256': 'not-a-hash'}],
                         [{**self.identity, 'model_files_sha256': {'../config.json': 'a' * 64}}]]
        for allowed in registrations:
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                runner.validate_locked_run({**self.spec, 'allowed_runs': allowed}, **self.identity)

    def test_unlocked_development_is_explicit_and_spec_types_are_checked(self):
        self.assertFalse(runner.validate_locked_run(spec(), **self.identity)['locked'])
        for key, value in (('locked', 'true'), ('seed', True), ('max_new_tokens', 0), ('max_input_tokens', 1.5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_spec({**spec(), key: value})


class RunnerFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'fixture only; never loaded')
        protocol = self.root / 'spec.json'
        protocol.write_text(json.dumps(spec()))
        data = self.root / 'data.jsonl'
        records = [dict(id=str(index), context='A cat waits.', question='What waits?',
                        is_impossible=False, answers=['cat'], family_id=str(index)) for index in range(2)]
        data.write_text(''.join(json.dumps(row) + '\n' for row in records))
        self.args = SimpleNamespace(spec=protocol, data=data, model=model, label='fixture',
                                    bits=None, mode='grounded', output_dir=self.root / 'output')

    def result(self):
        result = json.loads((self.args.output_dir / 'run.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertIsNotNone(result['finished'])
        self.assertTrue((self.args.output_dir / 'checksums.json').is_file())
        return result

    def test_missing_spec_failure_is_saved_before_runtime_import(self):
        self.args.spec = self.root / 'missing.json'
        with self.assertRaises(FileNotFoundError):
            runner.run(self.args)
        self.assertEqual(self.result()['stage'], 'preflight')

    def test_dependency_import_failure_retains_source_and_failed_record(self):
        with patch.object(runner, '_load_runtime', side_effect=ImportError('test runtime missing')):
            with self.assertRaises(ImportError):
                runner.run(self.args)
        result = self.result()
        self.assertEqual(result['stage'], 'runtime_import')
        self.assertEqual(result['error_type'], 'ImportError')
        self.assertTrue((self.args.output_dir / 'source/lab/grounded_qa.py').is_file())

    def test_prequantized_source_is_rejected_even_when_bits_is_omitted(self):
        for field in ('quantization', 'quantization_config'):
            with self.subTest(field=field):
                (self.args.model / 'config.json').write_text(json.dumps({field: {'bits': 4}}))
                with self.assertRaises(ValueError):
                    runner.source_model_files(self.args.model)

    def test_existing_output_is_never_modified(self):
        self.args.output_dir.mkdir()
        marker = self.args.output_dir / 'keep.txt'
        marker.write_text('unchanged')
        with self.assertRaises(FileExistsError):
            runner.run(self.args)
        self.assertEqual(list(self.args.output_dir.iterdir()), [marker])
        self.assertEqual(marker.read_text(), 'unchanged')

    def fake_runtime(self):
        mx = SimpleNamespace(eval=lambda *a: None, float16='float16', floating='floating',
                             random=SimpleNamespace(seed=lambda *a: None),
                             metal=SimpleNamespace(device_info=lambda: {'fixture': True}))
        model = SimpleNamespace(parameters=lambda: {}, update=lambda *a: None)
        return (mx, lambda *a, **k: (model, {}, object()), None,
                lambda fn, params: params, None, None)

    def test_inference_failure_preserves_prior_rows_and_current_id(self):
        with patch.object(runner, '_load_runtime', return_value=self.fake_runtime()), \
                patch.object(runner, 'model_layout', return_value={'fixture': True}), \
                patch.object(runner.importlib.metadata, 'version', return_value='fixture'), \
                patch('lab.grounded_qa.predict', side_effect=[{'prediction': 'cat', 'confidence': .9}, RuntimeError('audit failed')]):
            with self.assertRaises(RuntimeError):
                runner.run(self.args)
        result = self.result()
        self.assertEqual(result['stage'], 'inference')
        self.assertEqual(result['current_id'], '1')
        self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(len((self.args.output_dir / 'predictions.jsonl').read_text().splitlines()), 1)
        self.assertNotIn('metrics', result)

    def test_nonfinite_prediction_is_rejected_without_corrupting_failure_json(self):
        with patch.object(runner, '_load_runtime', return_value=self.fake_runtime()), \
                patch.object(runner, 'model_layout', return_value={'fixture': True}), \
                patch.object(runner.importlib.metadata, 'version', return_value='fixture'), \
                patch('lab.grounded_qa.predict', return_value={'prediction': 'cat', 'confidence': math.nan}):
            with self.assertRaises(ValueError):
                runner.run(self.args)
        self.assertEqual(self.result()['predictions'], [])


class ModelLayoutTests(unittest.TestCase):
    def test_tensor_storage_and_layout_are_measured_independently_of_files(self):
        class Linear: pass
        class QuantizedLinear:
            bits = 8
            group_size = 64
        class Embedding: pass
        class QuantizedEmbedding: pass
        nn = SimpleNamespace(Linear=Linear, QuantizedLinear=QuantizedLinear, Embedding=Embedding,
                             QuantizedEmbedding=QuantizedEmbedding, Module=SimpleNamespace(is_module=lambda x: True))
        tensors = [('model.norm.weight', SimpleNamespace(dtype='float16', nbytes=8)),
                   ('model.linear.weight', SimpleNamespace(dtype='uint32', nbytes=32))]
        model = SimpleNamespace(parameters=lambda: tensors,
                                leaf_modules=lambda: [('linear', QuantizedLinear()), ('embedding', Embedding())])
        result = runner.model_layout(model, SimpleNamespace(float16='float16'), lambda values, **k: values, nn)
        self.assertEqual(result['parameter_tensor_bytes'], 40)
        self.assertEqual(result['module_counts']['quantized_linear'], 1)
        self.assertEqual(result['module_counts']['floating_embedding'], 1)
        self.assertTrue(result['normalizations_all_fp16'])
        self.assertEqual(result['dtype_distribution']['uint32']['tensor_bytes'], 32)


if __name__ == '__main__':
    unittest.main()
