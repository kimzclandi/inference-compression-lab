"""Synthetic runtime/runner fault injection; not model-quality evidence."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import qa_specialist as runner
from lab.artifact_integrity import file_hashes
from lab import qa_specialist_runtime as runtime_module
from lab.qa_specialist_runtime import ExtractiveRuntime, LIMITS, RUNTIME_PACKAGES, validate_assets
from lab.quantization_diagnostics import sha


class FakeVector:
    def __init__(self, values):
        self.values = values

    def astype(self, kind):
        return self

    def tolist(self):
        return list(self.values)


class FakeTensor:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = (1, len(values)) if shape is None else shape

    def __getitem__(self, index):
        assert index == 0
        return FakeVector(self.values)


class FakeEncoding(dict):
    def __init__(self, count=1):
        super().__init__(input_ids=[[0, 10, 2, 20, 21, 2]] * count,
                         attention_mask=[[1, 1, 1, 1, 1, 0]] * count,
                         offset_mapping=[[[0, 0], [0, 40], [0, 0], [0, 3], [4, 7], [0, 0]]] * count)
        self.sequence = [[None, 0, None, 1, 1, None] for _ in range(count)]

    def sequence_ids(self, index):
        return self.sequence[index]


def fake_runtime(encoding=None, question_tokens=2, logits=None):
    instance = ExtractiveRuntime.__new__(ExtractiveRuntime)
    encoded = FakeEncoding() if encoding is None else encoding
    tokenizer = Mock(cls_token_id=0)
    tokenizer.encode.return_value = list(range(question_tokens))
    tokenizer.return_value = encoded
    instance.tokenizer = tokenizer
    instance.session = Mock()
    instance.session.run.return_value = (logits if logits is not None else
        [FakeTensor([0, 100, 90, 8, 0, 99]), FakeTensor([0, 100, 90, 7, 0, 99])])
    return instance


@contextmanager
def fake_numpy():
    module = SimpleNamespace(asarray=lambda values, dtype: deepcopy(values), int64='int64')
    with patch.dict('sys.modules', {'numpy': module}):
        yield


class RuntimeInputTests(unittest.TestCase):
    def test_exact_offsets_masks_and_raw_logits_preserved(self):
        engine = fake_runtime()
        with fake_numpy():
            result = engine.predict('cat dog', 'What animal?')
        self.assertEqual(result['prediction'], 'cat')
        raw = result['raw_windows'][0]
        self.assertEqual(raw['context_mask'], [False, False, False, True, True, False])
        self.assertEqual(raw['offsets'][1], [0, 40])
        self.assertEqual(raw['start_logits'], [0, 100, 90, 8, 0, 99])
        self.assertEqual((result['start'], result['end']), (0, 3))
        self.assertEqual(result['feature_count'], 1)
        self.assertEqual(result['input_tokens'], [6])
        called = engine.tokenizer.call_args.kwargs
        self.assertEqual(called['truncation'], 'only_second')
        self.assertEqual(called['max_length'], 384)
        self.assertEqual(called['stride'], 128)
        self.assertTrue(called['return_overflowing_tokens'])
        self.assertTrue(called['return_offsets_mapping'])
        self.assertFalse(called['padding'])

    def test_padding_token_in_context_sequence_is_still_excluded(self):
        encoded = FakeEncoding()
        encoded.sequence[0][-1] = 1
        engine = fake_runtime(encoded)
        with fake_numpy():
            result = engine.predict('cat dog', 'What animal?')
        self.assertFalse(result['raw_windows'][0]['context_mask'][-1])
        self.assertEqual(result['prediction'], 'cat')

    def test_optional_logit_omission_keeps_answer_but_not_raw_windows(self):
        with fake_numpy():
            result = fake_runtime().predict('cat dog', 'Q', include_logits=False)
        self.assertEqual(result['prediction'], 'cat')
        self.assertNotIn('raw_windows', result)

    def test_invalid_context_and_question_fail_before_tokenization_or_inference(self):
        cases = [(None, 'Q'), ('', 'Q'), (' \t', 'Q'), ('x' * 12001, 'Q'),
                 ('cat', None), ('cat', ''), ('cat', '\t'), ('cat', 'x' * 1001)]
        for context, question in cases:
            engine = fake_runtime()
            with self.subTest(context=context, question=question), self.assertRaises(ValueError):
                engine.predict(context, question)
            engine.tokenizer.assert_not_called()
            engine.session.run.assert_not_called()

    def test_question_token_limit_is_exact_and_never_silently_truncates(self):
        with fake_numpy():
            self.assertEqual(fake_runtime(question_tokens=64).predict('cat dog', 'Q')['prediction'], 'cat')
        engine = fake_runtime(question_tokens=65)
        with self.assertRaisesRegex(ValueError, 'token limit'):
            engine.predict('cat dog', 'Q')
        engine.tokenizer.assert_not_called()
        engine.session.run.assert_not_called()

    def test_window_limit_rejects_zero_and_nine_without_partial_inference(self):
        for count in (0, 9):
            engine = fake_runtime(FakeEncoding(count))
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'window count'):
                engine.predict('cat dog', 'Q')
            engine.session.run.assert_not_called()
        with fake_numpy():
            result = fake_runtime(FakeEncoding(8)).predict('cat dog', 'Q')
        self.assertEqual(result['feature_count'], 8)

    def test_missing_cls_rejected_before_session(self):
        encoded = FakeEncoding()
        encoded['input_ids'][0][0] = 100
        engine = fake_runtime(encoded)
        with fake_numpy(), self.assertRaisesRegex(ValueError, 'CLS'):
            engine.predict('cat dog', 'Q')
        engine.session.run.assert_not_called()

    def test_malformed_model_tensor_shape_rejected(self):
        for shape in ((6,), (1, 5), (2, 6)):
            engine = fake_runtime(logits=[FakeTensor([0] * 6, shape), FakeTensor([0] * 6)])
            with fake_numpy(), self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'tensor shape'):
                engine.predict('cat dog', 'Q')

    def test_tokenizer_shape_and_value_contracts_fail_before_session(self):
        mutations = [
            lambda e: e['attention_mask'].append([1] * 6),
            lambda e: e['offset_mapping'].clear(),
            lambda e: e['attention_mask'][0].pop(),
            lambda e: e['offset_mapping'][0].pop(),
            lambda e: e.sequence[0].pop(),
            lambda e: e['input_ids'][0].__setitem__(1, True),
            lambda e: e['input_ids'][0].__setitem__(1, -1),
            lambda e: e['attention_mask'][0].__setitem__(3, '0'),
            lambda e: e['attention_mask'][0].__setitem__(3, True),
            lambda e: e['attention_mask'][0].__setitem__(3, 2),
            lambda e: e.sequence[0].__setitem__(3, True),
            lambda e: e.sequence[0].__setitem__(3, '1'),
            lambda e: e.sequence[0].__setitem__(3, 2),
        ]
        for index, mutate in enumerate(mutations):
            encoded = FakeEncoding()
            mutate(encoded)
            engine = fake_runtime(encoded)
            with self.subTest(mutation=index), fake_numpy(), self.assertRaises(ValueError):
                engine.predict('cat dog', 'Q')
            engine.session.run.assert_not_called()

    def test_oversized_feature_rejected_instead_of_relying_on_tokenizer(self):
        encoded = FakeEncoding()
        encoded['input_ids'][0] = [0] + [10] * 384
        encoded['attention_mask'][0] = [1] * 385
        encoded['offset_mapping'][0] = [[0, 0]] + [[0, 3]] * 384
        encoded.sequence[0] = [None] + [1] * 384
        engine = fake_runtime(encoded)
        with fake_numpy(), self.assertRaisesRegex(ValueError, 'feature length'):
            engine.predict('cat dog', 'Q')
        engine.session.run.assert_not_called()

    def test_decoder_rejects_nonfinite_logits_even_on_question_token(self):
        engine = fake_runtime(logits=[FakeTensor([0, float('nan'), 0, 4, 0, 0]),
                                     FakeTensor([0] * 6)])
        with fake_numpy(), self.assertRaisesRegex(ValueError, 'finite'):
            engine.predict('cat dog', 'Q')


class AssetIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in ('fp32.onnx', 'int8.onnx', 'tokenizer.json', 'UPSTREAM-README.md'):
            (self.root / name).write_text('synthetic identity fixture: ' + name)
        self.versions = {name: 'fixture-1' for name in RUNTIME_PACKAGES}
        self.manifest = dict(tokenizer_files={'tokenizer.json': sha(self.root / 'tokenizer.json')},
                             models={name: dict(file=name + '.onnx', sha256=sha(self.root / (name + '.onnx')))
                                     for name in ('fp32', 'int8')},
                             source_files={'README.md': dict(sha256=sha(self.root / 'UPSTREAM-README.md'))},
                             packages=self.versions)
        self.save_manifest()
        patched = patch.object(runtime_module.importlib.metadata, 'version', side_effect=self.versions.__getitem__)
        patched.start()
        self.addCleanup(patched.stop)

    def save_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        self.digest = sha(self.root / 'manifest.json')

    def test_complete_model_tokenizer_attribution_and_versions_match(self):
        self.assertEqual(validate_assets(self.root, self.digest), self.manifest)

    def test_modified_manifest_model_tokenizer_and_attribution_are_rejected(self):
        for name in ('manifest.json', 'fp32.onnx', 'int8.onnx', 'tokenizer.json', 'UPSTREAM-README.md'):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b' mutation')
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_assets(self.root, self.digest)
            path.write_bytes(original)

    def test_unlisted_file_and_missing_precision_are_rejected(self):
        (self.root / 'unlisted.bin').write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'Unexpected/missing'):
            validate_assets(self.root, self.digest)
        (self.root / 'unlisted.bin').unlink()
        del self.manifest['models']['int8']
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'precision'):
            validate_assets(self.root, self.digest)

    def test_wrong_runtime_version_is_rejected(self):
        self.versions['onnxruntime'] = 'fixture-2'
        with self.assertRaisesRegex(ValueError, 'dependency versions'):
            validate_assets(self.root, self.digest)

    def test_manifest_cannot_reference_parent_file(self):
        self.manifest['tokenizer_files'] = {'../tokenizer.json': 'a' * 64}
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            validate_assets(self.root, self.digest)

    def test_invalid_precision_or_thread_contract_fails_before_loading_assets(self):
        for variant, threads in (('fp16', 4), ('fp32', True), ('int8', 4.), ('fp32', 1)):
            with self.subTest(variant=variant, threads=threads), self.assertRaises(ValueError):
                ExtractiveRuntime(self.root / 'missing', variant, 'a' * 64, threads)

    def test_provider_and_fast_tokenizer_are_checked_on_initialization(self):
        session = Mock()
        session.get_providers.return_value = ['CPUExecutionProvider']
        fake_ort = SimpleNamespace(SessionOptions=lambda: SimpleNamespace(),
                                   ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
                                   GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL='all'),
                                   InferenceSession=Mock(return_value=session))
        auto = Mock()
        auto.from_pretrained.return_value = SimpleNamespace(is_fast=True)
        with patch.dict('sys.modules', {'onnxruntime': fake_ort, 'transformers': SimpleNamespace(AutoTokenizer=auto)}):
            ExtractiveRuntime(self.root, 'int8', self.digest)
            call = fake_ort.InferenceSession.call_args
            self.assertEqual(call.kwargs['providers'], ['CPUExecutionProvider'])
            options = call.kwargs['sess_options']
            self.assertEqual((options.intra_op_num_threads, options.inter_op_num_threads), (4, 1))
            self.assertEqual(options.execution_mode, 'sequential')
            self.assertEqual(options.graph_optimization_level, 'all')
            self.assertTrue(auto.from_pretrained.call_args.kwargs['local_files_only'])
            self.assertTrue(auto.from_pretrained.call_args.kwargs['use_fast'])
            session.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            with self.assertRaisesRegex(ValueError, 'provider'):
                ExtractiveRuntime(self.root, 'fp32', self.digest)
            auto.from_pretrained.return_value = SimpleNamespace(is_fast=False)
            with self.assertRaisesRegex(ValueError, 'Fast tokenizer'):
                ExtractiveRuntime(self.root, 'fp32', self.digest)


class RunnerEvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        for name in runner.SOURCE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic source identity fixture\n')
        self.records = [dict(id=str(i), split='calibration', context='cat dog', question='Q') for i in range(2)]
        data = self.root / 'data.jsonl'
        data.write_text(''.join(json.dumps(row) + '\n' for row in self.records))
        self.spec = dict(locked=True, data_sha256={'calibration': sha(data)},
                         source_sha256={name: sha(self.root / name) for name in runner.SOURCE_FILES},
                         input_limits=LIMITS, asset_manifest_sha256='a' * 64)
        spec = self.root / 'spec.json'
        spec.write_text(json.dumps(self.spec))
        self.args = SimpleNamespace(spec=spec, data=data, variant='fp32', selection=None,
                                    asset_root=self.root / 'assets', output_dir=self.root / 'output')
        self.args.asset_root.mkdir()
        (self.args.asset_root / 'manifest.json').write_text('{"fixture": true}')
        version = patch.object(runtime_module.importlib.metadata, 'version', return_value='fixture-1')
        version.start()
        self.addCleanup(version.stop)

    def result(self):
        root = self.args.output_dir
        record = json.loads((root / 'run.json').read_text())
        checksums = json.loads((root / 'checksums.json').read_text())
        self.assertEqual(checksums, file_hashes(root, exclude=('checksums.json',)))
        return record

    def test_wrong_model_precision_dataset_or_source_rejected_by_preflight(self):
        self.assertEqual(runner.validate_run(self.spec, self.args.data, 'fp32', self.root), 'calibration')
        for spec, variant in (({**self.spec, 'locked': False}, 'fp32'),
                              (self.spec, 'fp16'),
                              ({**self.spec, 'data_sha256': {'calibration': 'b' * 64}}, 'fp32'),
                              ({**self.spec, 'source_sha256': {}}, 'fp32')):
            with self.subTest(spec=spec, variant=variant), self.assertRaises(ValueError):
                runner.validate_run(spec, self.args.data, variant, self.root)
        (self.root / runner.SOURCE_FILES[0]).write_text('changed after freeze')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            runner.validate_run(self.spec, self.args.data, 'fp32', self.root)

    def test_missing_spec_failure_creates_auditable_preflight_record(self):
        self.args.spec = self.root / 'missing.json'
        with self.assertRaises(FileNotFoundError):
            runner.run(self.args)
        record = self.result()
        self.assertEqual((record['status'], record['stage'], record['completed_predictions']),
                         ('failed', 'preflight', 0))
        self.assertEqual(record['error_type'], 'FileNotFoundError')

    def test_identity_failure_also_leaves_a_preflight_record(self):
        with patch.object(runner, 'git_identity', side_effect=RuntimeError('synthetic identity fault')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic identity fault'):
                runner.run(self.args)
        record = self.result()
        self.assertEqual((record['status'], record['stage'], record['completed_predictions']),
                         ('failed', 'preflight', 0))

    def test_loader_failure_preserves_copied_protocol_data_and_source(self):
        with patch.object(runtime_module, 'ExtractiveRuntime', side_effect=RuntimeError('synthetic loader fault')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic loader fault'):
                runner.run(self.args)
        record = self.result()
        self.assertEqual((record['status'], record['stage']), ('failed', 'load'))
        self.assertEqual(sha(self.args.output_dir / 'protocol.json'), sha(self.args.spec))
        self.assertEqual(sha(self.args.output_dir / 'data.jsonl'), sha(self.args.data))
        self.assertEqual(file_hashes(self.args.output_dir / 'source'), self.spec['source_sha256'])

    def test_inference_failure_preserves_completed_append_only_predictions(self):
        engine = Mock(manifest={'fixture': True})
        engine.predict.side_effect = [dict(prediction='cat', confidence=.9), RuntimeError('synthetic inference fault')]
        with patch.object(runtime_module, 'ExtractiveRuntime', return_value=engine):
            with self.assertRaisesRegex(RuntimeError, 'synthetic inference fault'):
                runner.run(self.args)
        record = self.result()
        self.assertEqual((record['status'], record['stage'], record['current_id'], record['completed_predictions']),
                         ('failed', 'inference', '1', 1))
        raw = (self.args.output_dir / 'predictions.jsonl').read_text().splitlines()
        self.assertEqual(len(raw), 1)
        self.assertEqual(json.loads(raw[0]), dict(prediction='cat', confidence=.9, id='0'))
        engine.predict.assert_any_call('cat dog', 'Q')

    def test_existing_run_is_never_reused_or_overwritten(self):
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / 'sentinel'
        sentinel.write_bytes(b'do not overwrite')
        before = file_hashes(self.args.output_dir)
        with self.assertRaises(FileExistsError):
            runner.run(self.args)
        self.assertEqual(file_hashes(self.args.output_dir), before)

    def test_successful_mock_run_covers_all_ids_and_keeps_one_prediction_store(self):
        engine = Mock(manifest={'fixture': True})
        engine.predict.side_effect = [dict(prediction='cat', confidence=.9) for _ in self.records]
        with patch.object(runtime_module, 'ExtractiveRuntime', return_value=engine), redirect_stdout(io.StringIO()):
            runner.run(self.args)
        record = self.result()
        self.assertEqual((record['status'], record['completed_predictions']), ('complete', 2))
        self.assertNotIn('predictions', record)
        self.assertNotIn('current_id', record)
        predictions = [json.loads(line) for line in (self.args.output_dir / 'predictions.jsonl').read_text().splitlines()]
        self.assertEqual([item['id'] for item in predictions], ['0', '1'])


if __name__ == '__main__':
    unittest.main()
