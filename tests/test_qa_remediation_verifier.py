"""Statistical contracts and adversarial evidence checks for QA remediation."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.qa_remediation import SOURCE_FILES, validate_locked_run
from experiments.verify_qa_remediation import (
    paired_bootstrap, tensor_bytes_comparison, verify_run, verify, write_selection,
)
from lab.artifact_integrity import file_hashes
from lab.quantization_diagnostics import sha
from lab.qa_metrics import evaluate


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def fixture_run(folder):
    data = [dict(id='yes', source_title='A', family_id='f1', context='Answer is here.',
                 question='What is here?', answers=['Answer'], is_impossible=False),
            dict(id='no', source_title='A', family_id='f2', context='Nothing is stated.',
                 question='What is his name?', answers=[], is_impossible=True)]
    (folder / 'data.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in data))
    model = {'config.json': dict(bytes=25, sha256='1'*64),
             'model.safetensors': dict(bytes=100, sha256='2'*64)}
    spec = dict(locked=True, seed=42, max_input_tokens=2048, max_new_tokens=48,
                allowed_runs=[dict(label='test-fp16', mode='grounded', bits=None,
                                   model_files_sha256={k: v['sha256'] for k, v in model.items()},
                                   data_sha256=sha(folder/'data.jsonl'))])
    write_json(folder/'protocol.json', spec)
    predictions = []
    for row in data:
        predictions.append(dict(id=row['id'], prediction='NO_ANSWER' if row['is_impossible'] else 'Answer',
            prompt_sha256='3'*64, input_token_ids_sha256='4'*64, input_tokens=40,
            token_ids=[12], generated_tokens=1, stop_reason='eos', ttft_seconds=.1,
            decode_seconds=0., total_seconds=.1, total_pipeline_seconds=.2,
            confidence=0 if row['is_impossible'] else .5,
            audit=None if row['is_impossible'] else dict(yes_logit=0., no_logit=0.,
                confidence=.5, binary_mass=.8, yes_token=1, no_token=2, input_tokens=50,
                prompt_sha256='5'*64)))
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in predictions))
    metrics, scored = evaluate(data, predictions)
    layout = dict(parameter_tensor_bytes=200, parameter_tensor_count=2,
                  dtype_distribution={'mlx.core.float16': dict(tensors=2, tensor_bytes=200)},
                  module_counts=dict(quantized_linear=0, quantized_embedding=0,
                                     floating_linear=1, floating_embedding=0), quantizer_counts={},
                  normalization_tensors=[dict(name='norm.weight', dtype='mlx.core.float16',
                                             tensor_bytes=20, is_fp16=True)], normalizations_all_fp16=True)
    for name in SOURCE_FILES:
        path = folder/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    run = dict(status='complete', stage='complete', spec_sha256=sha(folder/'protocol.json'),
               dataset_sha256=sha(folder/'data.jsonl'), mode='grounded', model_label='test-fp16',
               bits=None, source_model_files=model, model_layout=layout, loaded_config={'torch_dtype':'float16'},
               source_sha256={name: sha(folder/'source'/name) for name in SOURCE_FILES},
               predictions=predictions, metrics=metrics, scored=scored)
    run['locked_spec_validation'] = validate_locked_run(spec, **{
        'label': run['model_label'], 'mode': run['mode'], 'bits': run['bits'],
        'model_files_sha256': {k:v['sha256'] for k,v in model.items()},
        'data_sha256': run['dataset_sha256']})
    write_json(folder/'run.json', run)
    write_json(folder/'checksums.json', file_hashes(folder))
    return run


class RunEvidenceTests(unittest.TestCase):
    def test_complete_run_recomputes_raw_and_contracted_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture_run(root)
            result = verify_run(root, threshold=.6)
            self.assertEqual(result['n'], 2)
            self.assertEqual(result['metrics']['overall']['em'], 1)
            self.assertEqual(result['selective']['system']['overall']['em'], .5)
            self.assertEqual(result['selective']['selective']['accepted'], 0)

    def test_resigned_source_hole_identity_score_or_audit_mutations_fail(self):
        mutations = [
            ('source manifest', lambda r: r['source_sha256'].pop(SOURCE_FILES[-1])),
            ('model precision', lambda r: r.update(bits=8)),
            ('score', lambda r: r['scored'][0].update(em=0)),
            ('incomplete', lambda r: r.update(status='running')),
            ('tensor sum', lambda r: r['model_layout'].update(parameter_tensor_bytes=199)),
            ('confidence', lambda r: r['predictions'][0].update(confidence=.9)),
        ]
        for name, mutation in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td); run = fixture_run(root); mutation(run)
                write_json(root/'run.json', run)
                if name == 'confidence':
                    (root/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in run['predictions']))
                write_json(root/'checksums.json', file_hashes(root, exclude=('checksums.json',)))
                with self.assertRaises(ValueError):
                    verify_run(root)

    def test_empty_checksums_or_unlisted_file_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture_run(root)
            (root/'unlisted.txt').write_text('cannot silently ignore me')
            with self.assertRaisesRegex(ValueError, 'exact artifact files'):
                verify_run(root)
            write_json(root/'checksums.json', {})
            with self.assertRaisesRegex(ValueError, 'Empty checksum'):
                verify_run(root)


@unittest.skipUnless((ROOT/'results/qa-remediation-v1/selection.json').exists(), 'Calibrated evidence not archived')
class FrozenCalibrationTests(unittest.TestCase):
    def test_complete_negative_evidence_cannot_be_called_success(self):
        result = verify(ROOT/'results/qa-remediation-v1', ROOT/'configs/qa-remediation/study.json')
        self.assertTrue(result['evidence_valid'])
        self.assertEqual(result['status'], 'calibration_failed')
        self.assertFalse(result['overall_success'])
        self.assertFalse(result['confirmation_evaluated'])
        self.assertEqual(result['answer_entrypoint'], 'unavailable_quality')

    def test_forged_pass_and_consumed_confirmation_are_rejected(self):
        for alteration in ('pass', 'consumed'):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as td:
                copy_root = Path(td)/'evidence'
                shutil.copytree(ROOT/'results/qa-remediation-v1', copy_root)
                if alteration == 'pass':
                    selection = json.loads((copy_root/'selection.json').read_text())
                    selection.update(calibration_feasible=True, confirmation_allowed=True,
                                     status='calibration_passed')
                    write_json(copy_root/'selection.json', selection)
                else:
                    (copy_root/'confirmation-fp16').mkdir()
                with self.assertRaises(ValueError):
                    verify(copy_root)

    def test_selection_writer_refuses_existing_record(self):
        with self.assertRaisesRegex(ValueError, 'never overwrite'):
            write_selection(ROOT/'results/qa-remediation-v1', ROOT/'configs/qa-remediation/study.json')


class PairedStatisticsTests(unittest.TestCase):
    def test_zero_and_constant_difference_are_exact(self):
        data = [dict(id=str(i), source_title='a', family_id=f'f{i // 2}') for i in range(8)]
        a = [dict(id=str(i), em=0., f1=.25) for i in range(8)]
        same = paired_bootstrap(data, a, a, replicates=100)
        self.assertEqual(same['em_ci95'], [0., 0.])
        self.assertEqual(same['f1_ci95'], [0., 0.])
        b = [dict(id=str(i), em=1., f1=.75) for i in range(8)]
        changed = paired_bootstrap(data, a, b, replicates=100)
        self.assertEqual(changed['em_ci95'], [1., 1.])
        self.assertEqual(changed['f1_ci95'], [.5, .5])
        self.assertEqual(changed['families'], 4)

    def test_questions_in_single_article_family_cannot_be_resampled_apart(self):
        data = [dict(id='a', source_title='A', family_id='only-A'),
                dict(id='b', source_title='A', family_id='only-A'),
                dict(id='c', source_title='B', family_id='only-B')]
        a = [dict(id=i, em=0, f1=0) for i in 'abc']
        b = [dict(id='a', em=1, f1=1), dict(id='b', em=0, f1=0), dict(id='c', em=0, f1=0)]
        result = paired_bootstrap(data, a, b, replicates=100)
        # One fixed family per article means every bootstrap draw is the same.
        self.assertEqual(result['em_ci95'], [1/3, 1/3])
        self.assertEqual(result['articles'], 2)

    def test_pairing_ignores_input_order_but_rejects_missing_duplicate_or_nan(self):
        data = [dict(id=str(i), source_title='A', family_id=f'f{i}') for i in range(4)]
        a = [dict(id=str(i), em=0., f1=0.) for i in range(4)]
        b = [dict(id=str(i), em=float(i % 2), f1=float(i % 2)) for i in range(4)]
        self.assertEqual(paired_bootstrap(data, a, b, replicates=100),
                         paired_bootstrap(data[::-1], a[::-1], b[::-1], replicates=100))
        for bad in [b[:-1], [b[0], *b], [dict(b[0], f1=float('nan')), *b[1:]]]:
            with self.assertRaises(ValueError):
                paired_bootstrap(data, a, bad, replicates=100)

    def test_tensor_ratio_includes_quantization_metadata_not_runtime_memory(self):
        result = tensor_bytes_comparison(
            {'tensors': {'weight': {'bytes': 200}, 'norm': {'bytes': 20}}},
            {'tensors': {'weight': {'bytes': 100}, 'scale': {'bytes': 4},
                         'bias': {'bytes': 4}, 'norm': {'bytes': 20}}})
        self.assertEqual(result['q8_to_fp16_ratio'], 128/220)
        for invalid in [{}, {'tensors': {}}, {'tensors': {'weight': {'bytes': 0}}},
                        {'tensors': {'weight': {'bytes': 100.5}}}]:
            with self.assertRaises(ValueError):
                tensor_bytes_comparison(invalid, {'tensors': {'a': {'bytes': 1}}})


if __name__ == '__main__':
    unittest.main()
