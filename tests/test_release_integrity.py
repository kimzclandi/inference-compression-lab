import json
from pathlib import Path
import shutil
import tempfile
import unittest
from lab.artifact_integrity import file_hashes, verify_hashes, safe_path, git_identity
from lab.confirmation import comparison
from experiments.verify_qwen_quantization import verify
from experiments.verify_qwen_confirmation import verify as confirm
from experiments.release_archive import validate_name, verify as verify_zip
from experiments.verify_release import verify as verify_release
from lab.model_identity import INFERENCE_FILES, verify_inference_files
from lab.quantization_diagnostics import read, aggregates_equal, sha
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class IntegrityTests(unittest.TestCase):
    def test_empty_and_incomplete_manifest_reject_real_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'evidence'; shutil.copytree(ROOT/'results/qwen-quantization-v1', p)
            for invalid in [{}, {'protocol.json': '0'*64}]:
                (p/'checksums.json').write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): verify(p)

    def test_added_file_or_symlink_or_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); (p/'a').write_text('data'); hashes = file_hashes(p)
            verify_hashes(p, hashes)
            (p/'extra').write_text('data')
            with self.assertRaises(ValueError): verify_hashes(p, hashes)
            for name in ['.','../outside','/tmp/outside','a/../b','a//b','a\\b']:
                with self.assertRaises(ValueError): safe_path(p, name)
            (p/'link').symlink_to(p/'a')
            with self.assertRaises(ValueError): file_hashes(p)

    def test_archive_does_not_capture_parent_git_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            self.assertIsNone(git_identity(td)['git_head'])

    def test_external_chat_template_is_part_of_model_identity(self):
        expected={name:{'sha256':'a','bytes':1} for name in INFERENCE_FILES}
        verify_inference_files(expected,expected)
        changed=dict(expected,**{'chat_template.jinja':{'sha256':'b','bytes':1}})
        with self.assertRaisesRegex(ValueError,'chat_template'):verify_inference_files(changed,expected)

    def test_release_rejects_weights_and_incomplete_archive_manifest(self):
        for name in ['.','runs/model.json','weights.safetensors','../outside','.git/config']:
            with self.assertRaises(ValueError):validate_name(name)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('release-manifest.json',json.dumps({'sha256':{}}))
                z.writestr('a.py','pass')
            with self.assertRaises(ValueError):verify_zip(p)

    def test_release_rejects_empty_or_partial_historical_ledger(self):
        # Rehashing the remaining files must not silently narrow preservation.
        for empty in [True, False]:
            with self.subTest(empty=empty), tempfile.TemporaryDirectory() as td:
                p = Path(td); dest = p/'configs/release/protected-results.json'
                dest.parent.mkdir(parents=True)
                ledger = read(ROOT/'configs/release/protected-results.json')
                if empty: ledger['sha256'] = {}
                else: ledger['sha256'].pop(next(iter(ledger['sha256'])))
                dest.write_text(json.dumps(ledger))
                with self.assertRaisesRegex(ValueError, 'ledger changed'): verify_release(p)

    def test_resigned_source_manifest_cannot_omit_files_or_escape(self):
        for source in [{}, {'../protocol.json': 'protocol'}]:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as td:
                p = Path(td)/'evidence'; shutil.copytree(ROOT/'results/qwen-quantization-v1', p)
                run = read(p/'run.json')
                run['source_sha256'] = {name: sha(p/'protocol.json') for name in source}
                (p/'run.json').write_text(json.dumps(run))
                (p/'checksums.json').write_text(json.dumps(file_hashes(p, exclude=('checksums.json','summary.json'))))
                with self.assertRaises(ValueError): verify(p)

    def test_self_consistent_zip_cannot_contain_symlink(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'bad.zip'; data = b'../outside'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('release-manifest.json', json.dumps({'sha256': {'link': hashlib.sha256(data).hexdigest()}}))
                link = zipfile.ZipInfo('link'); link.external_attr = 0o120777 << 16
                z.writestr(link, data)
            with self.assertRaisesRegex(ValueError, 'Non-regular'): verify_zip(p)


class BootstrapTests(unittest.TestCase):
    def test_identical_and_uniform_effects_have_known_intervals(self):
        data = [dict(id=str(i),source_title='article'+str(i//4),family_id=str(i//2)) for i in range(8)]
        zero = [dict(id=r['id'],em=0.) for r in data]
        one = [dict(id=r['id'],em=1.) for r in data]
        cfg = dict(seed=7,replicates=100)
        self.assertEqual(comparison(data,zero,zero,cfg)['ci95'], [0.,0.])
        positive = comparison(data,zero,one,cfg)
        self.assertEqual(positive['ci95'], [1.,1.])
        self.assertEqual((positive['families'],positive['articles']), (4,2))
        self.assertEqual(comparison(data,one,zero,cfg)['ci95'], [-1.,-1.])
        with self.assertRaises(ValueError): comparison(data,zero,one[:-1],cfg)


class ConfirmationEvidenceTests(unittest.TestCase):
    def test_frozen_recomputation_keeps_negative_gate(self):
        p=ROOT/'results/qwen-confirmation-v1';result=confirm(p)
        self.assertTrue(aggregates_equal(result,read(p/'summary.json')))
        self.assertFalse(result['gate']['passed'])
        self.assertEqual(result['selected_vs']['q4']['ci95'][0],0.)

    def test_resigned_missing_question_is_not_valid_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'evidence';shutil.copytree(ROOT/'results/qwen-confirmation-v1',p)
            result=read(p/'q4-quality.json');result['predictions'].pop()
            (p/'q4-quality.json').write_text(json.dumps(result))
            checks=read(p/'checksums.json');checks['q4-quality.json']=sha(p/'q4-quality.json')
            (p/'checksums.json').write_text(json.dumps(checks))
            with self.assertRaisesRegex(ValueError,'Duplicate, missing or extra IDs'):confirm(p)

    def test_resigned_tokenizer_mismatch_or_partial_source_is_rejected(self):
        for mutation in ['tokenizer', 'source']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                p=Path(td)/'evidence';shutil.copytree(ROOT/'results/qwen-confirmation-v1',p)
                if mutation == 'tokenizer':
                    name = 'q4-quality.json'; value = read(p/name)
                    value['model_files']['tokenizer.json']['sha256'] = '0'*64
                else:
                    name = 'run.json'; value = read(p/name)
                    value['source_sha256'].pop(next(iter(value['source_sha256'])))
                (p/name).write_text(json.dumps(value))
                (p/'checksums.json').write_text(json.dumps(file_hashes(p, exclude=('checksums.json','summary.json'))))
                with self.assertRaises(ValueError): confirm(p)


if __name__ == '__main__': unittest.main()
