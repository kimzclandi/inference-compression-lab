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
            for name in ['../outside','/tmp/outside','a/../b','a//b','a\\b']:
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
        for name in ['runs/model.json','weights.safetensors','../outside','.git/config']:
            with self.assertRaises(ValueError):validate_name(name)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('release-manifest.json',json.dumps({'sha256':{}}))
                z.writestr('a.py','pass')
            with self.assertRaises(ValueError):verify_zip(p)


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


if __name__ == '__main__': unittest.main()
