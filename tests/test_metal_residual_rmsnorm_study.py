"""Failure-path tests; no GPU, model execution, or performance samples."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import metal_residual_rmsnorm as study


class ReceiptTests(unittest.TestCase):
    def test_rejects_tampered_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            study.save(root/'run.json', {'status':'complete'})
            study.seal(root)
            study.save(root/'run.json', {'status':'failed'})
            with self.assertRaisesRegex(ValueError,'identities'):
                study.checked_receipt(root)

    def test_source_drift_rejected_even_with_valid_receipt_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); spec=root/'spec.json'; source=root/'source.py'
            spec.write_text('{}'); source.write_text('original')
            study.save(root/'run.json', {'status':'complete','spec_sha256':study.sha(spec),
                                        'source_sha256':{str(source):study.sha(source)}})
            # Audit has self-consistent identities; live executed source has drifted.
            audit=root/'audit'; audit.mkdir()
            (audit/'run.json').write_bytes((root/'run.json').read_bytes())
            study.seal(audit); source.write_text('changed')
            with patch.object(study,'SPEC',spec), patch.object(study,'SOURCES',[str(source)]):
                with self.assertRaisesRegex(ValueError,'Source drift'):
                    study.checked_receipt(audit)

    def test_output_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):
                study.reserve(path)
            self.assertEqual((path/'sentinel').read_text(),'keep')


class FixedScheduleFailureTests(unittest.TestCase):
    def setup_run(self, root):
        spec=json.loads(study.SPEC.read_text())
        protocol=root/'protocol.json'; study.save(protocol,spec)
        audit=root/'audit'; audit.mkdir(); study.save(audit/'checksums.json',{})
        args=argparse.Namespace(output_dir=root/'benchmark',audit_root=audit,model=root/'model')
        return protocol,args

    def test_failed_worker_is_retained_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol,args=self.setup_run(Path(tmp))
            with patch.object(study,'SPEC',protocol), \
                 patch.object(study,'checked_receipt',return_value={'source_sha256':{}}), \
                 patch.object(study.subprocess,'run',return_value=argparse.Namespace(returncode=2)) as run:
                with self.assertRaisesRegex(RuntimeError,'no retries'):
                    study.benchmark(args)
            self.assertEqual(run.call_count,1)
            result=json.loads((args.output_dir/'run.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['workers'],[{'round':0,'mode':'native','folder':'0-native','status':'failed','exit_code':2}])
            self.assertTrue((args.output_dir/'0-native.log').exists())
            self.assertTrue((args.output_dir/'checksums.json').exists())

    def test_budget_expiry_does_not_launch_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol,args=self.setup_run(Path(tmp))
            with patch.object(study,'SPEC',protocol), \
                 patch.object(study,'checked_receipt',return_value={'source_sha256':{}}), \
                 patch.object(study.time,'monotonic',side_effect=[0.,901.,902.]), \
                 patch.object(study.subprocess,'run') as run:
                with self.assertRaises(TimeoutError):
                    study.benchmark(args)
            run.assert_not_called()
            result=json.loads((args.output_dir/'run.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['workers'],[])
            self.assertEqual(result['wall_seconds'],902.)


if __name__=='__main__':
    unittest.main()
