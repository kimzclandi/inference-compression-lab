import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from experiments.verify_qwen_cache_reservation import ROOT,verify

class EvidenceTests(unittest.TestCase):
    def mutate(self,name,change):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'evidence';shutil.copytree(ROOT,root)
            data=json.loads((root/name).read_text());change(data)
            (root/name).write_text(json.dumps(data))
            info=json.loads((root/'run.json').read_text())
            info['artifacts'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
            (root/'run.json').write_text(json.dumps(info))
            with self.assertRaises(AssertionError):verify(root)
    def test_missing_trial_rejected(self):
        self.mutate('samples.json',lambda x:x.pop())
    def test_wrong_capacity_rejected(self):
        self.mutate('correctness.json',lambda x:x[0]['reserved']['capacities'][0].__setitem__(0,999))
    def test_timed_token_drift_rejected(self):
        self.mutate('samples.json',lambda x:x[0]['tokens'].__setitem__(0,-1))
