"""Offline recomputation and adversarial checks for real lifecycle evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments.verify_qwen_cache_lifecycle import verify

ROOT=Path(__file__).resolve().parents[1]/'results/qwen-cache-lifecycle-gpu-v1'


class LifecycleEvidence(unittest.TestCase):
    def test_recompute(self):
        self.assertEqual(verify(ROOT),json.loads((ROOT/'summary.json').read_text()))

    def corrupt(self,name,mutate):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for filename in ('manifest.json','timings.json','contracts.json','workloads.json'):
                data=json.loads((ROOT/filename).read_text())
                if filename==name:mutate(data)
                (out/filename).write_text(json.dumps(data))
            with self.assertRaises(ValueError):verify(out)

    def test_reject_false_success_counter(self):
        def mutate(c):c['faults'][0]['after']['stats']['hits']+=1
        self.corrupt('contracts.json',mutate)

    def test_reject_lru_change(self):
        def mutate(c):c['faults'][0]['after']['resident_tokens'].reverse()
        self.corrupt('contracts.json',mutate)

    def test_reject_trace_omission(self):
        self.corrupt('timings.json',lambda rows:rows.pop())
