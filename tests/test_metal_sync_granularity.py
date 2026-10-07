import copy
import json
from pathlib import Path
import unittest
from experiments.metal_sync_granularity import summarize,validate


class SyncGranularity(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(Path('configs/metal-sync-granularity-v1.json').read_text())
        self.records=[dict(rows=r,sync_every=c,round=n,arm=a,chain_seconds=[1./c]*self.spec['repeats'])
            for r in self.spec['rows'] for c in self.spec['sync_every'] for n in range(self.spec['rounds']) for a in self.spec['arms']]

    def test_synchronization_sensitivity_does_not_promote_kernel(self):
        result=summarize(self.records,self.spec)
        self.assertFalse(result['kernel_promoted'])
        self.assertEqual(result['shapes']['512']['sensitivity']['original']['sync1_over_sync32'],32.)
        self.assertEqual(result['shapes']['512']['original_over_masked']['32'],1.)

    def test_reject_changed_coverage_and_bad_samples(self):
        bad=[self.records[:-1],self.records+[self.records[0]]]
        for value in [0,-1,True,float('nan'),float('inf')]:
            rows=copy.deepcopy(self.records);rows[0]['chain_seconds'][0]=value;bad.append(rows)
        for rows in bad:
            with self.assertRaises(ValueError):summarize(rows,self.spec)

    def test_cadence_must_divide_fixed_work(self):
        spec=copy.deepcopy(self.spec);spec['sync_every']=[1,7,32]
        with self.assertRaises(ValueError):validate(spec)
