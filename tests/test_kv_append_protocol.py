import copy
import json
import unittest
from experiments.kv_append_mps import SPEC,logical_writes,summarize


class KVAppendProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(SPEC.read_text())
        self.rows=[dict(batch=b,prefix=p,scope=s,round=r,arm=a,
                        initialization_seconds=[0.1]*self.spec['repeats'],
                        chain_seconds=[1. if a=='preallocated' else 2.]*self.spec['repeats'])
                   for b in self.spec['batches'] for p in self.spec['prefixes'] for s in self.spec['scopes']
                   for r in range(self.spec['rounds']) for a in self.spec['arms']]

    def test_append_only_win_cannot_pass_attention_gate(self):
        for row in self.rows:
            if row['scope']=='append_attention' and row['arm']=='preallocated':row['chain_seconds']=[3.]*self.spec['repeats']
        self.assertTrue(all(not c['primary_speed_gate'] for c in summarize(self.rows,self.spec)['cases']))

    def test_missing_duplicate_and_nonfinite_rejected(self):
        bad=copy.deepcopy(self.rows);bad[0]['initialization_seconds'][0]=float('nan')
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],bad):
            with self.assertRaises(ValueError):summarize(rows,self.spec)

    def test_copy_model_is_sum_of_growing_lengths(self):
        spec=dict(self.spec,steps=3,heads=2,head_dim=4)
        counts=logical_writes(spec,1,5);unit=2*1*2*4*2
        self.assertEqual(counts['cat_append_output_bytes'],unit*(6+7+8))
        self.assertEqual(counts['preallocated_append_write_bytes'],unit*3)
        self.assertTrue(counts['not_measured_dram'])
