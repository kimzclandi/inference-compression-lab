import copy
import json
import unittest
import numpy as np
from experiments.attention_mps_study import SPEC, make_arms, summarize
from experiments.verify_attention_mps_reference import nonblas_reference


class MPSProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(SPEC.read_text())
        self.rows=[dict(batch=b,query_length=l,key_length=s,round=r,arm=a,
                        seconds=[1.0 if a=='sdpa_auto' else 2.0]*self.spec['repeats'])
                   for b in self.spec['batches'] for l,s in self.spec['shapes']
                   for r in range(self.spec['rounds']) for a in self.spec['arms']]

    def test_both_controls_required(self):
        for row in self.rows:
            if row['arm']=='eager_fp16':row['seconds']=[0.5]*self.spec['repeats']
        result=summarize(self.rows,self.spec)
        self.assertTrue(all(not c['speed_gate'] for c in result['shapes']))

    def test_four_faster_rounds_required(self):
        for row in self.rows:
            if row['arm']=='sdpa_auto' and row['round']<2:row['seconds']=[3.0]*self.spec['repeats']
        self.assertTrue(all(not c['speed_gate'] for c in summarize(self.rows,self.spec)['shapes']))

    def test_missing_duplicate_or_short_samples_rejected(self):
        short=copy.deepcopy(self.rows);short[0]['seconds'].pop()
        for records in (self.rows[:-1], self.rows+[self.rows[0]],short):
            with self.assertRaises(ValueError):summarize(records,self.spec)

    def test_nonfinite_and_zero_time_rejected(self):
        for v in (float('nan'),float('inf'),0,-1):
            rows=copy.deepcopy(self.rows);rows[0]['seconds'][0]=v
            with self.assertRaises(ValueError):summarize(rows,self.spec)

    def test_cached_chunk_is_out_of_performance_scope(self):
        with self.assertRaises(ValueError):
            make_arms(None,[np.zeros((1,1,3,2)),np.zeros((1,1,7,2)),np.zeros((1,1,7,2))])

    def test_nonblas_oracle_uniform_and_chunk_boundaries(self):
        k=np.zeros((1,1,4,1));v=np.arange(1,5).reshape(1,1,4,1)
        for length,expected in [(4,[1,1.5,2,2.5]),(2,[2,2.5]),(1,[2.5])]:
            q=np.zeros((1,1,length,1))
            np.testing.assert_array_equal(nonblas_reference(q,k,v,chunk=1).ravel(),expected)
