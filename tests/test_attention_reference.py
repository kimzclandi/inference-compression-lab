import copy
import json
import unittest
import numpy as np
from lab.attention_reference import attention, prefix_mask
from experiments.attention_backend_study import SPEC, summarize


class AttentionReferenceTests(unittest.TestCase):
    def test_decode_sees_every_key(self):
        np.testing.assert_array_equal(prefix_mask(1, 4), [[True]*4])
        result = attention(np.zeros((1,1,1,1)), np.zeros((1,1,4,1)), np.arange(1,5).reshape(1,1,4,1))
        self.assertEqual(result.item(), 2.5)

    def test_chunk_alignment(self):
        np.testing.assert_array_equal(prefix_mask(2, 4), [[1,1,1,0],[1,1,1,1]])

    def test_prefill_uniform_logits(self):
        result = attention(np.zeros((1,1,4,1)), np.zeros((1,1,4,1)), np.arange(1,5).reshape(1,1,4,1))
        np.testing.assert_array_equal(result.ravel(), [1,1.5,2,2.5])

    def test_cached_suffix_matches_full(self):
        rng = np.random.default_rng(7)
        q,k,v = [rng.normal(size=(2,3,9,5)) for _ in range(3)]
        full = attention(q,k,v)
        for length in (1,3):
            np.testing.assert_allclose(attention(q[...,-length:,:],k,v), full[...,-length:,:], atol=1e-14)

    def test_future_values_do_not_leak(self):
        q=k=np.zeros((1,1,4,1)); v=np.arange(4).reshape(1,1,4,1).astype(float)
        ref=attention(q,k,v); v[...,3,:]=10000
        np.testing.assert_array_equal(attention(q,k,v)[...,:3,:], ref[...,:3,:])

    def test_reject_invalid_shapes_and_nonfinite(self):
        x=np.zeros((1,1,3,2))
        for q,k,v in [(x,x[:,:,:1],x[:,:,:1]),(x,x,np.zeros((1,2,3,2))),(x+np.nan,x,x)]:
            with self.assertRaises(ValueError): attention(q,k,v)


class TimingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads(SPEC.read_text())
        self.rows=[dict(batch=b,query_length=l,key_length=s,round=r,arm=a,
                        event_ms_per_call=1 if a=='flash' else 2,wall_ms_per_call=3)
                   for b in self.spec['batches'] for l,s in self.spec['shapes']
                   for r in range(self.spec['rounds']) for a in self.spec['arms']]

    def test_math_is_primary_not_eager(self):
        for r in self.rows:
            if r['arm']=='math': r['event_ms_per_call']=0.5
        self.assertTrue(all(not r['speed_gate'] for r in summarize(self.rows,self.spec)['shapes']))

    def test_incomplete_duplicate_and_nan_rejected(self):
        broken=copy.deepcopy(self.rows);broken[0]['event_ms_per_call']=float('nan')
        for records in (self.rows[:-1],self.rows+[self.rows[0]],broken):
            with self.assertRaises(ValueError): summarize(records,self.spec)

    def test_gate_requires_consistent_rounds(self):
        for r in self.rows:
            if r['arm']=='flash' and r['round']<2:r['event_ms_per_call']=3
        self.assertTrue(all(not r['speed_gate'] for r in summarize(self.rows,self.spec)['shapes']))
