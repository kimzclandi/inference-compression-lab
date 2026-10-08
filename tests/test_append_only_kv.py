import importlib.util
import unittest
from unittest.mock import patch
from lab.append_only_kv import AppendOnlyKV


@unittest.skipUnless(importlib.util.find_spec('torch'),'Optional torch; mandatory in attention-cpu CI job')
class AppendOnlyKVTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;self.k=torch.arange(12,dtype=torch.float32).reshape(1,2,3,2);self.v=self.k+10

    def test_append_matches_cat_at_every_step(self):
        cache=AppendOnlyKV(self.k,self.v,6);k=self.k;v=self.v
        for i in range(3):
            nk=self.k[:,:,:1]+i;nv=nk+20
            ak,av=cache.append(nk,nv);k=self.t.cat((k,nk),dim=2);v=self.t.cat((v,nv),dim=2)
            self.assertTrue(self.t.equal(ak,k));self.assertTrue(self.t.equal(av,v))

    def test_capacity_rejection_is_atomic(self):
        cache=AppendOnlyKV(self.k,self.v,3)
        with self.assertRaises(OverflowError):cache.append(self.k[:,:,:1],self.v[:,:,:1])
        self.assertEqual(cache.length,3)
        self.assertTrue(self.t.equal(cache.view()[0],self.k))

    def test_dtype_shape_and_empty_rejected_before_mutation(self):
        cache=AppendOnlyKV(self.k,self.v,8)
        for k,v in [(self.k.double(),self.v.double()),(self.k[:,:1],self.v[:,:1]),(self.k,self.v[:,:,:1]),(self.k[:,:,:0],self.v[:,:,:0])]:
            with self.assertRaises(ValueError):cache.append(k,v)
            self.assertEqual(cache.length,3)
            self.assertTrue(self.t.equal(cache.view()[1],self.v))

    def test_second_copy_failure_keeps_visible_prefix_and_retry_works(self):
        cache=AppendOnlyKV(self.k,self.v,4);calls=[]
        def fail_second(dst,src):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('injected synchronous value-copy failure')
            dst.copy_(src)
        with patch.object(cache,'_copy',side_effect=fail_second):
            with self.assertRaises(RuntimeError):cache.append(self.k[:,:,:1],self.v[:,:,:1])
        self.assertEqual(cache.length,3)
        self.assertTrue(self.t.equal(cache.view()[0],self.k));self.assertTrue(self.t.equal(cache.view()[1],self.v))
        cache.append(self.k[:,:,:1]+7,self.v[:,:,:1]+7)
        self.assertEqual(cache.length,4)
        self.assertTrue(self.t.equal(cache.view()[0][:,:,-1:],self.k[:,:,:1]+7))

    def test_owns_initial_prefix_and_prior_view_length_is_stable(self):
        cache=AppendOnlyKV(self.k,self.v,4);old=cache.view()[0];expected=self.k.clone();self.k.fill_(999)
        self.assertTrue(self.t.equal(old,expected))
        cache.append(self.k[:,:,:1],self.v[:,:,:1]);self.assertEqual(old.shape[-2],3)

    def test_constructor_capacity_and_autograd_constraints(self):
        for capacity in (True,2,3.5):
            with self.assertRaises(ValueError):AppendOnlyKV(self.k,self.v,capacity)
        with self.assertRaises(ValueError):AppendOnlyKV(self.k.requires_grad_(),self.v,4)
