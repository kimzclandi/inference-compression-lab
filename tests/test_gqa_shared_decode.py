import unittest
from lab.gqa_shared_decode import validate_metadata

class GQAContracts(unittest.TestCase):
    def test_fixed_contract_and_partial_partition(self):
        self.assertEqual(validate_metadata((1,14,1,64),(1,2,257,64),(1,2,257,64),('float16',)*3),3)

    def test_rejects_mask_equivalent_shapes_mismatches_and_dtypes(self):
        for q,k,v,dtype in [((2,14,1,64),(1,2,1,64),(1,2,1,64),('float16',)*3),
                           ((1,14,2,64),(1,2,1,64),(1,2,1,64),('float16',)*3),
                           ((1,14,1,64),(1,2,0,64),(1,2,0,64),('float16',)*3),
                           ((1,14,1,64),(1,2,1,64),(1,2,2,64),('float16',)*3),
                           ((1,14,1,64),(1,2,1,64),(1,2,1,64),('float32',)*3)]:
            with self.subTest(q=q,k=k), self.assertRaises(ValueError): validate_metadata(q,k,v,dtype)

class GQAMath(unittest.TestCase):
    def test_independent_partition_algebra_and_gqa_head_mapping(self):
        try: import numpy as np
        except ImportError: self.skipTest('NumPy needed for float64 math')
        from lab.gqa_reference import dense_reference,partitioned_reference
        rng=np.random.default_rng(20261009)
        for n in [1,31,32,33,127,128,129,257,1024,4096,8192]:
            for scale in [1,16]:
                q=(rng.normal(size=(1,14,1,64))*scale).astype('float16')
                k=(rng.normal(size=(1,2,n,64))*scale).astype('float16')
                v=rng.normal(size=(1,2,n,64)).astype('float16')
                np.testing.assert_allclose(dense_reference(q,k,v),partitioned_reference(q,k,v),atol=1e-12,rtol=1e-12)
        q=np.zeros((1,14,1,64));k=np.zeros((1,2,127,64));v=np.zeros_like(k)
        v[:,0]=2;v[:,1]=5
        result=dense_reference(q,k,v)
        np.testing.assert_allclose(result[:, :7],2,atol=1e-12,rtol=0)
        np.testing.assert_allclose(result[:, 7:],5,atol=1e-12,rtol=0)

class AdapterContracts(unittest.TestCase):
    def test_restore_nested_rejection_and_pre_mutation_offset(self):
        import sys, types
        from unittest.mock import patch
        from lab.gqa_shared_decode import qwen_decode_adapter
        class Module: pass
        class Attention: pass
        class Model: pass
        class KVCache:
            def __init__(self,offset):self.offset=offset;self.updates=0
            def update_and_fetch(self,*args):self.updates+=1;raise AssertionError('must not update')
        modules={name:types.ModuleType(name) for name in ('mlx','mlx.nn','mlx_lm','mlx_lm.models','mlx_lm.models.cache','mlx_lm.models.qwen2')}
        modules['mlx.nn'].Module=Module
        modules['mlx_lm.models.cache'].KVCache=KVCache
        modules['mlx_lm.models.qwen2'].Model=Model
        modules['mlx_lm.models.qwen2'].Attention=Attention
        model=Model();model.args=types.SimpleNamespace(num_attention_heads=14,num_key_value_heads=2,hidden_size=896)
        original=Attention();layer=types.SimpleNamespace(self_attn=original)
        model.model=types.SimpleNamespace(layers=[layer])
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with qwen_decode_adapter(model,'shared_compiled',{}):
                    for mode in ('native','shared_compiled'):
                        with self.assertRaisesRegex(ValueError,'nested'):
                            with qwen_decode_adapter(model,mode,{}):pass
                    for offset in (8192,-1,True,1.5):
                        cache=KVCache(offset)
                        with self.assertRaisesRegex(ValueError,'offset'):
                            layer.self_attn(types.SimpleNamespace(shape=(1,1,896)),cache=cache)
                        self.assertEqual(cache.updates,0)
                    raise RuntimeError('injected')
            self.assertIs(layer.self_attn,original)
            with qwen_decode_adapter(model,'native',{}):self.assertIs(layer.self_attn,original)

    def test_optimized_runner_rejected_before_importing_mlx(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-O','-m','experiments.gqa_shared_decode','--help'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('optimized Python mode unsupported',result.stderr)

class FrozenGQAEvidence(unittest.TestCase):
    def test_preserved_correctness_failure_has_no_performance(self):
        from experiments.verify_gqa_shared_decode import verify
        r=verify('results/gqa-shared-decode-v1')
        self.assertTrue(r['evidence_valid']);self.assertFalse(r['accepted'])
        self.assertEqual(r['performance_trials'],0)
        self.assertEqual(len(r['model_kv_failed_records']),10)

    def test_modified_record_and_optimized_verifier_rejected(self):
        import json,shutil,subprocess,sys,tempfile
        from pathlib import Path
        from experiments.verify_gqa_shared_decode import verify
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'evidence';shutil.copytree('results/gqa-shared-decode-v1',root)
            p=root/'model-correctness.json';data=json.loads(p.read_text());data[0]['kv'][18]['allclose']=True
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError,'artifact modified'):verify(root)
        result=subprocess.run([sys.executable,'-O','-m','experiments.verify_gqa_shared_decode'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('optimized Python mode unsupported',result.stderr)
