import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'),'experiment dependencies unavailable')
class RuntimeTests(unittest.TestCase):
    def test_pool_ignores_padding_and_normalizes(self):
        import numpy as np
        from lab.minilm_runtime import pool
        h=np.array([[[3.,4.],[100.,-100.]]],dtype=np.float32)
        out=pool(h,np.array([[1,0]],dtype=np.int64))
        np.testing.assert_allclose(out,[[.6,.8]])
        self.assertEqual(out.shape,(1,2))

    def test_invalid_threads_fail_before_loading(self):
        from lab.minilm_runtime import MiniLMRuntime
        with self.assertRaises(ValueError):MiniLMRuntime('missing.onnx',threads=0)

    @unittest.skipUnless(Path('models/minilm/onnx/model.onnx').exists(),'local model unavailable')
    def test_hash_and_real_encode_contract(self):
        import numpy as np
        from lab.minilm_runtime import MiniLMRuntime
        with self.assertRaises(ValueError):
            MiniLMRuntime('models/minilm/onnx/model.onnx',expected_sha256='wrong')
        rt=MiniLMRuntime('models/minilm/onnx/model.onnx',threads=1,max_length=64,fixed_padding=True)
        texts=['A robot picks up a cup.','A robot picks up a cup.']
        tokens=rt.tokenize(texts)
        self.assertEqual(tokens['input_ids'].shape,(2,64))
        e=rt.encode(texts)
        self.assertEqual(e.shape,(2,384))
        np.testing.assert_allclose(np.linalg.norm(e,axis=1),1,atol=1e-8)
        np.testing.assert_array_equal(e[0],e[1])
        with self.assertRaises(ValueError):rt.tokenize([])
