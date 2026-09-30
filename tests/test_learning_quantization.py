import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('numpy'),'numpy unavailable')
class TeachingQuantizationTests(unittest.TestCase):
    def test_zero_channels_and_zeros_are_representable(self):
        import numpy as np
        from experiments.quantization_learning import weight_qdq,activation_qdq
        for pc in [False,True]:
            q,s=weight_qdq(np.zeros((8,4),dtype=np.float32),pc)
            np.testing.assert_array_equal(q,0)
            self.assertTrue(np.all(np.isfinite(s)))
        q,s=activation_qdq(np.zeros((1,32,8),dtype=np.float32))
        np.testing.assert_array_equal(q,0)

    def test_output_channel_scale_does_not_change_other_channel(self):
        import numpy as np
        from experiments.quantization_learning import weight_qdq
        w=np.array([[1.,1.],[.1,.1]],dtype=np.float32)
        before,_=weight_qdq(w,True)
        w[:,0]*=100
        after,_=weight_qdq(w,True)
        np.testing.assert_array_equal(before[:,1],after[:,1])

    def test_no_clipping_weight_rounding_bound(self):
        import numpy as np
        from experiments.quantization_learning import weight_qdq
        w=np.random.default_rng(4).normal(size=(8,4)).astype(np.float32)
        for pc in [False,True]:
            q,s=weight_qdq(w,pc)
            self.assertTrue(np.all(np.abs(q-w)<=s/2+1e-6))
