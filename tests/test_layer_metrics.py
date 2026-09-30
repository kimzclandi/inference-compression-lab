import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'experiment dependencies unavailable')
class LayerMetricsTests(unittest.TestCase):
    def test_padding_is_excluded_and_nmse_is_energy_normalized(self):
        import numpy as np
        from experiments.minilm_layer_errors import error_sums, metrics
        a = np.array([[[1.,2.],[100.,100.]]])
        b = np.array([[[2.,4.],[-500.,-500.]]])
        result = metrics(error_sums(a,b,np.array([[1,0]])))
        self.assertEqual(result['n'],2)
        self.assertEqual(result['mse'],2.5)
        self.assertEqual(result['nmse'],1.)
        self.assertAlmostEqual(result['cosine'],1.)

    def test_aggregate_uses_element_counts_not_mean_of_batch_means(self):
        import numpy as np
        from experiments.minilm_layer_errors import error_sums, metrics
        chunks = [error_sums(np.ones((1,n,1)),np.full((1,n,1),v),np.ones((1,n))) for n,v in [(1,2),(3,1)]]
        total = {k:sum(x[k] for x in chunks) for k in ['n','sse','reference_energy','candidate_energy','dot']}
        total['max_abs'] = max(x['max_abs'] for x in chunks)
        self.assertEqual(metrics(total)['mse'],.25)
