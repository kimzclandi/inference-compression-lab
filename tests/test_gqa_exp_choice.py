import unittest
import numpy as np
from lab.gqa_exp_choice import sources, metrics, aggregate


class ExpChoiceMetricTests(unittest.TestCase):
    def arrays(self):
        a = np.ones((1,14,1,64), dtype=np.float16)
        return {name:a.astype(dtype) for name,dtype in [('native',np.float16),
            ('original',np.float16),('standard_half',np.float16),('fast_half',np.float16),
            ('standard_float',np.float32),('fast_float',np.float32)]}

    def test_intervention_only_changes_three_calls(self):
        s = sources()
        for part,count in [('partial',2),('merge',1)]:
            a,b = s[f'standard_{part}.metal'],s[f'fast_{part}.metal']
            self.assertEqual(a.count('metal::exp('),count)
            self.assertEqual(b,a.replace('metal::exp(', 'metal::fast::exp('))
        self.assertEqual(s['standard_merge.metal'].count('fp32['),2)

    def test_native_agreement_can_worsen_reference_accuracy(self):
        a = self.arrays(); hi=np.nextafter(np.float16(1),np.float16(2))
        a['standard_half'].flat[0]=hi; a['standard_float'].flat[0]=float(hi)
        a['fast_half'].flat[1]=hi; a['fast_float'].flat[1]=float(hi)
        ref = a['standard_half'].astype(np.float64)
        m = metrics(a,ref)
        self.assertEqual((m['resolved'],m['persistent'],m['introduced']),(1,0,1))
        self.assertEqual(m['half_reference_change'],dict(improved=0,worsened=2,equal=894))
        b = m['original_disagreement_boundaries'][0]
        self.assertTrue(b['adjacent'])
        self.assertEqual(b['midpoint_ties_to_even'],1.0)
        self.assertEqual(b['reference_distance_gap_units'],.5)
        summary = aggregate([dict(metrics=m)])
        self.assertFalse(summary['hypothesis_supported'])

    def test_invalid_reference_and_output_rejected(self):
        a = self.arrays(); ref = a['native'].astype(np.float64)
        for bad in [ref[:,:,:,:1],np.full(ref.shape,np.nan)]:
            with self.assertRaises(ValueError): metrics(a,bad)
        a['fast_float']=a['fast_float'].astype(np.float16)
        with self.assertRaises(ValueError): metrics(a,ref)

    def test_nonadjacent_gap_is_not_labelled_ulp(self):
        a = self.arrays();a['standard_half'].flat[0]=1.25;a['standard_float'].flat[0]=1.25
        m=metrics(a,a['native'].astype(np.float64));b=m['original_disagreement_boundaries'][0]
        self.assertFalse(b['adjacent'])
        self.assertEqual(b['standard_distance_gap_units'],.5)
        self.assertFalse(any('ulps' in name for name in b))


if __name__ == '__main__':
    unittest.main()
