import json
import unittest
from experiments.release_archive import validate_payload

class HeadExclusionTests(unittest.TestCase):
    def test_json_suffix_cannot_hide_learned_weights(self):
        for key in ('weights','intercept','scaler'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_payload('evidence/model.json',json.dumps({'schema':'qa-risk-logistic-v1',key:[]}))
    def test_nonparametric_fit_trace_remains_distributable(self):
        validate_payload('fit.json',json.dumps({'schema':'qa-risk-logistic-v1','trace':[{'objective':1.2}]}))

if __name__=='__main__':unittest.main()
