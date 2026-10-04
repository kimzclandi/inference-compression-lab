import unittest
from experiments.qa_risk import select_variant
from lab.qa_gate import DEFAULT_CONSTRAINTS
from lab.qa_risk_calibration import FEATURE_NAMES,FIT_CONFIG

class RiskRunnerTests(unittest.TestCase):
    def test_selection_consumes_metric_pair_and_never_falls_back_to_best_em(self):
        model=dict(schema='qa-risk-logistic-v1',feature_names=list(FEATURE_NAMES),config=FIT_CONFIG,
                   convergence={'converged':True},scaler={'mean':[0]*5,'scale':[1]*5},weights=[0]*5,intercept=0)
        data=[dict(id='a',family_id='fa',context='alpha',question='what',answers=['alpha'],is_impossible=False),
              dict(id='b',family_id='fb',context='beta',question='what',answers=[],is_impossible=True)]
        records=[dict(id=r['id'],prediction=r['context'],features=[0]*5) for r in data]
        selected,predictions=select_variant(data,records,model,dict(threshold_grid=[.5,.9],quality_constraints=DEFAULT_CONSTRAINTS))
        self.assertFalse(selected['eligible']);self.assertIsNone(selected['threshold'])
        self.assertEqual(len(selected['candidates']),2);self.assertEqual(len(predictions),2)

if __name__=='__main__':unittest.main()
