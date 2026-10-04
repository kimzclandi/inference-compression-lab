import unittest
from experiments.qa_nonlinear import select

class NonlinearTests(unittest.TestCase):
    def test_coverage_cannot_be_replaced_by_precision(self):
        data=[dict(id=str(i),context='cat dog',answers=['cat'] if i<10 else [],is_impossible=i>=10) for i in range(20)]
        pred=[dict(id=str(i),prediction='cat',confidence=.95 if i==0 else .1) for i in range(20)]
        spec=dict(threshold_grid=[.5,.9],quality_constraints=dict(min_accepted_precision=.9,min_answerable_answer_coverage=.4,min_correct_answerable_coverage=.35,max_unanswerable_false_accept_rate=.1,max_invalid_rate=.05))
        self.assertFalse(select(data,pred,spec)['eligible'])
        for i in range(5):pred[i]['confidence']=.95
        selected=select(data,pred,spec)
        self.assertTrue(selected['eligible']);self.assertEqual(selected['threshold'],.5)
