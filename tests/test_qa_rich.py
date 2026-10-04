import math
import unittest
from lab.qa_rich_features import enrich

class RichTests(unittest.TestCase):
    def test_label_free_uncertainty_and_lexical_features(self):
        p=dict(window_index=0,start_token=1,end_token=1,prediction='cat',start=0,raw_windows=[dict(start_logits=[0,math.log(3),0],end_logits=[0,math.log(3),0],context_mask=[False,True,True],cls_index=0)])
        b=dict(features=[0]*5,feature_names=list('abcde'),target=0)
        a=enrich('Which cat?', 'cat dog',p,b)
        self.assertEqual(len(a['features']),16)
        self.assertAlmostEqual(a['features'][5],-(.6*math.log(.6)+2*.2*math.log(.2))/math.log(3))
        self.assertEqual(a['features'][13],1.)
        b['target']=1
        self.assertEqual(enrich('Which cat?','cat dog',p,b)['features'],a['features'])
