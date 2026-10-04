import unittest
import numpy as np
from unittest.mock import patch
from importlib.util import find_spec

class ProjectionTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('sklearn'),'semantic fitting dependency not installed')
    def test_classifier_trains_on_the_inference_projection(self):
        from experiments.qa_semantic_fixed import fit_head
        from sklearn.linear_model import LogisticRegression
        rng=np.random.default_rng(19);x=rng.normal(size=(90,1541));y=np.tile([0,1],45)
        observed=[];original=LogisticRegression.fit
        def fit(model,design,labels,*args,**kwargs):
            observed.append(design.copy());return original(model,design,labels,*args,**kwargs)
        with patch.object(LogisticRegression,'fit',fit):head=fit_head(x,y)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(observed[0],head.named_steps['features'].transform(x))
