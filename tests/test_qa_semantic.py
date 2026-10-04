import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from lab.qa_semantic import SemanticRuntime
from experiments import evaluate_qa_semantic as evaluate_module
from lab.quantization_diagnostics import sha

class Session:
    changed=False
    def run(self,*args):
        logits=np.asarray([[0.,1.,2.]])
        return [logits+(1. if self.changed else 0.),logits,np.arange(3*768,dtype=np.float32).reshape(1,3,768)]

class SemanticTests(unittest.TestCase):
    def test_hidden_pooling_and_logit_identity_guard(self):
        runtime=SemanticRuntime.__new__(SemanticRuntime);runtime.session=Session()
        p=dict(window_index=0,start_token=1,end_token=2,raw_windows=[dict(input_ids=[0,1,2],attention_mask=[1,1,1],cls_index=0,start_logits=[0.,1.,2.],end_logits=[0.,1.,2.])])
        v,d=runtime.vector(p);self.assertEqual(v.shape,(1536,));self.assertEqual(d,0.)
        self.assertEqual(v[768],1152.)
        runtime.session.changed=True
        with self.assertRaisesRegex(ValueError,'altered logits'):runtime.vector(p)

    def test_failed_selection_blocks_before_model_or_pickle_load(self):
        training=evaluate_module.ROOT/'results/qa-semantic-v1/training'
        with TemporaryDirectory() as tmp,patch.object(evaluate_module,'preflight',return_value={}):
            out=Path(tmp)/'out'
            with self.assertRaisesRegex(ValueError,'Eligible frozen selection'):
                evaluate_module.evaluate(out,training,Path(tmp)/'none.pkl',Path(tmp)/'assets',Path(tmp)/'tap.onnx',sha(training/'selection.json'))
            self.assertFalse(out.exists())
