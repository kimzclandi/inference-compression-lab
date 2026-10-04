import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from lab.artifact_integrity import file_hashes

class ReproductionFailureTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'semantic fitting dependency not installed')
    def test_unchanged_decisions_do_not_hide_score_reproduction_failure(self):
        from experiments.verify_qa_semantic import ROOT,audit
        from experiments import qa_semantic_fixed as api
        from experiments.qa_nonlinear import select
        import warnings
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training'
            shutil.copytree(ROOT/'results/qa-semantic-fixed-v1/training',training)
            file=training/'development-predictions.json'
            (x,y,_),(cal,_,records)=api.inputs(ROOT/'results/qa-semantic-v1/collection')
            with warnings.catch_warnings(record=True):
                pred=api.score(api.fit_head(x,y),cal,records)
            chosen=json.loads((training/'selection.json').read_text())
            chosen.update(select(api.base('calibration')[0],pred,api.preflight()))
            (training/'selection.json').write_text(json.dumps(chosen))
            grid=[r['threshold'] for r in chosen['curve']]
            row=next(r for r in pred if min(abs(r['confidence']-t) for t in grid)>1e-3 and r['confidence']<.99)
            row['confidence']+=2e-6;file.write_text(json.dumps(pred))
            (training/'checksums.json').write_text(json.dumps(file_hashes(training,exclude=('checksums.json',))))
            result=audit(root,fixed=True)
            self.assertFalse(result['evidence_valid'])
            self.assertFalse(result['archived_activation_rebuild'])
            self.assertTrue(result['all_grid_decisions_identical'])
            self.assertGreater(result['max_probability_difference'],1e-6)
