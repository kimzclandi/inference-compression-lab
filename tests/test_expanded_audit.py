import copy
import json
import shutil
import tempfile
from pathlib import Path
import unittest

from experiments.verify_qa_expanded import check_features, reference, independent_probability, check_diagnosis
from experiments.qa_expanded import evaluate, ROOT
from lab.quantization_diagnostics import sha, read, rows
from lab.artifact_integrity import file_hashes


class ExpandedAuditTests(unittest.TestCase):
    def test_independent_reference_detects_feature_and_label_corruption(self):
        ref=reference(); e=ref.examples()[0]
        features,raw,diagnostics=ref.independent(e['context'],e['windows']); raw['id']='one'
        data=[dict(id='one',context=e['context'],answers=['cat dog'],is_impossible=False)]
        matrix=[dict(id='one',prediction=raw['prediction'],features=features,diagnostics=diagnostics,feature_names=ref.NAMES,target=1)]
        self.assertEqual(check_features(data,[raw],matrix,ref)['rows'],1)
        broken=copy.deepcopy(matrix);broken[0]['features'][2] += .01
        with self.assertRaises(ValueError):check_features(data,[raw],broken,ref)
        broken=copy.deepcopy(matrix);broken[0]['target']=0
        with self.assertRaisesRegex(ValueError,'Correctness label'):check_features(data,[raw],broken,ref)

    def test_failed_calibration_blocks_before_head_or_model_load(self):
        training=ROOT/'results/qa-expanded-v1/training'
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'evaluation'
            with self.assertRaisesRegex(ValueError,'eligible frozen selection'):
                evaluate(training,Path(temp)/'missing-heads',Path(temp)/'missing-model',output,sha(training/'selection.json'))
            self.assertFalse(output.exists())

    def test_independent_logistic_score_handles_both_extremes(self):
        model=dict(scaler=dict(mean=[0]*5,scale=[1]*5),weights=[0]*5,intercept=0)
        self.assertEqual(independent_probability(model,[0]*5),.5)
        model['intercept']=1000;self.assertEqual(independent_probability(model,[0]*5),1.)
        model['intercept']=-1000;self.assertEqual(independent_probability(model,[0]*5),0.)

    def test_diagnosis_detects_false_decision_even_with_updated_checksum(self):
        source=ROOT/'results/qa-expanded-review-v1'
        data=rows(ROOT/'results/qa-expanded-v1/calibration/data.jsonl')
        predictions={name:read(ROOT/f'results/qa-expanded-v1/training/{name}-calibration-predictions.json') for name in ('original','expanded')}
        methods={name:dict(curve=[{k:v for k,v in p.items() if k not in ('per_article','failed_constraints')} for p in points])
                 for name,points in read(source/'diagnosis.json')['curves'].items()}
        spec=read(ROOT/'configs/qa-expanded/study.json')
        self.assertTrue(check_diagnosis(source,data,predictions,methods,spec)['all_pass'])
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'review';shutil.copytree(source,target)
            path=target/'calibration-decisions.json'; ledger=read(path)
            ledger[0]['accepted_correct']=not ledger[0]['accepted_correct'];path.write_text(json.dumps(ledger))
            (target/'checksums.json').write_text(json.dumps(file_hashes(target,exclude=('checksums.json',))))
            with self.assertRaisesRegex(ValueError,'Diagnosis row ledger'):
                check_diagnosis(target,data,predictions,methods,spec)


if __name__=='__main__':unittest.main()
