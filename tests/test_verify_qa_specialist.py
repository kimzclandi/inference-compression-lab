"""Mutation checks attack rehashed evidence, not only the outer checksum."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from lab.artifact_integrity import file_hashes
from lab.extractive_qa import decode
from lab.qa_specialist_runtime import LIMITS, RUNTIME_PACKAGES
from lab.quantization_diagnostics import sha
from experiments.qa_specialist import SOURCE_FILES
from experiments.verify_qa_specialist import (
    calibration_selection, independently_decode, verify_run, write_selection, verify,
)

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sample_window(positive=True):
    return dict(start_logits=[0.0 if positive else 4.0, -9.0, -9.0, 3.0, -9.0],
                end_logits=[0.0 if positive else 4.0, -9.0, -9.0, 3.0, -9.0],
                input_ids=[0, 7, 2, 12, 2], attention_mask=[1]*5,
                offsets=[[0,0],[0,4],[0,0],[0,5],[0,0]],
                sequence_ids=[None,0,None,1,None], context_mask=[False,False,False,True,False], cls_index=0)


class SpecialistVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets_bytes = (ROOT/'results/qa-specialist-assets-v1/assets.json').read_bytes()
        self.assets = json.loads(self.assets_bytes)
        self.spec = dict(locked=True, threshold_grid=[0.0, .5, .999999],
            model_id=self.assets['model_id'],revision=self.assets['revision'],
            dataset_sizes={'calibration':2,'evaluation':2},
            quality_constraints=dict(min_accepted_precision=.9, min_answerable_answer_coverage=.4,
                min_correct_answerable_coverage=.35, max_unanswerable_false_accept_rate=.1, max_invalid_rate=.05),
            asset_manifest_sha256=sha(ROOT/'results/qa-specialist-assets-v1/assets.json'),
            source_sha256={name:sha(ROOT/name) for name in SOURCE_FILES}, input_limits=LIMITS,
            compression_gate=dict(replicates=5000,confidence=.95,seed=8,minimum_ci_lower_bound=-.02,
                                  maximum_model_file_bytes_ratio=.6))
        self.data = {}
        for split in ('calibration','evaluation'):
            self.data[split] = [dict(id=split+str(i), context='Alpha', question='What?', answers=['Alpha'] if i==0 else [],
                is_impossible=i==1, family_id='family'+str(i), source_title='Synthetic', split=split) for i in range(2)]
            p=self.root/(split+'.jsonl'); p.write_text(''.join(json.dumps(r)+'\n' for r in self.data[split]))
        self.spec['data_sha256']={s:sha(self.root/(s+'.jsonl')) for s in self.data}
        self.study=self.root/'study.json'; put(self.study,self.spec)

    def make_run(self, split='calibration', variant='fp32'):
        folder=self.root/(split+'-'+variant);folder.mkdir()
        shutil.copy2(self.study,folder/'protocol.json')
        shutil.copy2(self.root/(split+'.jsonl'),folder/'data.jsonl')
        (folder/'assets.json').write_bytes(self.assets_bytes)
        for name in SOURCE_FILES:
            dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        preds=[]
        for i,row in enumerate(self.data[split]):
            windows=[sample_window(i==0)]; p=decode(row['context'],windows,LIMITS['max_answer_tokens'])
            p.update(id=row['id'], raw_windows=windows, feature_count=1,input_tokens=[5],
                     tokenization_seconds=.01,inference_seconds=.02,decode_seconds=.01,total_pipeline_seconds=.05)
            preds.append(p)
        self.save_predictions(folder,preds)
        run=dict(status='complete',stage='complete',variant=variant,split=split,
                 protocol_sha256=sha(self.study),dataset_sha256=self.spec['data_sha256'][split],
                 source_sha256=self.spec['source_sha256'],assets=self.assets,
                 provider='CPUExecutionProvider',intra_op_threads=4,inter_op_threads=1,
                 packages={name:self.assets['packages'][name] for name in RUNTIME_PACKAGES},
                 model_load_seconds=.1,completed_predictions=2)
        if split=='evaluation':
            shutil.copy2(self.root/'selection.json',folder/'selection.json')
            run['selection_sha256']=sha(folder/'selection.json')
        put(folder/'run.json',run); self.rehash(folder)
        return folder

    def save_predictions(self,folder,preds):
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in preds))

    def rehash(self,folder):
        put(folder/'checksums.json',file_hashes(folder,exclude=('checksums.json',)))

    def mutate_prediction(self,folder,fn):
        preds=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        fn(preds);self.save_predictions(folder,preds);self.rehash(folder)

    def test_valid_raw_and_calibration_selection(self):
        self.make_run();self.make_run(variant='int8')
        result=calibration_selection(self.root,self.spec,sha(self.study))
        self.assertIsInstance(result,dict)
        self.assertTrue(result['calibration_feasible'])
        self.assertEqual(result['variants']['fp32']['threshold'],.5)
        self.assertEqual(len(result['variants']['int8']['curve']),3)

    def test_complete_evaluation_uses_frozen_threshold(self):
        self.make_run();self.make_run(variant='int8');write_selection(self.root,self.study)
        self.make_run('evaluation','fp32');self.make_run('evaluation','int8')
        result=verify(self.root,self.study)
        self.assertTrue(result['overall_success'])
        self.assertEqual(result['answer_entrypoint'],'bounded_local_prototype')
        self.assertEqual(result['paired']['em_ci95'],[0.0,0.0])

    def test_selection_cannot_be_overwritten(self):
        self.make_run();self.make_run(variant='int8');write_selection(self.root,self.study)
        with self.assertRaises(ValueError):write_selection(self.root,self.study)

    def test_candidate_mutations_rejected_even_with_fresh_hashes(self):
        mutations=[lambda p:p[0].update(confidence=.999),lambda p:p[0].update(margin=9.0),
                   lambda p:p[0]['windows'][0].update(null_score=1.0),
                   lambda p:p[0].update(start=1),lambda p:p[0].update(context_sha256='0'*64),
                   lambda p:p[0]['raw_windows'][0]['start_logits'].__setitem__(3,float('nan')),
                   lambda p:p[0]['raw_windows'][0]['context_mask'].__setitem__(1,True),
                   lambda p:p[0]['raw_windows'][0]['offsets'].__setitem__(3,[0,7]),
                   lambda p:p[0].update(feature_count=2),lambda p:p.pop(),
                   lambda p:p[1].update(id=p[0]['id'])]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                folder=self.make_run();self.mutate_prediction(folder,mutation)
                with self.assertRaises((ValueError,KeyError)):verify_run(folder)
                shutil.rmtree(folder)

    def test_complete_count_status_source_and_asset_mutations_rejected(self):
        for field,value in [('completed_predictions',1),('status','failed'),('provider','CUDAExecutionProvider')]:
            with self.subTest(field=field):
                folder=self.make_run();run=json.loads((folder/'run.json').read_text());run[field]=value
                put(folder/'run.json',run);self.rehash(folder)
                with self.assertRaises(ValueError):verify_run(folder)
                shutil.rmtree(folder)
        folder=self.make_run();(folder/'source/lab/qa_metrics.py').write_text('fake');self.rehash(folder)
        with self.assertRaises(ValueError):verify_run(folder)
        shutil.rmtree(folder)
        folder=self.make_run();(folder/'assets.json').write_text('{}');self.rehash(folder)
        with self.assertRaises(ValueError):verify_run(folder)

    def test_empty_and_partial_checksums_rejected(self):
        folder=self.make_run()
        for manifest in ({},{'run.json':sha(folder/'run.json')},{'../escape':'0'*64}):
            put(folder/'checksums.json',manifest)
            with self.assertRaises(ValueError):verify_run(folder)

    def test_failed_calibration_stops_without_consuming_evaluation(self):
        self.spec['threshold_grid']=[.999999];put(self.study,self.spec)
        self.make_run();self.make_run(variant='int8');selection=write_selection(self.root,self.study)
        self.assertFalse(selection['calibration_feasible'])
        self.assertIsNone(selection['variants']['fp32']['threshold'])
        self.assertEqual(verify(self.root,self.study)['answer_entrypoint'],'unavailable_quality')
        (self.root/'evaluation-fp32').mkdir()
        with self.assertRaises(ValueError):verify(self.root,self.study)

    def test_rehashed_selection_change_and_partial_evaluation_rejected(self):
        self.make_run();self.make_run(variant='int8');write_selection(self.root,self.study)
        selection=json.loads((self.root/'selection.json').read_text())
        selection['variants']['int8']['threshold']=0.0
        put(self.root/'selection.json',selection)
        with self.assertRaises(ValueError):verify(self.root,self.study)

    def test_independent_decoder_ties_windows_and_zero_length_context_tokens(self):
        first=sample_window();second=deepcopy(first)
        second['start_logits'][3]=4.0
        expected=decode('Alpha',[first,second],30)
        self.assertEqual(independently_decode('Alpha',[first,second],LIMITS),expected)
        duplicate=deepcopy(first)
        duplicate['offsets'][3]=[0,0]
        with self.assertRaises(ValueError):independently_decode('Alpha',[duplicate],LIMITS)


if __name__=='__main__':unittest.main()
