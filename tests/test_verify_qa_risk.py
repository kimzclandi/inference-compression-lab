"""Fault injection for source-only head reconstruction and calibration evidence."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lab.artifact_integrity import file_hashes
from lab.extractive_qa import decode
from lab.qa_risk_calibration import fit, FEATURE_NAMES, FIT_CONFIG
from lab.quantization_diagnostics import sha
from experiments.qa_risk import SOURCE_FILES, model_digest
from experiments.verify_qa_risk import (assert_no_parameters, load_roles, reconstruct_matrix,
                                       select, verify_training, verify, verify_evaluation)

ROOT=Path(__file__).resolve().parents[1]
try: HAS_NUMPY=importlib.metadata.version('numpy')=='2.2.6'
except importlib.metadata.PackageNotFoundError: HAS_NUMPY=False


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def raw_prediction(identifier,positive):
    window=dict(start_logits=[0.0 if positive else 3.0,-8.0,-8.0,3.0,0.0,-8.0],
                end_logits=[0.0 if positive else 3.0,-8.0,-8.0,3.0,0.0,-8.0],
                input_ids=[0,7,2,12,13,2],attention_mask=[1]*6,sequence_ids=[None,0,None,1,1,None],
                offsets=[[0,0],[0,4],[0,0],[0,5],[6,10],[0,0]],
                context_mask=[False,False,False,True,True,False],cls_index=0)
    result=decode('Alpha Beta',[window],30);result.update(id=identifier,raw_windows=[window]);return result


class RiskContractTests(unittest.TestCase):
    def test_nested_parameter_disclosure_rejected(self):
        for value in [{'weights':[]},{'trace':[{'scaler':{}}]},{'variants':{'intercept':0}}]:
            with self.assertRaises(ValueError):assert_no_parameters(value)
        assert_no_parameters({'config':{'regularize_intercept':False},'model_sha256':'a'*64})

    def test_failed_calibration_cannot_consume_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training'
            put(training/'protocol.json',{'numpy_version':'2.2.6','scope':'synthetic'})
            put(training/'selection.json',{})
            selected=dict(any_eligible=False,both_eligible=False,variants={
                name:dict(eligible=False,threshold=None) for name in ('fp32','int8')})
            with patch('experiments.verify_qa_risk.verify_training',return_value={'selection':selected}):
                result=verify(root)
                self.assertEqual(result['status'],'calibration_failed')
                self.assertEqual(result['compression_gate']['status'],'not_evaluated')
                (root/'evaluation-int8').mkdir()
                with self.assertRaises(ValueError):verify(root)

    def test_only_int8_task_can_pass_without_compression_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training'
            put(training/'protocol.json',{'numpy_version':'2.2.6','scope':'synthetic'})
            put(training/'selection.json',{})
            (root/'evaluation-int8').mkdir()
            selected=dict(any_eligible=True,both_eligible=False,variants={
                'fp32':dict(eligible=False,threshold=None),'int8':dict(eligible=True,threshold=.7)})
            with patch('experiments.verify_qa_risk.verify_training',return_value={'selection':selected}), \
                 patch('experiments.verify_qa_risk.verify_evaluation',return_value={'gate':{'all_pass':True},'summary':{}}):
                result=verify(root)
                self.assertTrue(result['variants']['int8']['bounded_task_eligible'])
                self.assertEqual(result['variants']['int8']['answer_entrypoint'],'bounded_local_prototype')
                self.assertEqual(result['compression_gate']['status'],'not_evaluated')
                self.assertFalse(result['compression_default_recommendation'])

    def test_role_overlap_rejected_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);spec=dict(data_sha256={},dataset_sizes={})
            for role in ('train','calibration','evaluation'):
                path=repo/'configs/qa-risk/dataset'/role/'data.jsonl';path.parent.mkdir(parents=True)
                row=dict(id=role,source_title=role,family_id=role,context='Shared context',question=role,
                         answers=['Shared'],is_impossible=False,split=role)
                path.write_text(json.dumps(row)+'\n');spec['data_sha256'][role]=sha(path);spec['dataset_sizes'][role]=1
            with self.assertRaises(ValueError):load_roles(spec,repo)


@unittest.skipUnless(HAS_NUMPY,'Complete head reconstruction explicitly requires NumPy 2.2.6')
class RiskTrainingVerificationTests(unittest.TestCase):
    feature_extractor = None
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)/'training';self.folder.mkdir()
        self.roles={};self.raw={};self.parent={}
        for role,n in [('train',8),('calibration',4),('evaluation',4)]:
            data=[]
            for i in range(n):
                positive=i%2==0;identifier=role+str(i)
                row=dict(id=identifier,source_title=role,family_id=identifier,context='Alpha Beta',question='What?',
                         answers=['Alpha'] if positive else [],is_impossible=not positive,split=role)
                data.append(row)
                if role!='evaluation':
                    self.raw[identifier]=raw_prediction(identifier,positive)
                    self.parent[identifier]={**row,'split':'old'}
            self.roles[role]=data
        self.spec=dict(locked=True,feature_names=list(FEATURE_NAMES),fit_config=FIT_CONFIG,numpy_version='2.2.6',
            threshold_grid=[.5,.7,.9],scope='synthetic verification fixture',
            quality_constraints=dict(min_accepted_precision=.9,min_answerable_answer_coverage=.4,
                min_correct_answerable_coverage=.35,max_unanswerable_false_accept_rate=.1,max_invalid_rate=.05),
            source_sha256={n:sha(ROOT/n) for n in SOURCE_FILES},data_sha256={})
        for role in ('train','calibration'):
            p=self.folder/(role+'-data.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in self.roles[role]))
            self.spec['data_sha256'][role]=sha(p)
        put(self.folder/'protocol.json',self.spec);self.protocol_sha=sha(self.folder/'protocol.json')
        for name in SOURCE_FILES:
            dest=self.folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        selection=dict(protocol_sha256=self.protocol_sha,variants={},scope=self.spec['scope'])
        for variant in ('fp32','int8'):
            matrix=reconstruct_matrix(self.roles['train'],self.raw)
            cal=reconstruct_matrix(self.roles['calibration'],self.raw)
            model=fit([r['features'] for r in matrix],[r['target'] for r in matrix])
            put(self.folder/(variant+'-training-features.json'),matrix)
            put(self.folder/(variant+'-calibration-features.json'),cal)
            put(self.folder/(variant+'-fit.json'),{k:v for k,v in model.items() if k not in ('weights','intercept','scaler')})
            chosen,predictions=select(self.roles['calibration'],cal,model,self.spec)
            put(self.folder/(variant+'-calibration-predictions.json'),predictions)
            selection['variants'][variant]={**chosen,'model_sha256':model_digest(model)}
        selection['any_eligible']=any(v['eligible'] for v in selection['variants'].values())
        selection['both_eligible']=all(v['eligible'] for v in selection['variants'].values())
        put(self.folder/'selection.json',selection)
        put(self.folder/'run.json',dict(status='complete',variants=['fp32','int8'],numpy='2.2.6',protocol_sha256=self.protocol_sha))
        self.rehash()

    def rehash(self):put(self.folder/'checksums.json',file_hashes(self.folder,exclude=('checksums.json',)))

    def verify(self):
        with patch('experiments.verify_qa_risk.load_roles',return_value=self.roles), \
             patch('experiments.verify_qa_risk.verify_parents',return_value=(self.parent,{'fp32':self.raw,'int8':self.raw})):
            return verify_training(self.folder,self.spec,self.protocol_sha,feature_extractor=self.feature_extractor)

    def mutate_json(self,name,mutation):
        p=self.folder/name;value=json.loads(p.read_text());mutation(value);put(p,value);self.rehash()

    def test_rebuild_from_train_only_and_keep_private_parameters_internal(self):
        result=self.verify()
        self.assertEqual(result['heads']['fp32']['n_train'],8)
        assert_no_parameters(result['selection'])
        self.assertNotIn('weights',json.loads((self.folder/'fp32-fit.json').read_text()))

    def test_refit_replays_verified_serialized_training_values(self):
        actual_reconstruct=reconstruct_matrix
        def rounded_reconstruction(data,raw,**kwargs):
            result=actual_reconstruct(data,raw,**kwargs)
            if data[0]['split']=='train':
                result[0]['features'][0]+=1e-13
            return result
        with patch('experiments.verify_qa_risk.reconstruct_matrix',side_effect=rounded_reconstruction), \
             patch('experiments.verify_qa_risk.fit',wraps=fit) as fit_spy:
            self.verify()
        self.assertEqual(fit_spy.call_count,2)
        for call,variant in zip(fit_spy.call_args_list,('fp32','int8')):
            stored=json.loads((self.folder/(variant+'-training-features.json')).read_text())
            self.assertEqual(call.args[0],[record['features'] for record in stored])
            self.assertEqual(call.args[1],[record['target'] for record in stored])

    def test_rehashed_feature_target_score_and_model_identity_mutations(self):
        cases=[('fp32-training-features.json',lambda v:v[0].update(target=0)),
               ('fp32-calibration-features.json',lambda v:v[0]['features'].__setitem__(0,999.0)),
               ('fp32-calibration-predictions.json',lambda v:v[0].update(confidence=.01)),
               ('selection.json',lambda v:v['variants']['fp32'].update(model_sha256='0'*64)),
               ('selection.json',lambda v:v['variants']['int8'].update(threshold=.99)),
               ('fp32-fit.json',lambda v:v.update(weights=[0]*5))]
        for name,mutation in cases:
            with self.subTest(name=name):
                original=(self.folder/name).read_bytes();self.mutate_json(name,mutation)
                with self.assertRaises(ValueError):self.verify()
                (self.folder/name).write_bytes(original);self.rehash()

    def test_empty_manifest_and_partial_fit_fail_closed(self):
        put(self.folder/'checksums.json',{})
        with self.assertRaises(ValueError):self.verify()
        self.rehash();self.mutate_json('run.json',lambda v:v.update(status='failed'))
        with self.assertRaises(ValueError):self.verify()


@unittest.skipUnless(HAS_NUMPY,'Complete head reconstruction explicitly requires NumPy 2.2.6')
class RecordedEvaluationMutationTests(unittest.TestCase):
    feature_extractor = None
    def test_rehashed_raw_score_and_completion_mutations_fail(self):
        training_dir=ROOT/'results/qa-risk-v2/training'
        original=ROOT/'results/qa-risk-v2/evaluation-int8'
        if not original.exists():self.skipTest('Fixed completed evaluation is not available')
        spec=json.loads((training_dir/'protocol.json').read_text())
        matrix=json.loads((training_dir/'int8-training-features.json').read_text())
        model=fit([r['features'] for r in matrix],[r['target'] for r in matrix])
        training=dict(selection=json.loads((training_dir/'selection.json').read_text()),heads={'int8':model},
                      roles={'evaluation':[json.loads(x) for x in (original/'data.jsonl').read_text().splitlines()]})
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'evaluation-int8';shutil.copytree(original,folder)
            def check():return verify_evaluation(folder,'int8',spec,sha(training_dir/'protocol.json'),training_dir,training,
                                                feature_extractor=self.feature_extractor)
            run_bytes=(folder/'run.json').read_bytes();raw_bytes=(folder/'predictions.jsonl').read_bytes()
            mutations=[lambda p:p[0].update(confidence=.01),lambda p:p[0].update(base_confidence=.01),
                       lambda p:p[0]['risk_features']['features'].__setitem__(0,999.0),
                       lambda p:p[0]['raw_windows'][0]['context_mask'].__setitem__(0,True)]
            for mutation in mutations:
                predictions=[json.loads(x) for x in raw_bytes.decode().splitlines()];mutation(predictions)
                (folder/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in predictions))
                put(folder/'checksums.json',file_hashes(folder,exclude=('checksums.json',)))
                with self.assertRaises(ValueError):check()
                (folder/'predictions.jsonl').write_bytes(raw_bytes)
            run=json.loads(run_bytes);run['completed_predictions']=0;put(folder/'run.json',run)
            put(folder/'checksums.json',file_hashes(folder,exclude=('checksums.json',)))
            with self.assertRaises(ValueError):check()


if __name__=='__main__':unittest.main()
