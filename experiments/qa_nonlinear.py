"""Fixed nonlinear correctness ranking; gate evaluation after development only."""
import argparse
import importlib.metadata
import math
from pathlib import Path
import pickle
import shutil

from lab.artifact_integrity import file_hashes, verify_hashes, git_identity
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, rows, sha, write, aggregates_equal
from experiments.verify_qa_coverage_gap import metrics, passes, require
from experiments.verify_qa_expanded import check_features, reference

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'configs/qa-nonlinear/study.json'


def preflight():
    spec=read(SPEC)
    for name,version in spec['versions'].items():
        require(importlib.metadata.version(name)==version,'Dependency version: '+name)
    for name,digest in spec['sha256'].items():require(sha(ROOT/name)==digest,'Frozen identity: '+name)
    return spec


def model(spec,matrix):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fitted=HistGradientBoostingClassifier(**spec['estimator'])
        fitted.fit([r['features'] for r in matrix],[r['target'] for r in matrix])
    return fitted


def score(head,matrix):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):values=head.predict_proba([r['features'] for r in matrix])[:,1]
    return [dict(id=r['id'],prediction=r['prediction'],confidence=float(v)) for r,v in zip(matrix,values)]


def select(data,predictions,spec):
    curve=[dict(threshold=t,metrics=metrics(data,predictions,t)) for t in spec['threshold_grid']]
    for c in curve:c['passes']=passes(c['metrics'],spec['quality_constraints'])
    passing=[c for c in curve if c['passes']]
    return dict(eligible=bool(passing),threshold=passing[0]['threshold'] if passing else None,curve=curve)


def start(output,action):
    spec=preflight();out=reserve_directory(output)
    shutil.copy2(SPEC,out/'protocol.json');shutil.copy2(Path(__file__),out/'runner.py')
    state=dict(action=action,status='running',**git_identity(ROOT),protocol_sha256=sha(SPEC),
               parameters_distributed=False,authorship='AI-assisted implementation and execution')
    return spec,out,state


def finish(out,state):
    write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def train(output,head_path):
    spec,out,state=start(output,'train')
    try:
        head_path=head_path.resolve()
        require(head_path.is_relative_to(ROOT/'runs') and not (ROOT/'runs').is_symlink(),'Head must remain under local runs')
        require(not head_path.exists(),'Refuse overwrite of head')
        matrix=read(ROOT/spec['training_matrix']);cal=read(ROOT/spec['development_matrix']);data=rows(ROOT/spec['development_data'])
        require(len(matrix)==2304 and len(cal)==len(data)==192,'Fixed matrix sizes')
        head=model(spec,matrix); predictions=score(head,cal)
        chosen=select(data,predictions,spec)
        head_path.parent.mkdir(parents=True,exist_ok=True)
        with head_path.open('xb') as f:pickle.dump(head,f,protocol=5)
        write(out/'development-predictions.json',predictions)
        write(out/'selection.json',dict(**chosen,protocol_sha256=sha(SPEC),head_pickle_sha256=sha(head_path)))
        write(out/'fit.json',dict(n_train=len(matrix),n_features=5,iterations=int(head.n_iter_),estimator=spec['estimator'],
            stopping='Fixed iteration budget; no gradient convergence claim for boosting',fit_attempts=1))
        state.update(status='complete',development_passed=chosen['eligible'],evaluation_permitted=chosen['eligible'])
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:finish(out,state)
    return state


def evaluate(output,training,head_path,asset_root,selection_sha256):
    spec=preflight();verify_hashes(training,read(training/'checksums.json'),exclude=('checksums.json',))
    chosen=read(training/'selection.json')
    require(sha(training/'selection.json')==selection_sha256 and chosen['eligible'] is True,'Eligible frozen selection required')
    require(chosen['protocol_sha256']==sha(SPEC) and sha(head_path)==chosen['head_pickle_sha256'],'Head/selection identity')
    with head_path.open('rb') as f:head=pickle.load(f)  # Only this run's own SHA-checked, local head.
    spec,out,state=start(output,'evaluate')
    try:
        from lab.qa_specialist_runtime import ExtractiveRuntime
        from experiments.qa_expanded import collect_into
        runtime=ExtractiveRuntime(asset_root,'int8',spec['asset_manifest_sha256'])
        data,features=collect_into(runtime,'evaluation',out,state,spec)
        require(sha(out/'data.jsonl')==spec['sha256'][spec['evaluation_data']],'Evaluation dataset identity')
        pred=score(head,features); actual=metrics(data,pred,chosen['threshold'])
        write(out/'predictions.json',pred);write(out/'selection.json',chosen)
        write(out/'summary.json',dict(threshold=chosen['threshold'],metrics=actual,quality_gate_passed=passes(actual,spec['quality_constraints']),
            scope=spec['scope'],default_policy_changed=False))
        state.update(status='complete',selection_sha256=selection_sha256,quality_gate_passed=passes(actual,spec['quality_constraints']))
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:finish(out,state)
    return state


def verify(folder,output):
    import gzip,json
    spec=preflight();train=folder/'training'
    verify_hashes(train,read(train/'checksums.json'),exclude=('checksums.json',))
    require(sha(train/'protocol.json')==sha(SPEC) and sha(train/'runner.py')==sha(Path(__file__)),'Training source identity')
    chosen=read(train/'selection.json');data=rows(ROOT/spec['development_data']);cal=read(ROOT/spec['development_matrix'])
    pred=read(train/'development-predictions.json');expected=select(data,pred,spec)
    require(aggregates_equal({k:chosen[k] for k in expected},expected),'Development selection')
    rebuilt=model(spec,read(ROOT/spec['training_matrix']));rescored=score(rebuilt,cal)
    require(aggregates_equal(pred,rescored),'Archived-input reconstruction scores')
    eval_dir=folder/'evaluation';result=dict(evidence_valid=True,development_passed=chosen['eligible'],evaluation_run=eval_dir.exists(),
        archived_input_reconstruction=True,model_inference=False,quality_gate_passed=False)
    if chosen['eligible']:
        require(eval_dir.exists(),'Eligible candidate still requires reserved evaluation')
        verify_hashes(eval_dir,read(eval_dir/'checksums.json'),exclude=('checksums.json',))
        require(sha(eval_dir/'protocol.json')==sha(SPEC) and sha(eval_dir/'runner.py')==sha(Path(__file__)),'Evaluation source identity')
        require(read(eval_dir/'selection.json')==chosen,'Evaluation selection changed')
        require(sha(eval_dir/'data.jsonl')==spec['sha256'][spec['evaluation_data']],'Evaluation identity')
        data=rows(eval_dir/'data.jsonl');raw=[]
        for p in sorted(eval_dir.glob('predictions-*.jsonl.gz')):raw.extend(json.loads(s) for s in gzip.decompress(p.read_bytes()).decode().splitlines())
        matrix=read(eval_dir/'features.json');result['features']=check_features(data,raw,matrix,reference())
        pred=read(eval_dir/'predictions.json');require(aggregates_equal(score(rebuilt,matrix),pred),'Evaluation reconstructed scores')
        summary=read(eval_dir/'summary.json');actual=metrics(data,pred,chosen['threshold'])
        require(aggregates_equal(actual,summary['metrics']),'Evaluation metric arithmetic')
        require(summary['quality_gate_passed']==passes(actual,spec['quality_constraints']),'Evaluation gate')
        result.update(quality_gate_passed=summary['quality_gate_passed'],evaluation_metrics=actual)
    else:require(not eval_dir.exists(),'Evaluation forbidden after development failure')
    write(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train');t.add_argument('--head-path',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--head-path',type=Path,required=True);e.add_argument('--training',type=Path,required=True)
    e.add_argument('--asset-root',type=Path,required=True);e.add_argument('--selection-sha256',required=True)
    v=sub.add_parser('verify');v.add_argument('--folder',type=Path,required=True)
    for q in (t,e,v):q.add_argument('--output',type=Path,required=True)
    a=vars(p.parse_args());action=a.pop('action');print(globals()[action](**a))
