"""One frozen semantic correctness-head challenger using archived QA selections."""
import argparse
import importlib.metadata
from pathlib import Path
import pickle
import shutil
import warnings
import numpy as np
from lab.quantization_diagnostics import read,rows,sha,write,aggregates_equal
from lab.artifact_integrity import file_hashes,verify_hashes,git_identity
from lab.evidence import reserve_directory
from lab.qa_expanded_io import load_collection
from experiments.verify_qa_coverage_gap import metrics,passes,require
from experiments.qa_nonlinear import select

ROOT=Path(__file__).resolve().parents[1];SPEC=ROOT/'configs/qa-semantic/study.json'

def preflight():
    s=read(SPEC)
    for p,h in s['sha256'].items():require(sha(ROOT/p)==h,'Frozen input/source: '+p)
    for p,v in s['versions'].items():require(importlib.metadata.version(p)==v,'Version: '+p)
    return s

def start(output,action):
    s=preflight();out=reserve_directory(output);shutil.copy2(SPEC,out/'protocol.json')
    for p in s['source_files']:
        target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
    return s,out,dict(status='running',action=action,**git_identity(ROOT),protocol_sha256=sha(SPEC),parameters_distributed=False,authorship='AI-assisted implementation and execution')

def finish(out,state):
    write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))

def base(split):
    if split=='old_train':
        data=rows(ROOT/'configs/qa-risk/dataset/train/data.jsonl');features=read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
        raw={r['id']:r for s in ('calibration','evaluation') for r in rows(ROOT/f'results/qa-specialist-v1/{s}-int8/predictions.jsonl')}
        return data,[raw[d['id']] for d in data],features
    folder='train-new' if split=='train_new' else 'calibration'
    raw,features=load_collection(ROOT/'results/qa-expanded-v1'/folder,split,sha(ROOT/'configs/qa-expanded/study.json'))
    return rows(ROOT/f'configs/qa-expanded/dataset/{split}.jsonl'),raw,features

def collect(output,asset_root,model_path):
    s,out,state=start(output,'collect')
    try:
        from lab.qa_semantic import SemanticRuntime
        runtime=SemanticRuntime(asset_root,s['asset_manifest_sha256'],model_path)
        state['instrumented_model_sha256']=runtime.model_sha
        for name,splits in [('training',['old_train','train_new']),('development',['calibration'])]:
            matrix=[];ids=[];maxdiff=0.
            for split in splits:
                data,raw,features=base(split)
                for d,p,f in zip(data,raw,features):
                    require(d['id']==p['id']==f['id'],'Semantic row alignment')
                    vector,diff=runtime.vector(p);matrix.append(vector);ids.append(d['id']);maxdiff=max(maxdiff,diff)
                    if len(ids)%128==0:print(name,len(ids),flush=True)
            np.savez_compressed(out/(name+'.npz'),features=np.asarray(matrix,dtype=np.float32))
            write(out/(name+'-identity.json'),dict(ids=ids,max_logit_difference=maxdiff,shape=[len(ids),1536],labels_used_for_features=False))
        state.update(status='complete',inference_rows=2496,encoder_trained=False)
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:finish(out,state)
    return state

def inputs(collection):
    verify_hashes(collection,read(collection/'checksums.json'),exclude=('checksums.json',))
    require(read(collection/'run.json')['status']=='complete' and sha(collection/'protocol.json')==sha(SPEC),'Semantic collection identity')
    values=[]
    for name,splits in [('training',['old_train','train_new']),('development',['calibration'])]:
        records=[]
        for split in splits:records+=base(split)[2]
        require([r['id'] for r in records]==read(collection/(name+'-identity.json'))['ids'],'Embedding ID identity')
        with np.load(collection/(name+'.npz'),allow_pickle=False) as z:semantic=z['features']
        require(semantic.shape==(len(records),1536) and np.isfinite(semantic).all(),'Embedding shape/finiteness')
        x=np.column_stack([[r['features'] for r in records],semantic]);y=np.asarray([r['target'] for r in records])
        values.append((x,y,records))
    return values

def fit_head(x,y):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    transform=ColumnTransformer([('raw',StandardScaler(),list(range(5))),('semantic',Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=64,svd_solver='randomized',iterated_power=7,random_state=2026100417))]),list(range(5,1541)))])
    head=Pipeline([('features',transform),('classifier',LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,tol=1e-6,random_state=2026100417))])
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);head.fit(x,y)
    return head

def score(head,x,records):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):p=head.predict_proba(x)[:,1]
    return [dict(id=r['id'],prediction=r['prediction'],confidence=float(v)) for r,v in zip(records,p)]

def train(output,collection,head_path):
    s,out,state=start(output,'train')
    try:
        require(head_path.resolve().is_relative_to(ROOT/'runs') and not head_path.exists(),'Fresh local head required')
        (x,y,records),(cal,_,cal_records)=inputs(collection)
        head=fit_head(x,y);pred=score(head,cal,cal_records);chosen=select(base('calibration')[0],pred,s)
        head_path.parent.mkdir(parents=True,exist_ok=True)
        with head_path.open('xb') as f:pickle.dump(head,f,protocol=5)
        write(out/'selection.json',dict(**chosen,protocol_sha256=sha(SPEC),head_pickle_sha256=sha(head_path),collection_checksums_sha256=sha(collection/'checksums.json')))
        write(out/'development-predictions.json',pred)
        write(out/'fit.json',dict(n_train=len(y),input_features=1541,classifier_features=69,iterations=int(head.named_steps['classifier'].n_iter_[0]),fit_attempts=1,convergence_warning=False))
        state.update(status='complete',development_passed=chosen['eligible'],evaluation_permitted=chosen['eligible'])
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:finish(out,state)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect');c.add_argument('--asset-root',type=Path,required=True);c.add_argument('--model-path',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--collection',type=Path,required=True);t.add_argument('--head-path',type=Path,required=True)
    for x in (c,t):x.add_argument('--output',type=Path,required=True)
    a=vars(p.parse_args());f=a.pop('action');print(globals()[f](**a))
