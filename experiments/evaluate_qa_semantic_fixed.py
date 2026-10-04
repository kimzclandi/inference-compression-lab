"""Evaluate only an eligible committed selection, once on the reserved cohort."""
import argparse
from pathlib import Path
import pickle
import shutil
import numpy as np
from experiments.qa_semantic_fixed import ROOT,SPEC,preflight,start,finish,score
from lab.quantization_diagnostics import read,rows,sha,write
from lab.artifact_integrity import verify_hashes
from experiments.verify_qa_coverage_gap import require,metrics,passes

PLAN=ROOT/'configs/qa-semantic-fixed/evaluation.json'

def evaluate(output,training,head_path,asset_root,model_path,selection_sha256):
    spec=preflight();plan=read(PLAN)
    require(plan['training_protocol_sha256']==sha(SPEC),'Evaluation protocol binding')
    require(plan['evaluator_sha256']==sha(Path(__file__)),'Evaluation source binding')
    verify_hashes(training,read(training/'checksums.json'),exclude=('checksums.json',))
    selected=read(training/'selection.json')
    require(selected['eligible'] is True and sha(training/'selection.json')==selection_sha256,'Eligible frozen selection required')
    require(selected['protocol_sha256']==sha(SPEC) and sha(head_path)==selected['head_pickle_sha256'],'Selection/head identity')
    with head_path.open('rb') as f:head=pickle.load(f)
    spec,out,state=start(output,'evaluate')
    shutil.copy2(PLAN,out/'evaluation-plan.json');shutil.copy2(Path(__file__),out/'evaluator.py')
    state['selection_sha256']=selection_sha256
    try:
        from lab.qa_specialist_runtime import ExtractiveRuntime
        from lab.qa_semantic import SemanticRuntime
        from experiments.qa_expanded import collect_into
        import gzip,json
        runtime=ExtractiveRuntime(asset_root,'int8',spec['asset_manifest_sha256'])
        data,features=collect_into(runtime,'evaluation',out,state,spec)
        semantic=SemanticRuntime(asset_root,spec['asset_manifest_sha256'],model_path)
        raw=[]
        for path in sorted(out.glob('predictions-*.jsonl.gz')):raw.extend(json.loads(s) for s in gzip.decompress(path.read_bytes()).decode().splitlines())
        vectors=[];maximum=0.
        for p in raw:
            value,diff=semantic.vector(p);vectors.append(value);maximum=max(maximum,diff)
        matrix=np.asarray(vectors,dtype=np.float32);np.savez_compressed(out/'embeddings.npz',features=matrix)
        x=np.column_stack([[r['features'] for r in features],matrix]);pred=score(head,x,features)
        actual=metrics(data,pred,selected['threshold']);write(out/'predictions.json',pred);write(out/'selection.json',selected)
        # A frozen historical policy is a descriptive paired reference, not a
        # newly validated comparator or a revised primary success criterion.
        from lab.qa_risk_calibration import fit,predict_probability
        old=read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
        baseline=fit([r['features'] for r in old],[r['target'] for r in old])
        control=[dict(id=r['id'],prediction=r['prediction'],confidence=predict_probability(baseline,r['features'])) for r in features]
        write(out/'original-frozen-predictions.json',control)
        write(out/'summary.json',dict(threshold=selected['threshold'],metrics=actual,quality_gate_passed=passes(actual,spec['quality_constraints']),
            original_frozen_at_0_7=metrics(data,control,.7),paired_improvement_confirmed=False,
            max_logit_difference=maximum,default_policy_changed=False,scope=spec['scope']))
        state.update(status='complete',quality_gate_passed=passes(actual,spec['quality_constraints']),encoder_trained=False)
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:finish(out,state)
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','training','head-path','asset-root','model-path'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection-sha256',required=True);print(evaluate(**vars(p.parse_args())))
