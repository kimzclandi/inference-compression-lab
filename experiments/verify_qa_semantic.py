"""Reconstruct semantic heads and diagnose projection numerics from archives."""
import argparse
import importlib
import math
from pathlib import Path
import warnings
import numpy as np
from experiments.qa_nonlinear import select
from experiments.verify_qa_coverage_gap import require
from lab.artifact_integrity import verify_hashes
from lab.quantization_diagnostics import read,sha,write,aggregates_equal

ROOT=Path(__file__).resolve().parents[1]

def audit(folder,fixed=False):
    api=importlib.import_module('experiments.qa_semantic_fixed' if fixed else 'experiments.qa_semantic')
    s=api.preflight();collection=ROOT/'results/qa-semantic-v1/collection';training=folder/'training'
    verify_hashes(training,read(training/'checksums.json'),exclude=('checksums.json',))
    require(sha(training/'protocol.json')==sha(api.SPEC),'Training protocol')
    for name in s['source_files']:require(sha(training/'source'/name)==s['sha256'][name],'Training source snapshots')
    original=read(collection/'protocol.json')
    for name in original['source_files']:require(sha(collection/'source'/name)==original['sha256'][name],'Collection source snapshots')
    selected=read(training/'selection.json');require(selected['collection_checksums_sha256']==sha(collection/'checksums.json'),'Training collection identity')
    (x,y,_),(cal,_,records)=api.inputs(collection)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');head=api.fit_head(x,y);replay=api.score(head,cal,records)
    stored=read(training/'development-predictions.json');require(len(stored)==len(replay),'Score count');maximum=0.; changed_decisions=0
    for a,b in zip(stored,replay):
        require(a['id']==b['id'] and a['prediction']==b['prediction'],'Score row identity')
        delta=abs(a['confidence']-b['confidence']);maximum=max(maximum,delta)
        require(math.isfinite(delta),'Nonfinite replay difference')
        changed_decisions+=sum((a['confidence']>=t)!=(b['confidence']>=t) for t in s['threshold_grid'])
    expected=select(api.base('calibration')[0],stored,s)
    require(aggregates_equal({k:selected[k] for k in expected},expected),'Selection arithmetic')
    require(expected['eligible'] is False and not (folder/'evaluation').exists(),'This recorded failure must not evaluate')
    state=read(training/'run.json');require(state['status']=='complete' and state['evaluation_permitted'] is False,'Training failure guard')
    # Use a non-BLAS contraction to check the final feature projection/scores.
    transform=head.named_steps['features'];raw=transform.named_transformers_['raw'];sem=transform.named_transformers_['semantic']
    scaler=sem.named_steps['scale'];pca=sem.named_steps['pca'];lr=head.named_steps['classifier']
    def independent(values):
        z=(values[:,5:]-scaler.mean_)/scaler.scale_
        hidden=np.einsum('ij,kj->ik',z-pca.mean_,pca.components_,optimize=False)
        features=np.column_stack([(values[:,:5]-raw.mean_)/raw.scale_,hidden])
        logits=np.einsum('ij,j->i',features,lr.coef_[0],optimize=False)+lr.intercept_[0]
        require(np.isfinite(features).all() and np.isfinite(logits).all(),'Non-finite final numerical state')
        return features,1/(1+np.exp(-logits))
    design,p=independent(x);_,q=independent(cal)
    error=max(abs(float(v)-r['confidence']) for v,r in zip(q,replay));require(error<=1e-6,'Independent projection/scoring mismatch')
    gradient=np.einsum('ij,i->j',design,p-y,optimize=False)/len(y)+lr.coef_[0]/(.1*len(y))
    norm=float(max(np.max(abs(gradient)),abs(np.mean(p-y))))
    reproduced=maximum<=1e-6 and changed_decisions==0
    return dict(evidence_valid=reproduced,projection_repair=fixed,archived_activation_rebuild=reproduced,probability_tolerance=1e-6,
        max_probability_difference=maximum,independent_score_difference=error,final_inference_path_gradient_max=norm,
        runtime_warning_messages=sorted(set(str(w.message) for w in captured)),runtime_warning_count=len(captured),
        all_grid_decisions_identical=changed_decisions==0,changed_grid_decisions=changed_decisions,development_passed=False,evaluation_run=False,
        scope='Archived activation/score reproduction with fixed probability tolerance and exact grid decisions. Original fit_transform/transform discrepancy is retained; numerical warnings are disclosed, not suppressed into a clean-training claim. No arbitrary regenerated-activation portability or model quality acceptance.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/qa-semantic-v1');p.add_argument('--fixed',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.fixed);write(a.output,result);print(result)
    if not result['evidence_valid']: raise SystemExit('FAILED: archived head score/decision reproduction; tolerance unchanged')
