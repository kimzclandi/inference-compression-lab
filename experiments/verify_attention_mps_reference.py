"""Audit NumPy matmul reference warnings with independent non-BLAS einsum.

No GPU execution or timing; never writes the frozen performance study.
"""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from lab.attention_reference import attention
from experiments.attention_backend_study import save, digest
from experiments.attention_mps_study import SPEC, input_arrays, array_hash


def nonblas_reference(q,k,v,chunk=16):
    q,k,v=[x.astype('float64') for x in (q,k,v)]
    length,keys=q.shape[-2],k.shape[-2]
    result=np.empty((*q.shape[:-1],v.shape[-1]),dtype='float64')
    for start in range(0,length,chunk):
        end=min(length,start+chunk)
        scores=np.einsum('bhld,bhsd->bhls',q[:,:,start:end],k,optimize=False)/q.shape[-1]**0.5
        mask=np.arange(keys)[None,:] <= keys-length+np.arange(start,end)[:,None]
        scores=np.where(mask,scores,-np.inf)
        weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
        weights/=weights.sum(axis=-1,keepdims=True)
        result[:,:,start:end]=np.einsum('bhls,bhsd->bhld',weights,v,optimize=False)
    return result


def audit(root,full):
    spec=json.loads((root/'source'/SPEC).read_text())
    receipt=json.loads((root/'run.json').read_text())
    for name in ('inputs.json','witnesses.json'):
        if digest(root/name)!=receipt['artifact_sha256'][name]:raise ValueError('Source artifact changed')
    input_rows=json.loads((root/'inputs.json').read_text())
    witnesses=json.loads((root/'witnesses.json').read_text())
    cases=[(b,l,s) for b in spec['batches'] for l,s in spec['shapes']]
    rows=[]
    for index,(b,l,s) in enumerate(cases):
        arrays=input_arrays(spec,index,b,l,s)
        source=next(r for r in input_rows if r['index']==index)
        if [array_hash(x) for x in arrays]!=source['sha256']:raise ValueError('Input identity mismatch')
        item=dict(index=index,batch=b,query_length=l,key_length=s)
        if full:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                original=attention(*arrays)
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                independent=nonblas_reference(*arrays)
            if not np.isfinite(original).all() or not np.isfinite(independent).all():
                raise ValueError('Nonfinite reference')
            np.testing.assert_allclose(original,independent,atol=1e-12,rtol=1e-12)
            item.update(full_reference_max_abs_error=float(np.abs(original-independent).max()),
                        matmul_warnings=sorted({str(w.message) for w in caught}),
                        original_reference_sha256=array_hash(original))
        q,k,v=[x[0,0].astype('float64') for x in arrays]
        errors=[]
        for w in (w for w in witnesses if w['index']==index):
            p=w['position'];end=s-l+p+1
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                scores=np.einsum('d,sd->s',q[p],k[:end],optimize=False)/q.shape[-1]**0.5
                weights=np.exp(scores-scores.max());weights/=weights.sum()
                ref=np.einsum('s,sd->d',weights,v[:end],optimize=False)
            actual=np.asarray(w['actual'])
            np.testing.assert_allclose(actual,ref,atol=spec['atol'],rtol=spec['rtol'])
            errors.append(float(np.abs(actual-ref).max()))
        item.update(output_rows_checked=len(errors),max_witness_abs_error=max(errors))
        rows.append(item)
    return dict(evidence_valid=True,full_reference_checked=full,cases=rows,
                gpu_rerun=False,performance_rerun=False,
                cause='Floating-point warnings observed with NumPy matmul; underlying library cause not established',
                input_receipt_sha256=digest(root/'run.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/attention-mps-v1'))
    parser.add_argument('--full',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():raise ValueError('Audit output must be new')
    result=audit(args.root,args.full)
    if args.output:save(args.output,result)
    print(json.dumps(result))
