"""Preallocated versus cat K/V append: a separately frozen MPS study."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import numpy as np
from lab.append_only_kv import AppendOnlyKV
from experiments.attention_backend_study import digest,save
from experiments.attention_mps_study import array_hash

SPEC=Path('configs/kv-append-mps-v2.json')
SOURCES=[str(SPEC),'lab/append_only_kv.py','experiments/kv_append_mps.py',
         'experiments/attention_backend_study.py','experiments/attention_mps_study.py','lab/attention_reference.py']


def arrays(spec,index,b,p):
    rng=np.random.default_rng(spec['seed']+index)
    return [rng.normal(size=(b,spec['heads'],n,spec['head_dim'])).astype('float16')
            for n in (spec['steps'],p+spec['steps'],p+spec['steps'])]


def reference(q,k,v):
    q,k,v=[x.astype('float64') for x in (q,k,v)]
    scores=np.einsum('bhld,bhsd->bhls',q,k,optimize=False)/q.shape[-1]**0.5
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
    return np.einsum('bhls,bhsd->bhld',weights,v,optimize=False)


def logical_writes(spec,b,p):
    unit=2*b*spec['heads']*spec['head_dim']*2  # K and V, FP16 bytes.
    t=spec['steps']
    return dict(cat_append_output_bytes=unit*(t*p+t*(t+1)//2),
                preallocated_append_write_bytes=unit*t,initial_prefix_write_bytes=unit*p,
                preallocated_storage_bytes=unit*(p+t),not_measured_dram=True)


def decode_attention(q,k,v):
    """Same explicit FP32 attention for both cache arms after v1 SDPA failure."""
    scores=q.float()@k.float().transpose(-1,-2)/q.shape[-1]**0.5
    return (scores.softmax(dim=-1)@v.float()).to(q.dtype)


def summarize(rows,spec):
    expected={(b,p,s,r,a) for b in spec['batches'] for p in spec['prefixes']
              for s in spec['scopes'] for r in range(spec['rounds']) for a in spec['arms']}
    cells={}
    for row in rows:
        key=tuple(row[k] for k in ('batch','prefix','scope','round','arm'))
        if key not in expected or key in cells:raise ValueError('Duplicate or unexpected cell')
        for field in ('initialization_seconds','chain_seconds'):
            if len(row[field])!=spec['repeats'] or any(not np.isfinite(x) or x<=0 for x in row[field]):
                raise ValueError('Invalid samples')
        cells[key]=row
    if set(cells)!=expected:raise ValueError('Incomplete timing matrix')
    result=[]
    for b in spec['batches']:
        for p in spec['prefixes']:
            scopes={}
            for s in spec['scopes']:
                med={a:statistics.median(statistics.median(cells[b,p,s,r,a]['chain_seconds'])
                                        for r in range(spec['rounds'])) for a in spec['arms']}
                init={a:statistics.median(statistics.median(cells[b,p,s,r,a]['initialization_seconds'])
                                         for r in range(spec['rounds'])) for a in spec['arms']}
                total={a:statistics.median(statistics.median(x+y for x,y in zip(
                    cells[b,p,s,r,a]['initialization_seconds'],cells[b,p,s,r,a]['chain_seconds']))
                    for r in range(spec['rounds'])) for a in spec['arms']}
                faster=sum(statistics.median(cells[b,p,s,r,'preallocated']['chain_seconds'])<
                           statistics.median(cells[b,p,s,r,'cat']['chain_seconds']) for r in range(spec['rounds']))
                ratio=med['cat']/med['preallocated']
                scopes[s]=dict(median_chain_seconds=med,median_initialization_seconds=init,
                               median_combined_seconds=total,speedup=ratio,faster_rounds=faster,
                               passed=ratio>=spec['minimum_speedup'] and faster>=spec['minimum_faster_rounds'])
            result.append(dict(batch=b,prefix=p,scopes=scopes,logical_writes=logical_writes(spec,b,p),
                               primary_speed_gate=scopes['append_attention']['passed']))
    return dict(cases=result,model_speedup=False,cuda_measured=False,production_cache=False)


def run(root):
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit clean protocol first')
    spec=json.loads(SPEC.read_text());root.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    info=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              start_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
              source_sha256={p:digest(p) for p in SOURCES})
    save(root/'run.json',info)
    try:
        for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL'):
            if os.environ.get(key,'0')!='0':raise ValueError(key+' must be disabled')
        import torch
        if not torch.backends.mps.is_available() or torch.__version__!=spec['torch_version'] or np.__version__!=spec['numpy_version']:
            raise ValueError('Pinned torch/numpy and actual MPS required')
        torch.set_num_threads(1)
        hw=subprocess.check_output(['/usr/sbin/system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True)
        info.update(hardware=[x.strip() for x in hw.splitlines() if any(k in x for k in ('Chip:','Memory:','Chipset Model:','Total Number of Cores:'))],
                    torch=torch.__version__,numpy=np.__version__,fallback=False)
        rows=[];checks=[];witnesses=[];input_rows=[]
        cases=[(b,p) for b in spec['batches'] for p in spec['prefixes']]
        indices=list(range(len(cases)));order=random.Random(spec['seed']);order.shuffle(indices)
        with torch.inference_mode():
            for index in indices:
                b,p=cases[index];q,k,v=arrays(spec,index,b,p)
                input_rows.append(dict(index=index,sha256=[array_hash(x) for x in (q,k,v)]))
                save(root/'inputs.json',input_rows)
                tq,tk,tv=[torch.from_numpy(x).to('mps') for x in (q,k,v)]
                steps=[(tq[:,:,i:i+1],tk[:,:,p+i:p+i+1],tv[:,:,p+i:p+i+1]) for i in range(spec['steps'])]
                def new(arm):
                    return AppendOnlyKV(tk[:,:,:p],tv[:,:,:p],p+spec['steps']) if arm=='preallocated' else (tk[:,:,:p].clone(),tv[:,:,:p].clone())
                def append(cache,arm,nk,nv):
                    if arm=='preallocated':return cache,cache.append(nk,nv)
                    # Assign only after both allocations succeed.
                    pair=(torch.cat((cache[0],nk),dim=-2),torch.cat((cache[1],nv),dim=-2))
                    return pair,pair
                for arm in spec['arms']:
                    cache=new(arm);error=0.;norm=0.;keys_equal=True
                    for i,(nq,nk,nv) in enumerate(steps):
                        cache,pair=append(cache,arm,nk,nv)
                        ak,av=[x.cpu().numpy() for x in pair]
                        keys_equal &= np.array_equal(ak,k[:,:,:p+i+1]) and np.array_equal(av,v[:,:,:p+i+1])
                        actual=decode_attention(nq,*pair).float().cpu().numpy()
                        ref=reference(q[:,:,i:i+1],k[:,:,:p+i+1],v[:,:,:p+i+1])
                        e=np.abs(actual-ref);ratio=e/(spec['atol']+spec['rtol']*np.abs(ref))
                        if not np.isfinite(ratio).all() or not np.all(ratio<=1) or not keys_equal:
                            save(root/'failure.json',dict(index=index,arm=arm,step=i,kv_exact=bool(keys_equal),
                                all_finite=bool(np.isfinite(ratio).all()),
                                normalized_error=float(ratio.max()) if np.isfinite(ratio).all() else None))
                            raise ValueError('K/V or Attention correctness failure')
                        error=max(error,float(e.max()));norm=max(norm,float(ratio.max()))
                        if i in (0,spec['steps']//2,spec['steps']-1):witnesses.append(dict(index=index,arm=arm,step=i,actual=actual[0,0,0].tolist()))
                    checks.append(dict(index=index,arm=arm,steps=spec['steps'],all_kv_exact=bool(keys_equal),max_abs_error=error,normalized_error=norm))
                    del cache,pair
                save(root/'correctness.json',checks);save(root/'witnesses.json',witnesses);save(root/'inputs.json',input_rows)
                def timed(arm,scope):
                    torch.mps.synchronize();start=time.perf_counter();cache=new(arm);torch.mps.synchronize();init=time.perf_counter()-start
                    start=time.perf_counter()
                    for nq,nk,nv in steps:
                        cache,pair=append(cache,arm,nk,nv)
                        if scope=='append_attention':
                            y=decode_attention(nq,*pair)
                        torch.mps.synchronize()
                        if scope=='append_attention':del y
                    elapsed=time.perf_counter()-start
                    return init,elapsed
                for scope in spec['scopes']:
                    for arm in spec['arms']:
                        for _ in range(spec['warmup_chains']):timed(arm,scope)
                for r in range(spec['rounds']):
                    tasks=[(a,s) for a in spec['arms'] for s in spec['scopes']];order.shuffle(tasks)
                    for arm,scope in tasks:
                        samples=[timed(arm,scope) for _ in range(spec['repeats'])]
                        rows.append(dict(batch=b,prefix=p,arm=arm,scope=scope,round=r,
                                         initialization_seconds=[x for x,y in samples],chain_seconds=[y for x,y in samples]))
                        save(root/'timings.json',rows)
                print(f'completed batch={b} prefix={p}',flush=True)
        save(root/'summary.json',summarize(rows,spec));info['status']='complete'
    except Exception as exc:
        info.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        info['artifact_sha256']={p.name:digest(p) for p in root.glob('*.json') if p.name!='run.json'}
        info['end_utc']=datetime.now(timezone.utc).isoformat();save(root/'run.json',info)


def verify(root):
    info=json.loads((root/'run.json').read_text())
    if info['status']!='complete':raise ValueError('Incomplete study')
    if set(info['source_sha256'])!=set(SOURCES):raise ValueError('Source matrix mismatch')
    for name,value in info['source_sha256'].items():
        if digest(root/'source'/name)!=value:raise ValueError('Source mismatch')
    if set(info['artifact_sha256'])!={'inputs.json','correctness.json','witnesses.json','timings.json','summary.json'}:raise ValueError('Missing artifact')
    for name,value in info['artifact_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Artifact hash mismatch')
    spec=json.loads((root/'source'/SPEC).read_text());summary=summarize(json.loads((root/'timings.json').read_text()),spec)
    if summary!=json.loads((root/'summary.json').read_text()):raise ValueError('Summary mismatch')
    cases=[(b,p) for b in spec['batches'] for p in spec['prefixes']]
    checks=json.loads((root/'correctness.json').read_text());expected={(i,a) for i in range(len(cases)) for a in spec['arms']}
    if len(checks)!=len(expected) or {(c['index'],c['arm']) for c in checks}!=expected:raise ValueError('Correctness matrix mismatch')
    if any(not c['all_kv_exact'] or c['steps']!=spec['steps'] or not np.isfinite(c['normalized_error']) or not 0<=c['normalized_error']<=1 for c in checks):raise ValueError('Correctness failed')
    witnesses=json.loads((root/'witnesses.json').read_text());input_rows=json.loads((root/'inputs.json').read_text())
    wanted={(i,a,s) for i,a in expected for s in (0,spec['steps']//2,spec['steps']-1)}
    if len(witnesses)!=len(wanted) or {(w['index'],w['arm'],w['step']) for w in witnesses}!=wanted:raise ValueError('Witness matrix mismatch')
    if len(input_rows)!=len(cases) or {r['index'] for r in input_rows}!=set(range(len(cases))):raise ValueError('Input matrix mismatch')
    for index,(b,p) in enumerate(cases):
        q,k,v=arrays(spec,index,b,p)
        if [array_hash(x) for x in (q,k,v)]!=next(r['sha256'] for r in input_rows if r['index']==index):raise ValueError('Input identity mismatch')
        for w in (w for w in witnesses if w['index']==index):
            i=w['step'];ref=reference(q[:1,:1,i:i+1],k[:1,:1,:p+i+1],v[:1,:1,:p+i+1])[0,0,0]
            actual=np.asarray(w['actual'])
            if actual.shape!=ref.shape:raise ValueError('Witness shape mismatch')
            np.testing.assert_allclose(actual,ref,atol=spec['atol'],rtol=spec['rtol'])
    print(json.dumps(dict(evidence_valid=True,full_output_comparisons=len(checks)*spec['steps'],witness_rows=len(witnesses),summary=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','verify']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.output) if a.mode=='run' else verify(a.output)
