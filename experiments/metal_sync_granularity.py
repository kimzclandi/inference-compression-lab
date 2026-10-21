"""Keep a dependent chain fixed while varying only evaluation/synchronization cadence."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

from experiments.metal_mechanism_diagnostic import save, sha

SPEC = Path('configs/metal-sync-granularity-v1.json')
SOURCES = ['experiments/metal_sync_granularity.py',
           'experiments/metal_mechanism_diagnostic.py', str(SPEC),
           'lab/kernels/residual_rmsnorm.metal', 'lab/kernels/residual_rmsnorm_masked.metal']


def validate(spec):
    for key in ('chain_length','width','warmups','rounds','repeats'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError('Positive integer protocol fields required')
    if any(type(n) is not int or n < 1 or spec['chain_length'] % n for n in spec['sync_every']):
        raise ValueError('Cadence must divide chain length')
    if len(spec['sync_every']) != len(set(spec['sync_every'])):
        raise ValueError('Duplicate cadence')


def summarize(records, spec):
    validate(spec)
    expected = {(r,c,n,a) for r in spec['rows'] for c in spec['sync_every']
                for n in range(spec['rounds']) for a in spec['arms']}
    cells = {}
    for row in records:
        key = (row['rows'],row['sync_every'],row['round'],row['arm'])
        if key not in expected or key in cells:
            raise ValueError('Unexpected or duplicate timing cell')
        values = row['chain_seconds']
        if len(values) != spec['repeats'] or any(type(v) not in (float,int) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Invalid timing samples')
        cells[key] = statistics.median(values)
    if set(cells) != expected:
        raise ValueError('Incomplete timing matrix')
    results = {}
    for r in spec['rows']:
        medians = {str(c): {a: statistics.median(cells[r,c,n,a] for n in range(spec['rounds']))
                            for a in spec['arms']} for c in spec['sync_every']}
        sensitivity = {}
        for arm in spec['arms']:
            ratio = medians['1'][arm] / medians[str(spec['chain_length'])][arm]
            faster = sum(cells[r,spec['chain_length'],n,arm] < cells[r,1,n,arm] for n in range(spec['rounds']))
            sensitivity[arm] = dict(sync1_over_sync32=ratio, faster_rounds=faster,
                meets_sensitivity_rule=ratio >= spec['minimum_ratio'] and faster >= spec['minimum_faster_rounds'])
        results[str(r)] = dict(median_chain_seconds=medians, sensitivity=sensitivity,
            original_over_masked={str(c):medians[str(c)]['original']/medians[str(c)]['masked'] for c in spec['sync_every']},
            compiled_over_masked={str(c):medians[str(c)]['compiled']/medians[str(c)]['masked'] for c in spec['sync_every']})
    return dict(scope=spec['scope'], timing_scope=spec['timing_scope'], shapes=results,
        kernel_promoted=False, excludes_pure_gpu_time=True,
        causal_limit='Cadence changes graph scheduling, live outputs and GPU submission as well as host synchronization; cannot subtract timings to identify isolated host cost.')


def backend(spec):
    import mlx.core as mx
    if importlib.metadata.version('mlx') != spec['mlx_version'] or not mx.metal.is_available():
        raise RuntimeError('Pinned MLX and Metal GPU required')
    mx.set_default_device(mx.gpu)
    return mx


def make_arms(mx, spec):
    eps = mx.array([spec['epsilon']], dtype=mx.float32); mx.eval(eps)
    def native(x,residual,weight):
        h=x+residual
        return h,mx.fast.rms_norm(h,weight,spec['epsilon'])
    arms={'compiled':mx.compile(native,shapeless=False)}
    def make(name,suffix):
        kernel=mx.fast.metal_kernel(name='sync_diagnostic_'+name,
            input_names=['x','residual','weight','epsilon'],output_names=['h','y'],
            source=Path(f'lab/kernels/residual_rmsnorm{suffix}.metal').read_text(),
            header='#include <metal_simdgroup>\n',ensure_row_contiguous=True)
        def call(x,residual,weight):
            threads=32*((x.shape[-1]+127)//128)
            return kernel(inputs=[x,residual,weight,eps],template=[('T',x.dtype)],
                grid=(x.size//x.shape[-1]*threads,1,1),threadgroup=(threads,1,1),
                output_shapes=[x.shape,x.shape],output_dtypes=[x.dtype,x.dtype],stream=mx.gpu)
        return mx.compile(call,shapeless=False)
    arms['original']=make('original','');arms['masked']=make('masked','_masked')
    return arms


def inputs(mx,spec,rows):
    import numpy as np
    rng=np.random.default_rng(spec['seed']+rows)
    x=mx.array(rng.normal(size=(rows,spec['width'])).astype(spec['dtype']))
    residual=mx.full(x.shape,0.01,dtype=getattr(mx,spec['dtype']))
    weight=mx.ones((spec['width'],),dtype=getattr(mx,spec['dtype']))
    mx.eval(x,residual,weight);mx.synchronize()
    return x,residual,weight


def chain(mx,fn,args,spec,cadence):
    x,residual,weight=args;pending=[]
    for step in range(spec['chain_length']):
        h,x=fn(x,residual,weight);pending.extend([h,x])
        if (step+1)%cadence==0:
            mx.eval(*pending);mx.synchronize();pending.clear()
    return h,x


def run(output):
    spec=json.loads(SPEC.read_text());validate(spec)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Commit a clean protocol before execution')
    output.mkdir(parents=True,exist_ok=False)
    sources={}
    for path in SOURCES:
        dest=output/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(Path(path).read_bytes());sources[path]=sha(path)
    save(output/'protocol.json',spec)
    manifest=dict(status='running',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,platform=platform.platform(),pid=os.getpid())
    save(output/'run.json',manifest)
    try:
        import numpy as np
        mx=backend(spec);arms=make_arms(mx,spec)
        manifest.update(device=mx.metal.device_info(),versions={p:importlib.metadata.version(p) for p in ['mlx','numpy']})
        rng=random.Random(spec['seed']);records=[];checks=[]
        for rows in spec['rows']:
            args=inputs(mx,spec,rows);reference=None;original=None
            for cadence in spec['sync_every']:
                for arm in spec['arms']:
                    values=tuple(np.asarray(a).copy() for a in chain(mx,arms[arm],args,spec,cadence))
                    if reference is None:reference=values
                    for a,b in zip(values,reference):
                        np.testing.assert_allclose(a,b,atol=spec['atol'],rtol=spec['rtol'])
                    if arm=='original' and original is None:original=values
                    if arm in ('original','masked') and not all(np.array_equal(a,b) for a,b in zip(values,original)):
                        raise ValueError('Custom/cadence output mismatch')
                    checks.append(dict(rows=rows,sync_every=cadence,arm=arm,tolerance_pass=True,
                        output_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in values]))
                    for _ in range(spec['warmups']):chain(mx,arms[arm],args,spec,cadence)
            for n in range(spec['rounds']):
                order=[(c,a) for c in spec['sync_every'] for a in spec['arms']];rng.shuffle(order)
                for cadence,arm in order:
                    samples=[]
                    for _ in range(spec['repeats']):
                        mx.synchronize();start=time.perf_counter()
                        pair=chain(mx,arms[arm],args,spec,cadence)
                        samples.append(time.perf_counter()-start)
                        del pair
                    records.append(dict(rows=rows,sync_every=cadence,arm=arm,round=n,chain_seconds=samples,
                                        cell_order=order))
                    save(output/'timings.json',records)
            print(rows,'complete',flush=True)
        save(output/'correctness.json',checks);save(output/'summary.json',summarize(records,spec));manifest['status']='complete'
    except Exception as exc:manifest.update(status='failed',error=repr(exc));raise
    finally:
        save(output/'run.json',manifest)
        save(output/'checksums.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='checksums.json'})


def verify(output):
    actual={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'}
    if actual!=json.loads((output/'checksums.json').read_text()):raise ValueError('Evidence changed')
    spec=json.loads((output/'protocol.json').read_text());run_record=json.loads((output/'run.json').read_text())
    if spec!=json.loads(SPEC.read_text()) or run_record['status']!='complete':raise ValueError('Incomplete or changed protocol')
    for path,digest in run_record['source_sha256'].items():
        if sha(output/'source'/path)!=digest:raise ValueError('Source identity differs')
    checks=json.loads((output/'correctness.json').read_text())
    expected={(r,c,a) for r in spec['rows'] for c in spec['sync_every'] for a in spec['arms']}
    if len(checks)!=len(expected) or {(r['rows'],r['sync_every'],r['arm']) for r in checks}!=expected or not all(r['tolerance_pass'] is True for r in checks):raise ValueError('Correctness coverage differs')
    for rows in spec['rows']:
        hashes=[r['output_sha256'] for r in checks if r['rows']==rows and r['arm'] in ('original','masked')]
        if any(h!=hashes[0] for h in hashes):raise ValueError('Custom output hashes differ')
    result=summarize(json.loads((output/'timings.json').read_text()),spec)
    if result!=json.loads((output/'summary.json').read_text()):raise ValueError('Summary differs')
    return result


def profile(arm):
    spec=json.loads(SPEC.read_text());cfg=spec['profile']
    print(json.dumps(dict(stage='started',pid=os.getpid(),arm=arm)),flush=True)
    time.sleep(cfg['startup_delay_seconds'])
    mx=backend(spec);fn=make_arms(mx,spec)[arm];args=inputs(mx,spec,cfg['rows'])
    for _ in range(cfg['warmups']):chain(mx,fn,args,spec,cfg['sync_every'])
    print(json.dumps(dict(stage='warm',pid=os.getpid(),arm=arm)),flush=True)
    for _ in range(cfg['batches']):pair=chain(mx,fn,args,spec,cfg['sync_every'])
    print(json.dumps(dict(stage='complete',pid=os.getpid(),arm=arm,profiled_calls=cfg['batches']*spec['chain_length'],
        checksum=float(mx.sum(pair[1].astype(mx.float32)).item()),scope=cfg['scope'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','verify','profile'])
    p.add_argument('--output',type=Path);p.add_argument('--arm',choices=['compiled','original','masked']);a=p.parse_args()
    if a.action=='profile':
        if not a.arm:p.error('--arm required')
        profile(a.arm)
    else:
        if a.output is None:p.error('--output required')
        if a.action=='run':run(a.output)
        else:print(json.dumps(verify(a.output),indent=2))
