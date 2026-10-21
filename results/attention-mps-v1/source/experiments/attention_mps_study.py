"""Fixed MPS API-level attention comparison; all historical studies stay frozen."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import numpy as np
from lab.attention_reference import attention, prefix_mask
from experiments.attention_backend_study import save, digest

SPEC = Path('configs/attention-mps-v1.json')
SOURCES = [str(SPEC), 'experiments/attention_mps_study.py',
           'experiments/attention_backend_study.py', 'lab/attention_reference.py']


def input_arrays(spec, index, batch, length, keys):
    rng = np.random.default_rng(spec['seed'] + index)
    return [rng.normal(size=(batch, spec['heads'], n, spec['head_dim'])).astype('float16')
            for n in (length, keys, keys)]


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def make_arms(torch, inputs):
    q, k, v = inputs; length, keys = q.shape[-2], k.shape[-2]
    if length != 1 and length != keys:
        raise ValueError('This performance protocol covers prefill and single-token decode only')
    mask = None if length == 1 else torch.as_tensor(prefix_mask(length, keys), device=q.device)
    def explicit(fp32):
        # Input conversion is part of this implementation's cost.
        qt, kt, vt = [x.float() for x in inputs] if fp32 else inputs
        scores = qt @ kt.transpose(-1, -2) / q.shape[-1] ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return (scores.softmax(dim=-1) @ vt).to(q.dtype)
    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=length == keys)
    return dict(eager_fp16=lambda: explicit(False), eager_fp32=lambda: explicit(True), sdpa_auto=sdpa)


def summarize(records, spec):
    expected = {(b,l,s,r,a) for b in spec['batches'] for l,s in spec['shapes']
                for r in range(spec['rounds']) for a in spec['arms']}
    cells = {}
    for row in records:
        key = tuple(row[k] for k in ('batch','query_length','key_length','round','arm'))
        samples = row['seconds']
        if key not in expected or key in cells:
            raise ValueError('Duplicate or unexpected timing cell')
        if len(samples) != spec['repeats'] or any(not np.isfinite(v) or v <= 0 for v in samples):
            raise ValueError('Invalid timing samples')
        cells[key] = statistics.median(samples)
    if set(cells) != expected:
        raise ValueError('Incomplete timing matrix')
    rows = []
    for b in spec['batches']:
        for l,s in spec['shapes']:
            med = {a: statistics.median(cells[b,l,s,r,a] for r in range(spec['rounds']))
                   for a in spec['arms']}
            controls = {}
            for a in ('eager_fp16','eager_fp32'):
                ratio = med[a]/med['sdpa_auto']
                faster = sum(cells[b,l,s,r,'sdpa_auto'] < cells[b,l,s,r,a] for r in range(spec['rounds']))
                controls[a] = dict(speedup=ratio, faster_rounds=faster,
                                   passed=ratio >= spec['minimum_speedup'] and faster >= spec['minimum_faster_rounds'])
            rows.append(dict(batch=b, query_length=l, key_length=s, median_seconds=med,
                             controls=controls, speed_gate=all(v['passed'] for v in controls.values())))
    return dict(shapes=rows, cuda_measured=False, custom_kernel=False, model_speedup_claim=False,
                timing_scope=spec['timing_scope'])


def run(output):
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Commit a clean protocol first')
    spec = json.loads(SPEC.read_text())
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(name).read_bytes())
    info = dict(status='running', start_utc=datetime.now(timezone.utc).isoformat(),
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                platform=platform.platform(), source_sha256={p:digest(p) for p in SOURCES},
                mps_performance_measured=False, cuda_performance_measured=False)
    save(output/'run.json', info)
    try:
        # Validate before importing torch so a requested CPU fallback cannot slip in.
        for key in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL'):
            if os.environ.get(key, '0') != '0':
                raise RuntimeError(key + ' must be disabled')
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('An actual Apple MPS device is required')
        if torch.__version__ != spec['torch_version'] or np.__version__ != spec['numpy_version']:
            raise RuntimeError('Pinned torch/numpy required')
        torch.set_num_threads(1)
        hardware = subprocess.check_output(['/usr/sbin/system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True)
        safe_hardware = [line.strip() for line in hardware.splitlines()
                         if any(label in line for label in ('Chip:','Memory:','Chipset Model:','Total Number of Cores:','Metal Support:'))]
        info.update(torch_version=torch.__version__, numpy_version=np.__version__, hardware=safe_hardware,
                    fallback_enabled=False, fast_math=False, prefer_metal=False,
                    actual_sdpa_kernel='Not identified; framework automatic dispatch, no FlashAttention attribution')
        save(output/'run.json', info)
        cases = [(b,l,s) for b in spec['batches'] for l,s in spec['shapes']]
        indices = list(range(len(cases))); random.Random(spec['seed']).shuffle(indices)
        order = random.Random(spec['seed'] + 1)
        records, checks, witnesses, inputs_info = [], [], [], []
        with torch.inference_mode():
            for index in indices:
                b,l,s = cases[index]
                arrays = input_arrays(spec,index,b,l,s)
                reference = attention(*arrays)
                inputs = [torch.from_numpy(x).to('mps') for x in arrays]
                inputs_info.append(dict(index=index,batch=b,query_length=l,key_length=s,
                                        sha256=[array_hash(x) for x in arrays]))
                arms = make_arms(torch, inputs)
                for name, fn in arms.items():
                    y = fn(); assert y.device.type == 'mps'
                    actual = y.float().cpu().numpy().astype('float64'); del y
                    error = np.abs(actual-reference)
                    norm = error/(spec['atol']+spec['rtol']*np.abs(reference))
                    passed = bool(np.isfinite(actual).all() and np.all(norm <= 1))
                    checks.append(dict(index=index,arm=name,passed=passed,
                                       max_abs_error=float(error.max()),normalized_error=float(norm.max()),
                                       output_sha256=array_hash(actual)))
                    # Preserve entire output rows at deterministic coordinates for
                    # independent CI replay; full-tensor comparison remains a run receipt.
                    for position in sorted({0,l//2,l-1}):
                        witnesses.append(dict(index=index,arm=name,position=position,
                                              actual=actual[0,0,position].tolist()))
                    save(output/'correctness.json', checks)
                    save(output/'witnesses.json', witnesses)
                    if not passed:
                        raise ValueError('Correctness failure, no timing permitted')
                del reference, actual, error, norm
                for fn in arms.values():
                    for _ in range(spec['warmups']): y=fn(); del y
                    torch.mps.synchronize()
                for r in range(spec['rounds']):
                    names=list(arms);order.shuffle(names)
                    for name in names:
                        times=[]
                        for _ in range(spec['repeats']):
                            torch.mps.synchronize(); start=time.perf_counter()
                            y=arms[name](); torch.mps.synchronize()
                            times.append(time.perf_counter()-start); del y
                        records.append(dict(batch=b,query_length=l,key_length=s,round=r,arm=name,seconds=times))
                        save(output/'timings.json',records)
                save(output/'inputs.json',inputs_info)
                print(f'completed B={b}, L={l}, S={s}',flush=True)
                del inputs, arms, fn, arrays
        save(output/'summary.json',summarize(records,spec))
        info.update(status='complete',mps_performance_measured=True)
    except Exception as exc:
        info.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        info['artifact_sha256']={p.name:digest(p) for p in output.glob('*.json') if p.name!='run.json'}
        info['end_utc']=datetime.now(timezone.utc).isoformat()
        save(output/'run.json',info)


def verify(output):
    info=json.loads((output/'run.json').read_text())
    if info['status']!='complete' or not info['mps_performance_measured'] or info['cuda_performance_measured']:
        raise ValueError('Study did not complete on MPS')
    if set(info['source_sha256']) != set(SOURCES):raise ValueError('Missing archived source')
    for name,value in info['source_sha256'].items():
        if digest(output/'source'/name)!=value:raise ValueError('Source hash mismatch')
    required={'inputs.json','correctness.json','witnesses.json','timings.json','summary.json'}
    if set(info['artifact_sha256'])!=required:raise ValueError('Missing artifact')
    for name,value in info['artifact_sha256'].items():
        if digest(output/name)!=value:raise ValueError('Artifact hash mismatch')
    spec=json.loads((output/'source'/SPEC).read_text())
    summary=summarize(json.loads((output/'timings.json').read_text()),spec)
    if summary!=json.loads((output/'summary.json').read_text()):raise ValueError('Summary mismatch')
    cases=[(b,l,s) for b in spec['batches'] for l,s in spec['shapes']]
    checks=json.loads((output/'correctness.json').read_text())
    expected={(i,a) for i in range(len(cases)) for a in spec['arms']}
    if len(checks)!=len(expected) or {(c['index'],c['arm']) for c in checks}!=expected:
        raise ValueError('Incomplete correctness matrix')
    if any(not c['passed'] or not np.isfinite(c['normalized_error']) or not 0 <= c['normalized_error'] <= 1 for c in checks):
        raise ValueError('Correctness failed')
    input_rows=json.loads((output/'inputs.json').read_text())
    if len(input_rows)!=len(cases) or {r['index'] for r in input_rows}!=set(range(len(cases))):
        raise ValueError('Input matrix mismatch')
    witnesses=json.loads((output/'witnesses.json').read_text())
    expected_witnesses={(i,a,p) for i,(_,l,_) in enumerate(cases)
                        for a in spec['arms'] for p in {0,l//2,l-1}}
    if len(witnesses)!=len(expected_witnesses) or {(w['index'],w['arm'],w['position']) for w in witnesses}!=expected_witnesses:
        raise ValueError('Witness matrix mismatch')
    for index,(b,l,s) in enumerate(cases):
        arrays=input_arrays(spec,index,b,l,s)
        row=next(r for r in input_rows if r['index']==index)
        if [array_hash(x) for x in arrays]!=row['sha256']:raise ValueError('Input reconstruction mismatch')
        q,k,v=[x[0,0].astype('float64') for x in arrays]
        for w in (w for w in witnesses if w['index']==index):
            p=w['position'];end=s-l+p+1
            scores=q[p]@k[:end].T/spec['head_dim']**0.5
            weights=np.exp(scores-scores.max());weights/=weights.sum()
            reference=weights@v[:end]
            actual=np.asarray(w['actual'])
            if actual.shape!=reference.shape:raise ValueError('Witness shape mismatch')
            np.testing.assert_allclose(actual,reference,atol=spec['atol'],rtol=spec['rtol'])
    print(json.dumps(dict(evidence_valid=True,shape_cells=len(cases),correctness_cells=len(checks),
                         independently_replayed_output_rows=len(witnesses),cuda_measured=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output) if args.mode=='run' else verify(args.output)
