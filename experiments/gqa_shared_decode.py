"""One bounded, frozen shared-GQA Metal study with a hard parent timeout."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from lab.gqa_shared_decode import attention, qwen_decode_adapter
from lab.gqa_reference import dense_reference
from lab.gqa_summary import summarize

SPEC = Path('configs/gqa-shared-decode-v1.json')
SOURCES = [str(SPEC), 'lab/gqa_shared_decode.py', 'lab/gqa_reference.py', 'lab/gqa_summary.py',
           'lab/kernels/gqa_shared_partial.metal', 'lab/kernels/gqa_shared_merge.metal',
           'experiments/gqa_shared_decode.py', 'third_party/MLX-MIT.txt']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def execute(model_path, out):
    if not __debug__:raise RuntimeError('optimized Python mode unsupported')
    spec = json.loads(SPEC.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True):
        raise RuntimeError('commit a clean protocol/implementation before GPU execution')
    out.mkdir(parents=True, exist_ok=False)
    run = dict(status='running', phase='setup', protocol_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), spec=spec, source_sha256={}, artifacts={})
    for name in SOURCES:
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        run['source_sha256'][name] = sha(target)
    save(out/'run.json', run)
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.models.cache import make_prompt_cache
        import mlx_lm
        root = Path(mlx_lm.__file__).parent.parent
        for name, expected in spec['upstream_sha256'].items():
            assert sha(root/name) == expected, 'upstream identity changed'
        for pkg, expected in [('mlx',spec['mlx']),('mlx-lm',spec['mlx_lm']),('numpy',spec['numpy'])]:
            assert importlib.metadata.version(pkg) == expected, 'version changed'
        assert {p.name:sha(p) for p in model_path.iterdir() if p.is_file()} == spec['model_files_sha256'], 'model identity changed'
        assert mx.default_device() == mx.gpu and mx.metal.is_available(), 'Metal GPU required'
        run['device'] = mx.metal.device_info()
        run['hardware'] = {'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                           'physical_memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))}
        def budget():
            peak = int(mx.get_peak_memory())
            size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            run['peak_mlx_bytes'] = max(peak,run.get('peak_mlx_bytes',0))
            run['disk_bytes_at_last_check'] = size
            assert run['peak_mlx_bytes'] <= spec['budget']['peak_mlx_soft_bytes'], 'memory budget'
            assert size <= spec['budget']['disk_bytes'], 'disk budget'
        mx.reset_peak_memory()
        rng = np.random.default_rng(spec['seed'])
        fixtures, operator_checks = {}, []
        run['phase']='operator_correctness';save(out/'run.json',run)
        for n in spec['lengths']:
            for family in spec['families']:
                gain = 16 if family == 'large_scores' else 1
                q = (rng.normal(size=(1,14,1,64))*gain).astype('float16')
                capacity = ((n+255)//256)*256+256
                kb = np.full((1,2,capacity,64),60000,dtype='float16')
                vb = np.full_like(kb,-60000)
                kb[:,:,:n] = (rng.normal(size=(1,2,n,64))*gain).astype('float16')
                vb[:,:,:n] = rng.normal(size=(1,2,n,64)).astype('float16')
                qd,kd,vd = mx.array(q),mx.array(kb),mx.array(vb)
                mx.eval(qd,kd,vd)
                k,v=kd[:,:,:n,:],vd[:,:,:n,:]
                reference=dense_reference(q,kb[:,:,:n],vb[:,:,:n])
                outputs={}
                for mode in spec['operator_modes']:
                    value=np.array(attention(qd,k,v,mode))
                    outputs[mode]=value
                    finite=bool(np.isfinite(value).all())
                    check=dict(length=n,family=family,mode=mode,finite=finite,
                               allclose=bool(np.allclose(value,reference,atol=spec['operator_atol'],rtol=spec['operator_rtol'])),
                               max_abs=float(np.max(np.abs(value.astype('float64')-reference))) if finite else None)
                    operator_checks.append(check)
                    save(out/'operator-correctness.json',operator_checks)
                    # Archive before asserting so a numerical failure is inspectable.
                    np.savez_compressed(out/f'operator-{n}-{family}.npz',q=q,k_storage=kb,v_storage=vb,
                                        reference=reference,**outputs)
                    assert finite and check['allclose'], 'operator correctness failed'
                immutable=bool(np.array_equal(np.array(qd),q) and np.array_equal(np.array(kd),kb) and np.array_equal(np.array(vd),vb))
                save(out/f'operator-{n}-{family}-input-check.json',dict(unchanged=immutable,capacity=capacity,visible=n,sentinel_k=60000,sentinel_v=-60000))
                assert immutable, 'input or sentinel modified'
                if family=='normal': fixtures[n]=(qd,k,v)
                budget()
        # Whole-model correctness precedes every timing arm.
        run['phase']='model_correctness';save(out/'run.json',run)
        model,tokenizer=load(str(model_path));mx.eval(model.parameters());mx.synchronize()
        all_ids=tokenizer.encode(spec['prompt_text']*600)
        assert len(all_ids)>=max(spec['prompt_lengths'])
        run['prompt_token_ids']={str(n):all_ids[:n] for n in spec['prompt_lengths']}
        save(out/'run.json',run)
        request_attempts=[]
        def request(n,mode,audit=False):
            budget();mx.reset_peak_memory()
            counts={};tokens=[];stamps=[];scores=[];caches=[];iterator=None
            row=dict(prompt=n,mode=mode,audit=audit,status='running')
            start=time.perf_counter()
            try:
                caches=make_prompt_cache(model)
                with qwen_decode_adapter(model,mode,counts):
                    iterator=generate_step(mx.array(all_ids[:n]),model,max_tokens=spec['delivered_tokens'],
                                           prefill_step_size=spec['prefill_step_size'],prompt_cache=caches)
                    try:
                        for token,logprobs in iterator:
                            tokens.append(int(token));stamps.append(time.perf_counter())
                            if audit:scores.append(np.array(logprobs).copy())
                    finally:
                        iterator.close()
                        mx.synchronize(generation_stream)
                        mx.synchronize()
                row['status']='complete'
            except BaseException as error:
                row.update(status='failed',error_type=type(error).__name__)
                raise
            finally:
                elapsed=time.perf_counter()-start
                row.update(tokens=tokens,total_s=elapsed,token_times_s=[s-start for s in stamps],
                    ttft_s=stamps[0]-start if stamps else None,
                    tpot_s=(stamps[-1]-stamps[0])/(len(tokens)-1) if len(tokens)>1 else None,
                    delivered_tokens=len(tokens),routing_counts=counts,
                    cache_offsets=[int(c.offset) for c in caches],peak_mlx_bytes=int(mx.get_peak_memory()))
                request_attempts.append(dict(row));save(out/'request-attempts.json',request_attempts)
                budget()
            assert len(tokens)==spec['delivered_tokens'] and row['cache_offsets']==[n+spec['delivered_tokens']]*24
            if mode=='shared_compiled':
                assert counts==dict(prefill=24*math.ceil(n/spec['prefill_step_size']),decode=24*spec['delivered_tokens']),counts
            if audit:
                states=[(np.array(c.state[0]).copy(),np.array(c.state[1]).copy()) for c in caches]
                return row,np.stack(scores),states
            return row
        model_checks=[];expected={}
        for n in spec['prompt_lengths']:
            native,ns,nkv=request(n,'native',True)
            candidate,cs,ckv=request(n,'shared_compiled',True)
            layer_checks=[]
            for layer,(left,right) in enumerate(zip(nkv,ckv)):
                for kind,a,b in zip(('K','V'),left,right):
                    layer_checks.append(dict(layer=layer,kind=kind,shape=list(a.shape),
                        finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                        allclose=bool(np.allclose(a,b,atol=spec['model_kv_atol'],rtol=spec['model_kv_rtol'])),
                        max_abs=float(np.max(np.abs(a.astype('float64')-b.astype('float64')))) if np.isfinite(a).all() and np.isfinite(b).all() else None,
                        native_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
            check=dict(prompt=n,tokens_equal=native['tokens']==candidate['tokens'],
                       logprobs_finite=bool(np.isfinite(ns).all() and np.isfinite(cs).all()),
                       logprobs_allclose=bool(np.allclose(ns,cs,atol=spec['model_logprobs_atol'],rtol=spec['model_logprobs_rtol'])),
                       logprobs_max_abs=float(np.max(np.abs(ns.astype('float64')-cs.astype('float64')))) if np.isfinite(ns).all() and np.isfinite(cs).all() else None,
                       native=native,candidate=candidate,kv=layer_checks)
            for row in (native,candidate):
                for key in ('total_s','ttft_s','tpot_s','token_times_s'):row.pop(key)
            model_checks.append(check);save(out/'model-correctness.json',model_checks)
            np.savez_compressed(out/f'model-logprobs-{n}.npz',native=ns,candidate=cs)
            assert check['tokens_equal'] and check['logprobs_finite'] and check['logprobs_allclose'] and all(x['finite'] and x['allclose'] for x in layer_checks), 'model correctness failed'
            expected[n]=native['tokens']
            del nkv,ckv,ns,cs
            budget()
        # All correctness gates passed; no candidate is selected from timings.
        run['phase']='benchmark';save(out/'run.json',run)
        order_rng=random.Random(spec['seed']);operator_samples=[];model_samples=[]
        for n in spec['lengths']:
            q,k,v=fixtures[n]
            for mode in spec['operator_modes']:
                for _ in range(spec['operator_warmups']):mx.eval(attention(q,k,v,mode))
            mx.synchronize()
            for rnd in range(spec['rounds']):
                for repeat in range(spec['operator_repeats']):
                    modes=list(spec['operator_modes']);order_rng.shuffle(modes)
                    for order,mode in enumerate(modes):
                        mx.synchronize();start=time.perf_counter()
                        for _ in range(spec['calls_per_block']):mx.async_eval(attention(q,k,v,mode))
                        mx.synchronize();elapsed=time.perf_counter()-start
                        operator_samples.append(dict(length=n,round=rnd,repeat=repeat,order=order,mode=mode,
                            calls=spec['calls_per_block'],block_s=elapsed,per_call_s=elapsed/spec['calls_per_block']))
                        save(out/'operator-samples.json',operator_samples)
                budget()
            print('operator complete',n,flush=True)
        for n in spec['prompt_lengths']:
            for mode in spec['model_modes']:
                for _ in range(spec['model_warmups']):assert request(n,mode)['tokens']==expected[n]
            for rnd in range(spec['rounds']):
                for repeat in range(spec['model_repeats']):
                    modes=list(spec['model_modes']);order_rng.shuffle(modes)
                    for order,mode in enumerate(modes):
                        row=request(n,mode)
                        row.update(round=rnd,repeat=repeat,order=order)
                        model_samples.append(row)
                        save(out/'model-samples.json',model_samples);budget()
                        assert row['tokens']==expected[n], 'timed token drift'
            print('model complete',n,flush=True)
        run['summary']=summarize(spec,operator_samples,model_samples)
        run['status']='complete'
    except BaseException as error:
        run['status']='failed';run['error_type']=type(error).__name__
        # Keep traceback/private paths in external terminal log only.
        raise
    finally:
        run['artifacts']={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
                          if p.is_file() and p.name!='run.json'}
        save(out/'run.json',run)


def main():
    if not __debug__:raise RuntimeError('optimized Python mode unsupported')
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        execute(args.model,args.output)
        return
    # This supervisor does not import MLX; killing the worker terminates its GPU process.
    if args.output.exists():raise FileExistsError('output must be new')
    spec=json.loads(SPEC.read_text())
    try:
        result=subprocess.run([sys.executable,'-m','experiments.gqa_shared_decode',
            '--model',str(args.model),'--output',str(args.output),'--worker'],
            timeout=spec['budget']['hard_subprocess_seconds'])
    except subprocess.TimeoutExpired:
        if args.output.exists():save(args.output/'timeout.json',{'status':'hard_timeout','seconds':spec['budget']['hard_subprocess_seconds']})
        raise SystemExit(124)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
