"""Frozen Q8 QKV projection study; full tensors stay in an explicit private root."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

SPEC = Path('configs/qkv-projection-v1.json')
SOURCES = [str(SPEC), 'lab/qkv_projection.py', 'experiments/qkv_projection.py',
           'experiments/verify_qkv_projection.py', 'third_party/MLX-MIT.txt']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def execute(model_path, root):
    if not __debug__:
        raise RuntimeError('optimized Python unsupported')
    spec = json.loads(SPEC.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('clean committed source required before GPU execution')
    root.mkdir(parents=True, exist_ok=False)
    run = dict(schema_version=1, status='running', phase='identity', spec=spec,
               protocol_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               source_sha256={})
    correctness, micro, samples, profiles = [], [], [], []
    manifest = {'files': {}}
    for name in SOURCES:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        run['source_sha256'][name] = sha(target)
    save(root/'run.json', run)
    try:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_lm
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.base import scaled_dot_product_attention
        from lab.qkv_projection import prepare_qkv, qkv_decode_adapter
        from experiments.verify_qkv_projection import summarize
        site = Path(mlx_lm.__file__).parent.parent
        for name, expected in spec['upstream_sha256'].items():
            assert sha(site/name) == expected, 'upstream source identity changed'
        for pkg, key in [('mlx', 'mlx'), ('mlx-lm', 'mlx_lm'), ('numpy', 'numpy')]:
            assert importlib.metadata.version(pkg) == spec[key], 'runtime version changed'
        assert {p.name: sha(p) for p in model_path.iterdir() if p.is_file()} == spec['model_files_sha256'], 'model identity changed'
        assert mx.default_device() == mx.gpu and mx.metal.is_available(), 'Metal required'
        run['device'] = mx.metal.device_info()
        run['upstream_sha256'] = dict(spec['upstream_sha256'])
        run['physical_memory_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))
        def checkpoint():
            run['peak_mlx_observed'] = max(run.get('peak_mlx_observed', 0), int(mx.get_peak_memory()))
            size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            run['disk_bytes_at_checkpoint'] = size
            save(root/'run.json', run)
            if run['peak_mlx_observed'] > spec['budget']['peak_mlx_bytes'] or size > spec['budget']['disk_bytes']:
                raise RuntimeError('memory or disk budget exceeded')
        def archive(name, arrays, kind, case, arm=None):
            np.savez_compressed(root/name, **arrays)
            row = dict(sha256=sha(root/name), kind=kind, case=case)
            if arm is not None:
                row['arm'] = arm
            manifest['files'][name] = row
            save(root/'manifest.json', manifest)
            checkpoint()
        def check(kind, case, arm, actual, expected):
            assert actual.keys() == expected.keys()
            finite = all(bool(np.isfinite(v).all() and np.isfinite(expected[k]).all()) for k, v in actual.items())
            bitwise = all(v.dtype == expected[k].dtype and v.shape == expected[k].shape and v.tobytes() == expected[k].tobytes() for k, v in actual.items())
            maximum = max(float(np.max(np.abs(v.astype(np.float64)-expected[k].astype(np.float64)))) for k, v in actual.items()) if finite else None
            row = dict(kind=kind, case=case, arm=arm, finite=finite, bitwise=bitwise,
                       arrays_checked=len(actual), max_abs=maximum)
            correctness.append(row)
            save(root/'correctness.json', correctness)
            assert finite and bitwise, 'bitwise correctness gate failed; saved arrays retained'
        mx.reset_peak_memory()
        model, tokenizer = load(str(model_path))
        mx.eval(model.parameters()); mx.synchronize()
        run['loaded_model_active_bytes'] = int(mx.get_active_memory())
        text_ids = tokenizer.encode(spec['prompt_text']*600)
        assert len(text_ids) >= max(spec['prompts'])
        ids = {n: mx.array(text_ids[:n])[None] for n in spec['prompts']}
        mx.eval(*ids.values())
        run['prompt_token_ids'] = {str(n): text_ids[:n] for n in spec['prompts']}
        activations = {}
        originals = list(model.model.layers)
        active_prompt = None
        profile_step = 0
        @contextmanager
        def instrument():
            class Block(nn.Module):
                def __init__(self, upstream, index):
                    super().__init__(); self.upstream = upstream; self.index = index
                def __call__(self, x, mask=None, cache=None):
                    if x.shape[1] != 1:
                        return self.upstream(x, mask, cache)
                    a = self.upstream.self_attn
                    def stage(name, fn):
                        start = time.perf_counter_ns(); out = fn(); mx.eval(out)
                        profiles.append(dict(prompt=active_prompt, step=profile_step, layer=self.index,
                                             stage=name, latency_ms=(time.perf_counter_ns()-start)/1e6))
                        return out
                    h = stage('input_norm', lambda: self.upstream.input_layernorm(x))
                    q, k, v = stage('qkv', lambda: (a.q_proj(h), a.k_proj(h), a.v_proj(h)))
                    if profile_step == 1:
                        activations[(active_prompt, self.index)] = np.array(h).copy()
                    def rope_cache():
                        qq=q.reshape(1,1,14,64).transpose(0,2,1,3)
                        kk=k.reshape(1,1,2,64).transpose(0,2,1,3)
                        vv=v.reshape(1,1,2,64).transpose(0,2,1,3)
                        qq=a.rope(qq,offset=cache.offset); kk=a.rope(kk,offset=cache.offset)
                        kk,vv=cache.update_and_fetch(kk,vv)
                        return qq,kk,vv
                    qq,kk,vv = stage('rope_cache', rope_cache)
                    out = stage('sdpa', lambda: scaled_dot_product_attention(qq,kk,vv,cache=cache,scale=a.scale,mask=mask))
                    r = stage('o_projection', lambda: a.o_proj(out.transpose(0,2,1,3).reshape(1,1,896)))
                    h = stage('attention_residual', lambda: x+r)
                    normalized = stage('post_norm', lambda: self.upstream.post_attention_layernorm(h))
                    out = stage('mlp_residual', lambda: h+self.upstream.mlp(normalized))
                    return out
            try:
                model.model.layers = [Block(v, i) for i, v in enumerate(originals)]
                yield
            finally:
                model.model.layers = originals
        prepared = None
        def request(n, arm, audit=False):
            nonlocal active_prompt, profile_step
            active_prompt=n; profile_step=0
            context = instrument() if arm == 'profile' else (nullcontext() if arm == 'native' else qkv_decode_adapter(model,prepared,arm,counts={}))
            checkpoint(); mx.reset_peak_memory(); mx.synchronize()
            arrays={}; tokens=[]; logits_arrays=[]
            started=time.perf_counter_ns()
            with context:
                caches=make_prompt_cache(model)
                current=ids[n]
                decode_ms=0.0
                for step in range(spec['decode_steps']+1):
                    profile_step=step
                    tick=time.perf_counter_ns()
                    logits=model(current,cache=caches)[:,-1,:]
                    tok=mx.argmax(logits,axis=-1)
                    mx.eval(logits,tok)
                    token=int(tok.item()); elapsed=(time.perf_counter_ns()-tick)/1e6
                    tokens.append(token)
                    if step == 0:
                        ttft_ms=(time.perf_counter_ns()-started)/1e6
                    else:
                        decode_ms+=elapsed
                    if audit:
                        logits_arrays.append(np.array(logits[0]).copy())
                    current=tok.reshape(1,1)
                mx.synchronize()
            row=dict(case=f'p{n}',arm=arm,latency_ms=(time.perf_counter_ns()-started)/1e6,
                     ttft_ms=ttft_ms,decode_ms=decode_ms,peak_mlx_bytes=int(mx.get_peak_memory()),
                     tokens=tokens,cache_offsets=[int(c.offset) for c in caches])
            assert row['cache_offsets']==[n+spec['decode_steps']]*24
            if audit:
                arrays={'tokens':np.array(tokens,dtype=np.int64),'logits':np.stack(logits_arrays)}
                for i,c in enumerate(caches):
                    for name,v in zip(('K','V'),c.state):
                        arrays[f'l{i:02d}_{name}']=np.array(v).copy()
                archive(f'model-p{n}-{arm}.npz',arrays,'model',f'p{n}',arm)
            checkpoint()
            return row,arrays
        # Profile is a separate perturbed measurement, never a time attribution of native execution.
        run['phase']='native_profile'; save(root/'run.json',run)
        expected={}; profile_receipts=[]
        for n in spec['prompts']:
            warm_native,_=request(n,'native')
            native, values=request(n,'native',True); expected[n]=values
            assert warm_native['tokens']==values['tokens'].tolist()
            before=len(profiles)
            warm_profile,_=request(n,'profile')
            assert warm_profile['tokens']==values['tokens'].tolist()
            del profiles[before:]
            perturbed, observed=request(n,'profile',True)
            check('profile',f'p{n}','profile',observed,values)
            warm_native['purpose']='warmup'; warm_profile['purpose']='warmup'
            native['purpose']='audit_with_numpy_copies'; perturbed['purpose']='audit_with_numpy_copies'
            profile_receipts.extend([warm_native,native,warm_profile,perturbed])
            save(root/'profile-samples.json',profiles)
            save(root/'profile-requests.json',profile_receipts)
        profile_summary={}
        for n in spec['prompts']:
            rows=[r for r in profiles if r['prompt']==n]
            totals={stage:math.fsum(r['latency_ms'] for r in rows if r['stage']==stage) for stage in spec['profile_stages']}
            profile_summary[str(n)]={'stage_totals_ms':totals,'qkv_fraction_instrumented':totals['qkv']/math.fsum(totals.values())}
        run['profile_summary']=profile_summary
        assert len(profiles)==2*16*24*8 and len(activations)==48
        if profile_summary['4096']['qkv_fraction_instrumented'] < spec['profile_min_qkv_fraction']:
            run['status']='profile_gate_failed'
            return
        # Pack once only after profiling supports spending the bounded candidate budget.
        run['phase']='packing'; save(root/'run.json',run)
        mx.synchronize()
        run['active_before_packing_bytes']=int(mx.get_active_memory())
        original_qkv={(i,name,part):hashlib.sha256(np.array(getattr(getattr(a.self_attn,name),part)).tobytes()).hexdigest()
            for i,a in enumerate(originals) for name in ('q_proj','k_proj','v_proj') for part in ('weight','scales','biases','bias')}
        tick=time.perf_counter_ns(); prepared=prepare_qkv(model); mx.synchronize()
        run['packing_ms']=(time.perf_counter_ns()-tick)/1e6
        run['packed_tensor_bytes']=prepared.packed_nbytes
        run['active_after_packing_bytes']=int(mx.get_active_memory())
        fixtures={}
        run['phase']='operator_correctness'
        for n in spec['prompts']:
            for layer in range(24):
                case=f'p{n}-l{layer:02d}'
                x=mx.array(activations[(n,layer)]); mx.eval(x)
                fixtures[(n,layer)]=x
                arrays={'x':activations[(n,layer)]}
                for arm in ('native','compiled_three','packed'):
                    outputs=prepared.projections[layer].project(x,arm)
                    mx.eval(*outputs)
                    arrays.update({f'{arm}_{key}':np.array(v).copy() for key,v in zip(('q','k','v'),outputs)})
                    archive(f'operator-{case}.npz',arrays,'operator',case)
                    if arm!='native':
                        check('operator',case,arm,{k:arrays[f'{arm}_{k}'] for k in ('q','k','v')},
                              {k:arrays[f'native_{k}'] for k in ('q','k','v')})
        run['phase']='model_correctness'
        for n in spec['prompts']:
            for arm in ('native_adapter','compiled_three','packed'):
                _,arrays=request(n,arm,True)
                check('model',f'p{n}',arm,arrays,expected[n])
        run['phase']='benchmark'; save(root/'run.json',run)
        order_rng=random.Random(spec['seed'])
        def micro_trial(n,arm):
            mx.synchronize(); tick=time.perf_counter_ns()
            for layer in range(24):
                mx.async_eval(*prepared.projections[layer].project(fixtures[(n,layer)],arm))
            mx.synchronize()
            return (time.perf_counter_ns()-tick)/1e6
        for n in spec['prompts']:
            for arm in ('native','compiled_three','packed'):
                for _ in range(spec['micro_warmups']): micro_trial(n,arm)
            for rnd in range(spec['micro_rounds']):
                for repeat in range(spec['micro_repeats']):
                    arms=['native','compiled_three','packed']; order_rng.shuffle(arms)
                    for order,arm in enumerate(arms):
                        micro.append(dict(case=f'p{n}',arm=arm,round=rnd,repeat=repeat,order=order,
                                          calls=24,latency_ms=micro_trial(n,arm)))
                    save(root/'micro-samples.json',micro); checkpoint()
        for n in spec['prompts']:
            for arm in ('native','native_adapter','compiled_three','packed'):
                for _ in range(spec['model_warmups']):
                    row,_=request(n,arm); assert row['tokens']==expected[n]['tokens'].tolist()
            for rnd in range(spec['model_rounds']):
                for repeat in range(spec['model_repeats']):
                    arms=['native','native_adapter','compiled_three','packed']; order_rng.shuffle(arms)
                    for order,arm in enumerate(arms):
                        row,_=request(n,arm)
                        row.update(round=rnd,repeat=repeat,order=order); samples.append(row)
                        save(root/'model-samples.json',samples)
                        assert row['tokens']==expected[n]['tokens'].tolist(), 'timed token drift'
                    checkpoint()
        # Recheck immutable model files and runtime source at completion.
        assert {p.name:sha(p) for p in model_path.iterdir() if p.is_file()}==spec['model_files_sha256']
        assert all(sha(site/n)==v for n,v in spec['upstream_sha256'].items())
        assert all(sha(n)==v for n,v in run['source_sha256'].items())
        final_qkv={(i,name,part):hashlib.sha256(np.array(getattr(getattr(a.self_attn,name),part)).tobytes()).hexdigest()
            for i,a in enumerate(originals) for name in ('q_proj','k_proj','v_proj') for part in ('weight','scales','biases','bias')}
        assert final_qkv==original_qkv, 'live original QKV parameters changed'
        run['original_qkv_arrays_unchanged']=len(original_qkv)
        run['status']='complete'
    except BaseException as error:
        run['status']='failed'; run['exception_type']=type(error).__name__
        raise
    finally:
        for name, value in [('correctness.json',correctness),('micro-samples.json',micro),
                            ('model-samples.json',samples),('profile-samples.json',profiles),('manifest.json',manifest)]:
            save(root/name,value)
        run['private_manifest_sha256']=sha(root/'manifest.json')
        try:
            from experiments.verify_qkv_projection import summarize
            save(root/'summary.json',summarize(spec,correctness,micro,samples))
        finally:
            save(root/'run.json',run)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        execute(args.model,args.output); return
    if args.output.exists(): raise FileExistsError('new output directory required')
    spec=json.loads(SPEC.read_text())
    try:
        result=subprocess.run([sys.executable,'-m','experiments.qkv_projection','--worker',
                               '--model',str(args.model),'--output',str(args.output)],timeout=spec['budget']['hard_seconds'])
    except subprocess.TimeoutExpired:
        args.output.mkdir(parents=True,exist_ok=True)
        save(args.output/'parent-failure.json',{'status':'timeout','hard_seconds':spec['budget']['hard_seconds']})
        raise
    if result.returncode: raise SystemExit(result.returncode)


if __name__=='__main__': main()
