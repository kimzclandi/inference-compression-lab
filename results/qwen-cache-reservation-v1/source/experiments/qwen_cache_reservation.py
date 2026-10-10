"""Frozen whole-model native/reserved MLX KV allocation comparison (Mac only)."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import subprocess
import time
import traceback

from lab.mlx_cache_reservation import reserve_fresh_caches

SPEC=Path('configs/qwen-cache-reservation-v1.json')
SOURCES=[str(SPEC),'lab/mlx_cache_reservation.py', 'experiments/qwen_cache_reservation.py']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(SPEC.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    info=dict(status='running',protocol_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={},spec=spec,artifacts={})
    for name in SOURCES:
        target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());info['source_sha256'][name]=digest(target)
    save(args.output/'run.json',info)
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx_lm.models import cache as cache_module
        from mlx_lm.models.cache import KVCache,make_prompt_cache
        for pkg,key in [('mlx','mlx'),('mlx-lm','mlx_lm')]:
            assert importlib.metadata.version(pkg)==spec[key],(pkg,'version changed')
        assert digest(inspect.getfile(cache_module))==spec['upstream_cache_sha256'],'cache source changed'
        assert all(digest(args.model/n)==h for n,h in spec['model_files_sha256'].items()),'model changed'
        info['device']=str(mx.default_device());assert mx.default_device()==mx.gpu,'GPU required'
        info['system']=subprocess.check_output(['sw_vers'],text=True)
        info['hardware']=subprocess.check_output(['system_profiler','SPHardwareDataType','SPDisplaysDataType'],text=True)
        # Keep identifying serial/UUID fields out of the published receipt.
        info['hardware']='\n'.join(s for s in info['hardware'].splitlines() if not any(x in s for x in ['Serial Number','UUID','Provisioning UDID']))
        model,tokenizer=load(str(args.model));mx.eval(model.parameters());mx.synchronize()
        prompt=tokenizer.encode(spec['prompt_text']*200)
        assert len(prompt)>=max(spec['prefix_lengths'])
        info['prompt_token_ids']={str(p):prompt[:p] for p in spec['prefix_lengths']}
        samples=[];checks=[]
        def request(p,arm,audit=False):
            x=mx.array([prompt[:p]]);mx.eval(x);mx.synchronize()
            start=time.perf_counter()
            caches=make_prompt_cache(model)
            if arm=='reserved':reserve_fresh_caches(caches,p+spec['generated_tokens'],KVCache)
            tokens=[];logits=[];capacities=[];ttft=None
            for step in range(spec['generated_tokens']):
                y=model(x,cache=caches)[:,-1,:]
                token=mx.argmax(y,axis=-1)
                mx.eval(token);mx.synchronize()
                if step==0:ttft=time.perf_counter()-start
                # Host token extraction is included for both arms.
                tokens.append(int(token.item()))
                x=token[:,None]
                if audit:
                    logits.append(np.array(y).copy())
                    capacities.append([int(c.keys.shape[2]) for c in caches])
            total=time.perf_counter()-start
            row=dict(prefix=p,arm=arm,ttft_s=ttft,total_s=total,tokens=tokens)
            if audit:
                states=[(np.array(c.state[0]).copy(),np.array(c.state[1]).copy()) for c in caches]
                row.update(capacities=capacities,offsets=[c.offset for c in caches],allocated_kv_bytes=sum(c.keys.nbytes+c.values.nbytes for c in caches),active_kv_bytes=sum(c.state[0].nbytes+c.state[1].nbytes for c in caches),kv_dtype=str(caches[0].keys.dtype))
                return row,np.stack(logits),states
            return row
        # Full-model correctness before any performance trials.
        for p in spec['prefix_lengths']:
            left,ll,ls=request(p,'native',True);right,rl,rs=request(p,'reserved',True)
            finite=bool(np.isfinite(ll).all() and np.isfinite(rl).all())
            same_tokens=left['tokens']==right['tokens']
            max_abs=float(np.max(np.abs(ll.astype('float64')-rl.astype('float64'))))
            kv_equal=all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(ls,rs))
            logits_ok=bool(np.allclose(ll,rl,atol=spec['atol'],rtol=spec['rtol']))
            for row in [left,right]:row.pop('ttft_s');row.pop('total_s')
            check=dict(prefix=p,finite=finite,tokens_equal=same_tokens,logits_allclose=logits_ok,logits_max_abs=max_abs,kv_exact=kv_equal,layers=len(ls),native=left,reserved=right)
            checks.append(check);save(args.output/'correctness.json',checks)
            assert finite and same_tokens and logits_ok and kv_equal,'whole-model correctness failed'
        rng=random.Random(spec['seed'])
        for p in spec['prefix_lengths']:
            for arm in spec['arms']:
                for _ in range(spec['warmups']):request(p,arm)
            for round_id in range(spec['rounds']):
                for repeat in range(spec['repeats']):
                    arms=list(spec['arms']);rng.shuffle(arms)
                    for order,arm in enumerate(arms):
                        row=request(p,arm);row.update(round=round_id,repeat=repeat,order=order)
                        assert row['tokens']==checks[spec['prefix_lengths'].index(p)]['native']['tokens'],'timed token drift'
                        samples.append(row);save(args.output/'samples.json',samples)
            print('completed prefix',p,flush=True)
        info['status']='complete'
    except BaseException as exc:
        info['status']='failed';info['error']=str(type(exc).__name__)+': '+str(exc)
        traceback.print_exc();raise
    finally:
        for name in ['correctness.json','samples.json']:
            if (args.output/name).exists():info['artifacts'][name]=digest(args.output/name)
        save(args.output/'run.json',info)

if __name__=='__main__':main()
