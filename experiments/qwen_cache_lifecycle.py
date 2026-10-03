"""Real-model lifecycle fault injection and capacity/byte-budget trace."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import time
from lab.evidence import reserve_directory, sha256, source_record
from lab.qwen_prefix import QwenPrefixRuntime
from experiments.qwen_prefix_study import identity, save
from experiments.verify_qwen_prefix_fair import require
from experiments.verify_qwen_cache_lifecycle import verify


def state(rt):
    return {'resident_tokens': [list(v[0]) for v in rt.store.entries.values()], 'stats': rt.store.stats()}


def contracts(model, fingerprint, prefixes, suffix):
    rows=[]
    def fail(_): raise RuntimeError('injected clone failure, not real allocator OOM')
    for case in ('hit_clone','miss_clone','bypass_clone','builder_after_prefill','invalid_size'):
        rt=QwenPrefixRuntime(model,fingerprint,max_entries=2,max_bytes=2*64*12288)
        references=[rt.generate(p+suffix,max_new_tokens=8,stop_at_eos=False)['token_ids'] for p in prefixes[:2]]
        for p in prefixes[:2]: rt.prepare(p)
        before=state(rt)
        tokens=prefixes[0] if case=='hit_clone' else prefixes[2]
        if case=='bypass_clone': tokens=prefixes[2]*3
        builder=rt.build
        if case=='builder_after_prefill':
            def builder(t):
                rt.build(t)
                raise RuntimeError('injected builder failure after actual prefill')
        elif case=='invalid_size':
            def builder(t):
                snapshot, _=rt.build(t)
                return snapshot,-1
        try: rt.store.acquire(tokens,builder,clone=fail if 'clone' in case else rt.clone)
        except (RuntimeError,ValueError) as error: message=str(error)
        else: raise AssertionError('Expected injected failure')
        after=state(rt)
        expected={**before['stats'],'failures':before['stats']['failures']+1}
        require(before['resident_tokens']==after['resident_tokens'] and after['stats']==expected,'Failure changed cache')
        replay=[rt.generate(p+suffix,prefix=p,max_new_tokens=8,stop_at_eos=False) for p in prefixes[:2]]
        require([r['token_ids'] for r in replay]==references,'Failure polluted snapshot')
        rows.append({'case':case,'before':before,'after':after,'error':message,'reference_token_ids':references,'replay':replay})
    tiny=QwenPrefixRuntime(model,fingerprint,max_bytes=1)
    bypass=[tiny.generate(prefixes[0]+suffix,prefix=prefixes[0],max_new_tokens=8,stop_at_eos=False) for _ in range(2)]
    require(all(r['status']=='bypass' and r['cache']['entries']==0 for r in bypass),'Bypass residency')
    require(all(r['token_ids']==references[0] for r in bypass),'Bypass parity')
    before=state(rt)
    try: rt.generate(prefixes[0]+suffix,prefix=prefixes[1])
    except ValueError: pass
    else: raise AssertionError('Mismatch accepted')
    require(state(rt)==before,'Prefix rejection touched cache')
    rt.store.clear(); cleared=state(rt)
    replay=rt.generate(prefixes[0]+suffix,prefix=prefixes[0],max_new_tokens=8,stop_at_eos=False)
    require(replay['status']=='miss' and replay['token_ids']==references[0],'Clear replay')
    return {'faults':rows,'bypass':bypass,'mismatch_state_preserved':True,'cleared':cleared,'after_clear':replay}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--contracts-only',action='store_true',help='Real-model fault/clear/bypass smoke without timing traces')
    args=p.parse_args();out=reserve_directory(args.output_dir)
    spec_path=Path('configs/qwen-prefix/lifecycle.json');spec=json.loads(spec_path.read_text())
    import mlx.core as mx
    from mlx_lm import load
    require(mx.metal.is_available(),'Apple GPU required')
    files,fingerprint=identity(args.model)
    model,tok=load(str(args.model));mx.eval(model.parameters());mx.synchronize()
    config=json.loads((args.model/'config.json').read_text())
    require(config['model_type']=='qwen2' and config['quantization']['bits']==8,'Qwen2 Q8 required')
    save(out/'manifest.json',{'source':source_record(),'spec':spec,'spec_sha256':sha256(spec_path),
        'model_fingerprint':fingerprint,'model_files_sha256':files,'device':mx.metal.device_info(),
        'versions':{p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')}})
    base=tok.encode('The robot inspection log contains maintenance and sensor records. '*300,add_special_tokens=False)
    suffix=tok.encode('\nSummarize:',add_special_tokens=False)
    prefixes=[base[:63]+[n] for n in (20,21,22)]
    save(out/'contracts.json',contracts(model,fingerprint,prefixes,suffix))
    print('fault contracts passed',flush=True)
    if args.contracts_only:
        save(out/'complete.json',{'sha256':{p.name:sha256(p) for p in sorted(out.glob('*.json'))}})
        return
    records=[];works=[];rng=random.Random(spec['seed'])
    for length in spec['prefix_lengths']:
        prefixes=[base[:length-1]+[n] for n in (20,21,22)]
        works.append({'length':length,'prefixes':prefixes,'suffix':suffix})
        runtimes={mode:QwenPrefixRuntime(model,fingerprint,max_entries=2 if mode=='entries2' else 3,
                    max_bytes=length*12288*(2 if mode=='entries3_bytes2' else 3)) for mode in spec['modes']}
        for mode,rt in runtimes.items():
            for prefix in prefixes:
                for _ in range(2):
                    rt.generate(prefix+suffix,prefix=prefix,max_new_tokens=spec['generated_tokens'],stop_at_eos=False,
                                reuse_prefix=mode!='direct',segmented_snapshot=False)
            rt.store.clear()
        for round_id in range(spec['rounds']):
            modes=list(spec['modes']);rng.shuffle(modes)
            for mode in modes:
                rt=runtimes[mode];rt.store.clear();before=rt.store.stats()
                mx.synchronize();start=time.perf_counter()
                outputs=[rt.generate(prefixes[i]+suffix,prefix=prefixes[i],max_new_tokens=spec['generated_tokens'],
                         stop_at_eos=False,reuse_prefix=mode!='direct',segmented_snapshot=False) for i in spec['trace']]
                mx.synchronize();elapsed=time.perf_counter()-start
                after=rt.store.stats()
                records.append({'length':length,'round':round_id,'mode':mode,'seconds':elapsed,'outputs':outputs,
                     'max_entries':rt.store.max_entries,'max_bytes':rt.store.max_bytes,
                     'cache_delta':{k:after[k]-before[k] for k in ('hits','misses','evictions','bypasses','failures')}})
                # Release resident snapshots of inactive paths outside timing.
                rt.store.clear()
            save(out/'timings.json',records)
        print('length',length,'done',flush=True)
    save(out/'workloads.json',works)
    save(out/'summary.json',verify(out))
    save(out/'complete.json',{'sha256':{p.name:sha256(p) for p in sorted(out.glob('*.json'))}})


if __name__=='__main__':main()
