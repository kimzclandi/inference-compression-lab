"""One frozen diagnostic run; intentionally contains no performance measurement."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from lab.gqa_diagnostic import array_sha,error_metrics,diagnostic_hooks
from lab.gqa_reference import dense_reference
from lab.gqa_shared_decode import qwen_decode_adapter

SPEC=Path('configs/gqa-shared-diagnostic-v1.json')
SOURCES=[str(SPEC),'experiments/gqa_shared_diagnostic.py','lab/gqa_diagnostic.py',
         'lab/kernels/gqa_diagnostic_layout.metal']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def execute(model_path,root):
    if not __debug__:raise RuntimeError('optimized mode unsupported')
    spec=json.loads(SPEC.read_text())
    if subprocess.check_output(['git','status','--porcelain'],text=True):raise RuntimeError('clean committed protocol required')
    root.mkdir(parents=True,exist_ok=False)
    run=dict(status='running',phase='identity',protocol_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),spec=spec,artifacts={},source_sha256={})
    for name in SOURCES:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());run['source_sha256'][name]=sha(target)
    save(root/'run.json',run)
    try:
        import mlx.core as mx
        import mlx_lm
        import mlx_lm.models.qwen2 as qwen2
        import lab.gqa_shared_decode as frozen
        import numpy as np
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import generation_stream
        site=Path(mlx_lm.__file__).parent.parent
        for name,value in spec['protected_source_sha256'].items():assert sha(name)==value,'frozen source changed'
        for name,value in spec['upstream_sha256'].items():assert sha(site/name)==value,'upstream changed'
        for name,key in [('mlx','mlx'),('mlx-lm','mlx_lm'),('numpy','numpy')]:assert importlib.metadata.version(name)==spec[key]
        assert {p.name:sha(p) for p in model_path.iterdir() if p.is_file()}==spec['model_files_sha256']
        assert mx.default_device()==mx.gpu and mx.metal.is_available()
        run['device']=mx.metal.device_info();mx.reset_peak_memory()
        def budget():
            run['peak_mlx_bytes']=max(run.get('peak_mlx_bytes',0),int(mx.get_peak_memory()))
            run['disk_bytes_at_checkpoint']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            save(root/'run.json',run)
            assert run['peak_mlx_bytes']<=spec['budget']['peak_mlx_soft_bytes'],'memory budget'
            assert run['disk_bytes_at_checkpoint']<=spec['budget']['disk_bytes'],'disk budget'
        layout_kernel=mx.fast.metal_kernel(name='icl_gqa_diagnostic_layout_v1',input_names=['q','k','v'],output_names=['meta'],
            source=Path('lab/kernels/gqa_diagnostic_layout.metal').read_text(),ensure_row_contiguous=False)
        layout_calls=0
        def layout(q,k,v):
            nonlocal layout_calls
            meta=np.array(layout_kernel(inputs=[q,k,v],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(24,)],output_dtypes=[mx.int32])[0])
            layout_calls+=1
            return {key:[int(x) for x in meta[start:start+4]] for key,start in [('q_shape',0),('q_strides',4),('k_shape',8),('k_strides',12),('v_shape',16),('v_strides',20)]}
        model,_=load(str(model_path));mx.eval(model.parameters());mx.synchronize()
        original_attention=frozen.attention
        originals=[x.self_attn for x in model.model.layers]
        trajectories={};live=[];capture_records=[]
        run['phase']='trajectories';save(root/'run.json',run)
        for arm in spec['trajectory_order']:
            caches=make_prompt_cache(model);cache_ids={id(c):i for i,c in enumerate(caches)}
            folder=root/arm;folder.mkdir();compact={};logits_rows=[];logprobs_rows=[];prefill=[];routes={};current_step=0;seen=[]
            def callback(is_native,q,k,v,out,cache):
                layer=len(seen)
                save(root/'last-route.json',dict(arm=arm,step=current_step,layer=layer,q_shape=list(q.shape),k_shape=list(k.shape),v_shape=list(v.shape),native_route=is_native))
                assert layer<24
                if cache is not None:assert cache_ids[id(cache)]==layer
                cache=caches[layer]
                assert cache.offset==128+current_step
                assert tuple(k.shape)==(1,2,128+current_step,64) and k.shape==v.shape
                meta=layout(q,k,v)
                save(root/'last-capture.json',dict(arm=arm,step=current_step,layer=layer,offset=cache.offset,layout=meta,capacity_shape=list(cache.keys.shape)))
                assert meta['k_strides'][1:]==meta['v_strides'][1:]==[16384,64,1]
                assert tuple(cache.keys.shape)==tuple(cache.values.shape)==(1,2,256,64)
                arrays=[np.array(x).copy() for x in (q,k,v,out)]
                if not all(np.isfinite(x).all() for x in arrays):
                    np.savez_compressed(folder/'nonfinite-capture.npz',q=arrays[0],k=arrays[1],v=arrays[2],out=arrays[3])
                    save(root/'failure.json',dict(reason='nonfinite capture',arm=arm,step=current_step,layer=layer,layout=meta))
                    raise ValueError('nonfinite capture')
                qn,kn,vn,on=arrays
                record=dict(arm=arm,step=current_step,layer=layer,native_route=is_native,offset=cache.offset,capacity=256,layout=meta,
                    q_sha256=array_sha(qn),k_sha256=array_sha(kn),v_sha256=array_sha(vn),out_sha256=array_sha(on))
                capture_records.append(record);save(root/'captures.json',capture_records)
                if current_step==0:
                    assert is_native and q.shape[2]==128
                    prefill.append(record)
                else:
                    assert q.shape==(1,14,1,64)
                    if arm in ('native_a','native_b'):assert is_native
                    else:assert not is_native
                    key=f's{current_step:02d}-l{layer:02d}'
                    for name,value in [('q',qn),('k_new',kn[:,:,-1:]),('v_new',vn[:,:,-1:]),('out',on)]:compact[key+'-'+name]=value
                    if arm=='native_a':
                        ks,vs=np.array(cache.keys).copy(),np.array(cache.values).copy()
                        np.savez_compressed(folder/(key+'.npz'),q=qn,k_storage=ks,v_storage=vs,native=on)
                        assert np.array_equal(ks[:,:,:cache.offset],kn) and np.array_equal(vs[:,:,:cache.offset],vn),'storage/view capture mismatch'
                        assert (ks[:,:,cache.offset:]==0).all() and (vs[:,:,cache.offset:]==0).all(),'nonzero unused capacity'
                        live.append((current_step,layer,q,k,v,meta,array_sha(qn),array_sha(kn),array_sha(vn)))
                seen.append(layer)
            adapter_mode='native' if arm in ('native_a','native_b') else 'shared_compiled'
            route_mode='native' if arm=='adapter_native' else 'shared_compiled'
            with qwen_decode_adapter(model,adapter_mode,routes),diagnostic_hooks(qwen2,frozen,callback,route_mode):
                inputs=[spec['prompt_ids']]+[[x] for x in spec['forced_decode_tokens']]
                for step,ids in enumerate(inputs):
                    current_step=step;seen=[]
                    with mx.stream(generation_stream):
                        logits=model(mx.array(ids)[None],cache=caches)[:,-1,:]
                        logprobs=logits-mx.logsumexp(logits,keepdims=True)
                        mx.eval(logits,logprobs)
                        la,pa=np.array(logits[0]).copy(),np.array(logprobs[0]).copy()
                    assert seen==list(range(24)) and all(c.offset==128+step for c in caches)
                    logits_rows.append(la);logprobs_rows.append(pa)
                    np.savez_compressed(folder/'predictions.npz',logits=np.stack(logits_rows),logprobs=np.stack(logprobs_rows))
                    assert np.isfinite(la).all() and np.isfinite(pa).all(),'nonfinite prediction'
                    np.savez_compressed(folder/'compact.npz',**compact)
                    budget()
            mx.synchronize(generation_stream);mx.synchronize()
            assert all(layer.self_attn is old for layer,old in zip(model.model.layers,originals))
            if adapter_mode=='shared_compiled':assert routes==dict(prefill=24,decode=384)
            final={f'l{layer:02d}-{kind}':np.array(value).copy() for layer,c in enumerate(caches) for kind,value in zip(('K','V'),c.state)}
            np.savez_compressed(folder/'final-kv.npz',**final)
            assert all(np.isfinite(v).all() and v.shape==(1,2,144,64) for v in final.values()),'nonfinite or invalid final cache'
            trajectories[arm]=dict(prefill=prefill,logits=np.stack(logits_rows),logprobs=np.stack(logprobs_rows),compact=compact,final=final)
            save(folder/'trajectory.json',dict(prefill_calls=24,decode_calls=384,prediction_rows=17,cache_offsets=[c.offset for c in caches],routes=routes,
                 greedy_prediction_ids=[int(np.argmax(x)) for x in logits_rows],prefill_hashes=prefill))
            del caches
        assert len(live)==384
        run['phase']='same_input_shadow';save(root/'run.json',run)
        shadow=[]
        for step,layer,q,k,v,oldmeta,qsha,ksha,vsha in live:
            key=f's{step:02d}-l{layer:02d}';path=root/'native_a'/(key+'.npz')
            with np.load(path,allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
            with mx.stream(generation_stream):
                before=layout(q,k,v)
                values=[np.array(x).copy() for x in (q,k,v)]
                if [array_sha(x) for x in values]!=[qsha,ksha,vsha] or before!=oldmeta:
                    np.savez_compressed(root/'capture-mutation.npz',q=values[0],k=values[1],v=values[2])
                    save(root/'failure.json',dict(reason='retained capture changed',step=step,layer=layer,before=oldmeta,after=before))
                    raise ValueError('retained capture changed')
                candidate=np.array(original_attention(q,k,v,'shared_compiled')).copy()
                after=layout(q,k,v)
                after_values=[np.array(x).copy() for x in (q,k,v)]
            if before!=after or [array_sha(x) for x in after_values]!=[qsha,ksha,vsha]:
                np.savez_compressed(root/'shadow-mutation.npz',q=after_values[0],k=after_values[1],v=after_values[2],candidate=candidate)
                save(root/'failure.json',dict(reason='shadow mutated input/layout',step=step,layer=layer,before=before,after=after))
                raise ValueError('shadow mutated input/layout')
            reference=dense_reference(*values)
            arrays.update(candidate=candidate,reference=reference)
            np.savez_compressed(path,**arrays)
            assert np.isfinite(candidate).all() and np.isfinite(reference).all(),'nonfinite shadow/reference'
            row=dict(step=step,layer=layer,layout=before,inputs_unchanged=True,
                native_reference=error_metrics(arrays['native'],reference,spec['operator_atol'],spec['operator_rtol']),
                candidate_reference=error_metrics(candidate,reference,spec['operator_atol'],spec['operator_rtol']),
                native_candidate=error_metrics(arrays['native'],candidate,spec['operator_atol'],spec['operator_rtol']))
            shadow.append(row);save(root/'shadow.json',shadow)
            budget()
        assert layout_calls==spec['metadata_helper_calls'],layout_calls
        run['phase']='summarize';save(root/'run.json',run)
        # All summary formulas are fixed before this diagnostic execution.
        summary=build_summary(spec,trajectories)
        save(root/'summary.json',summary)
        run.update(status='complete',diagnostic_complete=True,metadata_helper_calls=layout_calls,performance_trials=0)
        budget()
    except BaseException as error:
        run.update(status='failed',diagnostic_complete=False,error_type=type(error).__name__,performance_trials=0)
        raise
    finally:
        run['artifacts']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'run.json'}
        save(root/'run.json',run)


def build_summary(spec,trajectories):
    import numpy as np
    old=json.loads(Path('results/gqa-shared-decode-v1/model-correctness.json').read_text())[0]
    comparisons={}
    for left,right in [('native_a','native_b'),('native_a','adapter_native'),('adapter_native','candidate'),('native_a','candidate')]:
        a,b=trajectories[left],trajectories[right];rows=[];kv=[]
        for key in a['compact']:
            atol,rtol=(spec['kv_atol'],spec['kv_rtol']) if key.endswith(('k_new','v_new')) else (spec['operator_atol'],spec['operator_rtol'])
            rows.append(dict(key=key,**error_metrics(a['compact'][key],b['compact'][key],atol,rtol)))
        for key in a['final']:kv.append(dict(key=key,**error_metrics(a['final'][key],b['final'][key],spec['kv_atol'],spec['kv_rtol'])))
        comparisons[left+'__'+right]=dict(compact=rows,final_kv=kv,logprobs=error_metrics(a['logprobs'],b['logprobs'],spec['logprobs_atol'],spec['logprobs_rtol']),
            raw_logits=error_metrics(a['logits'],b['logits'],0,0),raw_logits_scope='diagnostic exact comparison, no original logits tolerance',
            prefill_hash_equal=[all(x[key]==y[key] for key in ('q_sha256','k_sha256','v_sha256','out_sha256')) for x,y in zip(a['prefill'],b['prefill'])],
            prefill_layout_equal=[x['layout']==y['layout'] for x,y in zip(a['prefill'],b['prefill'])])
    fidelity={}
    with np.load('results/gqa-shared-decode-v1/model-logprobs-128.npz',allow_pickle=False) as z:
        for arm,oldarm in [('native_a','native'),('native_b','native'),('adapter_native','native'),('candidate','candidate')]:
            t=trajectories[arm]
            matches={f"l{r['layer']:02d}-{r['kind']}":array_sha(t['final'][f"l{r['layer']:02d}-{r['kind']}"])==r[oldarm+'_sha256'] for r in old['kv']}
            fidelity[arm]=dict(logprobs_first16=error_metrics(z[oldarm],t['logprobs'][:16],spec['logprobs_atol'],spec['logprobs_rtol']),final_kv_hash_matches=matches,
                all_final_kv_exact=all(matches.values()),greedy_first16_match_forced=bool(np.array_equal(np.argmax(t['logits'][:16],axis=-1),spec['forced_decode_tokens'])))
    current=[r['key'] for r in comparisons['native_a__candidate']['final_kv'] if not r['allclose']]
    previous=[f"l{r['layer']:02d}-{r['kind']}" for r in old['kv'] if not r['allclose']]
    return dict(diagnostic_complete=True,original_kv_gate_pass=not current,original_kv_gate_scope='new diagnostic trajectory using original thresholds; does not replace v1 failure',original_v1_model_gate_pass=False,original_failed_kv=current,old_failed_kv=previous,failed_set_reproduced=current==previous,
        comparisons=comparisons,fidelity=fidelity,first16_native_candidate_logprobs=error_metrics(trajectories['native_a']['logprobs'][:16],trajectories['candidate']['logprobs'][:16],spec['logprobs_atol'],spec['logprobs_rtol']),causal_inference='Same-input/operator-vs-adapter/trajectory separation only; no unique reduction/exp/FMA root cause or performance claim.',performance_trials=0)


def main():
    if not __debug__:raise RuntimeError('optimized mode unsupported')
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:execute(a.model,a.output);return
    if a.output.exists():raise FileExistsError('new output required')
    spec=json.loads(SPEC.read_text())
    try:r=subprocess.run([sys.executable,'-m','experiments.gqa_shared_diagnostic','--model',str(a.model),'--output',str(a.output),'--worker'],timeout=spec['budget']['hard_subprocess_seconds'])
    except subprocess.TimeoutExpired:
        if a.output.exists():save(a.output/'timeout.json',dict(status='hard_timeout',seconds=spec['budget']['hard_subprocess_seconds']))
        raise SystemExit(124)
    raise SystemExit(r.returncode)

if __name__=='__main__':main()
