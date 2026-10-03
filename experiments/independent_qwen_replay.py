"""Reviewer replay: independent safetensors reader and greedy loop, no lab imports.

Checks all local model tensors, then replays a small fixed published cohort.
This is same-device reproduction, never a fresh confirmation or speed benchmark.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def tensors(path):
    # Read the public safetensors container directly, independently of MLX loader.
    with Path(path).open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
        for name, info in header.items():
            if name == '__metadata__':
                continue
            start, end = info['data_offsets']
            f.seek(8 + length + start)
            raw = f.read(end - start)
            if len(raw) != end - start:
                raise ValueError('Truncated tensor: ' + name)
            yield name, info['dtype'], info['shape'], raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    plan = read(a.plan)
    state = {'status': 'running', 'started': datetime.now(timezone.utc).isoformat(),
             'scope': 'Published-sample reproduction, existing pinned environment, same M4 Max',
             'plan': plan, 'plan_sha256': digest(a.plan), 'script_sha256': digest(__file__),
             'python': platform.python_version(), 'platform': platform.platform(),
             'packages': {n: importlib.metadata.version(n) for n in ['mlx','mlx-lm','numpy','transformers']},
             'identity': {}, 'records': []}
    write(a.output_dir/'started.json', state)
    try:
        import numpy as np
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        for name, meta in read('configs/qwen-quantization/upstream-identity.json').items():
            if digest(a.upstream/name) != meta['sha256']:
                raise ValueError('Upstream file mismatch: ' + name)
        paths = {v:a.model_root/(f'student-{v}' if v in ['fp16','q4','q8'] else v) for v in plan['variants']}
        summaries = {}
        dtypes = {'F16':'mlx.core.float16', 'F32':'mlx.core.float32', 'U32':'mlx.core.uint32'}
        expected = read('configs/qwen-quantization/tensor-identities.json')
        base_config = read(paths['fp16']/'config.json')
        for v, path in paths.items():
            actual = {n:{'dtype':dtypes[t], 'shape':s, 'bytes':len(b), 'sha256':hashlib.sha256(b).hexdigest()}
                      for n,t,s,b in tensors(path/'model.safetensors')}
            if actual != expected[v]['tensors']:
                raise ValueError('Independent tensor identity mismatch: ' + v)
            cfg = read(path/'config.json')
            if cfg != expected[v]['config']:
                raise ValueError('Model configuration mismatch: ' + v)
            semantic = {k:val for k,val in cfg.items() if k not in ['quantization','quantization_config']}
            if semantic != base_config:
                raise ValueError('Non-quantization configuration drift: ' + v)
            frozen_files = read(f'results/qwen-confirmation-v1/{v}-quality.json')['model_files']
            live_files = {f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in path.iterdir() if f.is_file()}
            if live_files != frozen_files:
                raise ValueError('Model file inventory drift: ' + v)
            summaries[v] = actual
            state['identity'][v] = {'tensors':len(actual), 'tensor_bytes':sum(t['bytes'] for t in actual.values()),
                                     'all_tensor_hashes_match':True, 'all_files_match':True,
                                     'non_quantization_config_matches_fp16':True}
        # Independently reconstruct original BF16 -> FP16 conversion one tensor at a time.
        converted = 0
        for n,t,s,b in tensors(a.upstream/'model.safetensors'):
            if t == 'BF16':
                data = (np.frombuffer(b, dtype='<u2').astype(np.uint32) << 16).view(np.float32).astype('<f2').tobytes()
            elif t == 'F16':
                data = b
            else:
                raise ValueError('Unexpected upstream dtype: ' + t)
            if hashlib.sha256(data).hexdigest() != summaries['fp16'][n]['sha256'] or s != summaries['fp16'][n]['shape']:
                raise ValueError('Original source conversion differs: ' + n)
            converted += 1
        if converted != len(summaries['fp16']):
            raise ValueError('Source tensor inventory mismatch')
        state['upstream_tensors_converted_independently'] = converted
        for v,block in [('selected',10),('control',22)]:
            prefix=f'model.layers.{block}.'
            wanted = {n:x for n,x in summaries['q4'].items() if not n.startswith(prefix)}
            wanted.update({n:x for n,x in summaries['fp16'].items() if n.startswith(prefix)})
            if summaries[v] != wanted:
                raise ValueError('Restoration source mismatch: ' + v)
            state['identity'][v]['original_fp16_restored_tensors'] = sum(n.startswith(prefix) for n in wanted)
            state['identity'][v]['other_tensors_identical_to_q4'] = True
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache, KVCache
        mx.random.seed(plan['seed'])
        state['device'] = mx.metal.device_info()
        data={r['id']:r for r in map(json.loads,Path('configs/qwen-confirmation/dataset/data.jsonl').read_text().splitlines())}
        cfg=read('configs/qwen-prefix/source-qa-protocol.json')
        for v,path in paths.items():
            model,tok=load(str(path), tokenizer_config={'local_files_only':True})
            old={r['id']:r for r in read(f'results/qwen-confirmation-v1/{v}-quality.json')['predictions']}
            old_logits={r['id']:r for r in read(f'results/qwen-confirmation-v1/{v}-logits.json')}
            for sid in plan['ids']:
                row=data[sid]
                messages=[{'role':'system','content':cfg['system_prompt']},
                          {'role':'user','content':f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"}]
                prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                ids=tok.encode(prompt,add_special_tokens=False)
                cache=make_prompt_cache(model)
                if not all(type(c) is KVCache for c in cache):
                    raise ValueError('Non-floating KV cache')
                tokens=[]; x=mx.array([ids]); first=None
                for step in range(plan['max_new_tokens']):
                    logits=model(x,cache=cache)[:, -1, :]
                    token=mx.argmax(logits,axis=-1); mx.eval(token); mx.synchronize()
                    if step == 0:
                        first=np.array(logits[0].astype(mx.float32))
                    tokens.append(int(token.item()))
                    if tokens[-1] in tok.eos_token_ids:
                        break
                    x=token.reshape(1,1)
                top=np.argsort(-first,kind='stable')[:10].tolist()
                rec={'variant':v,'id':sid,'input_shape':[1,len(ids)],'token_ids':tokens,
                     'prediction':tok.decode(tokens,skip_special_tokens=True),
                     'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                     'input_token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     'top10':[{'token':i,'text':tok.decode([i]),'logit':float(first[i])} for i in top]}
                rec['matches']={'tokens':tokens==old[sid]['token_ids'], 'text':rec['prediction']==old[sid]['prediction'],
                                'prompt':rec['prompt_sha256']==old[sid]['prompt_sha256'],
                                'input_tokens':rec['input_token_ids_sha256']==old[sid]['input_token_ids_sha256'],
                                'top10':rec['top10']==old_logits[sid]['top10']}
                state['records'].append(rec)
                write(a.output_dir/'run.json',state)
                if not all(rec['matches'].values()):
                    raise ValueError('Published replay drift: ' + v + '/' + sid)
            print(v + ': identity and 8 published prompts match',flush=True)
            del model,tok,cache,logits,first; gc.collect(); mx.clear_cache()
        state['status']='complete'
    except BaseException as e:
        state.update(status='failed',error=repr(e)); raise
    finally:
        state['finished']=datetime.now(timezone.utc).isoformat()
        write(a.output_dir/'run.json',state)


if __name__=='__main__':
    main()
