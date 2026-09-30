"""Tune, freeze, confirm and independently evaluate a MiniLM CPU runtime."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from urllib.request import urlopen
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import psutil
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from lab.benchmark import measure
from lab.evidence import reserve_directory, source_record, sha256
from lab.minilm_runtime import MiniLMRuntime

VARIANTS=['fp32','int8_per_channel','int8_per_channel_excluded']
SPEC=Path('configs/minilm-runtime-study.json')
REVISION='ab7a5ac0e35aa22088bdcf23e7fd99b220e53308'


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(path):
    return json.loads(Path(path).read_text())


def model_path(assets,variant):
    return Path('models/minilm/onnx/model.onnx') if variant=='fp32' else assets/(variant+'.onnx')


def environment():
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':source_record(),
            'platform':platform.platform(),'architecture':platform.machine(),'python':sys.version,
            'memory_bytes':psutil.virtual_memory().total,'logical_cpus':os.cpu_count(),
            'versions':{p:importlib.metadata.version(p) for p in ['onnxruntime','onnx','numpy','scipy','tokenizers','pyarrow','psutil']}}
    if sys.platform=='darwin':
        result['cpu']=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    return result


def prepare(args):
    out=reserve_directory(args.output_dir)
    reserve_directory(args.assets_dir)
    manifest=read('models/minilm/manifest.json')
    for name,info in manifest['files'].items():
        if sha256(Path('models/minilm')/name)!=info['sha256']:raise ValueError('Model source hash mismatch')
    if sha256('data/stsb-validation.parquet')!=manifest['dataset']['sha256']:raise ValueError('Validation hash mismatch')
    exclusion=read('results/minilm-mac-m4max-layer-errors-optimized/exclusion.json')['nodes_to_exclude']
    for v in VARIANTS[1:]:
        quantize_dynamic(str(model_path(args.assets_dir,'fp32')),str(model_path(args.assets_dir,v)),
            op_types_to_quantize=['MatMul'],per_channel=True,reduce_range=False,weight_type=QuantType.QInt8,
            nodes_to_exclude=exclusion if v.endswith('excluded') else [],extra_options={'MatMulConstBOnly':True})
        onnx.checker.check_model(str(model_path(args.assets_dir,v)))
    save(out/'manifest.json',{'environment':environment(),'spec':read(SPEC),'spec_sha256':sha256(SPEC),
         'upstream':manifest,'assets_dir':str(args.assets_dir),'excluded_nodes':exclusion,
         'model_sha256':{v:sha256(model_path(args.assets_dir,v)) for v in VARIANTS},
         'prior_validation_spearman':{r['variant']:r['spearman'] for r in read('results/minilm-mac-m4max-ablation/summary.json')}})


def request_texts(spec,phase,batch):
    rows=pq.read_table('data/stsb-validation.parquet').to_pylist()
    low,high=spec['tune_rows' if phase=='tune' else 'confirm_rows']
    source=[r['sentence1'] for r in rows[low:high]]+[r['sentence2'] for r in rows[low:high]]
    return [[source[(i+j)%len(source)] for j in range(batch)] for i in range(spec['requests'])]


def summarize_rounds(paths):
    records=[read(p) for p in paths]
    medians=[r['timing']['repeats'][0]['median_ms'] for r in records]
    p95s=[r['timing']['repeats'][0]['p95_ms'] for r in records]
    return {'round_medians_ms':medians,'median_ms':float(np.median(medians)),
            'round_p95_ms':p95s,'median_round_p95_ms':float(np.median(p95s)),
            'round_mean_cpu_cores': [r['cpu_seconds_during_warmup_and_measurement']/r['wall_seconds_during_warmup_and_measurement'] for r in records],
            'rss_after_warmup_and_measurement_bytes':[r['rss_after_measurement_bytes'] for r in records],
            'lifetime_peak_rss_bytes':[r['lifetime_peak_rss_bytes'] for r in records]}


def worker(args):
    spec=read(SPEC); plan=read(args.plan); item=plan['jobs'][args.job]
    runtime=MiniLMRuntime(model_path(args.assets_dir,item['variant']),threads=item['threads'],
        max_length=item['shape'][1],fixed_padding=True)
    texts=request_texts(spec,plan['phase'],item['shape'][0]); feeds=[runtime.tokenize(t) for t in texts]
    process=psutil.Process(); memory_before=process.memory_info().rss
    counter=0
    def call():
        nonlocal counter
        idx=counter%len(feeds); counter+=1
        if item['scope']=='session':return runtime.run_tokens(feeds[idx])
        return runtime.encode(texts[idx])
    before_cpu=time.process_time(); before_wall=time.perf_counter()
    timing=measure(call,warmup=spec['warmup'],samples=spec['samples'],repeats=1,
        scope='ORT session.run plus Python cyclic dispatch' if item['scope']=='session' else
              'tokenization + ORT session.run + masked mean pooling + L2 normalization; excludes session load, queue/network')
    cpu=time.process_time()-before_cpu; wall=time.perf_counter()-before_wall
    result={**item,'timing':timing,'cpu_seconds_during_warmup_and_measurement':cpu,
            'wall_seconds_during_warmup_and_measurement':wall,
            'rss_before_warmup_bytes':memory_before,'rss_after_measurement_bytes':process.memory_info().rss,
            'lifetime_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'memory_scope':'entire isolated worker includes Python, dataset, tokenizer, model/session; no ONNX inspection or profiler in this worker',
            'providers':runtime.session.get_providers(),
            'feed_sha256':[__import__('hashlib').sha256(b''.join(f[k].tobytes() for k in sorted(f))).hexdigest() for f in feeds],
            'valid_tokens':[int(f['attention_mask'].sum()) for f in feeds]}
    save(args.plan.parent/'raw'/f'{args.job:04d}.json',result)


def execute(args,phase,configs,shapes,rounds,scopes):
    spec=read(SPEC); out=reserve_directory(args.output_dir/phase); (out/'raw').mkdir()
    jobs=[]
    # Deterministic seeded permutation per round; isolated process for every cell.
    rng=np.random.default_rng(20260930)
    for shape in shapes:
        for scope in scopes:
            for round_id in range(rounds):
                for idx in rng.permutation(len(configs)):
                    conf=configs[int(idx)]
                    jobs.append({**conf,'shape':shape,'scope':scope,'round':round_id})
    plan={'environment':environment(),'spec_sha256':sha256(SPEC),'phase':phase,'jobs':jobs,
          'order':'seed 20260930 permutation each round; serial execution; new worker per cell',
          'manifest_sha256':sha256(args.output_dir/'manifest.json')}
    save(out/'plan.json',plan)
    for i,job in enumerate(jobs):
        subprocess.run([sys.executable,'-m','experiments.minilm_runtime_study','worker',
            '--output-dir',str(args.output_dir),'--assets-dir',str(args.assets_dir),
            '--spec',str(SPEC),'--plan',str(out/'plan.json'),'--job',str(i)],check=True)
        if (i+1)%max(len(configs),1)==0:print(phase,'completed',i+1,'/',len(jobs),flush=True)
    summary=[]
    for shape in shapes:
        for scope in scopes:
            for conf in configs:
                files=[out/'raw'/f'{i:04d}.json' for i,j in enumerate(jobs)
                       if j['shape']==shape and j['scope']==scope and j['name']==conf['name']]
                summary.append({**conf,'shape':shape,'scope':scope,**summarize_rounds(files)})
    save(out/'summary.json',summary)
    return summary


def tune(args):
    spec=read(SPEC); prior=read(args.output_dir/'manifest.json')['prior_validation_spearman']
    for v in VARIANTS[1:]:
        if prior['fp32']-prior[v]>spec['quality_acceptance_max_absolute_spearman_drop']:
            raise ValueError('Predefined quantization candidate fails development quality criterion')
    configs=[{'name':f'{v}-t{t}','variant':v,'threads':t} for v in VARIANTS for t in spec['threads']]
    summary=execute(args,'tune',configs,[spec['primary_shape']],spec['tune_rounds'],['session'])
    winners={v:min((r for r in summary if r['variant']==v),key=lambda r:(r['median_ms'],r['threads'])) for v in VARIANTS}
    quantized=min((winners[v] for v in VARIANTS[1:]),key=lambda r:(r['median_ms'],r['name']))
    selection={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'spec_sha256':sha256(SPEC),
               'tune_summary_sha256':sha256(args.output_dir/'tune/summary.json'),
               'per_variant':winners,'selected_quantized':quantized,
               'test_split_accessed_by_this_study':False,
               'rule':spec['selection'],'frozen_before_confirmation':True}
    save(args.output_dir/'selection.json',selection)
    print(json.dumps(selection,indent=2))


def confirmation_configs(selection):
    return [{'name':'fp32-original','variant':'fp32','threads':4},
            {'name':'quantized-original','variant':selection['selected_quantized']['variant'],'threads':4},
            {'name':'fp32-tuned','variant':'fp32','threads':selection['per_variant']['fp32']['threads']},
            {'name':'quantized-tuned','variant':selection['selected_quantized']['variant'],
             'threads':selection['selected_quantized']['threads']}]


def confirm(args,shapes=False):
    spec=read(SPEC);selection=read(args.output_dir/'selection.json')
    execute(args,'shapes' if shapes else 'confirm',confirmation_configs(selection),
        spec['generalization_shapes'] if shapes else [spec['primary_shape']],
        spec['shape_rounds'] if shapes else spec['confirm_rounds'],['session'] if shapes else ['session','pipeline'])


def quality(args):
    # Refuse to reuse an output directory; do not revise selection after inspecting test.
    out=reserve_directory(args.output_dir/'quality')
    selection=read(args.output_dir/'selection.json')
    url=f'https://huggingface.co/datasets/sentence-transformers/stsb/resolve/{REVISION}/data/test-00000-of-00001.parquet'
    path=Path('data/stsb-test.parquet')
    if not path.exists():
        with urlopen(url,timeout=180) as response,path.open('xb') as handle:handle.write(response.read())
    rows=pq.read_table(path).to_pylist()
    save(out/'protocol.json',{'environment':environment(),'selection_sha256':sha256(args.output_dir/'selection.json'),
        'url':url,'revision':REVISION,'sha256':sha256(path),'pairs':len(rows),'max_length':256,'batch':16,
        'note':'First use by this study after selection; public test split, not newly created blind data; pretraining overlap unaudited'})
    texts=[r['sentence1'] for r in rows]+[r['sentence2'] for r in rows]
    gold=np.array([r['score'] for r in rows]); outputs={};embeddings={};summary=[]
    for conf in confirmation_configs(selection):
        rt=MiniLMRuntime(model_path(args.assets_dir,conf['variant']),threads=conf['threads'])
        e=np.concatenate([rt.encode(texts[start:start+16]) for start in range(0,len(texts),16)])
        pred=(e[:len(rows)]*e[len(rows):]).sum(1)
        embeddings[conf['name']]=e;outputs[conf['name']]=pred
        save(out/(conf['name']+'-predictions.json'),[{'row':i,'gold':float(gold[i]),'cosine':float(pred[i])} for i in range(len(rows))])
        summary.append({**conf,'pairs':len(rows),'spearman':float(spearmanr(gold,pred).statistic),
                        'model_sha256':sha256(model_path(args.assets_dir,conf['variant'])),
                        'model_bytes':model_path(args.assets_dir,conf['variant']).stat().st_size})
        print('test quality',conf['name'],summary[-1]['spearman'],flush=True)
        del rt
    baseline=embeddings['fp32-original']
    for result in summary:
        result['embedding_mse_vs_fp32_original']=float(np.mean((embeddings[result['name']]-baseline)**2))
    rng=np.random.default_rng(20260930);bootstrap=[]
    for _ in range(2000):
        idx=rng.integers(0,len(rows),len(rows))
        bootstrap.append(float(spearmanr(gold[idx],outputs['quantized-tuned'][idx]).statistic-
                               spearmanr(gold[idx],outputs['fp32-tuned'][idx]).statistic))
    save(out/'summary.json',summary)
    save(out/'paired-bootstrap.json',{'seed':20260930,'replicates':2000,'unit':'paired rows; repeated-sentence dependence ignored',
         'deltas':bootstrap,'percentile_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()})


def profile(args):
    out=reserve_directory(args.output_dir/'profile');work=reserve_directory(args.assets_dir/'profiles')
    spec=read(SPEC); configs=confirmation_configs(read(args.output_dir/'selection.json'))
    for conf in configs:
        rt=MiniLMRuntime(model_path(args.assets_dir,conf['variant']),threads=conf['threads'],max_length=64,
                          fixed_padding=True,profile_prefix=work/conf['name'])
        feed=rt.tokenize(request_texts(spec,'confirm',1)[0])
        for _ in range(10):rt.run_tokens(feed)
        events=read(rt.session.end_profiling());nodes=[e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
        operators=Counter((e['args']['op_name'],e['args']['provider']) for e in nodes)
        save(out/(conf['name']+'.json'),{**conf,'events':nodes,'summary':[{'op':k[0],'provider':k[1],'events':n} for k,n in sorted(operators.items())],
             'note':'10 profiled requests, distinct from latency workers; event durations include profiler overhead'})


def main(args):
    global SPEC
    SPEC=args.spec
    if args.phase not in ['prepare','worker']:
        manifest=read(args.output_dir/'manifest.json')
        if sha256(SPEC)!=manifest['spec_sha256']:raise ValueError('Frozen specification changed')
        for v,h in manifest['model_sha256'].items():
            if sha256(model_path(args.assets_dir,v))!=h:raise ValueError('Model changed')
    {'prepare':prepare,'tune':tune,'confirm':confirm,'shapes':lambda a:confirm(a,True),
     'quality':quality,'profile':profile,'worker':worker}[args.phase](args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','tune','confirm','shapes','quality','profile','worker'])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--spec',type=Path,default=SPEC)
    p.add_argument('--plan',type=Path)
    p.add_argument('--job',type=int)
    main(p.parse_args())
