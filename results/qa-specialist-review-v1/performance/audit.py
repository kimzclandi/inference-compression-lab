"""Read-only standard-library timing/RSS recalculation and RC3 byte audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


RC3='e3d311e'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(path.read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check_tree(folder):
    expected=read(folder/'checksums.json')
    need(bool(expected),'empty file manifest')
    need(not any(p.is_symlink() for p in folder.rglob('*')),'symlink')
    actual={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
            if p.is_file() and p.relative_to(folder).as_posix()!='checksums.json'}
    need(actual==expected,'file hash or coverage mismatch '+str(folder))


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def same(a,b,name):
    if isinstance(b,dict):
        need(isinstance(a,dict),name)
        for key,value in b.items():same(a[key],value,name+'.'+key)
    elif isinstance(b,list):
        need(isinstance(a,list) and len(a)==len(b),name)
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,name+'.'+str(i))
    elif isinstance(b,float):need(finite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),name)
    else:need(type(a) is type(b) and a==b,name)


def history(repository):
    baseline=subprocess.check_output(['git','rev-parse',RC3],cwd=repository,text=True).strip()
    tree=subprocess.check_output(['git','ls-tree','-r','-z',baseline,'--','results'],cwd=repository)
    objects=[];changed=[];missing=[]
    for record in tree.split(b'\0'):
        if not record:continue
        metadata,name=record.split(b'\t',1);mode,kind,object_id=metadata.decode().split()
        name=name.decode();path=repository/name
        need(kind=='blob','unexpected historical tree entry')
        if not path.is_file() or path.is_symlink():missing.append(name);continue
        size=path.stat().st_size;git_hash=hashlib.sha1(('blob '+str(size)+'\0').encode());sha256=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):git_hash.update(block);sha256.update(block)
        actual=git_hash.hexdigest()
        item=dict(path=name,baseline_blob=object_id,current_blob=actual,current_sha256=sha256.hexdigest(),bytes=size)
        objects.append(item)
        if actual!=object_id:changed.append(item)
    return dict(baseline_commit=baseline,baseline_tracked_results=len(objects)+len(missing),
                all_preserved=not missing and not changed,missing=missing,changed=changed,files=objects,
                method='Independently compute Git blob SHA1 over header plus current bytes and compare RC3 tree; also record current SHA256. New result paths are outside this historical set.')


def audit_performance(repository):
    repository=Path(repository).resolve()
    root=repository/'results/qa-specialist-performance-v1';check_tree(root)
    study=read(root/'protocol.json');execution=read(root/'run.json');summary=read(root/'summary.json')
    need(execution['status']=='complete','benchmark incomplete')
    need(sha(root/'protocol.json')==sha(repository/'configs/qa-specialist/study.json'),'study changed')
    order=study['performance']['precision_order'];need(order==['fp32','int8','int8','fp32','fp32','int8'],'precision order')
    need(len(execution['commands'])==len(order),'worker command coverage')
    benchmark_sha=sha(root/'benchmark_source.py')
    need(benchmark_sha==execution['source_sha256']['experiments/benchmark_qa_specialist.py'],'benchmark snapshot')
    frozen_sources={**study['source_sha256'],**execution['source_sha256']}
    source_evidence={}
    for name,expected in frozen_sources.items():
        need(sha(repository/name)==expected,'runtime source differs from frozen protocol '+name)
        source_evidence[name]=expected
    data_path=repository/'configs/qa-specialist/dataset/calibration/data.jsonl'
    need(sha(data_path)==study['data_sha256']['calibration'],'calibration identity')
    data=[json.loads(line) for line in data_path.read_text().splitlines()]
    chosen=sorted(data,key=lambda r:hashlib.sha256(('2026100407'+r['id']).encode()).hexdigest())[:8]
    ids=[row['id'] for row in chosen]
    references={variant:{r['id']:r for line in (repository/f'results/qa-specialist-v1/calibration-{variant}/predictions.jsonl').read_text().splitlines() for r in [json.loads(line)]} for variant in ('fp32','int8')}
    runs=[];rounds=study['performance']['repetitions_per_sample'];need(rounds==5,'repetition count')
    expected_sequence=[(rep,identifier) for rep in range(rounds) for identifier in ids]
    variant_hashes={variant:{} for variant in ('fp32','int8')};slack=[]
    for index,variant in enumerate(order):
        folder=root/f'{index:02d}-{variant}';check_tree(folder);run=read(folder/'run.json')
        need(run['status']=='complete' and run['variant']==variant,'worker status/variant')
        need(run['spec_sha256']==sha(root/'protocol.json'),'worker protocol identity')
        need(run['sample_ids']==ids,'hash-ranked sample selection')
        command=execution['commands'][index]
        need(command[command.index('--variant')+1]==variant,'worker command variant')
        need(command[command.index('--output-dir')+1]==f'results/qa-specialist-performance-v1/{index:02d}-{variant}','worker output ordering')
        records=run['records']
        need([(r['repetition'],r['id']) for r in records]==expected_sequence,'record order/repetition/ID coverage')
        for row in records:
            need(all(finite(row[k]) and row[k]>=0 for k in ('seconds','tokenization_seconds','inference_seconds','decode_seconds')),'negative/nonfinite time')
            need(row['seconds']>0,'zero elapsed')
            remainder=row['seconds']-sum(row[k] for k in ('tokenization_seconds','inference_seconds','decode_seconds'))
            need(remainder>=-1e-9,'component times exceed entire request');slack.append(remainder)
            ref=references[variant][row['id']]
            need(row['prediction_sha256']==hashlib.sha256(ref['prediction'].encode()).hexdigest(),'prediction differs from calibration')
            need(row['input_tokens']==ref['input_tokens'] and row['feature_count']==ref['feature_count'],'feature/tokens differ from calibration')
            need(row['feature_count']==len(row['input_tokens']) and all(type(n)is int and 0<n<=384 for n in row['input_tokens']),'invalid feature shape')
            variant_hashes[variant].setdefault(row['id'],set()).add(row['prediction_sha256'])
        actual_median=statistics.median([r['seconds'] for r in records])
        same(run['median_seconds'],actual_median,'worker median')
        need(type(run['process_peak_rss_bytes'])is int and run['process_peak_rss_bytes']>=run['process_peak_before_load_bytes']>0,'RSS ordering/bytes')
        need(finite(run['initialization_seconds']) and run['initialization_seconds']>0,'load time')
        need(run['platform'].startswith('macOS-'),'evidence uses Darwin byte conversion')
        runs.append(dict(index=index,variant=variant,n=len(records),median_seconds=actual_median,
                         peak_rss_bytes=run['process_peak_rss_bytes'],initialization_seconds=run['initialization_seconds'],
                         feature_counts=sorted({r['feature_count'] for r in records}),
                         input_token_lengths=sorted({tuple(r['input_tokens']) for r in records}),
                         median_inference_seconds=statistics.median(r['inference_seconds'] for r in records),
                         median_tokenization_seconds=statistics.median(r['tokenization_seconds'] for r in records),
                         median_decode_seconds=statistics.median(r['decode_seconds'] for r in records)))
    need(all(len(hashes)==1 for lookup in variant_hashes.values() for hashes in lookup.values()),'within-variant output drift')
    medians={variant:[r['median_seconds'] for r in runs if r['variant']==variant] for variant in ('fp32','int8')}
    central={variant:statistics.median(values) for variant,values in medians.items()}
    rss={variant:[r['peak_rss_bytes'] for r in runs if r['variant']==variant] for variant in medians}
    rss_median={variant:statistics.median(values) for variant,values in rss.items()}
    ratio=central['fp32']/central['int8']
    recomputed=dict(process_median_seconds=medians,median_of_process_medians_seconds=central,
                    fp32_over_int8_speedup=ratio,speedup_point_gate_passed=ratio>=study['performance']['minimum_median_speedup_for_claim'],
                    process_peak_rss_bytes=rss,initialization_seconds={v:[r['initialization_seconds'] for r in runs if r['variant']==v] for v in medians},
                    samples_per_process=40,processes=6,scope=study['performance']['timing'])
    same(summary,recomputed,'summary')
    return dict(all_pass=True,source_hashes_verified_current=source_evidence,git_required=False,
                benchmark_commit=execution['git_head'],protocol_sha256=sha(root/'protocol.json'),
                processes_checked=6,measurements_checked=sum(r['n'] for r in runs),warmup_requests_per_process=8,
                order=order,sample_ids=ids,summary_matches=True,recomputed_summary=recomputed,
                independent_rss_median_bytes=rss_median,int8_over_fp32_peak_rss_median_ratio=rss_median['int8']/rss_median['fp32'],
                request_component_unaccounted_seconds_range=[min(slack),max(slack)],per_process=runs,
                outputs_stable_and_equal_to_prior_calibration=True,
                review=dict(synchronization='Synchronous ORT CPU Session.run; no asynchronous device synchronization is missing.',
                            timing='Outer perf_counter encloses runtime.predict: tokenizer, every window, logit conversion and exhaustive span decode. Warmup/loading/evidence writes are outside request timing.',
                            excluded='Quality evidence verification, threshold serving gate, later risk-head inference, I/O/network, concurrency and service queueing are not benchmarked.',
                            rss='Darwin ru_maxrss is treated as bytes; lifetime peak includes interpreter, tokenizer, ORT initialization and inference. Not model-only allocation, resident steady state, file bytes or logical KV bytes.',
                            provenance_limit='Workers do not individually snapshot runtime sources or OS PIDs. Parent sequential subprocess calls plus saved command order establish intended isolation; recorded clean tracked source state and Git content match protocol. This is not an external process trace.',
                            generalization='Three processes per precision, eight fixed published single-window calibration requests on one Mac; no tail SLO, concurrent throughput, long-window risk, energy or cross-device claim. Speed does not override failed quality evaluation.'),
                preservation_scope='Historical Git baseline checked separately by audit(); this portable function validates performance only.')


def audit(repository):
    repository=Path(repository).resolve()
    result=audit_performance(repository)
    preservation=history(repository)
    result['rc3_preservation']=preservation
    result['all_pass']=result['all_pass'] and preservation['all_preserved']
    for name,expected in result['source_hashes_verified_current'].items():
        committed=subprocess.check_output(['git','show',result['benchmark_commit']+':'+name],cwd=repository)
        need(hashlib.sha256(committed).hexdigest()==expected,'source differs from benchmark Git commit '+name)
    result['source_hashes_at_benchmark_commit_match']=True
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.repo.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(all_pass=result['all_pass'],measurements=result['measurements_checked'],speedup=result['recomputed_summary']['fp32_over_int8_speedup'],rc3_results=result['rc3_preservation']['baseline_tracked_results'],historical_changed=result['rc3_preservation']['changed'],historical_missing=result['rc3_preservation']['missing'])))
