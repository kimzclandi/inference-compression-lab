"""Independent read-only recalculation of 120 fixed risk-gated request timings."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(path.read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def finite_positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0

def tree(folder):
    need(not any(p.is_symlink() for p in folder.rglob('*')),'symlink')
    expected=read(folder/'checksums.json')
    actual={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
            if p.is_file() and p.relative_to(folder).as_posix()!='checksums.json'}
    need(bool(expected) and expected==actual,'checksum coverage/content mismatch '+str(folder))


def compare(actual,expected,name):
    if isinstance(expected,dict):
        for key,value in expected.items():compare(actual[key],value,name+'.'+key)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),name+' count')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,name+'.'+str(i))
    elif isinstance(expected,float):need(type(actual) in (int,float) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),name+' numeric')
    else:need(actual==expected,name+' value')


def audit(repo_path):
    repository=Path(repo_path).resolve()
    root=repository/'results/qa-risk-performance-v1';tree(root)
    protocol=read(root/'protocol.json');execution=read(root/'run.json');published=read(root/'summary.json')
    need(execution['status']=='complete','incomplete parent run')
    need(sha(root/'protocol.json')==sha(repository/'configs/qa-risk/performance.json'),'protocol identity')
    need(protocol['variant']=='int8' and protocol['threshold']==.7,'frozen variant/threshold')
    need((protocol['processes'],protocol['warmup_requests'],protocol['repetitions_per_sample'])==(3,8,5),'frozen process/sample counts')
    source_hashes={}
    for name,expected in protocol['source_sha256'].items():
        need(expected==sha(repository/name),'source drift '+name)
        source_hashes[name]=expected
    need(sha(root/'benchmark_source.py')==protocol['source_sha256']['experiments/benchmark_qa_risk.py'],'benchmark snapshot')
    training=repository/'results/qa-risk-v2/training'
    need(sha(training/'selection.json')==protocol['selection_sha256'],'selection changed')
    selection=read(training/'selection.json')['variants']['int8']
    need(selection['eligible'] is True and selection['threshold']==.7,'selection not eligible')
    dataset=repository/'configs/qa-specialist/dataset/calibration/data.jsonl'
    original_study=read(repository/'configs/qa-specialist/study.json')
    need(sha(dataset)==original_study['data_sha256']['calibration'],'original fixed input data changed')
    data=[json.loads(line) for line in dataset.read_text().splitlines()]
    ids=[r['id'] for r in sorted(data,key=lambda row:hashlib.sha256(('2026100407'+row['id']).encode()).hexdigest())[:8]]
    need(ids==protocol['sample_ids'],'sample selection differs')
    sequence=[(rep,identifier) for rep in range(5) for identifier in ids]
    need(len(execution['commands'])==3,'worker command count')
    processes=[];all_hashes=defaultdict(set);all_statuses=defaultdict(set)
    for index in range(3):
        folder=root/str(index);tree(folder);record=read(folder/'run.json')
        need(record['status']=='complete','worker incomplete')
        need(record['protocol_sha256']==sha(root/'protocol.json'),'worker protocol hash')
        command=execution['commands'][index]
        need(command[command.index('--output-dir')+1]==f'results/qa-risk-performance-v1/{index}','worker order/output')
        need(command[command.index('--training-dir')+1]=='results/qa-risk-v2/training','worker head source')
        need(command[command.index('--protocol')+1]=='configs/qa-risk/performance.json','worker protocol path')
        need(type(record['pid'])is int and record['pid']>0,'missing valid PID')
        need(record['platform'].startswith('macOS-'),'Darwin RSS units expected')
        need(type(record['peak_rss_bytes'])is int and record['peak_rss_bytes']>0,'RSS bytes')
        need(finite_positive(record['initialization_seconds']),'initialization time')
        measurements=record['records']
        need([(m['repetition'],m['id']) for m in measurements]==sequence,'sample/repetition order and exact coverage')
        for m in measurements:
            need(finite_positive(m['seconds']),'duration nonpositive/nonfinite')
            need(m['status'] in ('answer','abstain'),'invalid benchmark decision status')
            need(isinstance(m['decision_sha256'],str) and len(m['decision_sha256'])==64 and all(c in '0123456789abcdef' for c in m['decision_sha256']),'malformed decision digest')
            all_hashes[m['id']].add(m['decision_sha256']);all_statuses[m['id']].add(m['status'])
        values=[m['seconds'] for m in measurements];middle=statistics.median(values)
        compare(record['median_seconds'],middle,'per-process median')
        processes.append(dict(index=index,pid=record['pid'],n=len(values),median_seconds=middle,min_seconds=min(values),max_seconds=max(values),
                              peak_rss_bytes=record['peak_rss_bytes'],initialization_seconds=record['initialization_seconds'],
                              decision_counts=dict(Counter(m['status'] for m in measurements))))
    need(len({p['pid'] for p in processes})==3,'workers share PID')
    need(all(len(values)==1 for values in all_hashes.values()),'decision digest changed across repeats/processes')
    need(all(len(values)==1 for values in all_statuses.values()),'status changed across repeats/processes')
    recomputed=dict(median_of_process_medians_seconds=statistics.median(p['median_seconds'] for p in processes),
                    process_medians_seconds=[p['median_seconds'] for p in processes],peak_rss_bytes=[p['peak_rss_bytes'] for p in processes],
                    initialization_seconds=[p['initialization_seconds'] for p in processes],processes=3,measurements=120,scope=protocol['timing_scope'])
    compare(published,recomputed,'published summary')
    return dict(all_pass=True,benchmark_commit=execution['git_head'],protocol_sha256=sha(root/'protocol.json'),
                source_hashes_verified_current=source_hashes,git_required=False,dataset_sha256=sha(dataset),selection_sha256=protocol['selection_sha256'],
                sample_ids=ids,measurements_checked=120,processes_checked=3,distinct_pids=True,all_per_process_checksums_and_parent_coverage_pass=True,
                request_order='Per process: one untimed warmup per selected input, then five repetitions of the same fixed eight-ID order; processes started sequentially 0,1,2.',
                summary_matches=True,recomputed_summary=recomputed,process_details=processes,
                peak_rss_median_bytes=statistics.median(p['peak_rss_bytes'] for p in processes),
                decision_hashes={identifier:next(iter(values)) for identifier,values in all_hashes.items()},
                decision_statuses={identifier:next(iter(values)) for identifier,values in all_statuses.items()},
                observations=dict(included='Tokenizer, synchronous CPU ORT for all windows, logit/offset materialization, exhaustive decoder, all fixed risk features including alternative-span search, fitted logistic arithmetic, exact-span parser and threshold decision.',
                                  excluded='Model load and head rebuild are separately timed. Full evidence verification at serving startup, CLI JSON-schema/input/output conversion, JSON I/O, HTTP/network, queuing and concurrent service are excluded.',
                                  memory='Darwin lifetime ru_maxrss bytes includes process initialization and inference, not model-only memory or steady-state RSS.',
                                  comparison='This is a single INT8-plus-head request path. The earlier 54.19 ms and 1.33x belong to the base extractor benchmark; neither is transferred to this full risk-gated path. No equivalent passing FP32-head baseline exists.',
                                  input_limits='Eight published original calibration inputs are single-window. This does not characterize maximum-window latency, tail SLOs, concurrent throughput or an entire CLI launch.',
                                  provenance='Workers verify frozen runtime sources and selection; distinct saved OS PIDs and sequential parent commands support process isolation. Timing evidence records decision digests, not full decisions, so stability is checked but semantic decision correctness relies on the separate QA evidence.'),
                scope='Read-only raw timing/RSS arithmetic audit, no model run, no threshold changes, no new quality acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);args=p.parse_args();result=audit(args.repo)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(all_pass=True,measurements=120,median_ms=result['recomputed_summary']['median_of_process_medians_seconds']*1000,peak_rss_median_bytes=result['peak_rss_median_bytes'],distinct_pids=True)))
