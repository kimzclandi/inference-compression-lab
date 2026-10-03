"""Independent final selective QA scoring; standard library only, no inference."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import string
import subprocess

ROOT=Path(__file__).resolve().parents[2]
FREEZE_COMMIT='c0e28ffdacc1c7bd5a6910020b3771195f4317f3'
SELECTION_SHA='b2999477ee82318a98266ef0674c22a526cddea1eec541c591a7bec7e354b74a'


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(path.read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]

def norm(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return re.sub(r'\b(?:a|an|the)\b',' ',text).split()


def pair(prediction,gold):
    a,b=norm(prediction),norm(gold);em=float(a==b)
    if not a or not b:return em,em
    common=sum(min(a.count(word),b.count(word)) for word in set(a))
    return em,2*common/(len(a)+len(b))


def score(row,text):
    text=text.strip();abstain=text=='NO_ANSWER'
    valid=bool(text) and (abstain or text in row['context'])
    if abstain:em=f1=float(row['is_impossible'])
    elif not text or row['is_impossible']:em=f1=0.
    else:
        scores=[pair(text,gold) for gold in row['answers']]
        em=max(s[0] for s in scores);f1=max(s[1] for s in scores)
    if not valid:category='format_or_nonextractive'
    elif row['is_impossible'] and not abstain:category='unsupported_answer'
    elif not row['is_impossible'] and abstain:category='over_abstention'
    elif em<1:category='wrong_or_partial_span'
    else:category='correct'
    return dict(id=row['id'],em=em,f1=f1,format_valid=valid,abstain=abstain,is_impossible=row['is_impossible'],family_id=row['family_id'],category=category)


def summary(scored):
    def group(items):
        return dict(n=len(items),em=sum(x['em'] for x in items)/len(items),f1=sum(x['f1'] for x in items)/len(items),format_valid_rate=sum(x['format_valid'] for x in items)/len(items))
    yes=[x for x in scored if not x['is_impossible']];no=[x for x in scored if x['is_impossible']]
    families=defaultdict(list)
    for item in scored:families[item['family_id']].append(item['em'])
    return dict(overall=group(scored),answerable=group(yes),unanswerable=group(no),
                unsupported_answer_rate=sum(not x['abstain'] for x in no)/len(no),
                over_abstention_rate=sum(x['abstain'] for x in yes)/len(yes),
                categories=dict(Counter(x['category'] for x in scored)),
                family_macro_em=sum(sum(v)/len(v) for v in families.values())/len(families),always_abstain_em=len(no)/len(scored))


def same(actual,expected,path):
    if isinstance(expected,dict):
        need(set(actual)==set(expected),path+' exact keys')
        for key,value in expected.items():same(actual[key],value,path+'.'+key)
    elif isinstance(expected,float):need(type(actual) in (int,float) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),path+' number')
    else:need(actual==expected,path+' value')


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(lower=0. if k==0 else c-h,upper=1. if k==n else c+h)


def head_audit():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0');tracked=[p for p in tracked if p]
    paths={ROOT/name for name in tracked if name.endswith('.json')}
    for directory in ('configs','results'):
        paths.update((ROOT/directory).rglob('*.json'))
    matches=[];scanned=0
    def inspect(value,where):
        if isinstance(value,dict):
            if value.get('schema')=='qa-risk-logistic-v1' and any(k in value for k in ('weights','intercept','scaler')):matches.append(where)
            elif all(k in value for k in ('weights','intercept','scaler')):matches.append(where)
            for k,v in value.items():inspect(v,where+'/'+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):inspect(v,where+'/'+str(i))
    for path in sorted(paths):
        if path.is_file():inspect(read(path),path.relative_to(ROOT).as_posix());scanned+=1
    local=[]
    for path in sorted((ROOT/'runs').glob('qa-risk*/*.json')):
        relative=path.relative_to(ROOT).as_posix()
        if path.name not in ('fp32.json','int8.json'):continue
        ignored=subprocess.run(['git','check-ignore','--quiet',relative],cwd=ROOT).returncode==0
        local.append(dict(path=relative,ignored=ignored,tracked=relative in tracked))
    need(not matches,'learned head parameters in distributable JSON')
    need(all(x['ignored'] and not x['tracked'] for x in local),'local head accidentally tracked')
    binaries=[p for p in tracked if Path(p).suffix.lower() in ('.onnx','.safetensors','.pt','.pth','.bin')]
    need(not binaries,'tracked model binary')
    return dict(all_pass=True,json_files_scanned=scanned,parameter_matches=matches,tracked_model_binaries=binaries,local_head_files=local,
                scope='Current Git-tracked and configs/results JSON candidates only. Local learned heads are ignored/untracked. Final package payload still requires separate archive verification; no claim about external storage.')


def audit(folder):
    state=read(folder/'run.json');protocol=read(folder/'protocol.json');selection=read(folder/'selection.json');published=read(folder/'summary.json')
    need(state['status']=='complete' and state['completed_predictions']==128 and state['variant']=='int8','run completeness')
    need(sha(folder/'selection.json')==state['selection_sha256']==SELECTION_SHA,'selection identity')
    frozen=subprocess.check_output(['git','show',FREEZE_COMMIT+':results/qa-risk-v2/training/selection.json'],cwd=ROOT)
    need(hashlib.sha256(frozen).hexdigest()==SELECTION_SHA,'selection not fixed in committed object')
    need(state['git_head']==FREEZE_COMMIT,'run not bound to selection commit')
    need(selection['variants']['int8']['eligible'] is True and selection['variants']['int8']['threshold']==.7,'frozen selected threshold')
    need(selection['variants']['fp32']['eligible'] is False,'failed FP32 calibration overwritten')
    need(sha(folder/'data.jsonl')==state['dataset_sha256']==protocol['data_sha256']['evaluation'],'dataset identity')
    data=rows(folder/'data.jsonl');predictions=rows(folder/'predictions.jsonl')
    need(len(data)==len(predictions)==128 and len({r['id'] for r in data})==len({p['id'] for p in predictions})==128,'ID uniqueness/count')
    need({r['id'] for r in data}=={p['id'] for p in predictions},'ID coverage')
    lookup={p['id']:p for p in predictions};raw=[];system=[];records=[];counts=Counter();statuses=Counter()
    for row in data:
        pred=lookup[row['id']];text=pred['prediction'];confidence=pred['confidence']
        need(isinstance(text,str) and bool(text) and text==text.strip(),'candidate text')
        need(type(pred['start']) is int and type(pred['end']) is int and 0<=pred['start']<pred['end']<=len(row['context']),'candidate offsets')
        need(row['context'][pred['start']:pred['end']]==text,'candidate exact span')
        need(pred['context_sha256']==hashlib.sha256(row['context'].encode()).hexdigest(),'candidate context identity')
        need(type(confidence) in (int,float) and math.isfinite(confidence) and 0<=confidence<=1,'head score')
        status='abstain' if text=='NO_ANSWER' else 'answer' if text in row['context'] else 'invalid';statuses[status]+=1
        accepted=status=='answer' and confidence>=.7
        emitted=text if accepted else '' if status=='invalid' else 'NO_ANSWER'
        raw_item=score(row,text);system_item=score(row,emitted);raw.append(raw_item);system.append(system_item)
        counts['n']+=1;counts['unanswerable' if row['is_impossible'] else 'answerable']+=1
        if accepted:
            counts['accepted']+=1;counts['accepted_correct']+=int(system_item['em']==1.)
            counts['accepted_unanswerable' if row['is_impossible'] else 'accepted_answerable']+=1
        records.append(dict(id=row['id'],source_title=row['source_title'],is_impossible=row['is_impossible'],candidate=text,
                            start=pred['start'],end=pred['end'],context_sha256=pred['context_sha256'],head_score=confidence,
                            accepted=accepted,emitted=emitted,raw_em=raw_item['em'],raw_f1=raw_item['f1'],
                            em=system_item['em'],f1=system_item['f1']))
    c={k:counts[k] for k in ('n','accepted','accepted_correct','answerable','unanswerable','accepted_answerable','accepted_unanswerable')}
    metrics=dict(c,accepted_precision=c['accepted_correct']/c['accepted'],
                 answerable_answer_coverage=c['accepted_answerable']/c['answerable'],correct_answerable_coverage=c['accepted_correct']/c['answerable'],
                 unanswerable_false_accept_rate=c['accepted_unanswerable']/c['unanswerable'],acceptance_rate=c['accepted']/c['n'],
                 valid_span_rate=statuses['answer']/c['n'],invalid_rate=statuses['invalid']/c['n'],model_abstention_rate=statuses['abstain']/c['n'],
                 threshold_abstention_rate=sum(not r['accepted'] for r in records)/c['n'],model_status_counts={k:statuses[k] for k in ('answer','abstain','invalid')})
    calculated=dict(raw=summary(raw),system=summary(system),selective=metrics,threshold=.7)
    same(published['summary'],calculated,'summary')
    gate_keys=[('accepted_precision','min_accepted_precision','>='),('answerable_answer_coverage','min_answerable_answer_coverage','>='),
               ('correct_answerable_coverage','min_correct_answerable_coverage','>='),('unanswerable_false_accept_rate','max_unanswerable_false_accept_rate','<='),('invalid_rate','max_invalid_rate','<=')]
    criteria={}
    for metric,limit,op in gate_keys:
        value=metrics[metric];bound=protocol['quality_constraints'][limit]
        passed=value>=bound if op=='>=' else value<=bound
        need(published['gate']['criteria'][metric]['passed']==passed,'published gate disagreement')
        criteria[metric]=dict(value=value,limit=bound,operator=op,passed=passed)
    need(published['gate']['all_pass']==all(v['passed'] for v in criteria.values()),'aggregate gate disagreement')
    articles={}
    for title in sorted({r['source_title'] for r in records}):
        group=[r for r in records if r['source_title']==title]
        yes=sum(not r['is_impossible'] for r in group);accepted=sum(r['accepted'] for r in group)
        correct=sum(r['accepted'] and r['em']==1 for r in group);false=sum(r['accepted'] and r['is_impossible'] for r in group)
        articles[title]=dict(n=len(group),answerable=yes,unanswerable=len(group)-yes,accepted=accepted,accepted_correct=correct,
                             accepted_precision=correct/accepted if accepted else None,answerable_coverage=(accepted-false)/yes,
                             accepted_unanswerable=false,system_em=sum(r['em'] for r in group)/len(group),
                             refused_answerable=sum(not r['accepted'] and not r['is_impossible'] for r in group))
    categories={'accepted_correct':lambda r:r['accepted'] and r['em']==1.,'refused_impossible':lambda r:not r['accepted'] and r['is_impossible'],
                'refused_answerable':lambda r:not r['accepted'] and not r['is_impossible']}
    demos={};by_id={r['id']:r for r in data}
    for name,predicate in categories.items():
        selected=min((r for r in records if predicate(r)),key=lambda r:hashlib.sha256(('qa-risk-demo-2026100408'+r['id']).encode()).hexdigest())
        demos[name]=dict(selected,question=by_id[selected['id']]['question'],gold_answers=by_id[selected['id']]['answers'])
    intervals={'accepted_precision':wilson(c['accepted_correct'],c['accepted']),
               'false_accept_rate':wilson(c['accepted_unanswerable'],c['unanswerable'])}
    return dict(all_pass=True,quality_point_gate_passed=all(v['passed'] for v in criteria.values()),threshold=.7,
                freeze_commit=FREEZE_COMMIT,selection_sha256=SELECTION_SHA,protocol_sha256=sha(folder/'protocol.json'),
                data_sha256=sha(folder/'data.jsonl'),prediction_sha256=sha(folder/'predictions.jsonl'),
                summary_recomputed=calculated,criteria=criteria,per_article=articles,per_example=records,
                refused_total=sum(not r['accepted'] for r in records),
                refused_answerable_raw_em_correct=sum(not r['accepted'] and not r['is_impossible'] and r['raw_em']==1 for r in records),
                always_refuse_baseline=dict(em=.5,accepted=0,accepted_precision=None,answerable_coverage=0.,fails_coverage_gate=True),
                descriptive_wilson_intervals=intervals,interval_scope='Report only; binomial intervals ignore context/article dependence and are not population guarantees.',
                demo_selection=dict(rule='Smallest SHA256(qa-risk-demo-2026100408 + id) within each named outcome category; illustration chosen after outcomes, never evaluation evidence.',cases=demos),
                head_distribution_check=head_audit(),
                scope='All 128 fixed final public-benchmark predictions independently scored. Head scores treated as recorded and checked only against threshold; another verifier rebuilds head. No new inference or threshold search.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/qa-risk-v2/evaluation-int8');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.root);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(all_pass=True,quality_point_gate_passed=result['quality_point_gate_passed'],metrics=result['summary_recomputed']['selective'],demos={k:v['id'] for k,v in result['demo_selection']['cases'].items()},head_parameters_absent=result['head_distribution_check']['all_pass'])))
