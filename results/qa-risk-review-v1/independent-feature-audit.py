"""Independently hand-check the five fixed risk features; never evaluate a model.

Expected values use closed-form synthetic answers and an independent exhaustive
standard-library implementation. The project extractor is invoked only as the
implementation under test. No fit or new evaluation operation is imported/run.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import string
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
NAMES=['selected_span_null_margin','selected_joint_context_cls_log_probability',
       'different_normalized_answer_margin_gap','log1p_answer_token_count','log1p_window_count']


def require(condition,message):
    if not condition:raise ValueError(message)


def normalize(text):
    return ' '.join(re.sub(r'\b(?:a|an|the)\b',' ',text.lower().translate(str.maketrans('','',string.punctuation))).split())


def em_label(prediction,answers):
    return int(any(normalize(prediction)==normalize(answer) for answer in answers))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(context,windows):
    candidates=[];best_per_window=[]
    for wi,w in enumerate(windows):
        active=[i for i,m in enumerate(w['context_mask']) if m and w['offsets'][i][0]<w['offsets'][i][1]]
        null=w['start_logits'][w['cls_index']]+w['end_logits'][w['cls_index']]
        current=[]
        for start,end in itertools.combinations_with_replacement(active,2):
            if end-start+1>30 or not all(w['context_mask'][start:end+1]):continue
            left,right=w['offsets'][start][0],w['offsets'][end][1]
            text=context[left:right];trim=text.strip()
            if not trim:continue
            left+=len(text)-len(text.lstrip());right=left+len(trim)
            score=w['start_logits'][start]+w['end_logits'][end]
            record=dict(prediction=trim,start=left,end=right,start_token=start,end_token=end,
                        window_index=wi,span_score=score,null_score=null,margin=score-null)
            current.append(record);candidates.append(record)
        require(bool(current),'no legal candidates')
        best_per_window.append(sorted(current,key=lambda c:(-c['span_score'],c['start_token'],c['end_token']))[0])
    best=sorted(best_per_window,key=lambda c:(-c['margin'],c['window_index']))[0]
    other=[candidate for candidate in candidates if normalize(candidate['prediction'])!=normalize(best['prediction'])]
    require(bool(other),'undefined alternative gap')
    alternative=sorted(other,key=lambda c:(-c['margin'],c['window_index'],c['start_token'],c['end_token']))[0]
    chosen=windows[best['window_index']]
    indices=set(i for i,m in enumerate(chosen['context_mask']) if m)|{chosen['cls_index']}
    # Compute a start/end probability ratio directly after max shifts, rather
    # than reusing the runtime's subtract-two-logsumexp implementation.
    probabilities=[];log_normalizers=[]
    for axis,position in [('start_logits',best['start_token']),('end_logits',best['end_token'])]:
        maximum=max(chosen[axis][i] for i in indices)
        denominator=sum(math.exp(chosen[axis][i]-maximum) for i in sorted(indices))
        log_normalizers.append(maximum+math.log(denominator))
        probabilities.append(chosen[axis][position]-maximum-math.log(denominator))
    features=[best['margin'],sum(probabilities),best['margin']-alternative['margin'],
              math.log(2+best['end_token']-best['start_token']),math.log(1+len(windows))]
    prediction=dict(best,context_sha256=hashlib.sha256(context.encode()).hexdigest(),max_answer_tokens=30,raw_windows=windows)
    diagnostics=dict(selected_window_index=best['window_index'],selected_normalized_answer=normalize(best['prediction']),
                     start_logsumexp=log_normalizers[0],end_logsumexp=log_normalizers[1],
                     normalization_token_count=len(indices),alternative={key:alternative[key] for key in
                       ['prediction','margin','window_index','start_token','end_token','start','end']})
    diagnostics['alternative']['normalized']=normalize(alternative['prediction'])
    return features,prediction,diagnostics


def compare(actual,expected,label):
    if isinstance(expected,dict):
        for key,value in expected.items():compare(actual[key],value,label+'.'+key)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'.'+str(i))
    elif isinstance(expected,float):
        require(type(actual) in (float,int) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-11),label+' numeric mismatch')
    else:require(actual==expected,label+' mismatch')


def weighted_window(starts,ends,offsets,mask):
    return dict(start_logits=[math.log(w) for w in starts],end_logits=[math.log(w) for w in ends],
                offsets=offsets,context_mask=mask,cls_index=0)


def examples():
    masked=weighted_window([2,1,3,1],[2,1,1,4],[[0,0],[0,99],[0,3],[4,7]],[False,False,True,True])
    masked['start_logits'][1]=masked['end_logits'][1]=100.
    first=weighted_window([1,8,1],[1,1,8],[[0,0],[0,3],[4,7]],[False,True,True])
    second=weighted_window([1,4,3],[1,4,3],[[0,0],[8,11],[12,15]],[False,True,True])
    zero=weighted_window([1,4,100,1],[1,1,100,4],[[0,0],[0,1],[2,2],[3,4]],[False,True,True,True])
    return [
        dict(name='masked_question_excluded_cls_included',context='cat dog',windows=[masked],
             expected_prediction='cat dog',expected_alternative='dog',
             expected=[math.log(3),math.log(12/42),math.log(3),math.log(3),math.log(2)],
             derivation='start weights CLS/cat/dog=2/3/1, end=2/1/4; selected 3*4=12, null 2*2=4; log joint=log(12/(6*7)); best different answer dog score4/null4, gaplog3. Masked question logits100 are excluded.'),
        dict(name='cross_window_same_normalized_answer_excluded',context='The cat cat dog',windows=[first,second],
             expected_prediction='The cat',expected_alternative='cat dog',
             expected=[math.log(64),math.log(.64),math.log(64/12),math.log(3),math.log(3)],
             derivation='WindowA selected The cat score64/null1 and normalized cat. WindowB cat score16 is excluded as same normalized answer; cat dog score12 is best different answer across both windows, above A The score8. Selected-window joint=64/(10*10).'),
        dict(name='zero_width_context_in_normalizer_and_token_budget',context='a  b',windows=[zero],
             expected_prediction='a  b',expected_alternative='a',
             expected=[math.log(16),math.log(16/(106*106)),math.log(4),math.log(4),math.log(2)],
             derivation='Context zero-width token weight100 is not an endpoint but participates in each context+CLS normalizer, totaling106; answer spans3tokens, score16/null1; alternative a or b score4, earliest a wins.')]


def synthetic_audit():
    from lab.qa_risk_calibration import extract_features
    result=[]
    for example in examples():
        expected,prediction,diagnostics=independent(example['context'],example['windows'])
        compare(expected,example['expected'],example['name']+'.closed_form')
        require(prediction['prediction']==example['expected_prediction'],'hand span')
        require(diagnostics['alternative']['prediction']==example['expected_alternative'],'hand alternative')
        actual=extract_features(example['context'],prediction)
        compare(actual['feature_names'],NAMES,'feature order')
        compare(actual['features'],example['expected'],example['name']+'.implementation')
        compare(actual['diagnostics'],diagnostics,example['name']+'.diagnostics')
        result.append(dict(name=example['name'],derivation=example['derivation'],features=expected,
                           prediction=prediction['prediction'],diagnostics=diagnostics,all_pass=True))
    label_cases=[('The, CAT!',['cat'],1),('cat dog',['cat'],0),('cat',[],0),('The cat',['a dog','cat'],1)]
    for prediction,gold,expected in label_cases:require(em_label(prediction,gold)==expected,'EM label hand case')
    return dict(examples=result,em_label_cases=[dict(prediction=p,gold=a,expected=e) for p,a,e in label_cases])


def real_audit(training_dir):
    protocol=json.loads((training_dir/'protocol.json').read_text())
    state=json.loads((training_dir/'run.json').read_text());require(state['status']=='complete','training not complete')
    train={r['id']:r for line in (training_dir/'train-data.jsonl').read_text().splitlines() for r in [json.loads(line)]}
    result=[]
    for variant in ['fp32','int8']:
        features=json.loads((training_dir/(variant+'-training-features.json')).read_text())
        selected=min(features,key=lambda r:hashlib.sha256(('independent-feature-audit-'+variant+r['id']).encode()).hexdigest())
        raw={r['id']:r for split in ['calibration','evaluation']
             for line in (ROOT/protocol['development_root']/(split+'-'+variant)/'predictions.jsonl').read_text().splitlines()
             for r in [json.loads(line)]}
        row=train[selected['id']];record=raw[selected['id']]
        expected,prediction,diagnostics=independent(row['context'],record['raw_windows'])
        compare(selected['features'],expected,variant+'.features')
        compare(selected['diagnostics'],diagnostics,variant+'.diagnostics')
        require(selected['prediction']==record['prediction']==prediction['prediction'],'selected span mismatch')
        target=em_label(record['prediction'],row['answers'])
        require(selected['target']==target,'EM target mismatch')
        if row['is_impossible']:require(target==0,'impossible target must be zero')
        result.append(dict(variant=variant,id=row['id'],features=expected,diagnostics=diagnostics,
                           target=target,answerability=row['is_impossible'],all_pass=True,
                           features_file_sha256=sha(training_dir/(variant+'-training-features.json'))))
    return dict(training_dir=str(training_dir),protocol_sha256=sha(training_dir/'protocol.json'),
                selection_rule='Per precision smallest SHA256(independent-feature-audit- + variant + id); no outcome filtering.',
                records=result,evaluation_run_read=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-dir',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=dict(all_pass=True,synthetic=synthetic_audit(),
                feature_implementation_sha256=sha(ROOT/'lab/qa_risk_calibration.py'),
                runner_sha256=sha(ROOT/'experiments/qa_risk.py'),audit_sha256=sha(Path(__file__)),
                scope='Feature/EM-label audit only, not head convergence, score calibration, quality acceptance, or new inference.')
    if args.training_dir:result['real_training_examples']=real_audit(args.training_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(all_pass=True,synthetic_examples=3,real_examples=2 if args.training_dir else 0,output=str(args.output))))


if __name__=='__main__':main()
