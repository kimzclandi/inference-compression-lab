"""Independently reconstruct added endpoint/lexical features, without fitting."""
import argparse
import math
from pathlib import Path
import re
from lab.qa_expanded_io import load_collection
from lab.quantization_diagnostics import read,rows,sha,write
from experiments.verify_qa_coverage_gap import require
ROOT=Path(__file__).resolve().parents[1]

def extra(question,context,p):
    w=p['raw_windows'][p['window_index']];ids=[i for i in range(len(w['context_mask'])) if w['context_mask'][i] or i==w['cls_index']]
    answers=[]
    for axis,k in [('start_logits',p['start_token']),('end_logits',p['end_token'])]:
        logits=w[axis];peak=max(logits[i] for i in ids);exp={i:math.exp(logits[i]-peak) for i in ids};total=sum(exp.values())
        prob={i:v/total for i,v in exp.items()}
        h=sum(-v*math.log(v) for v in prob.values() if v)/math.log(len(ids))
        answers.append([h,math.log(prob[w['cls_index']]),math.log(prob[k]),min(logits[k]-logits[j] for j in ids if j!=k)])
    result=[answers[a][j] for j in range(4) for a in (0,1)]
    words=set(re.findall(r'\w+',p['prediction'].lower()));query=set(re.findall(r'\w+',question.lower()))
    return result+[len(words&query)/len(words) if words else 0.,math.log(1+len(re.findall(r'\w+',question))),p['start']/max(1,len(context))]

def audit():
    old_data=rows(ROOT/'configs/qa-risk/dataset/train/data.jsonl');old=read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
    raw={p['id']:p for s in ('calibration','evaluation') for p in rows(ROOT/f'results/qa-specialist-v1/{s}-int8/predictions.jsonl')}
    expected=[(d,raw[d['id']],f) for d,f in zip(old_data,old)]
    for split,folder in [('train_new','train-new'),('calibration','calibration')]:
        raw,base=load_collection(ROOT/'results/qa-expanded-v1'/folder,split,sha(ROOT/'configs/qa-expanded/study.json'))
        data=rows(ROOT/f'configs/qa-expanded/dataset/{split}.jsonl');items=list(zip(data,raw,base))
        if split=='train_new':expected+=items
        else:cal=items
    n=0;maximum=0.
    for name,records in [('training',expected),('development',cal)]:
        matrix=read(ROOT/f'results/qa-rich-input-v1/{name}-features.json')
        require(len(matrix)==len(records),'Rich row count')
        for stored,(d,p,b) in zip(matrix,records):
            require(stored['id']==d['id']==p['id']==b['id'],'Rich row alignment')
            require(stored['features'][:5]==b['features'] and stored['target']==b['target'] and stored['prediction']==b['prediction'],'Base fields changed')
            values=extra(d['question'],d['context'],p);require(len(stored['features'])==16,'Rich dimensions')
            error=max(abs(a-b) for a,b in zip(values,stored['features'][5:]));maximum=max(maximum,error)
            require(error<=1e-11,'Independent added features mismatch');n+=1
    return dict(evidence_valid=True,rows=n,extra_features=11,maximum_difference=maximum,model_inference=False,fit=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit();write(a.output,r);print(r)
