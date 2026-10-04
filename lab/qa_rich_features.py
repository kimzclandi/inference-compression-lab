"""Label-free endpoint uncertainty and question/answer lexical features."""
import math
import re

EXTRA_NAMES=['start_entropy','end_entropy','start_cls_logp','end_cls_logp','selected_start_logp','selected_end_logp','start_runner_up_gap','end_runner_up_gap','question_answer_token_overlap','log1p_question_tokens','answer_start_fraction']


def enrich(question,context,prediction,base):
    w=prediction['raw_windows'][prediction['window_index']]
    indices=[i for i,m in enumerate(w['context_mask']) if m or i==w['cls_index']]
    extras=[]; axis_values=[]
    for axis,position in [('start_logits',prediction['start_token']),('end_logits',prediction['end_token'])]:
        x=w[axis];maximum=max(x[i] for i in indices)
        lse=maximum+math.log(math.fsum(math.exp(x[i]-maximum) for i in indices))
        entropy=-math.fsum(math.exp(x[i]-lse)*(x[i]-lse) for i in indices)/math.log(len(indices))
        others=[x[i] for i in indices if i!=position]
        axis_values.append((entropy,x[w['cls_index']]-lse,x[position]-lse,x[position]-max(others)))
    for j in range(4):extras.extend(a[j] for a in axis_values)
    q=set(re.findall(r'\w+',question.lower()));a=set(re.findall(r'\w+',prediction['prediction'].lower()))
    extras.extend([len(q&a)/len(a) if a else 0.,math.log1p(len(re.findall(r'\w+',question))),prediction['start']/max(len(context),1)])
    if len(extras)!=len(EXTRA_NAMES) or not all(math.isfinite(x) for x in extras):raise ValueError('Invalid rich features')
    return dict(base,features=base['features']+extras,feature_names=base['feature_names']+EXTRA_NAMES)
