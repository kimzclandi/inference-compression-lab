"""Read-only audit of ablation features and calibration decisions; no refitting.

Independently enumerate legal spans from pinned raw logits, then recount every
threshold decision. This validates feature/metric arithmetic, not a new model
run, independent dataset, or arbitrary-platform optimizer convergence.
"""
import argparse
import json
import math
from pathlib import Path

from lab.artifact_integrity import verify_hashes
from lab.qa_metrics import normalize
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal, write

ROOT = Path(__file__).resolve().parents[1]


def need(value, message):
    if not value: raise ValueError(message)


def gaps(context, prediction):
    candidates = []
    for wi, window in enumerate(prediction['raw_windows']):
        mask, offsets = window['context_mask'], window['offsets']
        start, end = window['start_logits'], window['end_logits']
        cls = window['cls_index']; null = start[cls] + end[cls]
        for first in range(len(mask)):
            if not mask[first] or offsets[first][0] == offsets[first][1]: continue
            for last in range(first, min(first+30, len(mask))):
                if not mask[last]: break
                if offsets[last][0] == offsets[last][1]: continue
                left, right = offsets[first][0], offsets[last][1]
                while left < right and context[left].isspace(): left += 1
                while left < right and context[right-1].isspace(): right -= 1
                if left == right: continue
                text = context[left:right]
                candidates.append(dict(window=wi, first=first, last=last, start=left, end=right,
                                       text=text, normalized=normalize(text), margin=start[first]+end[last]-null))
    selected = max(candidates, key=lambda c:c['margin'])
    need(selected['text']==prediction['prediction'] and selected['window']==prediction['window_index'] and
         selected['start']==prediction['start'] and selected['end']==prediction['end'], 'Raw winner mismatch')
    alt = [c for c in candidates if c['normalized'] != selected['normalized']]
    need(alt, 'Global alternative undefined')
    local = [c for c in alt if c['window']==selected['window']]
    return selected['margin'] - max(c['margin'] for c in alt), (
        selected['margin'] - max(c['margin'] for c in local) if local else None)


def assert_no_parameters(value):
    if isinstance(value, dict):
        need(not ({'weights','intercept','scaler'} & set(value)), 'Learned parameters found')
        for v in value.values(): assert_no_parameters(v)
    elif isinstance(value, list):
        for v in value: assert_no_parameters(v)


def counts(data, features, predictions, threshold):
    need([r['id'] for r in data]==[r['id'] for r in features]==[r['id'] for r in predictions], 'Calibration order')
    accepted = []
    for row, feature, pred in zip(data, features, predictions):
        need(pred['prediction']==feature['prediction'] and pred['prediction'] in row['context'], 'Prediction identity/span')
        score=pred['confidence'];need(type(score) in (float,int) and math.isfinite(score) and 0<=score<=1, 'Score')
        accepted.append(score>=threshold)
    total=sum(accepted); correct=sum(a and f['target']==1 for a,f in zip(accepted,features))
    answerable=sum(not r['is_impossible'] for r in data)
    answered=sum(a and not r['is_impossible'] for a,r in zip(accepted,data))
    impossible=len(data)-answerable; false=total-answered
    return dict(n=len(data),accepted=total,accepted_correct=correct,answerable=answerable,unanswerable=impossible,
                accepted_answerable=answered,accepted_unanswerable=false,
                accepted_precision=correct/total if total else None,
                answerable_answer_coverage=answered/answerable,
                correct_answerable_coverage=correct/answerable,
                unanswerable_false_accept_rate=false/impossible,invalid_rate=0.)


def gate_pass(c, limits):
    return (c['accepted_precision'] is not None and c['accepted_precision']>=limits['min_accepted_precision'] and
            c['answerable_answer_coverage']>=limits['min_answerable_answer_coverage'] and
            c['correct_answerable_coverage']>=limits['min_correct_answerable_coverage'] and
            c['unanswerable_false_accept_rate']<=limits['max_unanswerable_false_accept_rate'] and
            c['invalid_rate']<=limits['max_invalid_rate'])


def verify(folder):
    folder=Path(folder);verify_hashes(folder,read(folder/'checksums.json'),exclude=('checksums.json',))
    protocol=read(folder/'protocol.json');run=read(folder/'run.json');comparison=read(folder/'comparison.json')
    need(run['status'] in ('complete','complete_with_arm_failures'), 'Incomplete ablation run')
    need(run['evaluation_consumed'] is False and run['model_inference_run'] is False, 'Study scope changed')
    need(sha(folder/'protocol.json')==run['protocol_sha256'], 'Protocol digest')
    for name,digest in protocol['input_sha256'].items():need(sha(ROOT/name)==digest,'Parent input changed')
    for name,digest in protocol['source_sha256'].items():
        need(sha(ROOT/name)==digest and sha(folder/'source'/name)==digest,'Source changed')
    for file in folder.rglob('*.json'):assert_no_parameters(read(file))
    output={}
    parent=read(ROOT/'configs/qa-risk/study.json')
    archived_selection=read(ROOT/'results/qa-risk-v2/training/selection.json')
    for variant in ('fp32','int8'):
        raw={r['id']:r for role in ('calibration','evaluation') for r in rows(
            ROOT/parent['development_root']/(role+'-'+variant)/'predictions.jsonl')}
        data_by_role={}; feature_changes={}; matrices={method:{} for method in ('all_windows','selected_window')}
        for split,file_name in (('train','training-features.json'),('calibration','calibration-features.json')):
            data=rows(ROOT/'configs/qa-risk/dataset'/split/'data.jsonl');data_by_role[split]=data
            old=read(ROOT/'results/qa-risk-v2/training'/(variant+'-'+file_name))
            baseline=read(folder/variant/'all_windows'/file_name)
            local=read(folder/variant/'selected_window'/file_name)
            need(baseline==old,'Baseline did not replay exact archived inputs')
            by_local={r['id']:r for r in local};changed=[];missing=[];multi=0
            for row,base in zip(data,baseline):
                need(row['id']==base['id'], 'Dataset order')
                pred=raw[row['id']];a,b=gaps(row['context'],pred)
                need(math.isclose(a,base['features'][2],rel_tol=0,abs_tol=1e-12),'Global gap arithmetic')
                target=int(any(normalize(pred['prediction'])==normalize(gold) for gold in row['answers']))
                need(base['target']==target,'EM target mismatch')
                multi+=len(pred['raw_windows'])>1
                if b is None:
                    missing.append(row['id']);need(row['id'] not in by_local,'Undefined feature was imputed');continue
                local_row=by_local[row['id']]
                need(math.isclose(b,local_row['features'][2],rel_tol=0,abs_tol=1e-12),'Local gap arithmetic')
                need(local_row['target']==target and local_row['prediction']==base['prediction'],'Candidate target/span')
                need(all(base['features'][i]==local_row['features'][i] for i in (0,1,3,4)),'Other feature changed')
                if base['features'][2]!=local_row['features'][2]:changed.append(row['id'])
            need(len(local)==len(data)-len(missing),'Local row coverage')
            feature_changes[split]=dict(n=len(data),multi_window_rows=multi,changed_gap_ids=changed,undefined_ids=missing)
            matrices['all_windows'][split]=baseline;matrices['selected_window'][split]=local
        arms={}; predictions_by_method={}
        for method in ('all_windows','selected_window'):
            arm=folder/variant/method;status=read(arm/'status.json')
            need(status==comparison['variants'][variant][method],'Comparison status drift')
            if status['status']!='complete':
                need(status['status'] in ('fit_failed','feature_failed'),'Unknown arm failure')
                arms[method]=dict(status=status['status'],eligible=False,error=status.get('error',status.get('reason')))
                continue
            fit_record=read(arm/'fit.json')
            need(fit_record['convergence']['gradient_max_abs']<=1e-8 and fit_record['config']==protocol['fit_config'], 'Fit gate')
            selection=read(arm/'selection.json');predictions=read(arm/'calibration-predictions.json')
            predictions_by_method[method]=predictions
            need([c['threshold'] for c in selection['candidates']]==protocol['threshold_grid'], 'Threshold grid')
            passing=[];fixed=None
            for candidate in selection['candidates']:
                actual=counts(data_by_role['calibration'],matrices[method]['calibration'],predictions,candidate['threshold'])
                for k,v in actual.items():need(v==candidate['summary']['selective'][k],'Count/rate mismatch: '+k)
                passed=gate_pass(actual,protocol['quality_constraints'])
                need(passed==candidate['gate']['all_pass'],'Gate arithmetic')
                if passed:passing.append(candidate['threshold'])
                if candidate['threshold']==.7:fixed=actual
            chosen=passing[0] if passing else None
            need(chosen==selection['threshold']==status['threshold'] and bool(passing)==selection['eligible']==status['eligible'],'Selection')
            if method=='all_windows':
                need(status['model_sha256']==archived_selection['variants'][variant]['model_sha256'],'Baseline head identity')
                old_predictions=read(ROOT/'results/qa-risk-v2/training'/(variant+'-calibration-predictions.json'))
                need(aggregates_equal(predictions,old_predictions),'Baseline calibration changed')
            arms[method]=dict(status='complete',eligible=bool(passing),threshold=chosen,fixed_0_7=fixed)
        paired=None
        if len(predictions_by_method)==2:
            a,b=[predictions_by_method[m] for m in ('all_windows','selected_window')]
            paired=dict(max_score_difference=max(abs(x['confidence']-y['confidence']) for x,y in zip(a,b)),
                        score_changed_ids=[x['id'] for x,y in zip(a,b) if x['confidence']!=y['confidence']],
                        decision_changed_at_0_7=[x['id'] for x,y in zip(a,b) if (x['confidence']>=.7)!=(y['confidence']>=.7)])
        output[variant]=dict(features=feature_changes,arms=arms,paired_calibration=paired)
    return dict(all_pass=True,variants=output,
                scope='Independent legal-span enumeration and threshold counts; saved local fit trace checked without refitting. No evaluation or model inference.',
                ablation_run_sha256=sha(folder/'run.json'),protocol_sha256=run['protocol_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'results/qa-span-gap-ablation-v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=verify(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
