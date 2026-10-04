"""Standard-library independent exhaustive logit/score audit; no lab imports.

This does not run a model, select a new grid, or optimize on evaluation data.
The committed protocol defines the only permitted threshold candidates.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import string


GATES = (
    ('accepted_precision', 'min_accepted_precision', '>='),
    ('answerable_answer_coverage', 'min_answerable_answer_coverage', '>='),
    ('correct_answerable_coverage', 'min_correct_answerable_coverage', '>='),
    ('unanswerable_false_accept_rate', 'max_unanswerable_false_accept_rate', '<='),
    ('invalid_rate', 'max_invalid_rate', '<='),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    require(type(value) in (int, float), 'non-numeric or boolean scalar')
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError('number outside finite range') from error
    require(math.isfinite(value), 'nonfinite numeric scalar')
    return value


def close(actual, expected, path='result'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), path + ': not a mapping')
        for key, value in expected.items():
            require(key in actual, path + ': missing ' + key)
            close(actual[key], value, path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ': list coverage mismatch')
        for index, (a, e) in enumerate(zip(actual, expected)):
            close(a, e, path + '[' + str(index) + ']')
    elif type(expected) is float:
        require(math.isclose(number(actual), expected, rel_tol=2e-12, abs_tol=2e-12), path + ': numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, path + ': value/type mismatch')


def independently_decode(context, windows, max_tokens=30):
    require(isinstance(context, str) and bool(context.strip()), 'empty context')
    require(type(max_tokens) is int and max_tokens > 0, 'invalid span limit')
    require(isinstance(windows, list) and bool(windows), 'empty windows')
    candidates = []
    evaluated_spans = 0
    for index, window in enumerate(windows):
        keys = ('start_logits', 'end_logits', 'offsets', 'context_mask')
        require(all(isinstance(window.get(key), list) for key in keys), 'token arrays must be lists')
        n = len(window['start_logits'])
        require(n > 0 and all(len(window[key]) == n for key in keys), 'unequal/empty token arrays')
        starts = [number(x) for x in window['start_logits']]
        ends = [number(x) for x in window['end_logits']]
        mask = window['context_mask']; offsets = window['offsets']; cls = window['cls_index']
        require(type(cls) is int and 0 <= cls < n, 'CLS index')
        require(all(type(value) is bool for value in mask) and any(mask) and not mask[cls], 'mask/CLS validity')
        for position, pair in enumerate(offsets):
            require(isinstance(pair, (list, tuple)) and len(pair) == 2
                    and all(type(x) is int for x in pair), 'invalid offset pair')
            require(0 <= pair[0] <= pair[1], 'invalid offset ordering')
            if mask[position]:
                require(pair[0] <= pair[1] <= len(context), 'context offset range')
                if position > 0 and mask[position - 1]:
                    require(pair[0] >= offsets[position - 1][0] and pair[1] >= offsets[position - 1][1],
                            'context offsets go backwards')
        # Independently enumerate endpoint combinations, then reject interiors.
        # This differs from the runtime's nested-start/early-break traversal.
        legal = []
        active = [i for i, allowed in enumerate(mask) if allowed and offsets[i][0] < offsets[i][1]]
        for first, last in itertools.combinations_with_replacement(active, 2):
            if last - first + 1 > max_tokens or not all(mask[first:last + 1]):
                continue
            left, right = offsets[first][0], offsets[last][1]
            raw = context[left:right]
            trimmed = raw.strip()
            if not trimmed:
                continue
            left += len(raw) - len(raw.lstrip())
            right = left + len(trimmed)
            summed = number(starts[first] + ends[last])
            legal.append((summed, -first, -last, left, right))
        require(bool(legal), 'no legal nonempty span')
        evaluated_spans += len(legal)
        score, neg_first, neg_last, left, right = max(legal)
        first, last = -neg_first, -neg_last
        null = number(starts[cls] + ends[cls]); margin = number(score - null)
        # Algebraically sigmoid(margin); separate expression from runtime.
        confidence = .5 * (1 + math.tanh(margin / 2))
        candidates.append(dict(prediction=context[left:right], start=left, end=right,
                               start_token=first, end_token=last, span_score=score,
                               start_logit=starts[first], end_logit=ends[last],
                               window_index=index, null_score=null, margin=margin,
                               cls_start_logit=starts[cls], cls_end_logit=ends[cls], confidence=confidence))
    winner = sorted(candidates, key=lambda row: (-row['margin'], row['window_index']))[0]
    result = dict(winner, windows=candidates,
                  context_sha256=hashlib.sha256(context.encode('utf-8')).hexdigest(),
                  max_answer_tokens=max_tokens)
    return result, evaluated_spans


def normalized_words(text):
    lower = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\b(?:a|an|the)\b', ' ', lower).split()


def independent_score(row, prediction):
    prediction = prediction.strip()
    if prediction == 'NO_ANSWER':
        value = float(row['is_impossible'])
        return value, value
    if not prediction or row['is_impossible']:
        return 0., 0.
    predicted = normalized_words(prediction)
    ems, f1s = [], []
    for gold in row['answers']:
        expected = normalized_words(gold)
        ems.append(float(predicted == expected))
        denominator = len(predicted) + len(expected)
        common = sum(min(predicted.count(word), expected.count(word)) for word in set(predicted))
        f1s.append(2 * common / denominator if denominator and predicted and expected else float(predicted == expected))
    require(bool(ems), 'answerable row has no gold answers')
    return max(ems), max(f1s)


def independent_metrics(rows, predictions, threshold):
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
            'invalid fixed threshold')
    n = len(rows); require(n > 0 and len(predictions) == n, 'empty or mismatched scoring cohort')
    counts = dict(n=n, accepted=0, accepted_correct=0, answerable=0, unanswerable=0,
                  accepted_answerable=0, accepted_unanswerable=0)
    statuses = Counter(); raw_scores = []; system_scores = []; scored = []
    for row, prediction in zip(rows, predictions):
        raw = prediction['prediction'].strip()
        status = 'abstain' if raw == 'NO_ANSWER' else 'answer' if raw and raw in row['context'] else 'invalid'
        statuses[status] += 1
        conf = number(prediction['confidence']); require(0 <= conf <= 1, 'confidence out of range')
        accepted = status == 'answer' and conf >= threshold
        output = raw if accepted else '' if status == 'invalid' else 'NO_ANSWER'
        raw_em, raw_f1 = independent_score(row, raw)
        em, f1 = independent_score(row, output)
        raw_scores.append((raw_em, raw_f1)); system_scores.append((em, f1))
        impossible = row['is_impossible']; require(type(impossible) is bool, 'nonboolean answerability')
        counts['unanswerable' if impossible else 'answerable'] += 1
        if accepted:
            counts['accepted'] += 1; counts['accepted_correct'] += int(em == 1.)
            counts['accepted_unanswerable' if impossible else 'accepted_answerable'] += 1
        scored.append(dict(id=row['id'], raw_em=raw_em, raw_f1=raw_f1, em=em, f1=f1,
                           accepted=accepted, output=output))
    rate = lambda numerator, denominator: numerator / denominator if denominator else None
    metrics = dict(counts,
                   accepted_precision=rate(counts['accepted_correct'], counts['accepted']),
                   answerable_answer_coverage=rate(counts['accepted_answerable'], counts['answerable']),
                   correct_answerable_coverage=rate(counts['accepted_correct'], counts['answerable']),
                   unanswerable_false_accept_rate=rate(counts['accepted_unanswerable'], counts['unanswerable']),
                   invalid_rate=statuses['invalid'] / n,
                   model_status_counts={key: statuses[key] for key in ('answer', 'abstain', 'invalid')})
    return dict(raw=dict(em=sum(x[0] for x in raw_scores)/n, f1=sum(x[1] for x in raw_scores)/n),
                system=dict(em=sum(x[0] for x in system_scores)/n, f1=sum(x[1] for x in system_scores)/n),
                selective=metrics, scored=scored)


def five_gates(metrics, constraints):
    criteria = {}
    for metric, constraint, op in GATES:
        limit = number(constraints[constraint]); require(0 <= limit <= 1, 'invalid constraint')
        value = metrics[metric]
        passed = value is not None and math.isfinite(value) and (value >= limit if op == '>=' else value <= limit)
        criteria[metric] = dict(value=value, limit=limit, comparator=op, passed=passed)
    return dict(all_pass=all(row['passed'] for row in criteria.values()), criteria=criteria)


def verify_files(folder, checksum_manifest):
    require(not any(p.is_symlink() for p in folder.rglob('*')), 'symlink in evidence')
    actual = {p.relative_to(folder).as_posix(): digest(p)
              for p in folder.rglob('*') if p.is_file() and p.relative_to(folder).as_posix() != 'checksums.json'}
    require(isinstance(checksum_manifest, dict) and bool(checksum_manifest), 'empty checksums')
    require(actual == checksum_manifest, 'file coverage/hash mismatch at ' + str(folder))


def audit_run(folder):
    state = load(folder / 'run.json'); protocol = load(folder / 'protocol.json')
    require(state['status'] == 'complete', 'run incomplete: ' + str(folder))
    verify_files(folder, load(folder / 'checksums.json'))
    data = [json.loads(line) for line in (folder / 'data.jsonl').read_text().splitlines()]
    records = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    ids = [row['id'] for row in data]; pred_ids = [row['id'] for row in records]
    require(bool(ids) and len(set(ids)) == len(ids) and len(set(pred_ids)) == len(pred_ids)
            and set(ids) == set(pred_ids), 'exact ID coverage failed')
    require(state['completed_predictions'] == len(ids), 'run prediction count')
    require(state['protocol_sha256'] == digest(folder/'protocol.json'), 'protocol identity')
    require(state['dataset_sha256'] == digest(folder/'data.jsonl') == protocol['data_sha256'][state['split']],
            'partition identity')
    require(state['source_sha256'] == protocol['source_sha256'], 'source mapping mismatch')
    source_files = {p.relative_to(folder/'source').as_posix(): digest(p)
                    for p in (folder/'source').rglob('*') if p.is_file()}
    require(source_files == protocol['source_sha256'], 'source content/coverage mismatch')
    by_id = {record['id']: record for record in records}; ordered = [by_id[key] for key in ids]
    all_spans = 0; all_windows = 0
    max_tokens = protocol['input_limits']['max_answer_tokens']
    require(max_tokens == 30, 'unexpected fixed span token limit')
    for row, record in zip(data, ordered):
        windows = record['raw_windows']
        rebuilt, spans = independently_decode(row['context'], windows, max_tokens)
        close(record, rebuilt, row['id']); all_spans += spans; all_windows += len(windows)
        require(record['feature_count'] == len(windows), 'feature coverage mismatch')
        require(record['input_tokens'] == [len(w['start_logits']) for w in windows], 'input shape mismatch')
        for window in windows:
            n = len(window['start_logits'])
            require(len(window['input_ids']) == len(window['attention_mask']) == len(window['sequence_ids']) == n,
                    'token metadata length')
            expected_mask = [seq == 1 and bool(attention)
                             for seq, attention in zip(window['sequence_ids'], window['attention_mask'])]
            require(window['context_mask'] == expected_mask, 'context mask differs from sequence/attention')
    constraints = protocol['quality_constraints']
    if state['split'] == 'calibration':
        thresholds = protocol['threshold_grid']
    else:
        require(state['split'] == 'evaluation', 'unexpected split')
        selection = load(folder/'selection.json')
        require(state['selection_sha256'] == digest(folder/'selection.json'), 'selection snapshot identity')
        require(selection['protocol_sha256'] == digest(folder/'protocol.json')
                and selection['calibration_feasible'] is True, 'evaluation without passing calibration')
        thresholds = [selection['variants'][state['variant']]['threshold']]
        require(thresholds[0] in protocol['threshold_grid'], 'evaluation threshold outside frozen grid')
    require(isinstance(thresholds, list) and bool(thresholds), 'missing fixed threshold grid')
    curves = []
    for threshold in thresholds:
        summary = independent_metrics(data, ordered, threshold)
        curves.append(dict(threshold=threshold, raw=summary['raw'], system=summary['system'],
                           selective=summary['selective'], gate=five_gates(summary['selective'], constraints)))
    feasible = [row['threshold'] for row in curves if row['gate']['all_pass']]
    return dict(n=len(data), variant=state['variant'], split=state['split'],
                data_sha256=digest(folder/'data.jsonl'), protocol_sha256=digest(folder/'protocol.json'),
                all_logits_redecoded=True, windows_checked=all_windows, legal_spans_compared=all_spans,
                curves=curves, smallest_feasible_threshold=(min(feasible) if feasible else None)
                if state['split']=='calibration' else None,
                fixed_evaluation_threshold=thresholds[0] if state['split']=='evaluation' else None), data, ordered


def self_test():
    a = dict(start_logits=[10, 9, 0], end_logits=[10, 9, 0], offsets=[[0,0],[0,3],[4,7]],
             context_mask=[False,True,True], cls_index=0)
    b = dict(start_logits=[0, 0, 5], end_logits=[0, 0, 5], offsets=[[0,0],[0,3],[4,7]],
             context_mask=[False,True,True], cls_index=0)
    value, _ = independently_decode('cat dog', [a,b])
    require(value['window_index'] == 1 and value['prediction'] == 'dog' and value['margin'] == 10., 'margin fixture')
    require(max(value['windows'],key=lambda w:w['span_score'])['prediction'] == 'cat', 'minnull difference fixture')
    hole = dict(start_logits=[0,20,0,0],end_logits=[0,0,0,20],offsets=[[0,0],[0,3],[0,0],[4,7]],
                context_mask=[False,True,False,True],cls_index=0)
    require(independently_decode('cat dog',[hole])[0]['prediction']=='cat', 'masked-interior fixture')
    unicode_window = dict(start_logits=[0,4],end_logits=[0,5],offsets=[[0,0],[0,5]],context_mask=[False,True],cls_index=0)
    unicode, _ = independently_decode(' \t猫 \n',[unicode_window])
    require((unicode['start'],unicode['end'],unicode['prediction'])==(2,3,'猫'), 'Unicode trimming fixture')
    zero = dict(start_logits=[0,5,100,0],end_logits=[0,0,100,5],offsets=[[0,0],[0,1],[2,2],[3,4]],
                context_mask=[False,True,True,True],cls_index=0)
    require(independently_decode('a  b',[zero],3)[0]['prediction']=='a  b', 'zero interior fixture')
    require(independently_decode('a  b',[zero],2)[0]['prediction']=='a', 'zero token counted fixture')
    caught = 0
    for key, replacement in [('prediction','cat'),('confidence',.1),('start',1),('window_index',0)]:
        broken = dict(value); broken[key] = replacement
        try: close(broken,value)
        except ValueError: caught += 1
    require(caught == 4, 'tampered decoder records not rejected')
    rows = [dict(id=str(i),context='cat dog',answers=[] if i > 1 else [['cat'],['dog']][i],
                 is_impossible=i > 1) for i in range(4)]
    predictions = [dict(id=str(i),prediction='cat' if i % 2 == 0 else 'dog',confidence=conf)
                   for i,conf in enumerate([.99,.95,.2,.1])]
    constraints = dict(min_accepted_precision=.9,min_answerable_answer_coverage=.4,
                       min_correct_answerable_coverage=.35,max_unanswerable_false_accept_rate=.1,
                       max_invalid_rate=.05)
    baseline = independent_metrics(rows,predictions,0.)
    fixed = independent_metrics(rows,predictions,.5)
    require(baseline['selective']['accepted_precision']==.5 and not five_gates(baseline['selective'],constraints)['all_pass'], 'unsafe precision fixture')
    require(fixed['system']['em']==1. and five_gates(fixed['selective'],constraints)['all_pass'], 'selective gate fixture')
    malformed = [dict(item) for item in predictions]; malformed[2]['prediction']='invented'
    failed = independent_metrics(rows,malformed,.5)
    require(failed['selective']['invalid_rate']==.25 and failed['scored'][2]['em']==0.
            and not five_gates(failed['selective'],constraints)['all_pass'], 'invalid refusal reward fixture')
    require(independent_score(rows[0],'The cat') == (1.,1.), 'normalization fixture')
    return dict(all_pass=True, tests=['same_window_null_margin','minnull_counterexample','masked_interior',
                                    'unicode_trimming','zero_offset_internal_only','zero_offset_counts_as_token','four_prediction_mutations_rejected','five_gate_metrics','invalid_not_rewarded_as_refusal','independent_normalization'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
    else:
        require(args.root is not None, '--root required')
        names = ['calibration-fp32','calibration-int8','evaluation-fp32','evaluation-int8']
        present = [name for name in names if (args.root/name).exists()]
        require(set(names[:2]).issubset(present), 'both complete calibration variants required')
        runs = {}; protocols = set()
        for name in present:
            result, _, _ = audit_run(args.root/name); runs[name] = result; protocols.add(result['protocol_sha256'])
        require(len(protocols) == 1, 'variants use different protocols')
        selection_status = None
        if (args.root/'selection.json').exists():
            selected = load(args.root/'selection.json')
            require(selected['protocol_sha256'] in protocols, 'selection protocol identity')
            both_feasible = all(runs['calibration-'+variant]['smallest_feasible_threshold'] is not None
                                for variant in ('fp32','int8'))
            require(selected['calibration_feasible'] is both_feasible, 'calibration eligibility mismatch')
            for variant in ('fp32','int8'):
                threshold = runs['calibration-'+variant]['smallest_feasible_threshold']
                require(selected['variants'][variant]['threshold'] == threshold, 'selected threshold mismatch')
                if 'evaluation-'+variant in runs:
                    require(both_feasible and runs['evaluation-'+variant]['fixed_evaluation_threshold'] == threshold,
                            'evaluation changed threshold')
            selection_status = dict(all_pass=True,calibration_feasible=both_feasible,
                                    selection_sha256=digest(args.root/'selection.json'))
        require(not any(name.startswith('evaluation') for name in present) or selection_status is not None,
                'evaluation needs root selection record')
        report = dict(all_pass=True, runs=runs, audit_script_sha256=digest(Path(__file__)),
                      scope='Independent exhaustive logit decoding and public-benchmark scoring; no model inference or threshold tuning.',
                      evaluation_observed=any(name.startswith('evaluation') for name in present),
                      selection_validation=selection_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(report,target,ensure_ascii=False,indent=2,allow_nan=False); target.write('\n')
    print(json.dumps(dict(all_pass=report['all_pass'],output=str(args.output))))


if __name__ == '__main__':
    main()
