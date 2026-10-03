"""Independent stdlib-only audit; imports no repository scorer or verifier.

Reads only explicitly named completed runs. Never discovers or opens a held-out
confirmation dataset. Reports are written exclusively and cannot be replaced.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import string


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalized(text):
    lowered = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(?:a|an|the)\b', ' ', lowered).split())


def reference_score(row, text):
    raw = text.strip()
    if raw == 'NO_ANSWER':
        return int(row['is_impossible']), float(row['is_impossible'])
    if not raw or row['is_impossible']:
        return 0, 0.0
    p = normalized(raw).split()
    scores = []
    for answer in row['answers']:
        g = normalized(answer).split()
        exact = int(p == g)
        shared = sum(min(p.count(token), g.count(token)) for token in set(p))
        scores.append((exact, 2 * shared / (len(p) + len(g)) if p and g else float(exact)))
    return max(value[0] for value in scores), max(value[1] for value in scores)


def output_status(row, prediction):
    raw = prediction['prediction'].strip()
    if raw == 'NO_ANSWER':
        return 'abstain'
    return 'answer' if raw and raw in row['context'] else 'invalid'


def independent_confidence(prediction, status):
    audit = prediction['audit']
    if status != 'answer':
        assert audit is None and prediction['confidence'] == 0
        return 0.0
    assert isinstance(audit, dict)
    yes, no = audit['yes_logit'], audit['no_logit']
    assert type(yes) in (float, int) and type(no) in (float, int)
    assert math.isfinite(yes) and math.isfinite(no)
    # Direct two-class softmax, independently formulated from the sigmoid code.
    offset = max(yes, no)
    numerator = math.exp(yes - offset)
    value = numerator / (numerator + math.exp(no - offset))
    assert math.isclose(value, prediction['confidence'], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(value, audit['confidence'], rel_tol=0, abs_tol=1e-12)
    assert math.isfinite(audit['binary_mass']) and 0 <= audit['binary_mass'] <= 1
    return value


def threshold_row(data, predictions, threshold, constraints):
    accepted = correct = accepted_answerable = accepted_impossible = invalid = 0
    system_em = system_f1 = 0
    answerable = sum(not row['is_impossible'] for row in data)
    impossible = len(data) - answerable
    assert answerable and impossible
    for row in data:
        pred = predictions[row['id']]
        status = output_status(row, pred)
        confidence = independent_confidence(pred, status)
        take = status == 'answer' and confidence >= threshold
        invalid += status == 'invalid'
        if take:
            accepted += 1
            accepted_impossible += row['is_impossible']
            accepted_answerable += not row['is_impossible']
            emitted = pred['prediction'].strip()
        else:
            emitted = '' if status == 'invalid' else 'NO_ANSWER'
        em, f1 = reference_score(row, emitted)
        correct += take and em == 1
        system_em += em
        system_f1 += f1
    metrics = dict(accepted=accepted, accepted_correct=correct,
                   accepted_answerable=accepted_answerable, accepted_unanswerable=accepted_impossible,
                   accepted_precision=correct / accepted if accepted else None,
                   answerable_answer_coverage=accepted_answerable / answerable,
                   correct_answerable_coverage=correct / answerable,
                   unanswerable_false_accept_rate=accepted_impossible / impossible,
                   invalid_rate=invalid / len(data), system_em=system_em / len(data),
                   system_f1=system_f1 / len(data))
    gates = dict(
        min_accepted_precision=accepted > 0 and metrics['accepted_precision'] >= constraints['min_accepted_precision'],
        min_answerable_answer_coverage=metrics['answerable_answer_coverage'] >= constraints['min_answerable_answer_coverage'],
        min_correct_answerable_coverage=metrics['correct_answerable_coverage'] >= constraints['min_correct_answerable_coverage'],
        max_unanswerable_false_accept_rate=metrics['unanswerable_false_accept_rate'] <= constraints['max_unanswerable_false_accept_rate'],
        max_invalid_rate=metrics['invalid_rate'] <= constraints['max_invalid_rate'])
    return dict(threshold=threshold, metrics=metrics, gates=gates,
                feasible=all(gates.values()), failed_gates=[name for name, passed in gates.items() if not passed])


def audit_run(folder, study, development=False):
    data = records(folder / 'data.jsonl')
    predictions_list = records(folder / 'predictions.jsonl')
    state = read(folder / 'run.json')
    assert state['status'] == 'complete', 'Refuse partial or failed run'
    assert state['predictions'] == predictions_list
    predictions = {record['id']: record for record in predictions_list}
    assert data and len(predictions) == len(predictions_list) == len(data)
    assert set(predictions) == {row['id'] for row in data}
    assert sha(folder / 'data.jsonl') == state['dataset_sha256']
    if not development:
        assert state['dataset_sha256'] == study['data_hashes']['calibration'], 'Only calibration is authorized here'
        assert sha(folder / 'protocol.json') == state['spec_sha256']
        assert read(folder / 'protocol.json') == study
        identity = dict(label=state['model_label'], mode=state['mode'], bits=state['bits'],
                        model_files_sha256={name: value['sha256'] for name, value in state['source_model_files'].items()},
                        data_sha256=state['dataset_sha256'])
        assert sum(candidate == identity for candidate in study['allowed_runs']) == 1
    raw_scores = [reference_score(row, predictions[row['id']]['prediction']) for row in data]
    raw_em = sum(value[0] for value in raw_scores) / len(data)
    raw_f1 = sum(value[1] for value in raw_scores) / len(data)
    assert math.isclose(raw_em, state['metrics']['overall']['em'], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(raw_f1, state['metrics']['overall']['f1'], rel_tol=0, abs_tol=1e-12)
    statuses = Counter(output_status(row, predictions[row['id']]) for row in data)
    archived_scores = {record['id']: record for record in state['scored']}
    assert set(archived_scores) == set(predictions)
    confidence_errors = []
    errors = []
    for row, (em, f1) in zip(data, raw_scores):
        pred = predictions[row['id']]
        status = output_status(row, pred)
        confidence = independent_confidence(pred, status)
        confidence_errors.append(abs(confidence - pred['confidence']))
        saved = archived_scores[row['id']]
        assert em == saved['em'] and math.isclose(f1, saved['f1'], rel_tol=0, abs_tol=1e-12)
        assert saved['format_valid'] == (status != 'invalid')
        assert saved['abstain'] == (status == 'abstain')
        if em == 0 or status == 'invalid':
            errors.append(dict(id=row['id'], context=row['context'], question=row['question'],
                               answers=row['answers'], is_impossible=row['is_impossible'],
                               prediction=pred['prediction'], status=status, raw_em=em, raw_f1=f1,
                               confidence=confidence, audit=pred['audit']))
    curve = [threshold_row(data, predictions, value, study['quality_constraints']) for value in study['threshold_grid']]
    feasible = [value['threshold'] for value in curve if value['feasible']]
    layout = state.get('model_layout')
    tensor_summary = None
    if layout is not None:
        byte_total = sum(value['tensor_bytes'] for value in layout['dtype_distribution'].values())
        tensor_count = sum(value['tensors'] for value in layout['dtype_distribution'].values())
        assert byte_total == layout['parameter_tensor_bytes']
        assert tensor_count == layout['parameter_tensor_count']
        assert byte_total > 0 and tensor_count > 0
        tensor_summary = dict(parameter_tensor_bytes=byte_total, parameter_tensor_count=tensor_count,
                              dtype_distribution=layout['dtype_distribution'], module_counts=layout['module_counts'],
                              normalizations_all_fp16=layout['normalizations_all_fp16'],
                              scope='Independent aggregation of saved dtype byte/count records, not a fresh tensor or RSS measurement.')
    elif not development:
        raise AssertionError('Formal run lacks physical parameter layout')
    examples = {}
    filters = {
        'unanswerable_but_supported_by_same_model': lambda r: r['is_impossible'] and r['status'] == 'answer',
        'wrong_answerable_span_with_high_score': lambda r: not r['is_impossible'] and r['status'] == 'answer' and r['raw_em'] == 0,
        'nonextractive_or_malformed_output': lambda r: r['status'] == 'invalid',
    }
    for name, select in filters.items():
        candidates = sorted((row for row in errors if select(row)), key=lambda row: (-row['confidence'], row['id']))
        examples[name] = candidates[0] if candidates else None
    return dict(n=len(data), answerable=sum(not row['is_impossible'] for row in data),
                raw_em=raw_em, raw_em_count=sum(value[0] for value in raw_scores), raw_f1=raw_f1,
                raw_status_counts=dict(statuses), maximum_confidence_recomputation_error=max(confidence_errors),
                calibration_curve=curve, selected_threshold=min(feasible) if feasible else None,
                any_feasible=bool(feasible), parameter_storage=tensor_summary, examples=examples,
                inputs_sha256={name: sha(folder / name) for name in ('data.jsonl', 'predictions.jsonl', 'run.json', 'protocol.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('configs/qa-remediation/study.json'))
    parser.add_argument('--run', action='append', required=True, help='NAME=directory; only development or calibration')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = read(args.study)
    assert len(study['threshold_grid']) == 9 and len(set(study['threshold_grid'])) == 9
    result = dict(scope='Independent stdlib audit; no model, scorer, gate, or verification imports; no confirmation data opened.',
                  dataset_role='published_development' if args.development else 'calibration',
                  study_sha256=sha(args.study), script_sha256=sha(__file__),
                  constraints=study['quality_constraints'], runs={})
    for item in args.run:
        name, folder = item.split('=', 1)
        assert name and name not in result['runs']
        assert 'confirmation' not in name and 'confirmation' not in Path(folder).name
        result['runs'][name] = audit_run(Path(folder), study, args.development)
    values = result['runs']
    if set(values) == {'fp16', 'q8'} and all(value['parameter_storage'] for value in values.values()):
        result['q8_to_fp16_parameter_tensor_bytes_ratio'] = (values['q8']['parameter_storage']['parameter_tensor_bytes'] /
                                                            values['fp16']['parameter_storage']['parameter_tensor_bytes'])
        result['both_calibration_feasible'] = all(value['any_feasible'] for value in values.values())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({name: dict(raw_em=value['raw_em'], raw_f1=value['raw_f1'],
                                  any_feasible=value['any_feasible'], selected_threshold=value['selected_threshold'])
                      for name, value in values.items()}, indent=2))


if __name__ == '__main__':
    main()
