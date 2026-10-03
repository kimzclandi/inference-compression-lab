"""Offline interview demonstration: a valid span can still be a wrong answer."""
import json
from pathlib import Path

from experiments.verify_qa_remediation import verify
from lab.quantization_diagnostics import read, rows
from lab.selective_qa import parse_output


def main():
    root = Path('results/qa-remediation-v1')
    verified = verify(root, Path('configs/qa-remediation/study.json'))
    identifier = '5ad0483977cf76001a686f8a'
    data = {row['id']: row for row in rows(root/'calibration-q8/data.jsonl')}
    predictions = {row['id']: row for row in rows(root/'calibration-q8/predictions.jsonl')}
    row, prediction = data[identifier], predictions[identifier]
    selection = read(root/'selection.json')
    curve = [dict(threshold=item['threshold'], feasible=item['feasible'],
                  accepted_precision=item['metrics']['selective']['accepted_precision'],
                  answerable_coverage=item['metrics']['selective']['answerable_answer_coverage'],
                  false_accept_rate=item['metrics']['selective']['unanswerable_false_accept_rate'])
             for item in selection['variants']['q8']['curve']]
    output = dict(scope='Offline replay of published calibration evidence; no model inference or new confirmation.',
                  research_status=verified['status'], quality_passed=verified['overall_success'],
                  example=dict(id=identifier, context=row['context'], question=row['question'],
                               gold_is_impossible=row['is_impossible'], prediction=prediction['prediction'],
                               raw_span_contract=parse_output(row['context'], prediction['prediction']),
                               uncalibrated_same_model_audit=prediction['audit'],
                               explanation='An exact substring and a high same-model score do not fix the reversed containment relation.'),
                  calibration_curve=curve, selected_threshold=selection['variants']['q8']['threshold'],
                  answer_entrypoint=verified['answer_entrypoint'])
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
