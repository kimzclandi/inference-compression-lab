"""Independent, stdlib-only audit of the 202 published QA cases.

Run from repository root. Imports no repository scorer or generation code.
This does not run a model, change any historical data, or define a new confirmation.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import string

DEST = Path(__file__).resolve().parent
VARIANTS = ('fp16', 'q4', 'q8', 'selected', 'control')
DATASETS = ('qwen-quantization-v1', 'qwen-confirmation-v1')


def load(path):
    return json.loads(path.read_text())


def normalize(text):
    # Independent transcription of the published SQuAD normalization definition.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', re.sub(r'\b(?:a|an|the)\b', ' ', text)).strip()


def pair(prediction, answer):
    pred, gold = normalize(prediction).split(), normalize(answer).split()
    em = int(pred == gold)
    common = sum(min(pred.count(t), gold.count(t)) for t in set(pred))
    return em, (2 * common / (len(pred) + len(gold)) if pred and gold else em)


def score(row, raw):
    raw = raw.strip()
    abstain = raw == 'NO_ANSWER'
    valid = bool(raw) and (abstain or raw in row['context'])
    if abstain:
        em = f1 = int(row['is_impossible'])
    elif not raw or row['is_impossible']:
        em = f1 = 0
    else:
        options = [pair(raw, answer) for answer in row['answers']]
        em = max(x[0] for x in options)
        f1 = max(x[1] for x in options)
    return dict(em=em, f1=f1, format_valid=valid, abstain=abstain)


def summary(items):
    return dict(n=len(items), em_count=sum(x['em'] for x in items),
                em=sum(x['em'] for x in items) / len(items),
                f1=sum(x['f1'] for x in items) / len(items),
                abstain_count=sum(x['abstain'] for x in items),
                format_valid_count=sum(x['format_valid'] for x in items))


all_records, report, inputs = [], {}, {}
for dataset in DATASETS:
    base = Path('results') / dataset
    rows = [json.loads(line) for line in (base / 'data.jsonl').read_text().splitlines()]
    by_id = {row['id']: row for row in rows}
    assert len(by_id) == len(rows)
    files = [base / 'data.jsonl', base / 'prompt.json', base / 'protocol.json']
    values = {v: load(base / f'{v}-quality.json') for v in VARIANTS}
    files.extend(base / f'{v}-quality.json' for v in VARIANTS)
    inputs.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    ref_preds = {p['id']: p for p in values['fp16']['predictions']}
    recomputed = {}
    for variant, value in values.items():
        predictions = {p['id']: p for p in value['predictions']}
        scored = {s['id']: s for s in value['scored']}
        assert len(predictions) == len(value['predictions']) == len(rows)
        assert set(predictions) == set(scored) == set(by_id)
        records = []
        for sample_id, row in by_id.items():
            pred = predictions[sample_id]
            actual = score(row, pred['prediction'])
            for metric in ('em', 'f1', 'format_valid', 'abstain'):
                assert abs(actual[metric] - scored[sample_id][metric]) < 1e-12
            for key in ('input_tokens', 'input_token_ids_sha256', 'prompt_sha256'):
                assert pred[key] == ref_preds[sample_id][key]
            record = dict(dataset=dataset, variant=variant, id=sample_id,
                          is_impossible=row['is_impossible'], **actual,
                          prediction=pred['prediction'], answers=row['answers'],
                          question=row['question'], source_title=row['source_title'],
                          generated_tokens=pred['generated_tokens'], stop_reason=pred['stop_reason'],
                          raw_equals_fp16=pred['prediction'] == ref_preds[sample_id]['prediction'])
            records.append(record)
        all_records.extend(records)
        recomputed[variant] = records
    report[dataset] = dict(n=len(rows), answerable_n=sum(not r['is_impossible'] for r in rows),
                           unanswerable_n=sum(r['is_impossible'] for r in rows), variants={})
    ref = {r['id']: r for r in recomputed['fp16']}
    for variant, records in recomputed.items():
        invalid = [r for r in records if not r['format_valid']]
        invalid_reject_scores = [score(by_id[r['id']], r['prediction'] if r['format_valid'] else 'NO_ANSWER')
                                 for r in records]
        report[dataset]['variants'][variant] = dict(
            all=summary(records),
            answerable=summary([r for r in records if not r['is_impossible']]),
            unanswerable=summary([r for r in records if r['is_impossible']]),
            fp16_em_gained=[r['id'] for r in records if r['em'] > ref[r['id']]['em']],
            fp16_em_lost=[r['id'] for r in records if r['em'] < ref[r['id']]['em']],
            raw_prediction_changes_vs_fp16=sum(not r['raw_equals_fp16'] for r in records),
            stop_reasons=dict(Counter(r['stop_reason'] for r in records)),
            invalid_reject_counterfactual=dict(
                scope='Offline deterministic simulation on published outputs, NOT new model evidence or deployment validation.',
                gains=sum(r['is_impossible'] for r in invalid),
                losses=sum(r['em'] for r in invalid),
                all=summary(invalid_reject_scores)),
        )

overall = {v: summary([r for r in all_records if r['variant'] == v]) for v in VARIANTS}
result = dict(scope='202 published cases, five variants; no model run; scorer unchanged.',
              verified_scored_records=len(all_records), prompt_and_input_identity_across_variants=True,
              by_dataset=report, combined=overall, input_sha256=inputs,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(DEST / 'recomputed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
(DEST / 'all-records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_records))
print(json.dumps(dict(verified_scored_records=len(all_records), combined=overall), indent=2))
