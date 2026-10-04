"""Audit disjoint-gap evidence without fitting or inference.

Re-enumerates legal spans and quality counts independently of the experimental
feature extractor and selective scorer. Stored training traces are consistency
checks, not an independent proof of the optimizer trajectory.
"""
import argparse
import math
from pathlib import Path

from lab.artifact_integrity import verify_hashes, safe_path
from lab.qa_metrics import normalize
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal, write

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def disjoint_gap(context, pred):
    alternatives = []
    for w in pred['raw_windows']:
        for i, (left0, end0) in enumerate(w['offsets']):
            if not w['context_mask'][i] or left0 == end0:
                continue
            for j in range(i, min(i + 30, len(w['offsets']))):
                if not all(w['context_mask'][i:j+1]):
                    break
                if w['offsets'][j][0] == w['offsets'][j][1]:
                    continue
                text0 = context[left0:w['offsets'][j][1]]
                text = text0.strip()
                if not text:
                    continue
                left = left0 + len(text0) - len(text0.lstrip())
                right = left + len(text)
                if left < pred['end'] and pred['start'] < right:
                    continue
                if normalize(text) == normalize(pred['prediction']):
                    continue
                k = w['cls_index']
                alternatives.append(w['start_logits'][i] + w['end_logits'][j] - w['start_logits'][k] - w['end_logits'][k])
    return pred['margin'] - max(alternatives) if alternatives else None


def metrics(data, predictions, threshold):
    mapped = {p['id']: p for p in predictions}
    require(len(mapped) == len(predictions) == len(data) and set(mapped) == {r['id'] for r in data}, 'Metric ID coverage')
    accepted = correct = answerable_accepted = impossible_accepted = invalid = 0
    answerable = sum(not r['is_impossible'] for r in data)
    for row in data:
        p = mapped[row['id']]; score = p['confidence']; text = p['prediction']
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, 'Invalid score')
        valid = bool(text and text.strip() and text != 'NO_ANSWER' and text.strip() in row['context'])
        invalid += not valid
        if valid and score >= threshold:
            accepted += 1
            answerable_accepted += not row['is_impossible']
            impossible_accepted += row['is_impossible']
            correct += any(normalize(text) == normalize(a) for a in row['answers'])
    impossible = len(data) - answerable
    return dict(n=len(data), accepted=accepted, accepted_correct=correct, answerable=answerable,
                unanswerable=impossible, accepted_answerable=answerable_accepted,
                accepted_unanswerable=impossible_accepted,
                accepted_precision=correct/accepted if accepted else None,
                answerable_answer_coverage=answerable_accepted/answerable,
                correct_answerable_coverage=correct/answerable,
                unanswerable_false_accept_rate=impossible_accepted/impossible, invalid_rate=invalid/len(data))


def passes(m, limits):
    return (m['accepted_precision'] is not None
            and m['accepted_precision'] >= limits['min_accepted_precision']
            and m['answerable_answer_coverage'] >= limits['min_answerable_answer_coverage']
            and m['correct_answerable_coverage'] >= limits['min_correct_answerable_coverage']
            and m['unanswerable_false_accept_rate'] <= limits['max_unanswerable_false_accept_rate']
            and m['invalid_rate'] <= limits['max_invalid_rate'])


def audit(folder):
    spec = read(folder / 'protocol.json')
    require(sha(folder / 'protocol.json') == sha(ROOT / 'configs/qa-coverage-gap/study.json'), 'Protocol identity')
    verify_hashes(folder, read(folder / 'checksums.json'), exclude=('checksums.json',))
    for name, digest in spec['source_sha256'].items():
        require(sha(safe_path(folder / 'source', name)) == digest, 'Source identity: ' + name)
    for name, digest in spec['input_sha256'].items():
        require(sha(safe_path(ROOT, name)) == digest, 'Input identity: ' + name)
    raw = {p['id']: p for split in ('calibration', 'evaluation')
           for p in rows(ROOT / f'results/qa-specialist-v1/{split}-int8/predictions.jsonl')}
    feature_audit = read(folder / 'feature-audit.json'); changed = {}; total = 0
    for split, suffix in (('train', 'training'), ('calibration', 'calibration')):
        data = rows(ROOT / f'configs/qa-risk/dataset/{split}/data.jsonl')
        archived = read(ROOT / f'results/qa-risk-v2/training/int8-{suffix}-features.json')
        base = read(folder / f'baseline/{suffix}-features.json')
        candidate = read(folder / f'disjoint/{suffix}-features.json')
        require(base == archived, 'Baseline matrix changed')
        cmap = {r['id']: r for r in candidate}; require(len(cmap) == len(candidate), 'Duplicate feature ID')
        changes = []; undefined = []; valid_ids = []
        for row, b in zip(data, base):
            require(row['id'] == b['id'], 'Matrix data order')
            p = raw[row['id']]
            require(b['target'] == int(any(normalize(p['prediction']) == normalize(a) for a in row['answers'])), 'Label mismatch')
            gap = disjoint_gap(row['context'], p)
            if gap is None:
                undefined.append(row['id']); continue
            valid_ids.append(row['id']); c = cmap[row['id']]
            require(math.isclose(gap, c['features'][2], rel_tol=0, abs_tol=1e-12), 'Disjoint feature mismatch')
            require(all(c['features'][j] == b['features'][j] for j in (0, 1, 3, 4)), 'Other features changed')
            require(c['target'] == b['target'] and c['prediction'] == b['prediction'], 'Candidate label/raw prediction changed')
            require(c['feature_names'][2] == 'disjoint_different_normalized_answer_margin_gap', 'Candidate semantic identity')
            if c['features'][2] != b['features'][2]: changes.append(row['id'])
            total += 1
        require([r['id'] for r in candidate] == valid_ids, 'Candidate matrix coverage/order')
        require(feature_audit[split]['changed_ids'] == changes and feature_audit[split]['changed_count'] == len(changes), 'Change count')
        require(feature_audit[split]['undefined_ids'] == undefined, 'Undefined count')
        changed[split] = len(changes)
    cal = rows(ROOT / 'configs/qa-risk/dataset/calibration/data.jsonl')
    compared = read(folder / 'comparison.json'); selected = {}
    for method in ('baseline', 'disjoint'):
        status = read(folder / method / 'status.json')
        require(status == compared['methods'][method], 'Arm status mismatch')
        if status['status'] != 'complete':
            require(status['status'] in ('feature_failed', 'fit_failed'), 'Unknown arm state')
            selected[method] = None; continue
        fit = read(folder / method / 'fit.json')
        require(not any(k in fit for k in ('weights', 'intercept', 'scaler')), 'Parameter distribution forbidden')
        require(fit['convergence']['converged'] and fit['trace'][-1]['gradient_max_abs'] <= spec['fit_config']['gradient_max_abs_tolerance'], 'Convergence record')
        prediction = read(folder / method / 'calibration-predictions.json')
        original = {r['id']: r['prediction'] for r in read(folder / method / 'calibration-features.json')}
        require(all(p['prediction'] == original[p['id']] for p in prediction), 'Scored raw answer changed')
        selection = read(folder / method / 'selection.json'); feasible = []
        require([c['threshold'] for c in selection['candidates']] == spec['threshold_grid'], 'Threshold grid')
        for c in selection['candidates']:
            actual = metrics(cal, prediction, c['threshold'])
            require(aggregates_equal(actual, {k: c['summary']['selective'][k] for k in actual}), 'Calibration counts')
            ok = passes(actual, spec['quality_constraints'])
            require(c['gate']['all_pass'] == ok, 'Quality gate')
            if ok: feasible.append(c)
        chosen = feasible[0] if feasible else None
        require(selection['eligible'] == bool(feasible) and selection['threshold'] == (chosen['threshold'] if chosen else None), 'Threshold selection')
        require(status['selected'] == chosen, 'Status selection')
        selected[method] = chosen
    b, c = (selected[k] for k in ('baseline', 'disjoint'))
    eligible = bool(b and c and all(c['summary']['selective'][k] - b['summary']['selective'][k] >= spec['min_coverage_gain'] for k in ('answerable_answer_coverage', 'correct_answerable_coverage')))
    require(compared['retrospective_replay_permitted'] == eligible, 'Replay gate')
    state = read(folder / 'run.json')
    require(state['status'] == 'complete' and state['candidate_calibration_improved'] == eligible and state['retrospective_replay_run'] == eligible, 'Run state')
    require((folder / 'retrospective').exists() == eligible, 'Replay presence')
    if eligible:
        data = rows(ROOT / 'results/qa-risk-v2/evaluation-int8/data.jsonl')
        summary = read(folder / 'retrospective/summary.json')['methods']
        for method in ('baseline', 'disjoint'):
            actual = metrics(data, read(folder / f'retrospective/{method}-predictions.json'), selected[method]['threshold'])
            require(aggregates_equal(actual, {k: summary[method]['summary']['selective'][k] for k in actual}), 'Retrospective counts')
    return dict(evidence_valid=True, candidate_calibration_improved=eligible, independently_checked_feature_rows=total,
                changed_feature_rows=changed, calibration_selected_thresholds={k: v['threshold'] if v else None for k,v in selected.items()},
                scope='Independent span-gap and metric arithmetic; no optimizer refit, inference, unseen evaluation or production permission',
                protocol_sha256=sha(folder / 'protocol.json'), run_sha256=sha(folder / 'run.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'results/qa-coverage-gap-v1')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = audit(a.root); write(a.output, result); print(result)
