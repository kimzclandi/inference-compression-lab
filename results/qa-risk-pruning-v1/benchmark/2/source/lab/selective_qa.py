"""Fail-closed extractive QA contract and calibration-only selective metrics.

Only ``NO_ANSWER`` (after stripping surrounding whitespace) is a model
abstention. Any other nonempty output must be an exact context substring. No
gold answer is used while parsing or deciding whether to emit an answer.

Offsets are Python Unicode character offsets with an exclusive end. If a span
occurs more than once, the first occurrence is recorded; this is provenance,
not evidence that the model selected the correct occurrence.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math

from lab.qa_metrics import evaluate


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar number.')
    return float(value)


def _unit_interval(value, name):
    value = _finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be between zero and one.')
    return value


class CalibrationError(ValueError):
    """No deployable candidate; retain the complete failed calibration curve."""

    def __init__(self, candidates, constraints, threshold_grid):
        super().__init__('No feasible threshold on calibration data; deployment gate remains closed.')
        self.candidates = candidates
        self.constraints = constraints
        self.threshold_grid = threshold_grid


def parse_output(context, raw_output):
    """Parse a model string into ``answer``, ``abstain``, or ``invalid``.

    Input: context and raw model output strings. Output: a JSON-safe record
    with the original string, stripped string, status, span, and context hash.
    Invalid outputs have no answer and must never be emitted by the caller.
    """
    if not isinstance(context, str) or not isinstance(raw_output, str):
        raise ValueError('context and raw_output must be strings.')
    output = raw_output.strip()
    result = dict(raw_output=raw_output, stripped_output=output,
                  context_sha256=hashlib.sha256(context.encode('utf-8')).hexdigest(),
                  status='invalid', answer=None, start=None, end=None,
                  reason='empty_output' if not output else 'not_exact_context_span')
    if output == 'NO_ANSWER':
        result.update(status='abstain', reason='model_abstention')
    elif output and output in context:
        start = context.index(output)
        result.update(status='answer', answer=output, start=start,
                      end=start + len(output), reason='exact_context_span')
    return result


def apply_threshold(parsed, confidence=None, threshold=None):
    """Apply a frozen scalar threshold; acceptance uses ``confidence >= threshold``.

    Confidence and threshold must be finite numbers in [0, 1]. Confidence is
    an ordering signal, not assumed to be a calibrated probability. Only valid
    span candidates can be accepted. Invalid outputs
    remain invalid even when the system safely suppresses them; they are not
    converted to a scored correct refusal.
    """
    if parsed.get('status') not in {'answer', 'abstain', 'invalid'}:
        raise ValueError('Unknown parsed output status.')
    if threshold is not None:
        threshold = _unit_interval(threshold, 'threshold')
        confidence = _unit_interval(confidence, 'confidence')
    elif confidence is not None:
        confidence = _unit_interval(confidence, 'confidence')
    result = deepcopy(parsed)
    result.update(model_status=parsed['status'], confidence=confidence,
                  threshold=threshold, rejection_reason=None)
    if parsed['status'] == 'answer' and threshold is not None and confidence < threshold:
        result.update(status='abstain', answer=None, start=None, end=None,
                      reason='below_confidence_threshold',
                      rejection_reason='below_confidence_threshold')
    if parsed['status'] == 'invalid':
        result['rejection_reason'] = parsed['reason']
    result['accepted'] = result['status'] == 'answer'
    result['action'] = 'answer' if result['accepted'] else 'abstain'
    # Empty invalid predictions score zero under the historical metric. Using
    # NO_ANSWER here would incorrectly reward malformed output on impossible QA.
    result['scoring_prediction'] = (result['answer'] if result['accepted'] else
                                    'NO_ANSWER' if result['status'] == 'abstain' else '')
    return result


def _validate_coverage(rows, predictions):
    if not rows:
        raise ValueError('Metrics require a nonempty dataset.')
    ids = [row['id'] for row in rows]
    pred_ids = [prediction['id'] for prediction in predictions]
    if (len(ids) != len(set(ids)) or len(pred_ids) != len(set(pred_ids))
            or set(ids) != set(pred_ids)):
        raise ValueError('Duplicate, missing or extra IDs: metrics require exact coverage.')
    if any(type(row['is_impossible']) is not bool for row in rows):
        raise ValueError('is_impossible must be a boolean.')


def evaluate_selective(rows, predictions, threshold=None):
    """Return ``(summary, per_example)`` for exactly covered QA IDs.

    Each prediction contains ``id`` and ``prediction``; a finite ``confidence``
    is also required for every row when a threshold is supplied. ``raw`` uses
    the unchanged historical qa_metrics score on the original model strings.
    ``system`` uses the same score after applying the output contract and
    threshold. ``selective`` is based on emitted valid spans, never on raw EM.

    Rates: accepted_precision = exact-correct accepted / all accepted;
    answerable_answer_coverage = accepted answerable / all answerable (NOT an
    accuracy measure); unanswerable_false_accept_rate = accepted unanswerable /
    all unanswerable; valid_span_rate = raw exact span candidates / all rows;
    invalid_rate = raw malformed/nonextractive outputs / all rows. Empty
    denominators are None, never perfect precision or zero risk.
    """
    rows, predictions = list(rows), list(predictions)
    _validate_coverage(rows, predictions)
    by_id = {prediction['id']: prediction for prediction in predictions}
    parsed = [parse_output(row['context'], by_id[row['id']]['prediction']) for row in rows]
    decisions = [apply_threshold(output, by_id[row['id']].get('confidence'), threshold)
                 for row, output in zip(rows, parsed)]
    raw, raw_scored = evaluate(rows, predictions)
    system_predictions = [dict(id=row['id'], prediction=decision['scoring_prediction'])
                          for row, decision in zip(rows, decisions)]
    system, system_scored = evaluate(rows, system_predictions)
    n = len(rows)
    answerable = sum(not row['is_impossible'] for row in rows)
    unanswerable = n - answerable
    accepted = sum(decision['accepted'] for decision in decisions)
    accepted_answerable = sum(decision['accepted'] and not row['is_impossible']
                              for row, decision in zip(rows, decisions))
    accepted_unanswerable = accepted - accepted_answerable
    accepted_correct = sum(decision['accepted'] and score['em'] == 1.0
                           for score, decision in zip(system_scored, decisions))
    counts = Counter(output['status'] for output in parsed)
    selective = dict(
        n=n, accepted=accepted, accepted_correct=accepted_correct,
        answerable=answerable, unanswerable=unanswerable,
        accepted_answerable=accepted_answerable,
        accepted_unanswerable=accepted_unanswerable,
        accepted_precision=accepted_correct / accepted if accepted else None,
        answerable_answer_coverage=accepted_answerable / answerable if answerable else None,
        correct_answerable_coverage=accepted_correct / answerable if answerable else None,
        unanswerable_false_accept_rate=accepted_unanswerable / unanswerable if unanswerable else None,
        acceptance_rate=accepted / n,
        valid_span_rate=counts['answer'] / n,
        invalid_rate=counts['invalid'] / n,
        model_abstention_rate=counts['abstain'] / n,
        threshold_abstention_rate=sum(d['rejection_reason'] == 'below_confidence_threshold'
                                      for d in decisions) / n,
        model_status_counts={key: counts[key] for key in ('answer', 'abstain', 'invalid')},
    )
    per_example = [dict(id=row['id'], is_impossible=row['is_impossible'],
                        model=output, decision=decision,
                        raw_score=raw_score, system_score=system_score)
                   for row, output, decision, raw_score, system_score
                   in zip(rows, parsed, decisions, raw_scored, system_scored)]
    return dict(raw=raw, system=system, selective=selective, threshold=threshold), per_example


def calibrate_threshold(rows, predictions, thresholds, *, min_accepted_precision,
                        min_answerable_answer_coverage,
                        max_unanswerable_false_accept_rate, max_invalid_rate):
    """Select the smallest feasible threshold from a predeclared calibration grid.

    Call only on the calibration partition, then freeze the returned threshold
    before any evaluation-partition inference or scoring. These are empirical
    calibration constraints, not population-risk confidence bounds. The caller
    owns split isolation and must record the grid before looking at outcomes.
    No feasible candidate raises CalibrationError (a ValueError subclass)
    carrying candidates, constraints, and threshold_grid; there is no
    maximum-EM fallback.
    Positive answerable coverage and at least one accepted output are required
    so that an always-abstain system cannot pass calibration.
    """
    constraints = dict(min_accepted_precision=min_accepted_precision,
                       min_answerable_answer_coverage=min_answerable_answer_coverage,
                       max_unanswerable_false_accept_rate=max_unanswerable_false_accept_rate,
                       max_invalid_rate=max_invalid_rate)
    for name, value in constraints.items():
        _unit_interval(value, name)
    if min_answerable_answer_coverage <= 0:
        raise ValueError('min_answerable_answer_coverage must be positive; all-abstain is not feasible.')
    grid = sorted(set(_unit_interval(value, 'threshold') for value in thresholds))
    if not grid:
        raise ValueError('A nonempty predeclared threshold grid is required.')
    rows, predictions = list(rows), list(predictions)
    _validate_coverage(rows, predictions)
    if not any(row['is_impossible'] for row in rows) or not any(not row['is_impossible'] for row in rows):
        raise ValueError('Calibration requires both answerable and unanswerable examples.')
    candidates = []
    for threshold in grid:
        summary, _ = evaluate_selective(rows, predictions, threshold)
        metrics = summary['selective']
        feasible = (metrics['accepted'] > 0
                    and metrics['accepted_precision'] >= min_accepted_precision
                    and metrics['answerable_answer_coverage'] >= min_answerable_answer_coverage
                    and metrics['unanswerable_false_accept_rate'] <= max_unanswerable_false_accept_rate
                    and metrics['invalid_rate'] <= max_invalid_rate)
        candidates.append(dict(threshold=threshold, feasible=feasible, metrics=metrics))
    feasible = [candidate for candidate in candidates if candidate['feasible']]
    if not feasible:
        raise CalibrationError(candidates, constraints, grid)
    return dict(threshold=feasible[0]['threshold'], constraints=constraints,
                threshold_grid=grid, candidates=candidates,
                selection_rule='smallest_feasible_calibration_threshold',
                risk_scope='empirical_calibration_only')
