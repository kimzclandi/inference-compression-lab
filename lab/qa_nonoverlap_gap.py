"""Experimental third feature: compete with disjoint answer spans.

No labels enter this function. This is not the default serving feature. Exact
character-interval disjointness removes boundary variants, but can also hide
genuine overlapping alternatives; its quality must be measured separately.
"""
from copy import deepcopy
import math

from lab.qa_metrics import normalize
from lab.qa_risk_calibration import extract_features


def extract_nonoverlap_features(context, prediction):
    """Validate all raw windows and replace only column 2 of five features.

    An alternative must have a different normalized answer AND no character
    overlap with the selected answer, even when it comes from another window.
    Touching half-open intervals are disjoint. Missing alternatives fail closed;
    there is no imputation, row dropping, raw-answer change or label access.
    """
    original = extract_features(context, prediction)
    selected = original['diagnostics']['selected_normalized_answer']
    best = None
    for wi, window in enumerate(prediction['raw_windows']):
        mask, offsets = window['context_mask'], window['offsets']
        start, end = window['start_logits'], window['end_logits']
        cls = window['cls_index']
        null = start[cls] + end[cls]
        for first in range(len(mask)):
            if not mask[first] or offsets[first][0] == offsets[first][1]:
                continue
            for last in range(first, min(first + 30, len(mask))):
                if not mask[last]:
                    break
                if offsets[last][0] == offsets[last][1]:
                    continue
                left, right = offsets[first][0], offsets[last][1]
                while left < right and context[left].isspace():
                    left += 1
                while left < right and context[right - 1].isspace():
                    right -= 1
                if left == right or max(left, prediction['start']) < min(right, prediction['end']):
                    continue
                text = context[left:right]
                normalized = normalize(text)
                if normalized == selected:
                    continue
                margin = start[first] + end[last] - null
                if not math.isfinite(margin):
                    raise ValueError('Non-finite disjoint alternative margin')
                if best is None or margin > best['margin']:
                    best = dict(prediction=text, normalized=normalized, margin=margin,
                                window_index=wi, start_token=first, end_token=last,
                                start=left, end=right)
    if best is None:
        raise ValueError('No disjoint differently normalized legal span')
    gap = prediction['margin'] - best['margin']
    if not math.isfinite(gap) or gap < original['features'][2]:
        raise ValueError('Disjoint-set margin gap is invalid')
    result = deepcopy(original)
    result['features'][2] = gap
    result['feature_names'][2] = 'disjoint_different_normalized_answer_margin_gap'
    result['diagnostics']['alternative'] = best
    result['diagnostics']['competitor_scope'] = 'all_windows_disjoint_character_intervals'
    return result
