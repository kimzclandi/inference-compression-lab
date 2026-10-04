"""Exact pruning for the frozen all-window risk features.

The pinned reference stays unchanged for historical reproduction. This small
specialization retains exhaustive legal-span traversal and decoder validation,
but avoids text normalization for provably dominated finite-margin candidates.
Worst-case traversal remains O(windows * tokens * max_answer_tokens).
"""
import math
import sys

from lab.extractive_qa import decode
from lab.qa_metrics import normalize
from lab.qa_risk_calibration import FEATURE_NAMES, _finite, _vector, _logsumexp


def extract_features(context, prediction):
    """Return the frozen five features with exact competitor-score pruning.

    ``prediction`` must contain the complete frozen decoder result plus
    ``raw_windows``. Its chosen span is checked against a fresh decoding of all
    raw windows. Windows are never silently dropped. A context with no legal
    differently normalized candidate is rejected: its margin gap is undefined.
    Zero-width context tokens may be internal to a span and count toward the
    fixed 30-token budget; they can never be endpoints.
    """
    if not isinstance(prediction, dict):
        raise ValueError('prediction must be a decoder record.')
    windows = prediction.get('raw_windows')
    selected = decode(context, windows, max_answer_tokens=30)
    keys = ('prediction', 'start', 'end', 'start_token', 'end_token',
            'window_index', 'margin', 'span_score', 'context_sha256', 'max_answer_tokens')
    for key in keys:
        if type(prediction.get(key)) is not type(selected[key]) or prediction[key] != selected[key]:
            raise ValueError('Stored prediction does not reproduce from raw logits: ' + key)
    chosen = windows[selected['window_index']]
    indices = [i for i, flag in enumerate(chosen['context_mask'])
               if flag or i == chosen['cls_index']]
    start_lse = _logsumexp([chosen['start_logits'][i] for i in indices])
    end_lse = _logsumexp([chosen['end_logits'][i] for i in indices])
    log_probability = _finite(selected['span_score'] - start_lse - end_lse,
                              'joint log probability')
    selected_normalized = normalize(selected['prediction'])
    alternative = None
    for window_index, window in enumerate(windows):
        mask = window['context_mask']
        offsets = window['offsets']
        start_logits, end_logits = window['start_logits'], window['end_logits']
        cls = window['cls_index']
        null_score = _finite(start_logits[cls] + end_logits[cls], 'null score')
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
                if left == right:
                    continue
                # Normalization cannot promote a candidate whose margin cannot
                # beat the first best *different* answer. Keep nonrepresentable
                # arithmetic on the reference path: duplicate answers may be
                # ignored there, but nonduplicate overflow must still fail.
                raw_margin = start_logits[first] + end_logits[last] - null_score
                if (alternative is not None and
                        -sys.float_info.max <= raw_margin <= alternative['margin']):
                    continue
                text = context[left:right]
                normalized = normalize(text)
                if normalized == selected_normalized:
                    continue
                margin = _finite(start_logits[first] + end_logits[last] - null_score,
                                 'alternative margin')
                if alternative is None or margin > alternative['margin']:
                    alternative = dict(prediction=text, normalized=normalized, margin=margin,
                                       window_index=window_index, start_token=first, end_token=last,
                                       start=left, end=right)
    if alternative is None:
        raise ValueError('No differently normalized legal span; margin-gap feature is undefined.')
    gap = _finite(selected['margin'] - alternative['margin'], 'alternative margin gap')
    features = _vector([
        selected['margin'], log_probability, gap,
        math.log1p(selected['end_token'] - selected['start_token'] + 1),
        math.log1p(len(windows)),
    ], 'features')
    return dict(features=features, feature_names=list(FEATURE_NAMES),
                diagnostics=dict(context_sha256=selected['context_sha256'],
                                 selected_window_index=selected['window_index'],
                                 selected_normalized_answer=selected_normalized,
                                 start_logsumexp=start_lse, end_logsumexp=end_lse,
                                 normalization_token_count=len(indices),
                                 alternative=alternative))

