"""Decode task-trained QA start/end logits without labels or generation.

Offsets are Python Unicode character offsets into the supplied ``context``;
their end is exclusive. A window contains equally long JSON-style lists of
``start_logits``, ``end_logits``, ``offsets`` and ``context_mask``, plus the
integer ``cls_index`` used by the model for its null-answer score. Question,
special and padding tokens must have a false context mask.

Each window contributes its highest start-plus-end scoring nonempty context
span of at most ``max_answer_tokens`` tokens. The chosen window maximizes that
span's score minus its own CLS null score. Ties prefer the lower window index,
then lower start and end token indices. Confidence is a stable sigmoid of the
margin: it is an ordering signal, NOT a calibrated correctness probability.
This module returns a span candidate even for a negative margin. A caller must
apply its separately calibrated, frozen acceptance threshold before serving it.
"""

import hashlib
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite scalar number.')
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'{name} must fit a finite scalar number.') from error
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be a finite scalar number.')
    return converted


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer, not a boolean.')
    return value


def _finite_arithmetic(value, name):
    if not math.isfinite(value):
        raise ValueError(f'{name} overflowed finite arithmetic.')
    return value


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate_window(context, window, index):
    name = f'windows[{index}]'
    if not isinstance(window, dict):
        raise ValueError(f'{name} must be a dictionary.')
    arrays = ('start_logits', 'end_logits', 'offsets', 'context_mask')
    if any(not isinstance(window.get(key), list) for key in arrays):
        raise ValueError(f'{name} requires four list-valued token arrays.')
    length = len(window['start_logits'])
    if length == 0 or any(len(window[key]) != length for key in arrays):
        raise ValueError(f'{name} requires nonempty, equal-length token arrays.')
    cls = _integer(window.get('cls_index'), f'{name}.cls_index')
    if not 0 <= cls < length:
        raise ValueError(f'{name}.cls_index is outside the token arrays.')
    start = [_number(value, f'{name}.start_logits[{i}]')
             for i, value in enumerate(window['start_logits'])]
    end = [_number(value, f'{name}.end_logits[{i}]')
           for i, value in enumerate(window['end_logits'])]
    mask = window['context_mask']
    if any(type(value) is not bool for value in mask):
        raise ValueError(f'{name}.context_mask must contain only booleans.')
    if mask[cls]:
        raise ValueError(f'{name} CLS token cannot also be a context token.')
    if not any(mask):
        raise ValueError(f'{name} has no context tokens.')
    offsets = []
    for i, pair in enumerate(window['offsets']):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'{name}.offsets[{i}] must be a two-integer pair.')
        left = _integer(pair[0], f'{name}.offsets[{i}][0]')
        right = _integer(pair[1], f'{name}.offsets[{i}][1]')
        if left < 0 or right < left:
            raise ValueError(f'{name}.offsets[{i}] has invalid ordering.')
        # Non-context offsets may refer to the question, whose length can
        # exceed context length. Never interpret them as context positions.
        if mask[i] and not (left <= right <= len(context)):
            raise ValueError(f'{name}.offsets[{i}] is outside the context.')
        if i and mask[i] and mask[i - 1]:
            prior_left, prior_right = offsets[-1]
            # Duplicate offsets are valid for byte-level subword pieces of
            # one Unicode character. Reversed positions are never valid.
            if left < prior_left or right < prior_right:
                raise ValueError(f'{name} context offsets must be monotonic.')
        offsets.append((left, right))
    return start, end, offsets, mask, cls


def decode(context, windows, max_answer_tokens=30):
    """Return a best exact span and auditable window scores; fail closed.

    No QA labels, gold strings or answerability labels enter this function.
    Any malformed window invalidates the whole request rather than silently
    dropping a window or emitting a candidate from incomplete evidence.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError('context must be a nonempty, non-whitespace string.')
    if not isinstance(windows, list) or not windows:
        raise ValueError('windows must be a nonempty list.')
    max_answer_tokens = _integer(max_answer_tokens, 'max_answer_tokens')
    if max_answer_tokens < 1:
        raise ValueError('max_answer_tokens must be positive.')
    candidates = []
    for window_index, window in enumerate(windows):
        start, end, offsets, mask, cls = _validate_window(context, window, window_index)
        null_score = _finite_arithmetic(start[cls] + end[cls], 'null score')
        best = None
        for first in range(len(mask)):
            # Fast byte-level tokenizers can retain zero-width tokens for
            # whitespace. They count toward a span's token budget but cannot
            # be an answer endpoint, even if their logits are the largest.
            if not mask[first] or offsets[first][0] == offsets[first][1]:
                continue
            for last in range(first, min(first + max_answer_tokens, len(mask))):
                # A false token anywhere inside the candidate breaks the
                # span, even if both of its endpoints would be context.
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
                span_score = _finite_arithmetic(start[first] + end[last], 'span score')
                if best is None or span_score > best['span_score']:
                    best = dict(prediction=context[left:right], start=left, end=right,
                                start_token=first, end_token=last, span_score=span_score,
                                start_logit=start[first], end_logit=end[last])
        if best is None:
            raise ValueError(f'windows[{window_index}] has no nonempty legal context span.')
        margin = _finite_arithmetic(best['span_score'] - null_score, 'span/null margin')
        best.update(window_index=window_index, null_score=null_score, margin=margin,
                    cls_start_logit=start[cls], cls_end_logit=end[cls],
                    confidence=_sigmoid(margin))
        candidates.append(best)
    # max returns the first equal key, preserving the specified window tie.
    result = dict(max(candidates, key=lambda candidate: candidate['margin']))
    result.update(context_sha256=hashlib.sha256(context.encode('utf-8')).hexdigest(),
                  max_answer_tokens=max_answer_tokens, windows=candidates)
    return result
