"""One fixed supervised correctness head on frozen extractive-QA evidence.

Feature extraction and scoring use only the Python standard library. NumPy is
imported only by ``fit``. Neither feature extraction nor prediction takes QA
gold answers or answerability labels. Training labels are explicit binary
correctness outcomes, supplied separately to ``fit``.

The fitted logistic score is NOT a guaranteed calibrated probability of answer
correctness. Threshold selection and evaluation require separate, fixed data.
No hyperparameters or feature choices are configurable in this module.
"""

import math

from lab.extractive_qa import decode
from lab.qa_metrics import normalize


FEATURE_NAMES = (
    'selected_span_null_margin',
    'selected_joint_context_cls_log_probability',
    'different_normalized_answer_margin_gap',
    'log1p_answer_token_count',
    'log1p_window_count',
)
FIT_CONFIG = dict(
    objective='sum_binary_logistic_loss_plus_half_l2_weight_squared',
    l2=1.0,
    regularize_intercept=False,
    standard_deviation_ddof=0,
    zero_variance_scale=1.0,
    max_iterations=100,
    gradient_max_abs_tolerance=1e-8,
    armijo=1e-4,
    backtracking_factor=0.5,
    max_backtracking_steps=40,
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number, not a boolean.')
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'{name} cannot be represented by a finite float.') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != len(FEATURE_NAMES):
        raise ValueError(f'{name} must contain exactly five numbers.')
    return [_finite(item, f'{name}[{index}]') for index, item in enumerate(value)]


def _logsumexp(values):
    maximum = max(values)
    return _finite(maximum + math.log(math.fsum(math.exp(value - maximum) for value in values)),
                   'logsumexp')


def extract_features(context, prediction):
    """Return five fixed label-free features, names and reproducible diagnostics.

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


def fit(features, labels):
    """Fit the fixed five-feature logistic head; return auditable JSON values.

    Objective: sum of binary logistic losses + 0.5 * ||weights||². The
    intercept is not regularized. Population means/stds are fitted only from
    this training matrix; zero-variance columns use scale one. Newton steps use
    the analytic gradient/Hessian and fixed Armijo backtracking. Any singular,
    non-finite or nonconverged fit fails closed with ValueError.
    """
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError('Training features must be a nonempty matrix.')
    matrix = [_vector(row, f'features[{index}]') for index, row in enumerate(features)]
    if not isinstance(labels, (list, tuple)) or len(labels) != len(matrix):
        raise ValueError('Training labels require exact nonempty row coverage.')
    outcomes = [_finite(value, f'labels[{index}]') for index, value in enumerate(labels)]
    if set(outcomes) != {0.0, 1.0}:
        raise ValueError('Training labels must contain both binary classes, zero and one.')

    import numpy as np

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            x = np.asarray(matrix, dtype=np.float64)
            y = np.asarray(outcomes, dtype=np.float64)
            mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
            scale = np.where(scale == 0, 1.0, scale)
            standardized = (x - mean) / scale
            design = np.column_stack((np.ones(len(x)), standardized))
            if not np.isfinite(design).all():
                raise ValueError('Standardization produced non-finite values.')
            beta = np.zeros(len(FEATURE_NAMES) + 1, dtype=np.float64)
            regularization = np.diag([0.0] + [FIT_CONFIG['l2']] * len(FEATURE_NAMES))

            def evaluate(parameters):
                logits = design @ parameters
                # logaddexp prevents exp overflow; select the sign by the
                # binary label to avoid subtracting two enormous numbers.
                loss = np.logaddexp(0.0, np.where(y == 1, -logits, logits)).sum()
                loss += 0.5 * FIT_CONFIG['l2'] * (parameters[1:] @ parameters[1:])
                probability = np.exp(-np.logaddexp(0.0, -logits))
                gradient = design.T @ (probability - y) + regularization @ parameters
                hessian = design.T @ ((probability * (1 - probability))[:, None] * design)
                hessian += regularization
                if not (np.isfinite(loss) and np.isfinite(gradient).all() and np.isfinite(hessian).all()):
                    raise ValueError('Non-finite logistic objective or derivatives.')
                return float(loss), gradient, hessian

            trace = []
            converged = False
            for iteration in range(FIT_CONFIG['max_iterations'] + 1):
                objective, gradient, hessian = evaluate(beta)
                maximum_gradient = float(np.abs(gradient).max())
                trace.append(dict(iteration=iteration, objective=objective,
                                  gradient_max_abs=maximum_gradient, step_size=None))
                if maximum_gradient <= FIT_CONFIG['gradient_max_abs_tolerance']:
                    converged = True
                    break
                if iteration == FIT_CONFIG['max_iterations']:
                    break
                direction = np.linalg.solve(hessian, gradient)
                decrement = float(gradient @ direction)
                if not np.isfinite(direction).all() or not math.isfinite(decrement) or decrement <= 0:
                    raise ValueError('Newton direction is not finite and strictly descending.')
                step = 1.0
                accepted = False
                for _ in range(FIT_CONFIG['max_backtracking_steps']):
                    trial = beta - step * direction
                    trial_objective, _, _ = evaluate(trial)
                    if trial_objective <= objective - FIT_CONFIG['armijo'] * step * decrement:
                        beta = trial
                        accepted = True
                        trace[-1]['step_size'] = step
                        break
                    step *= FIT_CONFIG['backtracking_factor']
                if not accepted:
                    raise ValueError('Newton backtracking did not find a valid descending step.')
            if not converged:
                raise ValueError('Logistic head did not converge in the fixed 100 iterations.')
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise ValueError('Non-finite or singular logistic fit.') from error
    return dict(schema='qa-risk-logistic-v1', feature_names=list(FEATURE_NAMES), config=dict(FIT_CONFIG),
                scaler=dict(mean=mean.tolist(), scale=scale.tolist()),
                weights=beta[1:].tolist(), intercept=float(beta[0]),
                n_train=len(matrix), class_counts={'0': outcomes.count(0.0), '1': outcomes.count(1.0)},
                convergence=dict(converged=True, iterations=iteration,
                                 gradient_max_abs=maximum_gradient), trace=trace,
                interpretation='Fitted logistic score; no guarantee of calibrated correctness probability.')


def predict_probability(model, features):
    """Return a fitted logistic score, not a guaranteed correctness probability.

    Only stored training scaler parameters are used. No adaptation or fitting
    is performed on these features, and no labels are accepted.
    """
    if not isinstance(model, dict) or model.get('schema') != 'qa-risk-logistic-v1':
        raise ValueError('Unknown correctness-head schema.')
    if model.get('feature_names') != list(FEATURE_NAMES) or model.get('config') != FIT_CONFIG:
        raise ValueError('Correctness-head features/config differ from the fixed mechanism.')
    convergence = model.get('convergence')
    if not isinstance(convergence, dict) or convergence.get('converged') is not True:
        raise ValueError('Only converged correctness heads can score.')
    scaler = model.get('scaler')
    if not isinstance(scaler, dict) or set(scaler) != {'mean', 'scale'}:
        raise ValueError('Correctness-head scaler must contain only mean and scale.')
    vector = _vector(features, 'features')
    mean = _vector(scaler.get('mean'), 'scaler.mean')
    scale = _vector(scaler.get('scale'), 'scaler.scale')
    weights = _vector(model.get('weights'), 'weights')
    intercept = _finite(model.get('intercept'), 'intercept')
    if any(value <= 0 for value in scale):
        raise ValueError('Correctness-head scales must be strictly positive.')
    try:
        terms = [_finite(((value - center) / unit) * weight, 'scaled linear term')
                 for value, center, unit, weight in zip(vector, mean, scale, weights)]
        logit = _finite(math.fsum([intercept, *terms]), 'linear score')
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError('Correctness-head scoring overflow.') from error
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)
