"""Fixed risk-head mechanism tests; synthetic fixtures are not QA evaluation."""
from copy import deepcopy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

from lab.extractive_qa import decode
from lab.qa_risk_calibration import (FEATURE_NAMES, FIT_CONFIG, extract_features,
                                     fit, predict_probability)


def window(start=None, end=None, offsets=None, mask=None):
    return dict(start_logits=[0., 4., 1.] if start is None else start,
                end_logits=[0., 1., 3.] if end is None else end,
                offsets=[[0, 0], [0, 3], [4, 7]] if offsets is None else offsets,
                context_mask=[False, True, True] if mask is None else mask, cls_index=0)


def prediction(context='cat dog', windows=None):
    windows = [window()] if windows is None else windows
    record = decode(context, windows)
    record['raw_windows'] = windows
    return record


def model():
    return dict(schema='qa-risk-logistic-v1', feature_names=list(FEATURE_NAMES), config=dict(FIT_CONFIG),
                scaler=dict(mean=[0.] * 5, scale=[1.] * 5), weights=[1., 0., 0., 0., 0.],
                intercept=-1., convergence=dict(converged=True))


class RiskFeatureTests(unittest.TestCase):
    def test_five_features_match_independent_hand_calculation(self):
        result = extract_features('cat dog', prediction())
        expected = [7., 7 - math.log(1 + math.exp(4) + math.exp(1))
                    - math.log(1 + math.exp(1) + math.exp(3)),
                    2., math.log(3), math.log(2)]
        self.assertEqual(result['feature_names'], list(FEATURE_NAMES))
        for actual, wanted in zip(result['features'], expected):
            self.assertAlmostEqual(actual, wanted, places=12)
        alternative = result['diagnostics']['alternative']
        self.assertEqual((alternative['prediction'], alternative['margin']), ('cat', 5.))

    def test_question_and_padding_logits_do_not_enter_normalization(self):
        record = window(start=[0., 4., 1000., 1.], end=[0., 1., 1000., 3.],
                        offsets=[[0, 0], [0, 3], [0, 0], [4, 7]],
                        mask=[False, True, False, True])
        result = extract_features('cat dog', prediction(windows=[record]))
        wanted = 5 - math.log(1 + math.exp(4) + math.exp(1)) - math.log(1 + math.exp(1) + math.exp(3))
        self.assertAlmostEqual(result['features'][1], wanted)
        self.assertEqual(result['diagnostics']['normalization_token_count'], 3)

    def test_alternative_can_come_from_another_window_and_uses_own_null(self):
        primary = window(start=[0., 4., 1.], end=[0., 1., 3.])
        second = window(start=[2., 0., 5.], end=[2., 0., 5.])
        result = extract_features('cat dog', prediction(windows=[primary, second]))
        self.assertEqual(result['features'][0], 7.)
        self.assertEqual(result['features'][2], 1.)
        self.assertEqual(result['diagnostics']['alternative']['window_index'], 1)
        self.assertEqual(result['features'][4], math.log(3))

    def test_same_normalized_answer_is_excluded_even_at_different_offsets(self):
        context = 'Cat cat dog'
        first = window(start=[0., 5., 0., 0.], end=[0., 5., 0., 0.],
                       offsets=[[0, 0], [0, 3], [4, 7], [8, 11]],
                       mask=[False, True, False, True])
        second = window(start=[0., 4.9, 0.], end=[0., 4.9, 0.],
                        offsets=[[0, 0], [4, 7], [8, 11]])
        result = extract_features(context, prediction(context, [first, second]))
        self.assertEqual(result['diagnostics']['selected_normalized_answer'], 'cat')
        self.assertEqual(result['diagnostics']['alternative']['prediction'], 'cat dog')
        self.assertAlmostEqual(result['features'][2], 5.1)

    def test_zero_width_tokens_cannot_win_alternative_endpoints(self):
        context = 'ab cd'
        record = window(start=[0., 10., 100., 0.], end=[0., 0., 100., 10.],
                        offsets=[[0, 0], [0, 2], [2, 2], [3, 5]],
                        mask=[False, True, True, True])
        result = extract_features(context, prediction(context, [record]))
        self.assertEqual(result['features'][0], 20.)
        self.assertEqual(result['features'][2], 10.)
        self.assertEqual(result['features'][3], math.log(4))
        self.assertEqual(result['diagnostics']['alternative']['prediction'], 'ab')
        self.assertEqual(result['diagnostics']['normalization_token_count'], 4)

    def test_missing_alternative_fails_instead_of_infinite_confidence(self):
        record = window(start=[0., 1.], end=[0., 1.], offsets=[[0, 0], [0, 3]], mask=[False, True])
        with self.assertRaisesRegex(ValueError, 'undefined'):
            extract_features('cat', prediction('cat', [record]))

    def test_gold_fields_are_ignored_and_input_is_not_mutated(self):
        record = prediction()
        plain = extract_features('cat dog', record)
        record.update(answers=['wrong'], is_impossible=True, em=float('nan'), gold=object())
        raw = deepcopy(record['raw_windows'])
        self.assertEqual(extract_features('cat dog', record), plain)
        self.assertEqual(record['raw_windows'], raw)

    def test_stale_span_margin_offsets_and_source_context_are_rejected(self):
        for key, bad in (('prediction', 'dog'), ('start', 1), ('end', 6), ('start_token', True),
                         ('margin', 99.), ('span_score', 99.), ('window_index', 1),
                         ('max_answer_tokens', 31), ('context_sha256', 'wrong')):
            record = prediction()
            record[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'does not reproduce'):
                extract_features('cat dog', record)

    def test_full_decoder_shape_mask_and_finite_checks_are_inherited(self):
        for mutate in (lambda r: r['raw_windows'][0]['start_logits'].__setitem__(1, float('nan')),
                       lambda r: r['raw_windows'][0]['context_mask'].__setitem__(0, True),
                       lambda r: r['raw_windows'][0]['offsets'].pop()):
            record = prediction()
            mutate(record)
            with self.assertRaises(ValueError):
                extract_features('cat dog', record)


class RiskPredictionTests(unittest.TestCase):
    def test_stdlib_prediction_uses_stored_training_scaler(self):
        fitted = model()
        fitted['scaler'] = dict(mean=[2., 0., 0., 0., 0.], scale=[2., 1., 1., 1., 1.])
        before = deepcopy(fitted)
        with patch.dict('sys.modules', {'numpy': None}):
            self.assertAlmostEqual(predict_probability(fitted, [6., 0., 0., 0., 0.]), 1 / (1 + math.exp(-1)))
        self.assertEqual(fitted, before)

    def test_extreme_finite_logits_do_not_overflow_sigmoid(self):
        self.assertEqual(predict_probability(model(), [1000., 0., 0., 0., 0.]), 1.)
        self.assertEqual(predict_probability(model(), [-1000., 0., 0., 0., 0.]), 0.)

    def test_bad_dimensions_nonfinite_bool_or_nonpositive_scale_rejected(self):
        mutations = [lambda m: m['weights'].pop(),
                     lambda m: m['weights'].__setitem__(0, True),
                     lambda m: m['scaler']['mean'].__setitem__(1, float('nan')),
                     lambda m: m['scaler']['scale'].__setitem__(2, 0.),
                     lambda m: m['scaler']['scale'].__setitem__(2, -1.),
                     lambda m: m.__setitem__('intercept', float('inf')),
                     lambda m: m['convergence'].__setitem__('converged', False),
                     lambda m: m['feature_names'].reverse(),
                     lambda m: m['config'].__setitem__('l2', 2.)]
        for index, mutate in enumerate(mutations):
            fitted = model()
            mutate(fitted)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                predict_probability(fitted, [0.] * 5)
        for values in ([0.] * 4, [True] + [0.] * 4, [float('nan')] + [0.] * 4):
            with self.assertRaises(ValueError):
                predict_probability(model(), values)

    def test_scaled_arithmetic_overflow_fails_closed(self):
        fitted = model()
        fitted['scaler']['scale'][0] = 1e-300
        with self.assertRaises(ValueError):
            predict_probability(fitted, [1e300, 0., 0., 0., 0.])

    def test_invalid_training_matrix_labels_fail_before_numpy_import(self):
        cases = [([], []), ([[0.] * 5], []), ([[0.] * 4], [1]),
                 ([[0.] * 5] * 2, [0, 0]), ([[0.] * 5] * 2, [1, 2]),
                 ([[0.] * 5] * 2, [False, True]), ([[0.] * 5] * 2, [0, float('nan')]),
                 ([[float('inf')] * 5, [0.] * 5], [0, 1])]
        with patch.dict('sys.modules', {'numpy': None}):
            for features, labels in cases:
                with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                    fit(features, labels)


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'NumPy required only for fitting')
class RiskTrainingTests(unittest.TestCase):
    def test_constant_features_fit_class_intercept_without_regularizing_it(self):
        fitted = fit([[3., 4., 5., 6., 7.]] * 4, [0, 0, 0, 1])
        self.assertEqual(fitted['scaler'], dict(mean=[3., 4., 5., 6., 7.], scale=[1.] * 5))
        self.assertEqual(fitted['weights'], [0.] * 5)
        self.assertAlmostEqual(fitted['intercept'], math.log(1 / 3), places=7)
        self.assertAlmostEqual(predict_probability(fitted, [3., 4., 5., 6., 7.]), .25, places=8)
        self.assertTrue(fitted['convergence']['converged'])
        self.assertLessEqual(fitted['convergence']['gradient_max_abs'], 1e-8)
        json.dumps(fitted, allow_nan=False)

    def test_nonconstant_fit_has_correct_standardization_and_reduces_objective(self):
        features = [[x, 0., 0., 0., 0.] for x in (-2., -1., 1., 2.)]
        fitted = fit(features, [0, 0, 1, 1])
        self.assertEqual(fitted['scaler']['mean'], [0.] * 5)
        self.assertAlmostEqual(fitted['scaler']['scale'][0], math.sqrt(2.5))
        self.assertGreater(fitted['weights'][0], 0.)
        self.assertAlmostEqual(fitted['intercept'], 0., places=8)
        trace = fitted['trace']
        self.assertAlmostEqual(trace[0]['objective'], 4 * math.log(2))
        self.assertTrue(all(a['objective'] >= b['objective'] for a, b in zip(trace, trace[1:])))
        self.assertLess(trace[-1]['objective'], trace[0]['objective'])
        self.assertLessEqual(fitted['convergence']['iterations'], 100)

    def test_independent_scalar_objective_has_stationary_solution_with_l2_one(self):
        inputs = [-2., -1., 1., 2.]
        fitted = fit([[x, 0., 0., 0., 0.] for x in inputs], [0, 0, 1, 1])
        scale = math.sqrt(2.5)
        def objective(weight):
            z = [weight * x / scale for x in inputs]
            return sum(math.log1p(math.exp(value)) for value in (z[0], z[1], -z[2], -z[3])) + .5 * weight**2
        weight = fitted['weights'][0]
        self.assertAlmostEqual(objective(weight), fitted['trace'][-1]['objective'], places=12)
        self.assertAlmostEqual((objective(weight + 1e-5) - objective(weight - 1e-5)) / 2e-5, 0., places=7)
        self.assertLess(objective(weight), objective(weight + .1))
        self.assertLess(objective(weight), objective(weight - .1))

    def test_fit_is_deterministic_and_permutation_invariant(self):
        x = [[value, value * value, 0., 0., 1.] for value in (-3., -1., 1., 2., 4.)]
        y = [0, 0, 1, 0, 1]
        first = fit(x, y)
        self.assertEqual(first, fit(x, y))
        reversed_fit = fit(x[::-1], y[::-1])
        for actual, expected in zip(first['weights'], reversed_fit['weights']):
            self.assertAlmostEqual(actual, expected, places=8)

    def test_singular_solver_or_overflow_cannot_return_a_model(self):
        import numpy as np
        with patch.object(np.linalg, 'solve', side_effect=np.linalg.LinAlgError('synthetic singularity')):
            with self.assertRaisesRegex(ValueError, 'singular'):
                fit([[-1.] * 5, [1.] * 5], [0, 1])
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            fit([[-1e308] * 5, [1e308] * 5], [0, 1])

    def test_256_row_finite_newton_fit_avoids_large_matmul_flag_failure(self):
        # NumPy 2.2.6 on the recorded macOS runtime raised a false
        # divide-by-zero in BLAS matmul for this shape at zero initialization.
        # Explicit contractions must fit the same fixed mathematical model.
        features = [[float(i % 17), float((i % 17)**2), float(i % 3),
                     float(i % 4), float(i % 5)] for i in range(256)]
        labels = [i % 2 for i in range(256)]
        fitted = fit(features, labels)
        self.assertTrue(fitted['convergence']['converged'])
        self.assertLessEqual(fitted['convergence']['gradient_max_abs'], 1e-8)
        self.assertEqual(fitted, fit(features, labels))
        # Recompute the regularized objective without NumPy/BLAS.
        weights, intercept = fitted['weights'], fitted['intercept']
        mean, scale = fitted['scaler']['mean'], fitted['scaler']['scale']
        logits = [intercept + math.fsum(w * (value - m) / s
                   for w, value, m, s in zip(weights, row, mean, scale)) for row in features]
        loss_terms = [max(z, 0) + math.log1p(math.exp(-abs(z)))
                      for z in ((-logit if label else logit) for logit, label in zip(logits, labels))]
        expected = math.fsum(loss_terms) + .5 * math.fsum(w * w for w in weights)
        self.assertAlmostEqual(fitted['trace'][-1]['objective'], expected, places=10)


if __name__ == '__main__':
    unittest.main()
