import hashlib
import math
import unittest

from lab.selective_qa import CalibrationError, apply_threshold, calibrate_threshold, evaluate_selective, parse_output


def row(identifier, impossible=False):
    return dict(id=identifier, context='猫 is a cat. A dog waits.',
                is_impossible=impossible, answers=[] if impossible else ['cat'],
                family_id='family-' + identifier)


class OutputContractTests(unittest.TestCase):
    def test_exact_span_and_unicode_character_offsets(self):
        context = '猫 cat cat'
        result = parse_output(context, ' cat \n')
        self.assertEqual(result['status'], 'answer')
        self.assertEqual((result['start'], result['end']), (2, 5))
        self.assertEqual(context[result['start']:result['end']], result['answer'])
        self.assertEqual(result['context_sha256'], hashlib.sha256(context.encode()).hexdigest())
        self.assertEqual(result['raw_output'], ' cat \n')

    def test_only_exact_reserved_token_means_model_abstention(self):
        self.assertEqual(parse_output('anything', ' NO_ANSWER \n')['status'], 'abstain')
        for value in ('no_answer', 'NO_ANSWER.', '"NO_ANSWER"', 'I cannot answer', '', '  '):
            with self.subTest(value=value):
                self.assertEqual(parse_output('anything', value)['status'], 'invalid')

    def test_no_case_normalization_quotes_or_substring_guessing(self):
        for value in ('Cat', '"cat"', 'The answer is cat.', 'cat\ncat'):
            with self.subTest(value=value):
                self.assertEqual(parse_output('a cat waits', value)['status'], 'invalid')

    def test_reserved_token_in_context_remains_abstention(self):
        self.assertEqual(parse_output('NO_ANSWER is reserved.', 'NO_ANSWER')['status'], 'abstain')

    def test_nonstring_inputs_rejected(self):
        for context, output in ((None, 'cat'), ('cat', None), ('cat', 3)):
            with self.assertRaises(ValueError):
                parse_output(context, output)

    def test_threshold_boundary_and_no_mutation(self):
        candidate = parse_output('a cat waits', 'cat')
        accepted = apply_threshold(candidate, confidence=.5, threshold=.5)
        rejected = apply_threshold(candidate, confidence=.499, threshold=.5)
        self.assertTrue(accepted['accepted'])
        self.assertEqual(rejected['status'], 'abstain')
        self.assertEqual(rejected['model_status'], 'answer')
        self.assertIsNone(rejected['answer'])
        self.assertEqual(candidate['answer'], 'cat')

    def test_invalid_is_suppressed_but_not_changed_to_correct_refusal(self):
        result = apply_threshold(parse_output('a cat', 'made up'), confidence=1, threshold=0)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['action'], 'abstain')
        self.assertEqual(result['scoring_prediction'], '')
        self.assertFalse(result['accepted'])

    def test_invalid_confidence_is_not_silently_accepted(self):
        candidate = parse_output('a cat', 'cat')
        for value in (None, math.nan, math.inf, -math.inf, '0.9', True, -.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_threshold(candidate, confidence=value, threshold=0.5)


class SelectiveMetricTests(unittest.TestCase):
    def test_hand_computed_metrics_and_original_model_scores(self):
        rows = [row('correct'), row('wrong'), row('rejected'), row('false', True),
                row('refusal', True), row('invalid', True)]
        predictions = [dict(id=r['id'], prediction=p, confidence=c) for r, p, c in zip(
            rows, ('cat', 'dog', 'cat', 'cat', 'NO_ANSWER', 'invented'), (.9, .8, .1, .7, .9, .9))]
        summary, scored = evaluate_selective(rows, predictions, threshold=.5)
        metrics = summary['selective']
        self.assertEqual(metrics['accepted'], 3)
        self.assertEqual(metrics['accepted_precision'], 1 / 3)
        self.assertEqual(metrics['answerable_answer_coverage'], 2 / 3)
        self.assertEqual(metrics['correct_answerable_coverage'], 1 / 3)
        self.assertEqual(metrics['unanswerable_false_accept_rate'], 1 / 3)
        self.assertEqual(metrics['valid_span_rate'], 4 / 6)
        self.assertEqual(metrics['invalid_rate'], 1 / 6)
        self.assertEqual(summary['raw']['overall']['em'], 3 / 6)
        self.assertEqual(summary['system']['overall']['em'], 2 / 6)
        self.assertEqual(scored[-1]['system_score']['em'], 0)
        self.assertEqual(scored[-1]['decision']['status'], 'invalid')

    def test_legacy_normalized_nonextractive_match_not_rewarded_by_system(self):
        # The unchanged historical normalization gives EM=1 to "the cat".
        # It is not a literal span of this context, so the contract emits nothing.
        summary, _ = evaluate_selective([row('x')], [dict(id='x', prediction='the cat')])
        self.assertEqual(summary['raw']['overall']['em'], 1)
        self.assertEqual(summary['system']['overall']['em'], 0)
        self.assertEqual(summary['selective']['invalid_rate'], 1)
        self.assertIsNone(summary['selective']['accepted_precision'])

    def test_threshold_abstention_separate_from_invalid_and_model_abstention(self):
        rows = [row('threshold', True), row('model', True), row('invalid', True)]
        predictions = [dict(id='threshold', prediction='cat', confidence=.1),
                       dict(id='model', prediction='NO_ANSWER', confidence=.9),
                       dict(id='invalid', prediction='invented', confidence=.9)]
        summary, _ = evaluate_selective(rows, predictions, threshold=.5)
        self.assertEqual(summary['system']['overall']['em'], 2 / 3)
        self.assertEqual(summary['selective']['threshold_abstention_rate'], 1 / 3)
        self.assertEqual(summary['selective']['model_abstention_rate'], 1 / 3)
        self.assertEqual(summary['selective']['invalid_rate'], 1 / 3)

    def test_all_abstention_has_undefined_precision_and_zero_coverage(self):
        rows = [row('yes'), row('no', True)]
        predictions = [dict(id=r['id'], prediction='NO_ANSWER') for r in rows]
        summary, _ = evaluate_selective(rows, predictions)
        self.assertEqual(summary['system']['overall']['em'], .5)
        self.assertIsNone(summary['selective']['accepted_precision'])
        self.assertEqual(summary['selective']['answerable_answer_coverage'], 0)

    def test_exact_id_coverage_and_nonempty_dataset_required(self):
        for rows, predictions in (([], []), ([row('a')], []),
                                  ([row('a')], [dict(id='b', prediction='cat')]),
                                  ([row('a'), row('a')], [dict(id='a', prediction='cat')]),
                                  ([row('a')], [dict(id='a', prediction='cat')] * 2)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_selective(rows, predictions)

    def test_predictions_are_aligned_by_id_not_list_order(self):
        rows = [row('yes'), row('no', True)]
        summary, _ = evaluate_selective(rows, [dict(id='no', prediction='NO_ANSWER'),
                                              dict(id='yes', prediction='cat')])
        self.assertEqual(summary['system']['overall']['em'], 1)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row('good'), row('lower_good'), row('false', True), row('refuse', True)]
        self.predictions = [dict(id=r['id'], prediction=p, confidence=c) for r, p, c in zip(
            self.rows, ('cat', 'cat', 'dog', 'NO_ANSWER'), (.9, .6, .3, .9))]
        self.constraints = dict(min_accepted_precision=.9, min_answerable_answer_coverage=.5,
                                max_unanswerable_false_accept_rate=0, max_invalid_rate=0)

    def test_smallest_feasible_predeclared_threshold_and_complete_curve(self):
        result = calibrate_threshold(self.rows, self.predictions, [.8, 0, .5], **self.constraints)
        self.assertEqual(result['threshold'], .5)
        self.assertEqual(result['threshold_grid'], [0, .5, .8])
        self.assertEqual([c['feasible'] for c in result['candidates']], [False, True, True])

    def test_no_feasible_grid_candidate_fails_without_em_fallback(self):
        with self.assertRaisesRegex(CalibrationError, 'No feasible') as failure:
            calibrate_threshold(self.rows, self.predictions, [0, 1], **self.constraints)
        self.assertEqual([c['feasible'] for c in failure.exception.candidates], [False, False])
        self.assertEqual(failure.exception.threshold_grid, [0, 1])
        self.assertEqual(failure.exception.constraints, self.constraints)

    def test_all_abstain_cannot_pass(self):
        predictions = [dict(id=r['id'], prediction='NO_ANSWER', confidence=.9) for r in self.rows]
        with self.assertRaisesRegex(ValueError, 'No feasible'):
            calibrate_threshold(self.rows, predictions, [0, .5], **self.constraints)

    def test_positive_coverage_both_classes_and_finite_grid_required(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            calibrate_threshold(self.rows, self.predictions, [0],
                                **dict(self.constraints, min_answerable_answer_coverage=0))
        with self.assertRaisesRegex(ValueError, 'both answerable'):
            calibrate_threshold(self.rows[:2], self.predictions[:2], [0], **self.constraints)
        for grid in ([], [math.nan], [math.inf], [True], [-.1], [1.1]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                calibrate_threshold(self.rows, self.predictions, grid, **self.constraints)

    def test_invalid_risk_configuration_fails(self):
        for value in (-.1, 1.1, math.nan, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate_threshold(self.rows, self.predictions, [0],
                                    **dict(self.constraints, max_invalid_rate=value))


if __name__ == '__main__':
    unittest.main()
