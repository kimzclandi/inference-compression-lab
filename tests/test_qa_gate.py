from copy import deepcopy
import math
import unittest

from lab.qa_gate import DEFAULT_CONSTRAINTS, evaluate_quality_gate, validate_policy, wilson_interval


def passing_metrics():
    return dict(n=200, accepted=50, accepted_correct=45,
                answerable=100, unanswerable=100,
                accepted_answerable=45, accepted_unanswerable=5,
                accepted_precision=.9, answerable_answer_coverage=.45,
                correct_answerable_coverage=.45,
                unanswerable_false_accept_rate=.05, invalid_rate=.01,
                model_status_counts=dict(answer=100, abstain=98, invalid=2))


def policy():
    return dict(enabled=True, mode='grounded', bits=8, threshold=.5,
                model_files_sha256={'model.safetensors': 'a' * 64},
                source_evidence_sha256='b' * 64)


class QualityGateTests(unittest.TestCase):
    def test_all_fixed_point_constraints_and_descriptive_intervals(self):
        result = evaluate_quality_gate(passing_metrics())
        self.assertTrue(result['all_pass'])
        self.assertEqual(result['constraints'], DEFAULT_CONSTRAINTS)
        self.assertLess(result['intervals']['accepted_precision']['lower'], .9)
        self.assertEqual(result['interval_role'], 'report_only_not_used_to_pass_the_gate')

    def test_false_reported_rate_fails_even_if_it_claims_perfection(self):
        values = passing_metrics()
        values['accepted_precision'] = 1
        result = evaluate_quality_gate(values)
        self.assertFalse(result['all_pass'])
        self.assertEqual(result['criteria']['accepted_precision']['recomputed_value'], .9)

    def test_missing_none_nonfinite_boolean_and_out_of_bounds_values_fail(self):
        for name in ('accepted_precision', 'answerable_answer_coverage', 'correct_answerable_coverage',
                     'unanswerable_false_accept_rate', 'invalid_rate'):
            for value in (None, math.nan, math.inf, -math.inf, True, -1, 2):
                with self.subTest(name=name, value=value):
                    values = passing_metrics()
                    values[name] = value
                    self.assertFalse(evaluate_quality_gate(values)['all_pass'])
            values = passing_metrics()
            del values[name]
            self.assertFalse(evaluate_quality_gate(values)['all_pass'])

    def test_missing_empty_and_inconsistent_counts_fail(self):
        for name in ('n', 'accepted', 'accepted_correct', 'answerable', 'unanswerable',
                     'accepted_answerable', 'accepted_unanswerable', 'model_status_counts'):
            values = passing_metrics()
            del values[name]
            self.assertFalse(evaluate_quality_gate(values)['all_pass'])
        for key, value in (('n', 0), ('accepted', 0), ('answerable', 0), ('unanswerable', 0),
                           ('accepted_correct', 46), ('n', 200.0), ('accepted', True)):
            values = passing_metrics()
            values[key] = value
            self.assertFalse(evaluate_quality_gate(values)['all_pass'])

    def test_always_abstain_fails_despite_fifty_percent_em(self):
        values = passing_metrics()
        values.update(accepted=0, accepted_correct=0, accepted_answerable=0, accepted_unanswerable=0,
                      accepted_precision=None, answerable_answer_coverage=0,
                      correct_answerable_coverage=0, unanswerable_false_accept_rate=0,
                      invalid_rate=0, model_status_counts=dict(answer=0, abstain=200, invalid=0))
        result = evaluate_quality_gate(values)
        self.assertFalse(result['all_pass'])
        self.assertIsNone(result['intervals']['accepted_precision'])

    def test_correct_coverage_is_a_separate_limit(self):
        constraints = dict(DEFAULT_CONSTRAINTS, min_correct_answerable_coverage=.5)
        result = evaluate_quality_gate(passing_metrics(), constraints)
        self.assertFalse(result['all_pass'])
        self.assertFalse(result['criteria']['correct_answerable_coverage']['passed'])
        self.assertTrue(result['criteria']['answerable_answer_coverage']['passed'])

    def test_missing_or_invalid_constraint_does_not_relax_gate(self):
        for constraints in ({}, {'min_accepted_precision': .9},
                            dict(DEFAULT_CONSTRAINTS, max_invalid_rate=math.nan)):
            self.assertFalse(evaluate_quality_gate(passing_metrics(), constraints)['all_pass'])
        self.assertFalse(evaluate_quality_gate(None)['all_pass'])

    def test_wilson_independent_known_reference_values(self):
        # Standard Wilson score interval for 5 successes out of 10.
        result = wilson_interval(5, 10)
        self.assertAlmostEqual(result['lower'], .236593090512564, places=12)
        self.assertAlmostEqual(result['upper'], .763406909487436, places=12)
        self.assertEqual(wilson_interval(0, 10)['lower'], 0)
        self.assertAlmostEqual(wilson_interval(0, 10)['upper'], .277532799862889, places=12)
        self.assertEqual(wilson_interval(10, 10)['upper'], 1)
        for success, total in ((0, 0), (11, 10), (-1, 10), (True, 10), (5, 10.0)):
            with self.assertRaises(ValueError):
                wilson_interval(success, total)


class FrozenPolicyTests(unittest.TestCase):
    def test_enabled_valid_policy_requires_recalculated_quality(self):
        result = validate_policy(policy(), passing_metrics())
        self.assertTrue(result['may_serve_answers'])
        self.assertFalse(result['file_hashes_verified'])
        values = passing_metrics()
        values['accepted_precision'] = .5
        candidate = dict(policy(), quality_gate={'all_pass': True})
        with self.assertRaisesRegex(ValueError, 'recalculated'):
            validate_policy(candidate, values)

    def test_disabled_policy_keeps_failed_gate_closed(self):
        result = validate_policy(dict(policy(), enabled=False), {})
        self.assertFalse(result['may_serve_answers'])
        self.assertFalse(result['quality_gate']['all_pass'])

    def test_fp16_and_q8_only(self):
        self.assertTrue(validate_policy(dict(policy(), bits=None), passing_metrics())['valid'])
        for bits in (4, 16, 8.0, True, '8'):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                validate_policy(dict(policy(), bits=bits), passing_metrics())
        for key in ('bits', 'enabled', 'threshold', 'mode', 'model_files_sha256', 'source_evidence_sha256'):
            candidate = policy()
            del candidate[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy(candidate, passing_metrics())

    def test_invalid_hashes_thresholds_modes_and_enabled_values_rejected(self):
        for key, value in (
                ('model_files_sha256', {}), ('model_files_sha256', {'x': ''}),
                ('model_files_sha256', {'': 'a' * 64}), ('source_evidence_sha256', ''),
                ('source_evidence_sha256', {}), ('source_evidence_sha256', 'z' * 64),
                ('threshold', math.nan), ('threshold', math.inf), ('threshold', -.1),
                ('threshold', 1.1), ('threshold', True), ('mode', 'unknown'), ('enabled', 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_policy(dict(policy(), **{key: value}), passing_metrics())
        self.assertTrue(validate_policy(dict(policy(), source_evidence_sha256={'records.json': 'c' * 64}),
                                        passing_metrics())['valid'])

    def test_block10_and_other_restoration_not_approved_as_default(self):
        for extra in ({'fallback_blocks': [10]}, {'block': 10}, {'restored_blocks': [22]},
                      {'variant': 'q4-block10'}, {'variant': 'Q4_BLOCK_10'}):
            candidate = deepcopy(policy())
            candidate.update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_policy(candidate, passing_metrics())


if __name__ == '__main__':
    unittest.main()
