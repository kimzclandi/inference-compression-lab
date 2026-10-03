from copy import deepcopy
import hashlib
import math
import unittest

from lab.extractive_qa import decode


def window(start=None, end=None, offsets=None, mask=None, cls=0):
    return dict(start_logits=[0, 4, 1] if start is None else start,
                end_logits=[0, 1, 3] if end is None else end,
                offsets=[[0, 0], [0, 3], [4, 7]] if offsets is None else offsets,
                context_mask=[False, True, True] if mask is None else mask,
                cls_index=cls)


class ExtractiveDecoderTests(unittest.TestCase):
    def test_hand_computed_logits_margin_and_exact_provenance(self):
        record = window(start=[2, 4, 1], end=[1, 1, 3])
        result = decode('cat dog', [record])
        self.assertEqual(result['prediction'], 'cat dog')
        self.assertEqual((result['start'], result['end']), (0, 7))
        self.assertEqual((result['start_token'], result['end_token']), (1, 2))
        self.assertEqual(result['span_score'], 7)
        self.assertEqual(result['null_score'], 3)
        self.assertEqual(result['margin'], 4)
        self.assertAlmostEqual(result['confidence'], 1 / (1 + math.exp(-4)))
        self.assertEqual(result['context_sha256'], hashlib.sha256(b'cat dog').hexdigest())
        self.assertEqual(result['windows'][0]['margin'], 4)

    def test_unicode_offsets_and_whitespace_trim_are_character_based(self):
        context = '\t猫 🐈 café\n'
        candidate = window(start=[0, 8], end=[0, 7],
                           offsets=[[0, 0], [0, len(context)]], mask=[False, True])
        result = decode(context, [candidate])
        self.assertEqual(result['prediction'], '猫 🐈 café')
        self.assertEqual((result['start'], result['end']), (1, len(context) - 1))
        self.assertEqual(context[result['start']:result['end']], result['prediction'])

    def test_window_selection_uses_each_own_null_and_not_raw_span(self):
        high_raw_high_null = window(start=[10, 9, 0], end=[10, 9, 0])
        lower_raw_good_margin = window(start=[0, 0, 5], end=[0, 0, 5])
        result = decode('cat dog', [high_raw_high_null, lower_raw_good_margin])
        self.assertEqual(result['window_index'], 1)
        self.assertEqual(result['prediction'], 'dog')
        self.assertEqual([item['margin'] for item in result['windows']], [-2, 10])

    def test_high_null_returns_candidate_with_low_confidence_not_fake_refusal(self):
        result = decode('cat dog', [window(start=[500, 1, 0], end=[500, 1, 0])])
        self.assertEqual(result['prediction'], 'cat')
        self.assertLess(result['confidence'], .5)
        self.assertTrue(math.isfinite(result['confidence']))
        self.assertEqual(result['confidence'], 0)

    def test_positive_extreme_sigmoid_does_not_overflow(self):
        result = decode('cat dog', [window(start=[-500, 1, 0], end=[-500, 1, 0])])
        self.assertEqual(result['confidence'], 1)

    def test_ties_choose_lower_window_then_start_then_end(self):
        tied = window(start=[0, 1, 1], end=[0, 1, 1])
        result = decode('cat dog', [tied, deepcopy(tied)])
        self.assertEqual((result['window_index'], result['start_token'], result['end_token']),
                         (0, 1, 1))

    def test_context_endpoints_cannot_cross_question_special_or_pad(self):
        record = window(start=[0, 20, 0, 0], end=[0, 0, 0, 20],
                        offsets=[[0, 0], [0, 3], [0, 0], [4, 7]],
                        mask=[False, True, False, True])
        result = decode('cat dog', [record])
        self.assertEqual(result['prediction'], 'cat')
        self.assertEqual(result['span_score'], 20)

    def test_question_tokens_with_large_logits_are_excluded(self):
        record = window(start=[0, 1000, 2], end=[0, 1000, 3],
                        offsets=[[0, 0], [100, 120], [4, 7]],
                        mask=[False, False, True])
        self.assertEqual(decode('cat dog', [record])['prediction'], 'dog')

    def test_max_token_span_is_inclusive_and_not_character_length(self):
        record = window(start=[0, 10, 0, 0], end=[0, 0, 0, 10],
                        offsets=[[0, 0], [0, 1], [2, 3], [4, 5]],
                        mask=[False, True, True, True])
        self.assertEqual(decode('a b c', [record], 3)['prediction'], 'a b c')
        self.assertEqual(decode('a b c', [record], 2)['prediction'], 'a')

    def test_duplicate_unicode_subword_offsets_are_legal(self):
        record = window(start=[0, 3, 0], end=[0, 0, 3],
                        offsets=[[0, 0], [0, 1], [0, 1]])
        self.assertEqual(decode('猫', [record])['prediction'], '猫')

    def test_zero_width_context_tokens_are_interior_only_and_count_toward_limit(self):
        record = window(start=[0, 10, 100, 0], end=[0, 0, 100, 10],
                        offsets=[[0, 0], [0, 2], [2, 2], [3, 5]],
                        mask=[False, True, True, True])
        result = decode('ab cd', [record], 3)
        self.assertEqual(result['prediction'], 'ab cd')
        self.assertEqual((result['start_token'], result['end_token'], result['span_score']), (1, 3, 20))
        shorter = decode('ab cd', [record], 2)
        self.assertEqual(shorter['prediction'], 'ab')
        self.assertEqual(shorter['span_score'], 10)

    def test_only_zero_width_context_tokens_provide_no_legal_answer(self):
        record = window(start=[0, 100], end=[0, 100], offsets=[[0, 0], [1, 1]],
                        mask=[False, True])
        with self.assertRaisesRegex(ValueError, 'no nonempty legal context span'):
            decode('ab cd', [record])

    def test_input_is_not_mutated(self):
        original = window()
        before = deepcopy(original)
        decode('cat dog', [original])
        self.assertEqual(original, before)

    def test_nonempty_context_and_windows_required(self):
        for context in (None, 1, '', ' \n\t'):
            with self.subTest(context=context), self.assertRaises(ValueError):
                decode(context, [window()])
        for windows in (None, {}, [], (window(),), [None]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                decode('cat dog', windows)

    def test_empty_mismatched_or_nonlist_arrays_fail(self):
        for key in ('start_logits', 'end_logits', 'offsets', 'context_mask'):
            for replacement in ([], [0], None, (0, 0, 0)):
                broken = window()
                broken[key] = replacement
                with self.subTest(key=key, replacement=replacement), self.assertRaises(ValueError):
                    decode('cat dog', [broken])

    def test_all_logits_including_masked_tokens_must_be_finite_numbers(self):
        for key in ('start_logits', 'end_logits'):
            for position in range(3):
                for bad in (True, False, '1', None, math.nan, math.inf, -math.inf, 10**1000):
                    broken = window()
                    broken[key][position] = bad
                    with self.subTest(key=key, position=position, bad=bad), self.assertRaises(ValueError):
                        decode('cat dog', [broken])

    def test_boolean_and_malformed_offset_rejected(self):
        for pair in ([True, 1], [0, False], [0., 1], ['0', 1], [-1, 1], [2, 1],
                     [0], [0, 1, 2], None, [0, 99]):
            broken = window()
            broken['offsets'][1] = pair
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                decode('cat dog', [broken])

    def test_context_offsets_cannot_go_backwards(self):
        for offsets in ([[0, 0], [4, 7], [0, 3]], [[0, 0], [0, 7], [4, 5]]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                decode('cat dog', [window(offsets=offsets)])

    def test_mask_values_must_be_boolean_and_cls_cannot_be_context(self):
        for mask in ([0, 1, 1], [False, True, 1], [True, True, True], [False] * 3):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                decode('cat dog', [window(mask=mask)])

    def test_cls_index_and_token_limit_must_be_bounded_integers(self):
        for cls in (None, True, 0., '0', -1, 3):
            with self.subTest(cls=cls), self.assertRaises(ValueError):
                decode('cat dog', [window(cls=cls)])
        for limit in (None, True, 1., '1', -1, 0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                decode('cat dog', [window()], limit)

    def test_whitespace_only_candidate_window_invalidates_request(self):
        whitespace = window(start=[0, 4], end=[0, 4], offsets=[[0, 0], [3, 4]],
                            mask=[False, True])
        with self.assertRaisesRegex(ValueError, 'no nonempty legal context span'):
            decode('cat dog', [window(), whitespace])

    def test_arithmetic_overflow_fails_instead_of_creating_confidence(self):
        for start, end in (([1e308, 0, 0], [1e308, 0, 0]),
                           ([0, 1e308, 0], [0, 1e308, 0]),
                           ([-1e308, 1e308, 0], [0, 0, 0])):
            with self.subTest(start=start), self.assertRaises(ValueError):
                decode('cat dog', [window(start=start, end=end)])


if __name__ == '__main__':
    unittest.main()
