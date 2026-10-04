"""Differential contracts, not evidence of unseen QA quality."""
from copy import deepcopy
import json
import random
import unittest
from unittest.mock import patch

from lab.extractive_qa import decode
from lab import qa_risk_calibration as reference, qa_risk_pruning as candidate
from experiments.my_span_gap_ablation import two_window_example
import test_qa_risk_calibration as original_tests


def record(context, windows):
    result = decode(context, windows)
    result['raw_windows'] = windows
    return result


def isolated(context, spans, scores, null=0.):
    offsets = [[0, 0]]
    logits = [null / 2]
    mask = [False]
    for offset, score in zip(spans, scores):
        offsets.extend([offset, [0, 0]])
        logits.extend([score / 2, 0.])
        mask.extend([True, False])
    return dict(start_logits=logits, end_logits=logits[:], offsets=offsets,
                context_mask=mask, cls_index=0)


class ExistingFeatureContracts(original_tests.RiskFeatureTests):
    """Run the unchanged reference feature contract against the specialization."""
    def setUp(self):
        override = patch.object(original_tests, 'extract_features', candidate.extract_features)
        override.start()
        self.addCleanup(override.stop)


class PruningTests(unittest.TestCase):
    def same(self, context, pred):
        before = deepcopy(pred)
        expected = reference.extract_features(context, pred)
        actual = candidate.extract_features(context, pred)
        # JSON also detects int/float representation changes hidden by ==.
        self.assertEqual(json.dumps(actual, sort_keys=True), json.dumps(expected, sort_keys=True))
        self.assertEqual(pred, before)
        return actual

    def test_cross_window_second_best_survives_duplicate_winner(self):
        example = two_window_example()
        result = self.same(example['context'], example['prediction'])
        self.assertEqual(result['features'][2], 1.)
        self.assertEqual(result['diagnostics']['alternative']['prediction'], 'Gamma')

    def test_first_equal_alternative_keeps_its_diagnostics(self):
        context = 'Alpha Beta Gamma'
        pred = record(context, [isolated(context, [[0, 5], [6, 10], [11, 16]], [10., 2., 2.])])
        result = self.same(context, pred)
        self.assertEqual(result['diagnostics']['alternative']['prediction'], 'Beta')

    def test_low_margin_duplicate_overflow_is_still_ignored(self):
        context = 'Alpha Beta Alpha Gamma'
        windows = [isolated(context, [[0, 5], [6, 10]], [10., 2.]),
                   isolated(context, [[11, 16], [17, 22]], [-1e308, 1e308], 1e308)]
        self.same(context, record(context, windows))

    def test_low_margin_nonduplicate_overflow_is_not_pruned(self):
        context = 'Alpha Beta Alpha Gamma'
        for integer in (False, True):
            windows = [isolated(context, [[0, 5], [6, 10]], [10., 2.]),
                       isolated(context, [[11, 16], [17, 22]], [1e308, -1e308], 1e308)]
            if integer:
                for key in ('start_logits', 'end_logits'):
                    windows[1][key] = [int(x) for x in windows[1][key]]
            pred = record(context, windows)
            for extractor in (reference.extract_features, candidate.extract_features):
                with self.subTest(integer=integer, extractor=extractor.__module__):
                    with self.assertRaisesRegex(ValueError, 'alternative margin'):
                        extractor(context, pred)

    def test_dominated_spans_skip_normalization(self):
        context = 'Alpha Beta Gamma'
        pred = record(context, [isolated(context, [[0, 5], [6, 10], [11, 16]], [10., 5., 1.])])
        with patch.object(reference, 'normalize', wraps=reference.normalize) as before:
            expected = reference.extract_features(context, pred)
        with patch.object(candidate, 'normalize', wraps=candidate.normalize) as after:
            actual = candidate.extract_features(context, pred)
        self.assertEqual(actual, expected)
        self.assertEqual((before.call_count, after.call_count), (4, 3))

    def test_undefined_normalized_alternative_still_fails(self):
        context = 'Cat cat! THE cat'
        pred = record(context, [isolated(context, [[0, 3], [4, 8], [9, 16]], [10., 5., 1.])])
        for extractor in (reference.extract_features, candidate.extract_features):
            with self.assertRaisesRegex(ValueError, 'undefined'):
                extractor(context, pred)

    def test_seeded_masks_offsets_ties_and_unicode(self):
        rng = random.Random(20261005)
        for trial in range(250):
            tokens = [rng.choice(['Alpha', 'alpha!', 'the', '猫', 'beta', ' café ']) for _ in range(rng.randint(3, 18))]
            context = ' '.join(tokens)
            offsets = []
            cursor = 0
            for token in tokens:
                offsets.append([cursor, cursor + len(token)])
                cursor += len(token) + 1
            windows = []
            for _ in range(rng.randint(1, 4)):
                off = [[0, 0]] + deepcopy(offsets)
                mask = [False] + [rng.random() > .2 for _ in tokens]
                mask[1] = True
                for j in range(2, len(off)):
                    if rng.random() < .15:
                        off[j][1] = off[j][0]
                windows.append(dict(start_logits=[rng.randint(-5, 5) for _ in off],
                    end_logits=[rng.randint(-5, 5) for _ in off], offsets=off,
                    context_mask=mask, cls_index=0))
            pred = record(context, windows)
            try:
                reference.extract_features(context, pred)
            except ValueError as error:
                with self.assertRaises(ValueError) as caught:
                    candidate.extract_features(context, pred)
                self.assertEqual(str(caught.exception), str(error))
            else:
                with self.subTest(trial=trial):
                    self.same(context, pred)


if __name__ == '__main__':
    unittest.main()
