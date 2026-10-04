"""Coverage must not be increased by counting more wrong accepted answers."""
import unittest
from experiments.qa_coverage_gap import permits_replay
from experiments.verify_qa_coverage_gap import metrics


def arm(answered, correct, passed=True):
    return {'selected': {'gate': {'all_pass': passed}, 'summary': {'selective': {
        'answerable_answer_coverage': answered, 'correct_answerable_coverage': correct}}}}


class CoverageAcceptanceTests(unittest.TestCase):
    def test_more_answers_without_more_correct_answers_cannot_proceed(self):
        spec = {'min_coverage_gain': .05}
        base = arm(.5, .5)
        self.assertFalse(permits_replay(base, arm(.7, .5), spec))
        self.assertFalse(permits_replay(base, arm(.7, .7, passed=False), spec))
        self.assertFalse(permits_replay(base, {'selected': None}, spec))
        self.assertTrue(permits_replay(base, arm(.625, .625), spec))

    def test_independent_counter_separates_wrong_answers_from_coverage(self):
        data = [dict(id='a', context='red blue', answers=['red'], is_impossible=False),
                dict(id='b', context='red blue', answers=['red'], is_impossible=False),
                dict(id='c', context='red blue', answers=[], is_impossible=True)]
        preds = [dict(id='a', prediction='red', confidence=.8),
                 dict(id='b', prediction='blue', confidence=.8),
                 dict(id='c', prediction='blue', confidence=.8)]
        m = metrics(data, preds, .7)
        self.assertEqual(m['answerable_answer_coverage'], 1.)
        self.assertEqual(m['correct_answerable_coverage'], .5)
        self.assertEqual(m['accepted_precision'], 1/3)
        self.assertEqual(m['unanswerable_false_accept_rate'], 1.)
        with self.assertRaises(ValueError): metrics(data, [preds[0]] * 3, .7)
        preds[0]['confidence'] = float('nan')
        with self.assertRaises(ValueError): metrics(data, preds, .7)
