from copy import deepcopy
import unittest

from lab.extractive_qa import decode
from lab.qa_nonoverlap_gap import extract_nonoverlap_features
from lab.qa_risk_calibration import extract_features


def example():
    context = 'Alpha Beta Gamma'
    # Alpha=10, Alpha Beta=9, Beta=7, Gamma=6. Independent Gamma
    # follows a false-mask separator; synthetic decoder-contract fixture.
    window = dict(start_logits=[0., 5., 3., 0., 3.],
                  end_logits=[0., 5., 4., 0., 3.],
                  offsets=[[0, 0], [0, 5], [6, 10], [0, 0], [11, 16]],
                  context_mask=[False, True, True, False, True], cls_index=0)
    pred = decode(context, [window]); pred['raw_windows'] = [window]
    return context, pred


class NonoverlapGapTests(unittest.TestCase):
    def test_boundary_competitor_is_replaced_and_other_features_unchanged(self):
        context, p = example(); before = deepcopy(p)
        old = extract_features(context, p); new = extract_nonoverlap_features(context, p)
        self.assertEqual(old['features'][2], 1.)
        self.assertEqual(new['features'][2], 3.)
        self.assertEqual(new['diagnostics']['alternative']['prediction'], 'Beta')
        self.assertEqual(p, before)
        for i in (0, 1, 3, 4): self.assertEqual(new['features'][i], old['features'][i])

    def test_cross_window_offsets_not_window_identity_determine_overlap(self):
        context, p = example(); second = deepcopy(p['raw_windows'][0])
        second['start_logits'][1] = 4.5  # overlapping Alpha Beta=8.5
        windows = [p['raw_windows'][0], second]
        p = decode(context, windows); p['raw_windows'] = windows
        new = extract_nonoverlap_features(context, p)
        self.assertEqual(new['features'][2], 3.)
        self.assertEqual(new['diagnostics']['alternative']['window_index'], 0)

    def test_disjoint_duplicate_answer_excluded(self):
        from experiments.my_span_gap_ablation import two_window_example
        fixture = two_window_example()
        result = extract_nonoverlap_features(fixture['context'], fixture['prediction'])
        self.assertEqual(result['diagnostics']['alternative']['prediction'], 'Gamma')
        self.assertEqual(result['features'][2], 1.)

    def test_touching_intervals_are_disjoint(self):
        context = 'AlphaBeta'
        w = dict(start_logits=[0., 5., 3.], end_logits=[0., 5., 4.],
                 offsets=[[0, 0], [0, 5], [5, 9]],
                 context_mask=[False, True, True], cls_index=0)
        p = decode(context, [w]); p['raw_windows'] = [w]
        self.assertEqual(extract_nonoverlap_features(context, p)['features'][2], 3.)

    def test_no_disjoint_alternative_fails(self):
        context = 'AB'
        w = dict(start_logits=[0., 5., 4.], end_logits=[0., 4., 5.],
                 offsets=[[0, 0], [0, 1], [1, 2]],
                 context_mask=[False, True, True], cls_index=0)
        p = decode(context, [w]); p['raw_windows'] = [w]
        with self.assertRaisesRegex(ValueError, 'No disjoint'):
            extract_nonoverlap_features(context, p)

    def test_mismatched_prediction_and_invalid_unselected_window_fail(self):
        context, p = example(); p['prediction'] = 'Gamma'
        with self.assertRaises(ValueError): extract_nonoverlap_features(context, p)
        context, p = example(); bad = deepcopy(p['raw_windows'][0]); bad['start_logits'][1] = float('nan')
        p['raw_windows'].append(bad)
        with self.assertRaises(ValueError): extract_nonoverlap_features(context, p)


if __name__ == '__main__':
    unittest.main()
