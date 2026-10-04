from copy import deepcopy
import math
import unittest

from experiments.my_span_gap_ablation import selected_window_features, two_window_example
from lab.extractive_qa import decode
from lab.qa_risk_calibration import extract_features


def record(context, windows):
    result = decode(context, windows); result['raw_windows'] = windows
    return result


class SpanGapAblationTests(unittest.TestCase):
    def test_two_window_counterexample_and_duplicate_answer_exclusion(self):
        example = two_window_example()
        original, local = example['all_windows'], example['selected_window']
        self.assertEqual(original['features'][2], 1.)
        self.assertEqual(local['features'][2], 8.)
        self.assertEqual(original['diagnostics']['alternative']['prediction'], 'Gamma')
        self.assertEqual(local['diagnostics']['alternative']['prediction'], 'Beta')
        self.assertEqual([original['features'][i] for i in (0,1,3,4)],
                         [local['features'][i] for i in (0,1,3,4)])
        self.assertEqual(local['features'][4], math.log1p(2))

    def test_single_window_is_identical_and_input_is_not_mutated(self):
        example = two_window_example(); context = example['context']
        pred = record(context, example['prediction']['raw_windows'][:1]); before = deepcopy(pred)
        self.assertEqual(extract_features(context,pred)['features'], selected_window_features(context,pred)['features'])
        self.assertEqual(pred, before)

    def test_selected_window_index_is_preserved_when_not_zero(self):
        example = two_window_example()
        pred = record(example['context'], list(reversed(example['prediction']['raw_windows'])))
        local = selected_window_features(example['context'],pred)
        self.assertEqual(local['features'][2], 8.)
        self.assertEqual(local['diagnostics']['selected_window_index'], 1)
        self.assertEqual(local['diagnostics']['alternative']['window_index'], 1)

    def test_no_local_alternative_fails_even_with_global_alternative(self):
        context='Alpha Beta'
        def window(offset, score):
            return dict(start_logits=[0.,score/2],end_logits=[0.,score/2],
                        offsets=[[0,0],offset],context_mask=[False,True],cls_index=0)
        pred=record(context,[window([0,5],10.),window([6,10],9.)])
        self.assertEqual(extract_features(context,pred)['features'][2],1.)
        with self.assertRaisesRegex(ValueError,'No differently normalized legal span'):
            selected_window_features(context,pred)

    def test_corrupt_unselected_window_still_invalidates_entire_record(self):
        example=two_window_example();pred=deepcopy(example['prediction'])
        pred['raw_windows'][1]['start_logits'][1]=float('nan')
        with self.assertRaises(ValueError):selected_window_features(example['context'],pred)

    def test_cannot_trust_a_forged_stored_candidate(self):
        example=two_window_example();pred=deepcopy(example['prediction']);pred['window_index']=1
        with self.assertRaisesRegex(ValueError,'does not reproduce'):
            selected_window_features(example['context'],pred)


if __name__=='__main__':unittest.main()
