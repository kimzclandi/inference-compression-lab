import unittest
import importlib.util
from pathlib import Path
from lab.batching import batch_indices


class PlanningTests(unittest.TestCase):
    def test_tail_duplicates_and_bounded_reordering(self):
        lengths = [40, 3, 40, 2, 5, 1, 90]
        batches = list(batch_indices(lengths, 2, 4, True))
        self.assertEqual(batches, [[3, 1], [0, 2], [5, 4], [6]])
        self.assertEqual(sorted(sum(batches, [])), list(range(7)))

    def test_invalid_limits(self):
        for batch, window in [(0, 4), (8, 4)]:
            with self.assertRaises(ValueError):
                list(batch_indices([1], batch, window, True))


@unittest.skipUnless(importlib.util.find_spec('onnxruntime') and
                     Path('models/minilm/onnx/model.onnx').exists(), 'local model unavailable')
class RealModelTests(unittest.TestCase):
    def test_order_truncation_padding_and_call_isolation(self):
        import numpy as np
        from lab.minilm_runtime import MiniLMRuntime
        rt = MiniLMRuntime('models/minilm/onnx/model.onnx', threads=2,
                           max_length=32, fixed_padding=True)
        texts = ['A robot.', 'a very long sentence ' * 60, '', 'A robot.',
                 'The dog runs.', 'Someone walks across the road.', 'End.']
        original = rt.tokenizer.to_str()
        expected = rt.encode(texts)
        actual, stats = rt.encode_batched(texts, batch_size=2, window_size=4)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(rt.tokenizer.to_str(), original)
        self.assertEqual(stats['requests'], len(texts))
        self.assertEqual(stats['batches'], 4)
        self.assertLessEqual(stats['padded_tokens'], len(texts) * 32)
        with self.assertRaises(ValueError):
            rt.encode_batched([])
