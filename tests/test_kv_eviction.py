import unittest

import numpy as np

from lab.kv_eviction import StreamingPolicy, evict, kv_bytes


class StreamingPolicyTests(unittest.TestCase):
    def test_short_sequence_untouched(self):
        self.assertEqual(StreamingPolicy(4, 8).keep_indices(10).tolist(), list(range(10)))

    def test_sink_plus_recent(self):
        self.assertEqual(StreamingPolicy(2, 3).keep_indices(10).tolist(), [0, 1, 7, 8, 9])

    def test_from_ratio_budget(self):
        p = StreamingPolicy.from_ratio(1000, 0.25, sink=4)
        self.assertEqual((p.budget, p.sink), (250, 4))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            StreamingPolicy(-1, 4)
        with self.assertRaises(ValueError):
            StreamingPolicy.from_ratio(100, 0.0)


class EvictTests(unittest.TestCase):
    def test_evict_selects_matching_slices(self):
        rng = np.random.default_rng(0)
        k = rng.standard_normal((2, 2, 12, 4)).astype(np.float16)
        v = rng.standard_normal((2, 2, 12, 4)).astype(np.float16)
        k2, v2, idx = evict(k, v, StreamingPolicy(1, 3))
        self.assertEqual(idx.tolist(), [0, 9, 10, 11])
        np.testing.assert_array_equal(k2, k[:, :, idx, :])
        np.testing.assert_array_equal(v2, v[:, :, idx, :])
        self.assertEqual(k.shape[2], 12)

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            evict(np.zeros((1, 1, 4, 2)), np.zeros((1, 1, 5, 2)), StreamingPolicy(1, 1))

    def test_kv_bytes_qwen05b_example(self):
        self.assertEqual(kv_bytes(1024, 24, 2, 64), 2 * 24 * 2 * 1024 * 64 * 2)


if __name__ == "__main__":
    unittest.main()
