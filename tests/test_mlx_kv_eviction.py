import importlib.util
import unittest

import numpy as np

from lab.kv_eviction import StreamingPolicy


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'Optional MLX; runs on Apple Silicon only')
class MLXEvictionTests(unittest.TestCase):
    def test_evict_keeps_offset_and_selects_positions(self):
        import mlx.core as mx
        from lab.mlx_kv_eviction import evict_caches

        class Native:
            def __init__(self, k, v, n):
                self.keys, self.values, self.offset = k, v, n

        k = mx.array(np.arange(2 * 10 * 3, dtype=np.float32).reshape(1, 2, 10, 3))
        native = Native(k, k + 1, 8)  # buffer longer than offset, like the upstream cache
        out = evict_caches([native], StreamingPolicy(sink=1, window=2))[0]
        self.assertEqual(out.offset, 8)
        np.testing.assert_array_equal(np.array(out.keys), np.array(k)[:, :, [0, 6, 7], :])
        nk = mx.ones((1, 2, 1, 3))
        keys, values = out.update_and_fetch(nk, nk)
        self.assertEqual((out.offset, out.physical_tokens), (9, 4))
        with self.assertRaises(ValueError):
            out.update_and_fetch(mx.ones((1, 2, 2, 3)), mx.ones((1, 2, 2, 3)))

    def test_per_head_eviction_gathers_each_head_independently(self):
        import mlx.core as mx
        from lab.mlx_kv_eviction import evict_caches_per_head

        class Native:
            def __init__(self, k, v, n):
                self.keys, self.values, self.offset = k, v, n

        base = np.arange(2 * 10 * 3, dtype=np.float32).reshape(1, 2, 10, 3)
        idx = np.array([[0, 4, 7], [1, 2, 7]])
        out = evict_caches_per_head([Native(mx.array(base), mx.array(base + 1), 8)], [idx])[0]
        self.assertEqual(out.offset, 8)
        np.testing.assert_array_equal(np.array(out.keys)[0, 0], base[0, 0, [0, 4, 7]])
        np.testing.assert_array_equal(np.array(out.keys)[0, 1], base[0, 1, [1, 2, 7]])


if __name__ == '__main__':
    unittest.main()
