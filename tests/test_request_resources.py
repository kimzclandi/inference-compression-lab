from types import SimpleNamespace
import unittest
from lab.request_resources import collect_after_sync

class RequestResourceTests(unittest.TestCase):
    def test_unallocated_and_partially_allocated_cache(self):
        tensor = SimpleNamespace(nbytes=16)
        cache = [SimpleNamespace(keys=None, values=None, offset=0),
                 SimpleNamespace(keys=tensor, values=None, offset=1)]
        result = collect_after_sync(cache, lambda: None)
        self.assertEqual(result['allocated_kv_bytes'], 16)
        self.assertEqual(result['unallocated_layers'], 2)
        self.assertEqual(result['layer_offsets'], [0, 1])

    def test_secondary_failure_preserves_original_exception(self):
        def sync(): raise RuntimeError('secondary sync failure')
        for cache, synchronize in [([], sync), ([object()], lambda: None)]:
            with self.subTest(cache=cache), self.assertRaisesRegex(ValueError, 'original backend failure'):
                try:
                    raise ValueError('original backend failure')
                except BaseException as original:
                    try:
                        raise
                    finally:
                        result = collect_after_sync(cache, synchronize, original)
                        self.assertIsNone(result['allocated_kv_bytes'])
                        self.assertEqual(len(result['cleanup_error_types']), 1)

    def test_new_sync_failure_is_not_swallowed(self):
        def sync(): raise RuntimeError('sync failed')
        with self.assertRaisesRegex(RuntimeError, 'sync failed'):
            collect_after_sync([], sync)
