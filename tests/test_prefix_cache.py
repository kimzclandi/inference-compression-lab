import copy
import unittest
from lab.prefix_cache import PrefixCache, token_key


class PrefixCacheTests(unittest.TestCase):
    def store(self, **kwargs):
        return PrefixCache('model-weight-tokenizer-config-hash', copy.deepcopy, **kwargs)

    def test_model_and_token_boundaries(self):
        self.assertNotEqual(token_key('a', [1, 23]), token_key('a', [12, 3]))
        self.assertNotEqual(token_key('a', [1]), token_key('b', [1]))
        for tokens in ([], [-1], [True], [1.5]):
            with self.assertRaises(ValueError): token_key('a', tokens)

    def test_hit_isolation_and_changed_prefix_miss(self):
        store = self.store()
        calls = []
        def build(tokens):
            calls.append(tokens)
            return [list(tokens)], 8
        a, status = store.acquire([1, 2], build)
        self.assertEqual(status, 'miss')
        a[0].append(900)
        b, status = store.acquire([1, 2], build)
        self.assertEqual(status, 'hit')
        self.assertEqual(b, [[1, 2]])
        store.acquire([1, 3], build)
        self.assertEqual(len(calls), 2)

    def test_lru_byte_budget_and_bypass(self):
        store = self.store(max_entries=2, max_bytes=16)
        build = lambda tokens: (list(tokens), 8)
        for tokens in ([1], [2], [1], [3]): store.acquire(tokens, build)
        self.assertEqual(store.evictions, 1)
        self.assertEqual(store.acquire([2], build)[1], 'miss')
        self.assertEqual(store.bytes, 16)
        self.assertEqual(store.acquire([4], lambda _: ([4], 17))[1], 'bypass')
        self.assertEqual(store.bytes, 16)
        store.clear()
        self.assertEqual(store.bytes, 0)

    def test_failed_build_never_inserts(self):
        store = self.store()
        def fail(_): raise RuntimeError('model failure')
        with self.assertRaises(RuntimeError): store.acquire([1], fail)
        self.assertEqual(len(store.entries), 0)
