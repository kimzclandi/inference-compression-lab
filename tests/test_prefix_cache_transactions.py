import copy
import unittest
from lab.prefix_cache import PrefixCache


class Transactions(unittest.TestCase):
    def store(self):
        c = PrefixCache('fixed', copy.deepcopy, max_entries=2, max_bytes=16)
        for t in ([1], [2]): c.acquire(t, lambda x: ([list(x)], 8))
        return c

    def test_failure_preserves_residents_lru_and_success_counters(self):
        def fail(_): raise RuntimeError('injected')
        for tokens, builder, clone in [([1], lambda t: ([], 8), fail),
                                       ([3], lambda t: ([], 8), fail),
                                       ([3], lambda t: ([], 17), fail),
                                       ([3], fail, copy.deepcopy),
                                       ([3], lambda t: ([], -1), copy.deepcopy)]:
            with self.subTest(tokens=tokens, builder=builder):
                c = self.store(); before = copy.deepcopy(c.entries); stats = c.stats()
                with self.assertRaises((RuntimeError, ValueError)): c.acquire(tokens, builder, clone=clone)
                self.assertEqual(c.entries, before)
                stats['failures'] += 1
                self.assertEqual(c.stats(), stats)
                c.acquire([3], lambda t: ([list(t)], 8))
                self.assertEqual([v[0] for v in c.entries.values()], [(2,), (3,)])

    def test_byte_limit_overrides_entry_capacity(self):
        for capacity, budget, expected_hits in [(2,24,0),(3,24,9),(3,16,0)]:
            c=PrefixCache('fixed',copy.deepcopy,max_entries=capacity,max_bytes=budget)
            for token in [0,1,2]*4: c.acquire([token],lambda t:(list(t),8))
            self.assertEqual(c.hits,expected_hits)
            self.assertEqual(c.bytes,sum(v[2] for v in c.entries.values()))
            stats=c.stats();c.clear()
            self.assertEqual(c.bytes,0);self.assertEqual(len(c.entries),0)
            self.assertEqual(c.hits,stats['hits'])
            self.assertEqual(c.acquire([0],lambda t:(list(t),8))[1],'miss')

    def test_identity_and_limit_validation(self):
        c=self.store()
        for field,value in [('model_id','new'),('max_entries',1),('max_bytes',1)]:
            with self.assertRaises(AttributeError):setattr(c,field,value)
        for value in [True,0,-1,1.5]:
            with self.assertRaises(ValueError): PrefixCache('fixed',copy.deepcopy,max_entries=value)
            with self.assertRaises(ValueError): PrefixCache('fixed',copy.deepcopy,max_bytes=value)
