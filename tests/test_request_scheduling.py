import unittest
from lab.request_scheduling import AdmissionQueue, Request, consume_stream, percentile, summarize

class SchedulingTests(unittest.TestCase):
    def test_admits_only_arrived_and_fifo_ties(self):
        q = AdmissionQueue([Request('a', 1, 20, 1), Request('b', 1, 1, 1)])
        self.assertIsNone(q.pop_ready(.9))
        self.assertEqual(q.pop_ready(1).request_id, 'a')
        self.assertEqual(q.pop_ready(1).request_id, 'b')

    def test_short_budget_bypasses_at_most_bound(self):
        q = AdmissionQueue([Request('large', 0, 1024, 32)] +
                           [Request(str(i), 0, 1, 1) for i in range(10)], 'short_budget')
        order = [q.pop_ready(0).request_id for _ in range(11)]
        self.assertEqual(order.index('large'), 3)
        self.assertEqual(max(q.bypasses.values()), 3)

    def test_cancel_pending_is_idempotent(self):
        q = AdmissionQueue([Request('x', 1, 1, 1)])
        self.assertTrue(q.cancel('x'))
        self.assertFalse(q.cancel('x'))
        self.assertIsNone(q.pop_ready(2))

    def test_zero_bypasses_reduces_to_fifo(self):
        q = AdmissionQueue([Request('a', 0, 999, 1), Request('b', 0, 1, 1)], 'short_budget', 0)
        self.assertEqual(q.pop_ready(0).request_id, 'a')

    def test_rejects_invalid_and_duplicate_requests(self):
        for arrival, prompt, output in [(float('nan'), 1, 1), (0, 0, 1), (0, 1, True)]:
            with self.assertRaises(ValueError): Request('x', arrival, prompt, output)
        r = Request('x', 0, 1, 1)
        with self.assertRaises(ValueError): AdmissionQueue([r, r])
        with self.assertRaises(ValueError): AdmissionQueue([r], 'unknown')

    def test_close_on_cancel_failure_and_complete(self):
        for mode in ('complete', 'cancel', 'failure', 'callback'):
            closed, seen = [], []
            def stream():
                try: yield from range(5)
                finally: closed.append(True)
            def callback(x):
                seen.append(x)
                if mode == 'callback': raise LookupError('callback')
            kwargs = {'cancel_after': 2} if mode == 'cancel' else {'fail_after': 2} if mode == 'failure' else {}
            if mode in ('failure', 'callback'):
                with self.assertRaises((RuntimeError, LookupError)): consume_stream(stream(), callback, **kwargs)
            else:
                self.assertEqual(consume_stream(stream(), callback, **kwargs), 'cancelled' if mode == 'cancel' else 'complete')
            self.assertEqual(closed, [True])
            self.assertEqual(len(seen), {'complete': 5, 'cancel': 2, 'failure': 2, 'callback': 1}[mode])

    def test_percentile_definition_and_metrics(self):
        self.assertAlmostEqual(percentile([0, 10], .95), 9.5)
        row = dict(tokens=[1, 2], queue_s=.1, ttft_s=.2, tpot_s=.01, total_s=.21, service_s=.11)
        self.assertEqual(summarize([row], 1)['throughput_tokens_s'], 2)
        with self.assertRaises(ValueError): summarize([dict(row, ttft_s=-1)], 1)
