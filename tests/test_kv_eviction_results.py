import hashlib
import json
import math
import unittest
from pathlib import Path

from experiments.kv_eviction_streaming import summarize

D = Path('results/kv-eviction-streaming-v1')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class KVEvictionResultReceiptTests(unittest.TestCase):
    """CPU replay of the published receipts; does not re-run the model."""

    def setUp(self):
        self.run = json.loads((D / 'run.json').read_text())
        self.spec = json.loads(Path('configs/kv-eviction-streaming-v1.json').read_text())

    def test_run_complete_clean_and_bound_to_protocol_commit(self):
        self.assertEqual(self.run['status'], 'complete')
        self.assertFalse(self.run['dirty'])
        self.assertEqual(self.run['protocol_commit'], 'd6007927af0c0c7d56f56b36ee30687dd02aa4ad')

    def test_artifact_hashes_match_receipt(self):
        for name, digest in self.run['artifacts'].items():
            self.assertEqual(sha(D / name), digest, name)

    def test_correctness_gate_passed(self):
        rows = json.loads((D / 'correctness.json').read_text())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['tokens_equal'] and r['logits_allclose'] for r in rows))

    def test_summary_recomputes_from_raw_rows_with_frozen_gate(self):
        quality = json.loads((D / 'quality.json').read_text())
        timing = json.loads((D / 'timing.json').read_text())
        self.assertEqual(len(quality), 74 * 3)
        replay = summarize(self.spec, quality, timing)
        stored = json.loads((D / 'summary.json').read_text())
        def close(a, b):
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
            if isinstance(a, float):
                return math.isclose(a, b, rel_tol=1e-12)
            return a == b
        self.assertTrue(close(replay, stored))
        self.assertFalse(stored['stream_50']['adopt'])
        self.assertFalse(stored['stream_25']['adopt'])


if __name__ == '__main__':
    unittest.main()
