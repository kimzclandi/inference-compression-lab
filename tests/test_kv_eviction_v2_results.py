import hashlib
import json
import math
import unittest
from pathlib import Path

from experiments.kv_eviction_v2 import summarize

D = Path('results/kv-eviction-v2')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-12)
    return a == b


class KVEvictionV2ReceiptTests(unittest.TestCase):
    """CPU replay of the published v2 receipts; does not re-run the model."""

    def setUp(self):
        self.run = json.loads((D / 'run.json').read_text())
        self.spec = json.loads(Path('configs/kv-eviction-v2.json').read_text())

    def test_run_complete_clean_and_bound_to_protocol_commit(self):
        self.assertEqual(self.run['status'], 'complete')
        self.assertFalse(self.run['dirty'])
        self.assertEqual(self.run['protocol_commit'], 'ce27a4bd83a6788d97ddb083ad6945338b6189be')

    def test_artifact_hashes_match_receipt(self):
        for name, digest in self.run['artifacts'].items():
            self.assertEqual(sha(D / name), digest, name)

    def test_correctness_gates_passed(self):
        rows = json.loads((D / 'correctness.json').read_text())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(all(v for k, v in r.items() if k != 'id') for r in rows))

    def test_summary_recomputes_and_only_snap_50_is_adopted(self):
        quality = json.loads((D / 'quality.json').read_text())
        timing = json.loads((D / 'timing.json').read_text())
        self.assertEqual(len(quality), 74 * 5)
        stored = json.loads((D / 'summary.json').read_text())
        self.assertTrue(close(summarize(self.spec, quality, timing), stored))
        self.assertFalse(stored['baseline_invalid'])
        adopted = [k for k, v in stored.items() if isinstance(v, dict) and v.get('adopt')]
        self.assertEqual(adopted, ['snap_50'])


if __name__ == '__main__':
    unittest.main()
