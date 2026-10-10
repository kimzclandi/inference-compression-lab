import json
import unittest
from pathlib import Path

import numpy as np

from experiments.kv_eviction_v2 import baseline, budget_for, summarize
from lab import snapkv
from lab.long_context_qa import arrange, passage_pool

ROWS = [json.loads(l) for l in Path('configs/qwen-prefix/qa-dev.jsonl').read_text().splitlines()]
SPEC = json.loads(Path('configs/kv-eviction-v2.json').read_text())


class SnapKVSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_planted_key_is_selected_and_window_kept(self):
        n, w, d = 40, 4, 8
        k = self.rng.standard_normal((2, n, d)) * 0.01
        q = self.rng.standard_normal((6, w, d)) * 0.01
        direction = np.ones(d)
        q[0:3] += direction          # heads of KV group 0 look along `direction`
        k[0, 10] += 5 * direction    # planted key at position 10 for KV head 0
        idx = snapkv.select(q, k, budget=w + 3, window=w, pool=1, scale=d ** -0.5)
        self.assertEqual(idx.shape, (2, w + 3))
        self.assertIn(10, idx[0].tolist())
        for h in range(2):
            self.assertEqual(idx[h, -w:].tolist(), list(range(n - w, n)))
            self.assertTrue(np.all(np.diff(idx[h]) > 0))

    def test_scores_respect_causality_inside_window(self):
        n, w, d = 10, 3, 4
        q = self.rng.standard_normal((2, w, d))
        k = self.rng.standard_normal((1, n, d))
        s = snapkv.window_scores(q, k, 0.5)
        # total mass = n_heads_in_group * window (each softmax row sums to 1)
        self.assertAlmostEqual(float(s.sum()), 2 * w)
        q2 = np.ones((2, w, d))
        k2 = np.zeros((1, n, d))
        k2[0, -1] = 10.0  # every query is aligned with it, but only the last may see it
        s2 = snapkv.window_scores(q2, k2, 0.5)
        # 2 heads x 1 visible query (the last one) -> mass close to 2, never 2 x 3.
        self.assertGreater(float(s2[0, -1]), 1.9)
        self.assertLessEqual(float(s2[0, -1]), 2.0 + 1e-9)

    def test_no_eviction_when_budget_covers_prompt(self):
        k = np.zeros((2, 5, 4)); q = np.zeros((4, 2, 4))
        self.assertEqual(snapkv.select(q, k, budget=5, window=2, pool=3, scale=1).tolist(), [list(range(5))] * 2)

    def test_max_pool(self):
        x = np.array([[0., 3., 0., 0., 1.]])
        np.testing.assert_array_equal(snapkv.max_pool_1d(x, 3), [[3., 3., 3., 1., 1.]])

    def test_budget_must_exceed_window(self):
        with self.assertRaises(ValueError):
            snapkv.select(np.zeros((2, 4, 2)), np.zeros((1, 10, 2)), budget=4, window=4, pool=1, scale=1)


class V2ProtocolTests(unittest.TestCase):
    def test_three_passage_prompts(self):
        pool = passage_pool(ROWS)
        for row in ROWS:
            passages, slot = arrange(row, pool, seed=SPEC['seed'], n_distractors=2)
            self.assertEqual(len(passages), 3)
            self.assertEqual(passages[slot], row['context'])
            self.assertEqual(len(set(passages)), 3)

    def test_v1_arrangement_unchanged_by_new_parameter(self):
        pool = passage_pool(ROWS)
        a = arrange(ROWS[5], pool, seed=20261009)
        self.assertEqual(len(a[0]), 9)

    def test_spec_frozen_with_baseline_gate(self):
        self.assertEqual(SPEC['status'], 'frozen-before-first-model-run')
        self.assertEqual(SPEC['baseline_gate']['answerable_em_min'], 40.0)
        self.assertEqual([c['name'] for c in SPEC['conditions']],
                         ['full_cache', 'stream_50', 'snap_50', 'stream_25', 'snap_25'])
        self.assertEqual(budget_for(SPEC['conditions'][2], 600), 300)

    def _rows(self, full_correct_answerable, snap_lost):
        quality, timing = [], []
        answerable = [r for r in ROWS if not r['is_impossible']]
        good = {r['id'] for r in answerable[:full_correct_answerable]}
        lost = {r['id'] for r in answerable[:snap_lost]}
        for r in ROWS:
            for c in SPEC['conditions']:
                em = 1.0 if r['id'] in good else 0.0
                if c['name'].startswith('snap') and r['id'] in lost:
                    em = 0.0
                if c['name'].startswith('stream') and r['id'] in good:
                    em = 0.0
                kv = 100 if c['name'] == 'full_cache' else 50 if c['name'].endswith('50') else 25
                quality.append(dict(id=r['id'], condition=c['name'], is_impossible=r['is_impossible'], em=em, kv_bytes=kv))
        for rnd in range(SPEC['timing']['rounds']):
            for c in SPEC['conditions']:
                timing.append(dict(round=rnd, condition=c['name'], tok_s=100.0 if c['name'] == 'full_cache' else 105.0,
                                   evict_s=0.001))
        return quality, timing

    def test_baseline_gate_stops_weak_baseline(self):
        q, t = self._rows(full_correct_answerable=10, snap_lost=0)  # 10/33 = 30.3%
        self.assertTrue(baseline(SPEC, q)['baseline_invalid'])
        self.assertTrue(summarize(SPEC, q, t)['baseline_invalid'])

    def test_snap_adopted_only_without_net_loss(self):
        q, t = self._rows(full_correct_answerable=20, snap_lost=0)
        s = summarize(SPEC, q, t)
        self.assertFalse(s['baseline_invalid'])
        self.assertTrue(s['snap_50']['adopt'])
        self.assertFalse(s['stream_50']['adopt'])
        q, t = self._rows(full_correct_answerable=20, snap_lost=1)
        self.assertFalse(summarize(SPEC, q, t)['snap_25']['gate_checks']['answerable'])


if __name__ == '__main__':
    unittest.main()
