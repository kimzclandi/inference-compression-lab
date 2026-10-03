"""Recompute published bucketing metrics; fail on missing or corrupted evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BucketingEvidenceTests(unittest.TestCase):
    def test_evidence_hashes_predictions_and_timing_denominators(self):
        for name, expected_requests in [('minilm-bucketing-v2', 256),
                                         ('minilm-bucketing-confirm-v1', 2758)]:
            directory = ROOT / 'results' / name
            complete = json.loads((directory / 'complete.json').read_text())
            for filename, expected in complete['files_sha256'].items():
                self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), expected)
            raw = json.loads((directory / 'timings.json').read_text())
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(len(raw), 28)
            self.assertEqual(len(summary), 4)
            for item in summary:
                matching = [r for r in raw if r['precision'] == item['precision'] and
                            r['bucket'] == item['bucket']]
                self.assertEqual(sorted(r['round'] for r in matching), list(range(7)))
                for record in matching:
                    self.assertGreater(record['seconds'], 0)
                    self.assertEqual(record['stats']['requests'], expected_requests)
                    self.assertEqual(record['stats'], item['benchmark_stats'])
                self.assertEqual(statistics.median(r['seconds'] for r in matching), item['median_seconds'])
                self.assertAlmostEqual(expected_requests/item['median_seconds'], item['sentences_per_second'])
                predictions = json.loads((directory / f"predictions-{item['precision']}-{int(item['bucket'])}.json").read_text())
                self.assertEqual([p['row'] for p in predictions], list(range(1379)))
                if importlib.util.find_spec('scipy'):
                    from scipy.stats import spearmanr
                    rho = spearmanr([p['label'] for p in predictions], [p['cosine'] for p in predictions]).statistic
                    self.assertAlmostEqual(rho, item['spearman'], places=13)
                if item['bucket']:
                    base = next(r for r in summary if r['precision'] == item['precision'] and not r['bucket'])
                    self.assertEqual(item['benchmark_stats']['valid_tokens'], base['benchmark_stats']['valid_tokens'])
                    self.assertEqual(item['benchmark_stats']['batches'], base['benchmark_stats']['batches'])
                    self.assertLess(item['benchmark_stats']['padded_tokens'], base['benchmark_stats']['padded_tokens'])
                    self.assertAlmostEqual(base['spearman']-item['spearman'], item['spearman_drop_vs_same_precision_consecutive'])

    def test_historical_results_remain_unchanged(self):
        import subprocess
        changed = subprocess.check_output(['git', 'diff', '--name-only', 'c6012d6', '--', 'results'],
                                          cwd=ROOT, text=True).splitlines()
        self.assertTrue(all(p.startswith(('results/minilm-bucketing-v2/',
                                         'results/minilm-bucketing-confirm-v1/')) for p in changed))
