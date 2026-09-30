"""Validate checked-in experimental artifacts without downloading models."""
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'results/minilm-cpu-dynamic-int8'


@unittest.skipUnless((ROOT / 'summary.json').exists(), 'real experiment not yet executed')
class EvidenceTests(unittest.TestCase):
    def test_predictions_measurements_and_integer_execution(self):
        summary = json.loads((ROOT / 'summary.json').read_text())
        self.assertEqual(len(summary), 3)
        gold = None
        for item in summary:
            name = item['variant']
            result = json.loads((ROOT / (name + '.json')).read_text())
            predictions = json.loads((ROOT / (name + '-predictions.json')).read_text())
            self.assertEqual(len(predictions), 1500)
            self.assertEqual([p['row'] for p in predictions], list(range(1500)))
            if gold is None:
                gold = [p['gold'] for p in predictions]
            self.assertEqual(gold, [p['gold'] for p in predictions])
            self.assertTrue(all(math.isfinite(p['cosine']) for p in predictions))
            for run in result['latency']['repeats']:
                self.assertEqual(len(run['samples_ms']), 200)
                self.assertEqual(statistics.median(run['samples_ms']), run['median_ms'])
            self.assertEqual(len(result['latency']['repeats']), 3)
            evidence = json.loads((ROOT / (name + '-execution.json')).read_text())
            if name != 'fp32':
                self.assertGreater(len(evidence['integer_matmul_nodes']), 0)
                self.assertTrue(any(('Integer' in x['op'] or 'QLinear' in x['op'])
                                    and x['provider'] == 'CPUExecutionProvider'
                                    for x in evidence['profile_executed_ops']))


if __name__ == '__main__':
    unittest.main()
