import copy
import json
from pathlib import Path
import unittest

from experiments.metal_mechanism_diagnostic import summarize


class TimingEvidence(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path('configs/mechanism-diagnostics-v1.json').read_text())
        self.records = [dict(rows=r, round=n, arm=a,
            seconds=[{'original': 2., 'compiled': 1., 'masked': 1.5}[a]] * self.spec['repeats'])
            for r in self.spec['timing_rows'] for n in range(self.spec['rounds']) for a in self.spec['arms']]

    def test_improving_custom_does_not_imply_beating_compiled(self):
        value = summarize(self.records, self.spec)['shapes']['512']['masked_vs']
        self.assertTrue(value['original']['diagnostic_speed_gate'])
        self.assertFalse(value['compiled']['diagnostic_speed_gate'])

    def test_reject_missing_duplicate_nonfinite_and_nonpositive(self):
        bad = [self.records[:-1], self.records + [self.records[0]]]
        for value in [float('nan'), float('inf'), 0, -1, True]:
            rows = copy.deepcopy(self.records); rows[0]['seconds'][0] = value; bad.append(rows)
        for rows in bad:
            with self.assertRaises(ValueError):
                summarize(rows, self.spec)

    def test_round_gate_is_not_replaced_by_ratio(self):
        rows = copy.deepcopy(self.records)
        for row in rows:
            if row['arm'] == 'masked' and row['round'] >= 4:
                row['seconds'] = [10.] * self.spec['repeats']
        result = summarize(rows, self.spec)['shapes']['512']['masked_vs']['original']
        self.assertGreater(result['speedup'], 1.05)
        self.assertFalse(result['diagnostic_speed_gate'])
