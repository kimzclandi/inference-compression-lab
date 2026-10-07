import copy
import json
from pathlib import Path
import unittest
from experiments.cache_workload_boundaries import simulate, summarize


class WorkloadBoundaries(unittest.TestCase):
    def test_fixed_hit_predictions_and_byte_pressure(self):
        spec = json.loads(Path('configs/cache-workload-boundaries-v1.json').read_text())
        for name, trace in spec['traces'].items():
            _, stats = simulate(trace, 3, spec['budget_prefixes'][name])
            self.assertEqual(stats['hits'], spec['expected_hits'][name])
        self.assertEqual(simulate([0,1,2]*4, 3, 3)[1]['hits'], 9)
        self.assertEqual(simulate([0,1,2]*4, 3, 2)[1]['hits'], 0)

    def test_reject_missing_coverage(self):
        spec = json.loads(Path('configs/cache-workload-boundaries-v1.json').read_text())
        with self.assertRaises(ValueError): summarize([], spec)

    def test_paired_tokens_and_counters_are_required(self):
        spec = dict(prefix_lengths=[64], traces={'tiny':[0,0]}, rounds=1,
            arms=['direct','cache'], max_entries=3, budget_prefixes={'tiny':3},
            expected_hits={'tiny':1}, generated_tokens=1, minimum_speedup=1.05,
            minimum_faster_rounds=1, scope='test')
        rows = [dict(length=64, trace='tiny', round=0, arm=arm,
            outputs=[dict(status=s, token_ids=[7], ttft_seconds=1., total_seconds=2.) for s in statuses],
            delta=dict(hits=1, misses=1, evictions=0, bypasses=0, failures=0))
            for arm, statuses in [('direct',['direct','direct']),('cache',['miss','hit'])]]
        self.assertTrue(summarize(rows, spec)['token_parity'])
        for change in ('token','counter','status'):
            bad = copy.deepcopy(rows)
            if change == 'token': bad[1]['outputs'][0]['token_ids']=[8]
            elif change == 'counter': bad[1]['delta']['hits']=2
            else: bad[1]['outputs'][0]['status']='hit'
            with self.assertRaises(ValueError): summarize(bad, spec)
