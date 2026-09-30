"""Validate real Mac artifacts; optional numerical dependencies for quality checks."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]/'results'


@unittest.skipUnless((ROOT/'minilm-mac-m4max-ablation/summary.json').exists(),'Mac experiment not run')
class MacEvidenceTests(unittest.TestCase):
    def test_predictions_protocol_and_execution(self):
        old=json.loads((ROOT/'minilm-cpu-dynamic-int8/provenance.json').read_text())
        for directory in ['minilm-mac-m4max-baseline','minilm-mac-m4max-ablation']:
            path=ROOT/directory
            self.assertEqual(json.loads((path/'provenance.json').read_text()),old)
            env=json.loads((path/'environment.json').read_text())
            self.assertEqual(env['architecture'],'arm64')
            self.assertEqual(env['cpu_model'],'Apple M4 Max')
            self.assertEqual(env['protocol']['provider'],'CPUExecutionProvider')
            for summary in json.loads((path/'summary.json').read_text()):
                name=summary['variant']
                result=json.loads((path/(name+'.json')).read_text())
                predictions=json.loads((path/(name+'-predictions.json')).read_text())
                self.assertEqual(len(predictions),1500)
                self.assertEqual([x['row'] for x in predictions],list(range(1500)))
                self.assertEqual(result['actual_providers'],['CPUExecutionProvider'])
                for run in result['latency']['repeats']:
                    self.assertEqual(len(run['samples_ms']),200)
                    self.assertEqual(statistics.median(run['samples_ms']),run['median_ms'])
                self.assertEqual(len(result['latency']['repeats']),3)
                self.assertEqual(summary['latency_median_of_round_medians_ms'],statistics.median(
                    r['median_ms'] for r in result['latency']['repeats']))
                if importlib.util.find_spec('scipy'):
                    from scipy.stats import spearmanr
                    score=spearmanr([p['gold'] for p in predictions],[p['cosine'] for p in predictions]).statistic
                    self.assertAlmostEqual(score,result['spearman'],places=13)
                evidence=json.loads((path/(name+'-execution.json')).read_text())
                expected=0 if name=='fp32' else (35 if name.endswith('excluded') else 36)
                self.assertEqual(len(evidence['integer_matmul_nodes']),expected)
                self.assertTrue(all(x['provider']=='CPUExecutionProvider' for x in evidence['profile_executed_ops']))
                if expected:
                    integer_events=sum(x['events'] for x in evidence['profile_executed_ops'] if 'Integer' in x['op'] or x['op']=='DynamicQuantizeMatMul')
                    self.assertEqual(integer_events,expected)

    def test_layer_sums_and_debug_fidelity(self):
        path=ROOT/'minilm-mac-m4max-layer-errors-optimized'
        raw=json.loads((path/'raw-errors.json').read_text())
        summary=json.loads((path/'summary.json').read_text())
        self.assertEqual(len(raw),8*(42*2+36))
        grouped=defaultdict(list)
        for r in raw:grouped[(r['kind'],r['variant'],r['node'])].append(r)
        for r in summary:
            rows=grouped[(r['kind'],r['variant'],r['node'])]
            self.assertEqual(len(rows),8)
            self.assertAlmostEqual(r['nmse'],sum(x['sse'] for x in rows)/sum(x['reference_energy'] for x in rows),places=14)
        batches=json.loads((path/'probe-batches.json').read_text())
        self.assertEqual(len(batches),8)
        for b in batches:
            for r in b['debug_vs_optimized_final'].values():self.assertLess(r['nmse'],1e-10)
        rank=json.loads((path/'local-ranking.json').read_text())
        exclusion=json.loads((path/'exclusion.json').read_text())
        self.assertEqual(exclusion['nodes_to_exclude'],[max(rank,key=lambda x:x['nmse'])['node']])
        env=json.loads((ROOT/'minilm-mac-m4max-ablation/environment.json').read_text())
        self.assertEqual(env['protocol']['excluded_nodes'],exclusion['nodes_to_exclude'])

    def test_crossover_positions_and_raw_samples(self):
        path=ROOT/'minilm-mac-m4max-latency-crossover'
        protocol=json.loads((path/'protocol.json').read_text())
        for v in protocol['order'][0]:
            self.assertEqual(sorted(r.index(v) for r in protocol['order']),list(range(4)))
            for i in range(4):
                r=json.loads((path/f'{i}-{v}.json').read_text())
                self.assertEqual(r['actual_providers'],['CPUExecutionProvider'])
                timing=r['latency']['repeats'][0]
                self.assertEqual(len(timing['samples_ms']),200)
                self.assertEqual(statistics.median(timing['samples_ms']),timing['median_ms'])
