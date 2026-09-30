"""Evidence contracts for the frozen runtime tuning/confirmation/test study."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/minilm-runtime-study-v1'


def read(p):return json.loads(p.read_text())


@unittest.skipUnless((STUDY/'audit.json').exists(),'runtime study not finalized')
class RuntimeStudyEvidenceTests(unittest.TestCase):
    def test_frozen_selection_and_dataset_boundary(self):
        manifest=read(STUDY/'manifest.json')
        spec=manifest['spec']
        selection=read(STUDY/'selection.json')
        quality=read(STUDY/'quality/protocol.json')
        self.assertFalse(selection['test_split_accessed_by_this_study'])
        self.assertLess(selection['timestamp_utc'],quality['environment']['timestamp_utc'])
        self.assertEqual(selection['spec_sha256'],manifest['spec_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'configs/minilm-runtime-study.json').read_bytes()).hexdigest(),manifest['spec_sha256'])
        self.assertEqual(quality['selection_sha256'],hashlib.sha256((STUDY/'selection.json').read_bytes()).hexdigest())
        self.assertTrue(set(range(*spec['tune_rows'])).isdisjoint(range(*spec['confirm_rows'])))
        self.assertIn('/data/test-',quality['url'])
        self.assertEqual(quality['pairs'],1379)

    def test_raw_timings_match_summaries(self):
        spec=read(STUDY/'manifest.json')['spec']
        for phase,rounds in [('tune',3),('confirm',8),('shapes',3)]:
            plan=read(STUDY/phase/'plan.json')
            records=[]
            for i,job in enumerate(plan['jobs']):
                r=read(STUDY/phase/'raw'/f'{i:04d}.json');records.append(r)
                self.assertEqual(r['providers'],['CPUExecutionProvider'])
                self.assertEqual(r['shape'],job['shape'])
                samples=r['timing']['repeats'][0]['samples_ms']
                self.assertEqual(len(samples),spec['samples'])
                self.assertTrue(all(x>=0 for x in samples))
                self.assertEqual(statistics.median(samples),r['timing']['repeats'][0]['median_ms'])
            for s in read(STUDY/phase/'summary.json'):
                match=[r for r in records if all(r[k]==s[k] for k in ['name','shape','scope'])]
                self.assertEqual(len(match),rounds)
                self.assertEqual(statistics.median(r['timing']['repeats'][0]['median_ms'] for r in match),s['median_ms'])
            # Every configuration uses identical token IDs for a matched workload.
            by_shape={}
            for r in records:
                shape=tuple(r['shape'])
                if shape in by_shape:self.assertEqual(r['feed_sha256'],by_shape[shape])
                else:by_shape[shape]=r['feed_sha256']

    def test_quality_and_deployment_hashes(self):
        gold=None
        for item in read(STUDY/'quality/summary.json'):
            predictions=read(STUDY/'quality'/(item['name']+'-predictions.json'))
            self.assertEqual(len(predictions),1379)
            values=[p['gold'] for p in predictions]
            if gold is None:gold=values
            self.assertEqual(gold,values)
            if importlib.util.find_spec('scipy'):
                from scipy.stats import spearmanr
                self.assertAlmostEqual(item['spearman'],spearmanr(gold,[p['cosine'] for p in predictions]).statistic,places=13)
        deployment=read(STUDY/'deployment.json')
        self.assertEqual(deployment['selection_sha256'],hashlib.sha256((STUDY/'selection.json').read_bytes()).hexdigest())
        self.assertEqual(deployment['quality_result_sha256'],hashlib.sha256((STUDY/'quality/summary.json').read_bytes()).hexdigest())
        selected=read(STUDY/'selection.json')['selected_quantized']
        self.assertEqual(selected['threads'],deployment['threads'])

    def test_integer_execution_and_acceptance_arithmetic(self):
        audit=read(STUDY/'audit.json')
        quality={x['name']:x for x in read(STUDY/'quality/summary.json')}
        fp,q=quality['fp32-tuned'],quality['quantized-tuned']
        self.assertAlmostEqual(audit['test_spearman_drop'],fp['spearman']-q['spearman'])
        self.assertAlmostEqual(audit['file_reduction_fraction'],1-q['model_bytes']/fp['model_bytes'])
        profile=read(STUDY/'profile/quantized-tuned.json')
        expected=350 if q['variant'].endswith('excluded') else 360
        self.assertEqual(sum(x['events'] for x in profile['summary'] if 'Integer' in x['op'] or x['op']=='DynamicQuantizeMatMul'),expected)
        self.assertTrue(all(x['provider']=='CPUExecutionProvider' for x in profile['summary']))
        self.assertEqual(audit['all_engineering_targets_pass'],all(audit['engineering_acceptance'].values()))
