"""Verify saved real-generation evidence without MLX, models, GPU or Git history."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())


class QwenEvidenceTests(unittest.TestCase):
    def test_checksums_and_identity(self):
        for name in ['qwen-prefix-v1','qwen-prefix-v2','qwen-prefix-locality-v1']:
            folder = ROOT/'results'/name
            for filename, expected in read(folder/'complete.json')['sha256'].items():
                self.assertEqual(hashlib.sha256((folder/filename).read_bytes()).hexdigest(), expected)
            manifest = read(folder/'manifest.json')
            expected_id = hashlib.sha256(json.dumps(manifest['model_files_sha256'],sort_keys=True).encode()).hexdigest()
            self.assertEqual(expected_id, manifest['model_fingerprint'])

    def test_quality_exact_coverage_and_unchanged_outputs(self):
        rows = [json.loads(line) for line in (ROOT/'configs/qwen-prefix/qa-dev.jsonl').read_text().splitlines()]
        gold = {r['id']: r for r in rows}
        self.assertEqual(len(gold), 74)
        for version in ['v1','v2']:
            folder = ROOT/'results'/('qwen-prefix-'+version)
            predictions = read(folder/'quality-predictions.json')
            summary = read(folder/'summary.json')['quality']
            self.assertEqual(len(predictions), 74)
            self.assertEqual({p['id'] for p in predictions}, set(gold))
            self.assertEqual(summary['token_parity_count'], sum(p['cold']['token_ids']==p['cached']['token_ids'] for p in predictions))
            for mode in ['cold','cached']:
                correct = 0
                for prediction in predictions:
                    row = gold[prediction['id']]
                    text = prediction[mode]['text'].strip()
                    score = text=='NO_ANSWER' if row['is_impossible'] else text in row['answers']
                    self.assertEqual(score,prediction[mode]['strict_correct'])
                    correct += score
                self.assertEqual(correct,summary[mode+'_strict_correct'])
            self.assertEqual(summary['always_abstain_correct'],sum(r['is_impossible'] for r in rows))
            self.assertEqual(sum(p['cached']['status']=='miss' for p in predictions),9)
            for p in read(folder/'native-decoder-parity.json'): self.assertTrue(p['matches'])

    def test_timing_denominators_modes_and_causal_control(self):
        folder = ROOT/'results/qwen-prefix-v2'
        records = read(folder/'timings.json')
        summary = read(folder/'summary.json')['performance']
        self.assertEqual(len(records),4*5*4)
        for item in summary:
            groups = {mode: [r for r in records if r['prefix_tokens']==item['prefix_tokens'] and r['mode']==mode]
                      for mode in ['cold','segmented_no_reuse','cached_workload','warm_hit']}
            for mode, group in groups.items():
                self.assertEqual(sorted(r['round'] for r in group),list(range(5)))
                for r in group:
                    self.assertEqual(len(r['outputs']),4)
                    for output in r['outputs']:
                        self.assertEqual(len(output['token_ids']),32)
                        self.assertGreater(output['total_seconds'],output['ttft_seconds'])
                        self.assertAlmostEqual(31/output['decode_seconds'],output['decode_tokens_per_second'])
                        self.assertLessEqual(output['cache']['entries'],2)
                        self.assertLessEqual(output['cache']['stored_tensor_bytes'],64*1024*1024)
                    expected_status = {'cold':['disabled']*4,'segmented_no_reuse':['rebuilt']*4,
                                       'cached_workload':['miss','hit','hit','hit'],'warm_hit':['hit']*4}[mode]
                    self.assertEqual([o['status'] for o in r['outputs']],expected_status)
            cold = statistics.median(o['ttft_seconds'] for r in groups['cold'] for o in r['outputs'])
            hit = statistics.median(o['ttft_seconds'] for r in groups['warm_hit'] for o in r['outputs'])
            self.assertEqual(cold,item['cold_ttft_seconds'])
            self.assertAlmostEqual(1-hit/cold,item['hit_ttft_reduction'])
            segmented = statistics.median(r['workload_seconds'] for r in groups['segmented_no_reuse'])
            cached = statistics.median(r['workload_seconds'] for r in groups['cached_workload'])
            self.assertAlmostEqual(1-cached/segmented,item['four_request_reduction_vs_segmented'])
            self.assertEqual(item['stored_tensor_bytes'],item['prefix_tokens']*24*2*2*64*2)

    def test_cache_failure_boundaries(self):
        contracts = read(ROOT/'results/qwen-prefix-v2/real-cache-contracts.json')
        self.assertTrue(contracts['mismatched_prefix_rejected'])
        self.assertTrue(contracts['oversized_bypassed'])
        self.assertTrue(contracts['oversized_output_matches_cold'])
        self.assertEqual([r['status'] for r in contracts['interleaved']],['miss','miss','hit','miss','miss','miss'])
        for r in contracts['interleaved']:
            if r['prefix_index']==0: self.assertTrue(r['matches_cold_a'])
        records = read(ROOT/'results/qwen-prefix-locality-v1/timings.json')
        self.assertEqual(len(records),2*2*5*2)
        for r in records:
            if r['mode']=='cached':
                self.assertEqual(r['cache_delta']['hits'],11 if r['scenario']=='hot' else 0)
                self.assertEqual(r['cache_delta']['evictions'],0 if r['scenario']=='hot' else 10)
