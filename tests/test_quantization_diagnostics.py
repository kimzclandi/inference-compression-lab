"""Guard evidence coverage, selection without quality fishing, and unchanged scoring."""
import unittest
from pathlib import Path
import tempfile
import shutil
import json
from lab.quantization_diagnostics import sha
from experiments.verify_qwen_quantization import verify
from lab.quantization_diagnostics import read, rows, rank_blocks, paired, performance
from lab.qa_metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticTests(unittest.TestCase):
    def test_historical_scores_are_identical_without_rescoring_rules(self):
        data = rows(ROOT / 'configs/qwen-prefix/qa-dev.jsonl')
        scored = {}
        for variant, expected in [('fp16', 19), ('q4', 16), ('q8', 18)]:
            base = ROOT / 'results/qwen-quantization-history'
            metrics, scored[variant] = evaluate(data, rows(base / f'{variant}-dev.predictions.jsonl'))
            self.assertEqual(metrics, read(base / f'{variant}-dev.metrics.json'))
            self.assertEqual(sum(r['em'] for r in scored[variant]), expected)
        changes = paired(scored['fp16'], scored['q4'])
        self.assertEqual(len(changes['lost']), 6)
        self.assertEqual(len(changes['gained']), 3)
        self.assertEqual(changes['lost'], read(ROOT / 'configs/qwen-quantization/study.json')['diagnosis_ids'])

    def test_format_flag_is_not_an_em_gate(self):
        data = [{'id': 'a', 'context': 'The answer is Apple.', 'answers': ['Apple'],
                 'is_impossible': False, 'family_id': 'f'}]
        _, scored = evaluate(data, [{'id': 'a', 'prediction': 'apple'}])
        self.assertEqual(scored[0]['em'], 1)
        self.assertFalse(scored[0]['format_valid'])

    def test_selection_coverage_and_tie_break_are_frozen(self):
        spec = {'diagnosis_ids': ['a', 'b'], 'screen_blocks': [0, 1]}
        records = [dict(block=b, id=i, top1=5, reference_top1=5,
                        reference_pair_margin=1., fp16_margin=1.)
                   for b in [0, 1] for i in ['a', 'b']]
        self.assertEqual(rank_blocks(records, spec)[0]['block'], 0)
        for invalid in [records[:-1], records + [records[0]], records[:-1] + [records[0]]]:
            with self.assertRaises(ValueError):
                rank_blocks(invalid, spec)

    def test_decode_denominator_excludes_prefill_and_first_token(self):
        r = dict(token_ids=list(range(32)), generated_tokens=32,
                 ttft_seconds=.1, decode_seconds=.31, total_seconds=.41)
        self.assertEqual(performance([r])['decode_tokens_per_second'], 100.)
        with self.assertRaises(ValueError):
            performance([dict(r, token_ids=[1])])


@unittest.skipUnless((ROOT / 'results/qwen-quantization-v1/run.json').exists(), 'GPU evidence not archived')
class FrozenEvidenceTests(unittest.TestCase):
    def test_recompute_and_preserve_parent_evidence(self):
        result = verify(ROOT / 'results/qwen-quantization-v1')
        self.assertEqual(result, read(ROOT / 'results/qwen-quantization-v1/summary.json'))
        for name, expected in read(ROOT / 'configs/qwen-quantization/parent-evidence.json').items():
            self.assertEqual(sha(ROOT / name), expected)

    def test_resigned_missing_screen_cell_is_rejected(self):
        self.reject_changed_file('screen.json', lambda x: x[:-1], 'Missing/duplicate screen cell')

    def test_resigned_changed_control_is_rejected(self):
        self.reject_changed_file('selection.json', lambda x: dict(x, control=x['selected']), 'Control changed')

    def test_resigned_shortened_benchmark_is_rejected(self):
        self.reject_changed_file('bench-0-q4/run.json', lambda x: dict(x, timings=x['timings'][:-1]),
                                 'Benchmark request coverage')

    def reject_changed_file(self, name, change, reason):
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / 'evidence'
            shutil.copytree(ROOT / 'results/qwen-quantization-v1', copy)
            (copy / name).write_text(json.dumps(change(read(copy / name))))
            checksums = read(copy / 'checksums.json'); checksums[name] = sha(copy / name)
            (copy / 'checksums.json').write_text(json.dumps(checksums))
            with self.assertRaisesRegex(ValueError, reason):
                verify(copy)


if __name__ == '__main__':
    unittest.main()
