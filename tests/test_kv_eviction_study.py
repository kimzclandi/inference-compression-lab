import json
import unittest
from pathlib import Path

from experiments.kv_eviction_streaming import policy_for, summarize
from lab.long_context_qa import arrange, build_prompt, build_user_message, passage_pool

ROWS = [json.loads(l) for l in Path('configs/qwen-prefix/qa-dev.jsonl').read_text().splitlines()]
SPEC = json.loads(Path('configs/kv-eviction-streaming-v1.json').read_text())


class LongContextPromptTests(unittest.TestCase):
    def test_every_row_sees_all_passages_once_with_own_at_slot(self):
        pool = passage_pool(ROWS)
        for row in ROWS:
            passages, slot = arrange(row, pool, seed=SPEC['seed'])
            self.assertEqual(sorted(passages), sorted(t for _, t in pool))
            self.assertEqual(passages[slot], row['context'])

    def test_deterministic_and_slots_vary(self):
        pool = passage_pool(ROWS)
        a = [arrange(r, pool, seed=SPEC['seed']) for r in ROWS]
        b = [arrange(r, pool, seed=SPEC['seed']) for r in ROWS]
        self.assertEqual(a, b)
        self.assertGreater(len({slot for _, slot in a}), 3)

    def test_build_prompt_uses_encoder_and_keeps_question_last(self):
        pool = passage_pool(ROWS)
        out = build_prompt(ROWS[0], pool, lambda s: list(s.encode()), seed=1)
        text = bytes(out['tokens']).decode()
        self.assertTrue(text.endswith('Question: ' + ROWS[0]['question']))
        self.assertEqual(text, build_user_message(out['passages'], ROWS[0]['question']))


class StudyLogicTests(unittest.TestCase):
    def test_spec_frozen_and_gate_present(self):
        self.assertEqual(SPEC['status'], 'frozen-before-first-model-run')
        self.assertEqual([c['name'] for c in SPEC['conditions']], ['full_cache', 'stream_50', 'stream_25'])
        self.assertEqual(sum(not r['is_impossible'] for r in ROWS), 33)

    def test_policy_for_budget(self):
        self.assertIsNone(policy_for(SPEC['conditions'][0], 1200))
        self.assertEqual(policy_for(SPEC['conditions'][1], 1200).budget, 600)
        self.assertEqual(policy_for(SPEC['conditions'][2], 1200).budget, 300)

    def _fake(self, lost_answerable, ratio):
        quality, timing = [], []
        for i, r in enumerate(ROWS):
            for cond in SPEC['conditions']:
                em = 1.0
                if cond['name'] != 'full_cache' and not r['is_impossible'] and i < lost_answerable * 3:
                    em = 0.0
                kv = 100 if cond['name'] == 'full_cache' else 50 if cond['name'] == 'stream_50' else 25
                quality.append(dict(id=r['id'], condition=cond['name'], is_impossible=r['is_impossible'], em=em,
                                    kv_bytes=kv, target_is_last_passage=False))
        for rnd in range(SPEC['timing']['rounds']):
            for cond in SPEC['conditions']:
                timing.append(dict(round=rnd, condition=cond['name'],
                                   tok_s=100.0 if cond['name'] == 'full_cache' else 100.0 * ratio))
        return summarize(SPEC, quality, timing)

    def test_gate_adopts_only_when_all_checks_pass(self):
        ok = self._fake(lost_answerable=0, ratio=1.02)
        self.assertTrue(ok['stream_50']['adopt'])
        self.assertAlmostEqual(ok['stream_50']['kv_reduction'], 0.5)

    def test_one_lost_answer_fails_answerable_gate(self):
        bad = self._fake(lost_answerable=1, ratio=1.02)
        self.assertFalse(bad['stream_50']['gate_checks']['answerable'])
        self.assertFalse(bad['stream_50']['adopt'])

    def test_slower_decode_fails_speed_gate(self):
        slow = self._fake(lost_answerable=0, ratio=0.97)
        self.assertFalse(slow['stream_25']['gate_checks']['speed'])


if __name__ == '__main__':
    unittest.main()
