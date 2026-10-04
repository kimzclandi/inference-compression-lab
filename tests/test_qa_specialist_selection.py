"""New specialist cohorts must be fixed, locally disjoint and honestly named."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.prepare_qa_remediation import KEYS, extend_exclusions
from experiments.prepare_qa_specialist import select_cohorts, checked_exclusions

ROOT = Path(__file__).resolve().parents[1]


def fixture(articles=4):
    return {'data': [dict(title=f'Article-{a}', paragraphs=[
        dict(context=f'Answer {a} passage {p}.', qas=[
            dict(id=f'{a}-{p}-{impossible}', question=f'Question {a} {p} {impossible}?',
                 is_impossible=impossible,
                 answers=[] if impossible else [{'text': 'Answer', 'answer_start': 0}])
            for impossible in (False, True)]) for p in range(3)])
        for a in range(articles)]}


def spec():
    return dict(seed=2026100405, splits=[dict(name='calibration', articles=1),
                                       dict(name='evaluation', articles=2)],
                per_class_per_article=2, max_context_chars=4000)


class SpecialistSelectionTests(unittest.TestCase):
    def test_deterministic_under_upstream_order(self):
        raw = fixture()
        first = select_cohorts(raw, spec(), {key: [] for key in KEYS})
        raw['data'].reverse()
        for article in raw['data']:
            article['paragraphs'].reverse()
            for paragraph in article['paragraphs']:
                paragraph['qas'].reverse()
        self.assertEqual(first, select_cohorts(raw, spec(), {key: [] for key in KEYS}))

    def test_no_fallback_when_quota_unavailable(self):
        with self.assertRaisesRegex(ValueError, 'evaluation: 1 < 2'):
            select_cohorts(fixture(2), spec(), {key: [] for key in KEYS})
        raw = fixture()
        for article in raw['data']:
            article['paragraphs'] = [article['paragraphs'][0]] * 3
        with self.assertRaisesRegex(ValueError, 'Insufficient eligible articles'):
            select_cohorts(raw, spec(), {key: [] for key in KEYS})

    def test_incorrect_split_claim_is_rejected(self):
        protocol = spec(); protocol['splits'][1]['name'] = 'confirmation'
        with self.assertRaisesRegex(ValueError, 'Invalid fixed specialist'):
            select_cohorts(fixture(), protocol, {key: [] for key in KEYS})

    def test_previous_inventory_and_all_384_rows_are_retained(self):
        frozen = json.loads((ROOT / 'configs/qa-specialist/exclusions.json').read_text())
        actual = json.loads((ROOT / frozen['prior_inventory']).read_text())
        count = 0
        for path in frozen['historical_datasets']:
            rows = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
            count += len(rows); actual = extend_exclusions(actual, rows)
        self.assertEqual(count, 384)
        for key in KEYS:
            self.assertEqual(actual[key], frozen[key])

    def test_inventory_hash_change_and_resigned_omission_fail(self):
        # Construct a complete independently located inventory; omission must fail
        # even after updating the top-level file hash.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            original = {key: [] for key in KEYS}; original['ids'] = ['must-remain']
            prior = tmp / 'prior.json'; prior.write_text(json.dumps(original))
            inventory = dict(original, prior_inventory=str(prior), historical_datasets=[],
                             local_sources_sha256={str(prior): hashlib.sha256(prior.read_bytes()).hexdigest()})
            target = tmp / 'inventory.json'; target.write_text(json.dumps(inventory))
            protocol = dict(exclusions=str(target), exclusions_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                checked_exclusions(protocol)
            inventory['ids'] = []; target.write_text(json.dumps(inventory))
            protocol['exclusions_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                checked_exclusions(protocol)

    def test_committed_cohorts_independent_balance_and_disjointness(self):
        base = ROOT / 'configs/qa-specialist'
        protocol = json.loads((base / 'selection.json').read_text())
        self.assertEqual(protocol['seed'], 2026100405)
        self.assertEqual(protocol['splits'], [{'name': 'calibration', 'articles': 4},
                                             {'name': 'evaluation', 'articles': 8}])
        self.assertIn('never an official hidden test or a model-unseen confirmation', protocol['scope'])
        inventory = json.loads((base / 'exclusions.json').read_text())
        forbidden = {key: set(inventory[key]) for key in KEYS}
        normalize = lambda text: ' '.join(text.lower().split())
        digest = lambda text: hashlib.sha256(normalize(text).encode()).hexdigest()
        for name, article_count in [('calibration', 4), ('evaluation', 8)]:
            data = (base / 'dataset' / name / 'data.jsonl').read_bytes()
            rows = [json.loads(line) for line in data.splitlines()]
            manifest = json.loads((base / 'dataset' / name / 'manifest.json').read_text())
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest['data_sha256'])
            self.assertEqual(len(rows), article_count * 32)
            actual = {'ids': {r['id'] for r in rows}, 'titles': {r['source_title'] for r in rows},
                      'context_hashes': {digest(r['context']) for r in rows},
                      'question_hashes': {digest(r['question']) for r in rows}}
            self.assertEqual(len(actual['ids']), len(rows))
            self.assertEqual(len(actual['question_hashes']), len(rows))
            self.assertEqual(len(actual['titles']), article_count)
            for key in KEYS:
                self.assertFalse(actual[key] & forbidden[key], (name, key))
                forbidden[key] |= actual[key]
            for title in actual['titles']:
                for impossible in (False, True):
                    group = [r for r in rows if r['source_title'] == title and r['is_impossible'] is impossible]
                    self.assertEqual(len(group), 16)
                    self.assertEqual(len({normalize(r['context']) for r in group}), 16)
                    self.assertTrue(all(bool(r['answers']) != impossible for r in group))
            self.assertTrue(all(r['split'] == name and r['family_id'] == digest(r['context'])
                                and len(r['context']) <= 4000 for r in rows))


if __name__ == '__main__':
    unittest.main()
