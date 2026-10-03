"""Leakage, allocation and source-integrity contracts for the fixed new QA cohorts."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from experiments.prepare_qa_remediation import (
    KEYS, extend_exclusions, normalized_hash, select_cohorts, validate_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture(articles=4):
    data = []
    for a in range(articles):
        paragraphs = []
        for p in range(3):
            context = f'Answer {a} passage {p}.'
            paragraphs.append(dict(context=context, qas=[
                dict(id=f'{a}-{p}-{impossible}', question=f'Question {a} {p} {impossible}?',
                     is_impossible=impossible,
                     answers=[] if impossible else [{'text': 'Answer', 'answer_start': 0}])
                for impossible in (False, True)]))
        data.append(dict(title=f'Article-{a}', paragraphs=paragraphs))
    return dict(data=data)


def protocol():
    return dict(seed=2026100403, splits=[dict(name='calibration', articles=1),
                                       dict(name='confirmation', articles=2)],
                per_class_per_article=2, max_context_chars=4000)


def empty_exclusions():
    return {key: [] for key in KEYS}


class SelectionContractTests(unittest.TestCase):
    def test_deterministic_under_upstream_order_and_no_class_fallback(self):
        raw = fixture()
        result = select_cohorts(raw, protocol(), empty_exclusions())
        reversed_raw = copy.deepcopy(raw)
        reversed_raw['data'].reverse()
        for article in reversed_raw['data']:
            article['paragraphs'].reverse()
            for paragraph in article['paragraphs']:
                paragraph['qas'].reverse()
        self.assertEqual(result, select_cohorts(reversed_raw, protocol(), empty_exclusions()))
        self.assertEqual(len(result['calibration']['records']), 4)
        self.assertEqual(len(result['confirmation']['records']), 8)

    def test_calibration_questions_are_excluded_even_in_other_articles(self):
        raw = fixture()
        before = select_cohorts(raw, protocol(), empty_exclusions())
        question = before['calibration']['records'][0]['question']
        confirmation_title = before['confirmation']['titles'][0]
        target = next(a for a in raw['data'] if a['title'] == confirmation_title)
        target['paragraphs'][0]['qas'][0]['question'] = '  ' + question.upper() + '  '
        after = select_cohorts(raw, protocol(), empty_exclusions())
        self.assertNotIn(normalized_hash(question), {
            normalized_hash(row['question']) for row in after['confirmation']['records']})

    def test_calibration_contexts_are_excluded_even_in_other_articles(self):
        raw = fixture()
        before = select_cohorts(raw, protocol(), empty_exclusions())
        cal_context = before['calibration']['records'][0]['context']
        target = next(a for a in raw['data'] if a['title'] == before['confirmation']['titles'][0])
        target['paragraphs'][0]['context'] = cal_context
        after = select_cohorts(raw, protocol(), empty_exclusions())
        self.assertNotIn(normalized_hash(cal_context), {
            row['family_id'] for row in after['confirmation']['records']})

    def test_insufficient_unique_contexts_fail_even_with_many_questions(self):
        raw = fixture()
        for article in raw['data']:
            article['paragraphs'] = [article['paragraphs'][0]] * 3
        with self.assertRaisesRegex(ValueError, 'Insufficient eligible articles'):
            select_cohorts(raw, protocol(), empty_exclusions())

    def test_insufficient_remaining_articles_cannot_shrink_confirmation(self):
        with self.assertRaisesRegex(ValueError, 'confirmation: 1 < 2'):
            select_cohorts(fixture(2), protocol(), empty_exclusions())

    def test_invalid_answer_offset_and_answerability_fail(self):
        raw = fixture()
        raw['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] = 2
        with self.assertRaisesRegex(ValueError, 'offsets'):
            select_cohorts(raw, protocol(), empty_exclusions())
        raw = fixture()
        raw['data'][0]['paragraphs'][0]['qas'][1]['answers'] = [{'text': 'Answer', 'answer_start': 0}]
        with self.assertRaisesRegex(ValueError, 'answerability'):
            select_cohorts(raw, protocol(), empty_exclusions())

    def test_each_exclusion_axis_filters_an_otherwise_eligible_record(self):
        raw = fixture()
        baseline = select_cohorts(raw, protocol(), empty_exclusions())
        row = baseline['calibration']['records'][0]
        for key, value in [('ids', row['id']), ('titles', row['source_title']),
                           ('context_hashes', row['family_id']),
                           ('question_hashes', normalized_hash(row['question']))]:
            with self.subTest(key=key):
                exclusions = empty_exclusions(); exclusions[key] = [value]
                result = select_cohorts(raw, protocol(), exclusions)
                selected_ids = {r['id'] for cohort in result.values() for r in cohort['records']}
                self.assertNotIn(row['id'], selected_ids)

    def test_cross_article_question_duplicate_rejected(self):
        raw = fixture(3)
        spec = protocol(); spec['splits'][0]['articles'] = 2; spec['splits'][1]['articles'] = 1
        for article in raw['data']:
            for p, paragraph in enumerate(article['paragraphs']):
                for q, question in enumerate(paragraph['qas']):
                    question['question'] = f'Duplicate question {p} {q}'
        with self.assertRaisesRegex(ValueError, 'Duplicate ID or normalized question'):
            select_cohorts(raw, spec, empty_exclusions())

    def test_removed_or_resigned_family_record_is_rejected(self):
        spec = protocol(); cohort = select_cohorts(fixture(), spec, empty_exclusions())['calibration']
        for altered in [cohort['records'][:-1],
                        [dict(cohort['records'][0], family_id='forged'), *cohort['records'][1:]]]:
            with self.assertRaises(ValueError):
                validate_cohort(altered, cohort['titles'], spec, 'calibration', empty_exclusions())


class CommittedCohortTests(unittest.TestCase):
    def test_frozen_data_independently_satisfies_balance_and_disjointness(self):
        base = ROOT / 'configs/qa-remediation'
        spec = json.loads((base / 'selection.json').read_text())
        self.assertEqual(spec['seed'], 2026100403)
        self.assertEqual(spec['splits'], [{'name': 'calibration', 'articles': 4},
                                         {'name': 'confirmation', 'articles': 8}])
        excluded = json.loads((base / 'exclusions.json').read_text())
        normalize = lambda text: ' '.join(text.lower().split())
        hash_text = lambda text: hashlib.sha256(normalize(text).encode()).hexdigest()
        forbidden = {key: set(excluded[key]) for key in KEYS}
        for name, article_count in [('calibration', 4), ('confirmation', 8)]:
            raw = (base / 'dataset' / name / 'data.jsonl').read_bytes()
            records = [json.loads(line) for line in raw.splitlines()]
            manifest = json.loads((base / 'dataset' / name / 'manifest.json').read_text())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['data_sha256'])
            self.assertEqual(len(records), article_count * 32)
            actual = {'ids': {r['id'] for r in records},
                      'titles': {r['source_title'] for r in records},
                      'context_hashes': {hash_text(r['context']) for r in records},
                      'question_hashes': {hash_text(r['question']) for r in records}}
            self.assertEqual(len(actual['ids']), len(records))
            self.assertEqual(len(actual['question_hashes']), len(records))
            self.assertEqual(len(actual['titles']), article_count)
            for key in KEYS:
                self.assertFalse(actual[key] & forbidden[key], (name, key))
                forbidden[key] |= actual[key]
            for title in actual['titles']:
                for impossible in (False, True):
                    group = [r for r in records if r['source_title'] == title and
                             r['is_impossible'] is impossible]
                    self.assertEqual(len(group), 16)
                    self.assertEqual(len({normalize(r['context']) for r in group}), 16)
                    self.assertTrue(all(bool(r['answers']) != impossible for r in group))
            self.assertTrue(all(r['split'] == name and r['family_id'] == hash_text(r['context'])
                                for r in records))

    def test_merged_exclusions_preserve_all_inventoried_and_202_old_rows(self):
        excluded = json.loads((ROOT / 'configs/qa-remediation/exclusions.json').read_text())
        original = json.loads((ROOT / excluded['prior_inventory']).read_text())
        old_count = 0
        for path in excluded['historical_datasets']:
            records = [json.loads(line) for line in (ROOT / path).read_text().splitlines()]
            old_count += len(records); original = extend_exclusions(original, records)
        self.assertEqual(old_count, 202)
        for key in KEYS:
            self.assertEqual(original[key], excluded[key])


if __name__ == '__main__':
    unittest.main()
