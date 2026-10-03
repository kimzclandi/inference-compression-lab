"""Risk-head development reuse and the last fixed public-dev articles."""
import hashlib
import json
from pathlib import Path
import unittest
from experiments.prepare_qa_risk_data import allocate_development

ROOT=Path(__file__).resolve().parents[1]


class RiskSelectionTests(unittest.TestCase):
    def setUp(self):
        self.base=ROOT/'configs/qa-risk'
        self.spec=json.loads((self.base/'selection.json').read_text())
        self.old=[json.loads(line) for name in self.spec['development_sources']
                  for line in (ROOT/name).read_text().splitlines()]

    def test_all_observed_old_rows_reallocated_deterministically_without_changes(self):
        first=allocate_development(self.old,self.spec)
        self.assertEqual(first,allocate_development(list(reversed(self.old)),self.spec))
        original={row['id']:{k:v for k,v in row.items() if k!='split'} for row in self.old}
        allocated={row['id']:{k:v for k,v in row.items() if k!='split'}
                   for cohort in first.values() for row in cohort['records']}
        self.assertEqual(original,allocated)
        ranked=sorted({row['source_title'] for row in self.old},
                      key=lambda title:hashlib.sha256(('2026100408'+title).encode()).hexdigest())
        self.assertEqual(first['train']['titles'],ranked[:8])
        self.assertEqual(first['calibration']['titles'],ranked[8:])
        self.assertEqual(len(first['train']['records']),256)
        self.assertEqual(len(first['calibration']['records']),128)

    def test_incomplete_duplicate_or_missing_article_pool_rejected(self):
        for bad in (self.old[:-1],self.old+[self.old[0]],
                    [row for row in self.old if row['source_title']!=self.old[0]['source_title']]):
            with self.subTest(n=len(bad)),self.assertRaises(ValueError):
                allocate_development(bad,self.spec)

    def test_committed_splits_have_exact_balance_disjointness_and_expected_identity(self):
        norm=lambda text:hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
        inventory=json.loads((self.base/'exclusions.json').read_text())
        fields=('ids','titles','context_hashes','question_hashes')
        seen={key:set() for key in fields}
        for name,article_count in [('train',8),('calibration',4),('evaluation',4)]:
            blob=(self.base/'dataset'/name/'data.jsonl').read_bytes()
            rows=[json.loads(line) for line in blob.splitlines()]
            manifest=json.loads((self.base/'dataset'/name/'manifest.json').read_text())
            self.assertEqual(hashlib.sha256(blob).hexdigest(),manifest['data_sha256'])
            self.assertEqual(len(rows),article_count*32)
            actual={'ids':{r['id'] for r in rows},'titles':{r['source_title'] for r in rows},
                    'context_hashes':{norm(r['context']) for r in rows},
                    'question_hashes':{norm(r['question']) for r in rows}}
            self.assertEqual(len(actual['ids']),len(rows))
            self.assertEqual(len(actual['question_hashes']),len(rows))
            self.assertEqual(len(actual['titles']),article_count)
            for key in fields:
                self.assertFalse(seen[key]&actual[key]);seen[key]|=actual[key]
                if name=='evaluation':self.assertFalse(actual[key]&set(inventory[key]))
            for title in actual['titles']:
                for impossible in (False,True):
                    group=[r for r in rows if r['source_title']==title and r['is_impossible'] is impossible]
                    self.assertEqual(len(group),16)
                    self.assertEqual(len({norm(r['context']) for r in group}),16)
            self.assertTrue(all(r['split']==name and r['family_id']==norm(r['context']) for r in rows))
        self.assertEqual(manifest['eligible_articles'],4)
        self.assertIn('All old specialist 384 rows',self.spec['scope'])


if __name__=='__main__':unittest.main()
