import tempfile
from pathlib import Path
import unittest

from experiments.prepare_qa_expanded import choose
from experiments.qa_expanded import reserve_heads
from lab.qa_expanded_io import paired_coverage, write_shard


class Tokenizer:
    def encode(self, text, add_special_tokens=False): return text.split()


class ExpandedRankingTests(unittest.TestCase):
    def test_selector_caps_contexts_and_prevents_cross_role_reuse(self):
        raw = {'data': []}
        for title in ('one', 'two', 'three'):
            paragraphs = []
            for j in range(3):
                context = title+' passage '+str(j)
                qa = [dict(id=f'{title}-{j}-{i}-{imp}', question=f'question {title} {j} {i} {imp}',
                           is_impossible=imp, answers=[] if imp else [dict(text=title, answer_start=0)])
                      for imp in (False,True) for i in range(2)]
                paragraphs.append(dict(context=context,qas=qa))
            raw['data'].append(dict(title=title, paragraphs=paragraphs))
        blocked = {k:set() for k in ('ids','context_hashes','question_hashes')}; titles=set()
        first, one = choose(raw, 'train_new', 1, 2, 1, blocked, titles, Tokenizer())
        second, two = choose(raw, 'calibration', 1, 2, 1, blocked, titles, Tokenizer())
        self.assertEqual(len(first),4); self.assertEqual(len(second),4)
        self.assertTrue(set(one['titles']).isdisjoint(two['titles']))
        self.assertTrue({r['family_id'] for r in first}.isdisjoint(r['family_id'] for r in second))
        self.assertTrue({r['id'] for r in first}.isdisjoint(r['id'] for r in second))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            choose(raw,'evaluation',2,2,1,blocked,titles,Tokenizer())

    def test_hierarchical_paired_bootstrap_known_outcomes_and_id_guard(self):
        data=[dict(id=str(i),source_title=str(i//4),family_id=str(i//2),is_impossible=False) for i in range(8)]
        no=[dict(id=r['id'],accepted_correct=False) for r in data]
        yes=[dict(id=r['id'],accepted_correct=True) for r in data]
        result=paired_coverage(data,no,yes,replicates=100)
        self.assertEqual(result['correct_coverage_difference'],1.)
        self.assertEqual(result['ci95'],[1.,1.])
        self.assertEqual(paired_coverage(data,yes,yes,replicates=100)['ci95'],[0.,0.])
        self.assertEqual(paired_coverage(data,yes,no,replicates=100)['ci95'],[-1.,-1.])
        with self.assertRaises(ValueError):paired_coverage(data,no[:-1],yes,replicates=100)

    def test_local_heads_cannot_be_written_in_distributable_directory(self):
        with self.assertRaisesRegex(ValueError,'local ignored runs'):
            reserve_heads(Path('results/accidental-head'))

    def test_raw_shards_are_deterministic_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.gz',Path(tmp)/'b.gz'
            write_shard(a,[dict(id='one',margin=1.)]);write_shard(b,[dict(id='one',margin=1.)])
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with self.assertRaises(FileExistsError):write_shard(a,[])


if __name__ == '__main__':unittest.main()
