"""Independent raw-source check: no selector/helper or model imports."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def verify(raw_path):
    root = Path(__file__).resolve().parents[2]
    base = root / 'configs/qa-specialist'
    raw_bytes = raw_path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'
    raw = json.loads(raw_bytes)
    indexed = {}
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                assert qa['id'] not in indexed
                indexed[qa['id']] = (article['title'], paragraph['context'], qa)
    digest = lambda text: hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
    inventory = json.loads((base / 'exclusions.json').read_text())
    forbidden = {key: set(inventory[key]) for key in ('ids', 'titles', 'context_hashes', 'question_hashes')}
    result = {}
    offsets = 0
    for name, articles in [('calibration', 4), ('evaluation', 8)]:
        blob = (base / 'dataset' / name / 'data.jsonl').read_bytes()
        rows = [json.loads(line) for line in blob.splitlines()]
        assert len(rows) == articles * 32
        counts = collections.Counter()
        families = collections.defaultdict(set)
        actual = {key: set() for key in forbidden}
        for row in rows:
            title, context, qa = indexed[row['id']]
            assert row['source_title'] == title and row['context'] == context
            assert row['question'] == qa['question'] and row['is_impossible'] is qa['is_impossible']
            assert row['answers'] == list(dict.fromkeys(a['text'] for a in qa['answers']))
            assert row['split'] == name and row['family_id'] == digest(context)
            assert len(context) <= 4000 and bool(row['answers']) != row['is_impossible']
            for answer in qa['answers']:
                start = answer['answer_start']; assert type(start) is int and start >= 0
                assert context[start:start + len(answer['text'])] == answer['text']
                offsets += 1
            key = (title, qa['is_impossible'])
            counts[key] += 1; families[key].add(digest(context))
            actual['ids'].add(row['id']); actual['titles'].add(title)
            actual['context_hashes'].add(digest(context)); actual['question_hashes'].add(digest(qa['question']))
        assert len(actual['ids']) == len(rows) and len(actual['question_hashes']) == len(rows)
        assert len(actual['titles']) == articles and len(counts) == articles * 2
        assert all(n == 16 for n in counts.values())
        assert all(len(group) == 16 for group in families.values())
        for key in forbidden:
            assert not actual[key] & forbidden[key], (name, key)
            forbidden[key] |= actual[key]
        result[name] = dict(n=len(rows), articles=articles, families=len(actual['context_hashes']),
                            unanswerable=sum(row['is_impossible'] for row in rows),
                            data_sha256=hashlib.sha256(blob).hexdigest(),
                            raw_fields_equal=True, local_disjointness=True)
    return dict(all_pass=True, source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                splits=result, answer_offsets_checked=offsets, model_outputs_read=False,
                scope='Local public-benchmark assessment; not model-unseen confirmation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().raw), ensure_ascii=False, indent=2, allow_nan=False))
