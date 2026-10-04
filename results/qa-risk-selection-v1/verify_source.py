"""Independent raw-SQuAD and old-role-conversion check; standard library only."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def verify(raw_path):
    root=Path(__file__).resolve().parents[2];base=root/'configs/qa-risk'
    raw_blob=raw_path.read_bytes();source_sha=hashlib.sha256(raw_blob).hexdigest()
    assert source_sha=='80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'
    raw=json.loads(raw_blob);indexed={}
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                assert qa['id'] not in indexed
                indexed[qa['id']]=(article['title'],paragraph['context'],qa)
    spec=json.loads((base/'selection.json').read_text())
    old={r['id']:r for path in spec['development_sources']
         for line in (root/path).read_text().splitlines() for r in [json.loads(line)]}
    assert len(old)==384
    ranked=sorted({row['source_title'] for row in old.values()},
                  key=lambda title:hashlib.sha256(('2026100408'+title).encode()).hexdigest())
    expected_titles={'train':set(ranked[:8]),'calibration':set(ranked[8:])}
    norm=lambda text:hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
    inventory=json.loads((base/'exclusions.json').read_text())
    seen={key:set() for key in ('ids','titles','context_hashes','question_hashes')}
    report={};offsets=0;reused={}
    for name,count in [('train',256),('calibration',128),('evaluation',128)]:
        blob=(base/'dataset'/name/'data.jsonl').read_bytes();rows=[json.loads(line) for line in blob.splitlines()]
        assert len(rows)==count
        counts=collections.Counter();families=collections.defaultdict(set);actual={key:set() for key in seen}
        for row in rows:
            title,context,qa=indexed[row['id']]
            assert row['source_title']==title and row['context']==context and row['question']==qa['question']
            assert row['is_impossible'] is qa['is_impossible'] and row['answers']==list(dict.fromkeys(a['text'] for a in qa['answers']))
            assert row['split']==name and row['family_id']==norm(context) and len(context)<=4000
            if name!='evaluation':
                original=old[row['id']]
                assert {k:v for k,v in row.items() if k!='split'}=={k:v for k,v in original.items() if k!='split'}
                reused[row['id']]=row
            for answer in qa['answers']:
                start=answer['answer_start'];assert type(start) is int and start>=0
                assert context[start:start+len(answer['text'])]==answer['text'];offsets+=1
            pair=(title,row['is_impossible']);counts[pair]+=1;families[pair].add(norm(context))
            actual['ids'].add(row['id']);actual['titles'].add(title)
            actual['context_hashes'].add(norm(context));actual['question_hashes'].add(norm(row['question']))
        assert len(actual['ids'])==count and len(actual['question_hashes'])==count
        assert len(actual['titles'])==count//32
        assert all(n==16 for n in counts.values()) and all(len(group)==16 for group in families.values())
        if name in expected_titles:assert actual['titles']==expected_titles[name]
        for key in seen:
            assert not seen[key]&actual[key];seen[key]|=actual[key]
            if name=='evaluation':assert not actual[key]&set(inventory[key])
        report[name]=dict(n=count,articles=len(actual['titles']),families=len(actual['context_hashes']),
                          data_sha256=hashlib.sha256(blob).hexdigest(),raw_fields_and_offsets_match=True,
                          cross_split_four_axis_disjointness=True)
    assert set(reused)==set(old)
    return dict(all_pass=True,source_sha256=source_sha,splits=report,answer_offsets_checked=offsets,
                old_384_rows_reused_exactly_once_with_only_split_changed=True,
                model_outputs_read_by_selector=False,
                scope='Previously observed specialist data is development; the four remaining public-dev articles are not model-unseen confirmation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().raw),ensure_ascii=False,indent=2))
