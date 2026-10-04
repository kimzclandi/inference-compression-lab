"""Deterministic input-only expansion: public SQuAD train plus reserved dev rows.

No model outputs are read. Dev titles may have appeared in older Qwen studies,
but cannot appear in any prior correctness-head cohort. IDs, exact normalized
contexts and questions are excluded against the cumulative historical inventory.
Upstream benchmark exposure and semantic near-duplicates are not ruled out.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, rows, sha, write
from experiments.prepare_qa_remediation import normalized_hash

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026100414
SOURCES = {'train-v2.0.json': '68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002',
           'dev-v2.0.json': '80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8'}


def rank(value):
    return hashlib.sha256((str(SEED) + value).encode()).hexdigest()


def exclusions():
    old = read(ROOT / 'configs/qa-risk/exclusions.json')
    blocked = {k: set(old[k]) for k in ('ids', 'context_hashes', 'question_hashes')}
    head_titles = set()
    for split in ('train', 'calibration', 'evaluation'):
        for r in rows(ROOT / f'configs/qa-risk/dataset/{split}/data.jsonl'):
            head_titles.add(r['source_title'])
            blocked['ids'].add(r['id']); blocked['context_hashes'].add(normalized_hash(r['context']))
            blocked['question_hashes'].add(normalized_hash(r['question']))
    return blocked, set(old['titles']) | head_titles, head_titles


def choose(raw, split, count, quota, per_context, blocked, denied_titles, tokenizer):
    selected = []; titles = []; rejected = Counter()
    for article in sorted(raw['data'], key=lambda a: rank(a['title'])):
        title = article['title']
        if title in denied_titles:
            continue
        candidates = {False: [], True: []}
        for paragraph in article['paragraphs']:
            context = paragraph['context']; family = normalized_hash(context)
            if not context.strip() or len(context) > 4000 or family in blocked['context_hashes']:
                continue
            for q in paragraph['qas']:
                if q['id'] in blocked['ids'] or normalized_hash(q['question']) in blocked['question_hashes']:
                    continue
                impossible = q['is_impossible']
                answers = list(dict.fromkeys(a['text'] for a in q['answers']))
                if type(impossible) is not bool or bool(answers) == impossible:
                    raise ValueError('Invalid source answerability')
                if any(context[a['answer_start']:a['answer_start']+len(a['text'])] != a['text'] for a in q['answers']):
                    raise ValueError('Invalid source answer offsets')
                candidates[impossible].append(dict(id=q['id'], context=context, question=q['question'],
                    answers=answers, is_impossible=impossible, source_title=title, family_id=family, split=split))
        picked = []; question_hashes = set()
        for impossible in (False, True):
            family_counts = Counter(); group = []
            for row in sorted(candidates[impossible], key=lambda r: rank(r['id'])):
                qh = normalized_hash(row['question'])
                if qh in question_hashes or family_counts[row['family_id']] >= per_context:
                    continue
                if len(row['question']) > 1000 or len(tokenizer.encode(row['question'], add_special_tokens=False)) > 64:
                    rejected['question_limit'] += 1; continue
                group.append(row); question_hashes.add(qh); family_counts[row['family_id']] += 1
                if len(group) == quota:
                    break
            picked.extend(group)
        if len(picked) != 2 * quota:
            rejected['insufficient_article_quota'] += 1; continue
        selected.extend(picked); titles.append(title); denied_titles.add(title)
        for row in picked:
            blocked['ids'].add(row['id']); blocked['context_hashes'].add(row['family_id'])
            blocked['question_hashes'].add(normalized_hash(row['question']))
        if len(titles) == count:
            break
    if len(titles) != count:
        raise ValueError(f'Insufficient fixed-input articles: {split} {len(titles)}/{count}; no quota relaxation')
    return sorted(selected, key=lambda r: r['id']), dict(titles=titles, per_class_per_article=quota,
        max_questions_per_context_per_class=per_context, n=len(selected),
        context_families=len({r['family_id'] for r in selected}), input_rejections=dict(rejected))


def prepare(raw_root, asset_root, output):
    for name, digest in SOURCES.items():
        if sha(raw_root / name) != digest:
            raise ValueError('Public source identity changed: ' + name)
    from transformers import AutoTokenizer
    from lab.qa_specialist_runtime import validate_assets
    asset_sha = read(ROOT / 'configs/qa-risk/study.json')['asset_manifest_sha256']
    validate_assets(asset_root, asset_sha)
    tokenizer = AutoTokenizer.from_pretrained(asset_root, local_files_only=True, use_fast=True)
    blocked, training_titles, head_titles = exclusions()
    out = reserve_directory(output); manifest = dict(seed=SEED, sources=SOURCES, splits={},
        source_urls={n:'https://rajpurkar.github.io/SQuAD-explorer/dataset/'+n for n in SOURCES},
        tokenizer_asset_manifest_sha256=asset_sha, model_outputs_read=False,
        selector_sha256=sha(Path(__file__)),
        scope='New ranking-head data roles; public upstream benchmark exposure remains. Dev titles exclude prior head cohorts, while exact IDs/contexts/questions exclude cumulative historical inventory. Older Qwen title overlap is explicitly permitted.',
        license=dict(name='CC BY-SA 4.0', url='https://creativecommons.org/licenses/by-sa/4.0/legalcode.en',
                     attribution='SQuAD2: Pranav Rajpurkar, Robin Jia, Percy Liang; Wikipedia contributors'))
    new, info = choose(read(raw_root/'train-v2.0.json'), 'train_new', 64, 16, 1, blocked, training_titles, tokenizer)
    (out/'train_new.jsonl').write_text(''.join(__import__('json').dumps(r, ensure_ascii=False)+'\n' for r in new))
    manifest['splits']['train_new'] = dict(info, sha256=sha(out/'train_new.jsonl'), source='official_train')
    dev_titles = set(head_titles) | set(info['titles'])
    dev = read(raw_root/'dev-v2.0.json')
    for split in ('calibration', 'evaluation'):
        data, info = choose(dev, split, 4, 24, 2, blocked, dev_titles, tokenizer)
        (out/(split+'.jsonl')).write_text(''.join(__import__('json').dumps(r, ensure_ascii=False)+'\n' for r in data))
        manifest['splits'][split] = dict(info, sha256=sha(out/(split+'.jsonl')), source='official_dev')
    # Selection does not infer independence from file names or from new IDs.
    write(out/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); print(prepare(a.raw_root, a.asset_root, a.output_dir))
