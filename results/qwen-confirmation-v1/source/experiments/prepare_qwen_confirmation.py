"""Select a new public SQuAD subset before model evaluation; no model imports."""
import argparse
import hashlib
import json
from pathlib import Path
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, write, sha


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized_hash(text):
    return digest(' '.join(text.lower().split()))


def select(raw, spec, exclusions):
    denied = {k: set(exclusions[k]) for k in ['ids', 'context_hashes', 'question_hashes', 'titles']}
    eligible = []
    for article in raw['data']:
        title = article['title']
        if title in denied['titles']:
            continue
        candidates = {False: [], True: []}
        for para in article['paragraphs']:
            context = para['context']; ch = normalized_hash(context)
            if ch in denied['context_hashes'] or len(context) > spec['max_context_chars']:
                continue
            for q in para['qas']:
                if q['id'] in denied['ids'] or normalized_hash(q['question']) in denied['question_hashes']:
                    continue
                answers = list(dict.fromkeys(a['text'] for a in q['answers']))
                if bool(answers) == q['is_impossible']:
                    raise ValueError('Invalid answerability annotation')
                for ans in q['answers']:
                    if context[ans['answer_start']:ans['answer_start'] + len(ans['text'])] != ans['text']:
                        raise ValueError('Answer offsets do not match passage')
                candidates[q['is_impossible']].append(dict(id=q['id'], context=context,
                    question=q['question'], answers=answers, is_impossible=q['is_impossible'],
                    source_title=title, family_id=ch, split='confirmation'))
        selected = []; used_questions = set()
        for impossible in [False, True]:
            used_contexts = set(); group = []
            for row in sorted(candidates[impossible], key=lambda r: digest(str(spec['seed']) + r['id'])):
                qh = normalized_hash(row['question'])
                if row['family_id'] in used_contexts or qh in used_questions:
                    continue
                group.append(row); used_contexts.add(row['family_id']); used_questions.add(qh)
                if len(group) == spec['per_class_per_article']:
                    break
            selected.extend(group)
        if len(selected) == 2 * spec['per_class_per_article']:
            eligible.append((title, selected))
    eligible.sort(key=lambda x: digest(str(spec['seed']) + x[0]))
    if len(eligible) < spec['articles']:
        raise ValueError('Insufficient eligible articles; no silent fallback')
    chosen = eligible[:spec['articles']]
    result = sorted([r for _, group in chosen for r in group], key=lambda r: r['id'])
    if len({normalized_hash(r['question']) for r in result}) != len(result):
        raise ValueError('Cross-article exact question duplicate; stop, do not adapt to results')
    return result, [t for t, _ in chosen]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=Path('configs/qwen-confirmation/selection.json'))
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); spec = read(a.spec); exclusions = read(spec['exclusions'])
    if sha(a.raw) != spec['source_sha256']:
        raise ValueError('Upstream source identity changed')
    records, titles = select(read(a.raw), spec, exclusions)
    out = reserve_directory(a.output_dir)
    with (out / 'data.jsonl').open('x') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write(out / 'manifest.json', {'selection': spec, 'exclusions_sha256': sha(spec['exclusions']),
        'data_sha256': sha(out / 'data.jsonl'), 'titles': titles, 'n': len(records),
        'families': len({r['family_id'] for r in records}),
        'unanswerable': sum(r['is_impossible'] for r in records),
        'scope': 'New to known project datasets, selected before inference. Public SQuAD dev subset, not official hidden test. Pretraining exposure unknown.'})
    print(json.dumps({'titles': titles, 'n': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
