"""Freeze disjoint public QA calibration/confirmation cohorts without model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from lab.evidence import reserve_directory, sha256


KEYS = ('ids', 'titles', 'context_hashes', 'question_hashes')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized_hash(text):
    """The historical exclusion contract: lower case and collapsed whitespace."""
    return digest(' '.join(text.lower().split()))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n', encoding='utf-8')


def exclusion_sets(exclusions):
    if any(not isinstance(exclusions.get(k), list) for k in KEYS):
        raise ValueError('All four exclusion lists must be present')
    return {key: set(exclusions[key]) for key in KEYS}


def extend_exclusions(exclusions, records):
    denied = exclusion_sets(exclusions)
    for row in records:
        denied['ids'].add(row['id'])
        denied['titles'].add(row['source_title'])
        denied['context_hashes'].add(normalized_hash(row['context']))
        denied['question_hashes'].add(normalized_hash(row['question']))
    return {key: sorted(value) for key, value in denied.items()}


def validate_cohort(records, titles, spec, split, exclusions):
    """Reject coverage and leakage defects independently of candidate ranking."""
    n_articles = next(s['articles'] for s in spec['splits'] if s['name'] == split)
    per_class = spec['per_class_per_article']
    if len(titles) != n_articles or len(set(titles)) != n_articles:
        raise ValueError('Incorrect article coverage')
    if len(records) != n_articles * per_class * 2:
        raise ValueError('Incorrect question coverage')
    denied = exclusion_sets(exclusions)
    ids, questions, family_articles = set(), set(), {}
    counts = {(title, impossible): [] for title in titles for impossible in (False, True)}
    for row in records:
        qh, ch = normalized_hash(row['question']), normalized_hash(row['context'])
        title = row['source_title']
        if row['id'] in ids or qh in questions:
            raise ValueError('Duplicate ID or normalized question in cohort')
        if (row['id'] in denied['ids'] or title in denied['titles'] or
                qh in denied['question_hashes'] or ch in denied['context_hashes']):
            raise ValueError('Cohort overlaps excluded data')
        if title not in titles or row['split'] != split or row['family_id'] != ch:
            raise ValueError('Incorrect title, split or context family identity')
        if type(row['is_impossible']) is not bool or bool(row['answers']) == row['is_impossible']:
            raise ValueError('Invalid answerability annotation')
        if len(row['context']) > spec['max_context_chars']:
            raise ValueError('Context exceeds fixed size limit')
        if ch in family_articles and family_articles[ch] != title:
            raise ValueError('Context family appears in more than one article')
        family_articles[ch] = title
        counts[title, row['is_impossible']].append(ch)
        ids.add(row['id']); questions.add(qh)
    if any(len(group) != per_class or len(set(group)) != per_class for group in counts.values()):
        raise ValueError('Each article/class requires the fixed number of unique contexts')


def select_split(raw, spec, exclusions, split):
    """Use the existing seed+ID/title hash rank; never read predictions or scores."""
    per_class = spec['per_class_per_article']
    n_articles = next(s['articles'] for s in spec['splits'] if s['name'] == split)
    denied = exclusion_sets(exclusions)
    eligible = []
    seen_titles = set()
    for article in raw['data']:
        title = article['title']
        if title in seen_titles:
            raise ValueError('Duplicate article title in upstream source')
        seen_titles.add(title)
        if title in denied['titles']:
            continue
        candidates = {False: [], True: []}
        for paragraph in article['paragraphs']:
            context = paragraph['context']; ch = normalized_hash(context)
            if ch in denied['context_hashes'] or len(context) > spec['max_context_chars']:
                continue
            for question in paragraph['qas']:
                if (question['id'] in denied['ids'] or
                        normalized_hash(question['question']) in denied['question_hashes']):
                    continue
                impossible = question['is_impossible']
                answers = list(dict.fromkeys(a['text'] for a in question['answers']))
                if type(impossible) is not bool or bool(answers) == impossible:
                    raise ValueError('Invalid answerability annotation')
                for answer in question['answers']:
                    start = answer['answer_start']
                    if start < 0 or context[start:start + len(answer['text'])] != answer['text']:
                        raise ValueError('Answer offsets do not match passage')
                candidates[impossible].append(dict(
                    id=question['id'], context=context, question=question['question'],
                    answers=answers, is_impossible=impossible, source_title=title,
                    family_id=ch, split=split))
        chosen = []; used_questions = set()
        for impossible in (False, True):
            used_contexts, group = set(), []
            for row in sorted(candidates[impossible], key=lambda r: digest(str(spec['seed']) + r['id'])):
                qh = normalized_hash(row['question'])
                if row['family_id'] in used_contexts or qh in used_questions:
                    continue
                group.append(row); used_contexts.add(row['family_id']); used_questions.add(qh)
                if len(group) == per_class:
                    break
            chosen.extend(group)
        if len(chosen) == 2 * per_class:
            eligible.append((title, chosen))
    eligible.sort(key=lambda group: digest(str(spec['seed']) + group[0]))
    if len(eligible) < n_articles:
        raise ValueError(f'Insufficient eligible articles for {split}: '
                         f'{len(eligible)} < {n_articles}; no silent fallback')
    titles = [title for title, _ in eligible[:n_articles]]
    records = sorted([row for _, group in eligible[:n_articles] for row in group],
                     key=lambda row: row['id'])
    validate_cohort(records, titles, spec, split, exclusions)
    return records, titles, len(eligible)


def select_cohorts(raw, spec, exclusions):
    if ([s['name'] for s in spec['splits']] != ['calibration', 'confirmation'] or
            any(type(s['articles']) is not int or s['articles'] <= 0 for s in spec['splits']) or
            type(spec['per_class_per_article']) is not int or spec['per_class_per_article'] <= 0 or
            type(spec['seed']) is not int or type(spec['max_context_chars']) is not int or
            spec['max_context_chars'] <= 0):
        raise ValueError('Invalid fixed selection protocol')
    result = {}; current = exclusions
    for split in spec['splits']:
        name = split['name']
        records, titles, eligible_count = select_split(raw, spec, current, name)
        result[name] = dict(records=records, titles=titles, eligible_articles=eligible_count,
                            effective_exclusion_counts={key: len(current[key]) for key in KEYS})
        current = extend_exclusions(current, records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/qa-remediation/selection.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    if sha256(args.raw) != spec['source_sha256']:
        raise ValueError('Upstream source identity changed')
    exclusions_path = Path(spec['exclusions'])
    if sha256(exclusions_path) != spec['exclusions_sha256']:
        raise ValueError('Exclusion inventory identity changed')
    exclusions = read_json(exclusions_path)
    # Verify the recorded sources even in a Git-free source archive.
    for path, expected in exclusions['local_sources_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Historical exclusion source identity changed: {path}')
    # Prevent a resigned omission from the merged exclusion lists.
    original = read_json(exclusions['prior_inventory'])
    for path in exclusions['historical_datasets']:
        original = extend_exclusions(original, read_rows(path))
    if any(sorted(exclusions[key]) != original[key] for key in KEYS):
        raise ValueError('Merged exclusion inventory is incomplete')
    selected = select_cohorts(read_json(args.raw), spec, exclusions)
    out = reserve_directory(args.output_dir)
    manifests = {}
    for name, cohort in selected.items():
        folder = reserve_directory(out / name)
        (folder / 'data.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
            for row in cohort['records']), encoding='utf-8')
        manifest = dict(split=name, selection_sha256=sha256(args.spec),
                        exclusions_sha256=sha256(exclusions_path),
                        data_sha256=sha256(folder / 'data.jsonl'), titles=cohort['titles'],
                        n=len(cohort['records']),
                        families=len({row['family_id'] for row in cohort['records']}),
                        unanswerable=sum(row['is_impossible'] for row in cohort['records']),
                        eligible_articles=cohort['eligible_articles'],
                        effective_exclusion_counts=cohort['effective_exclusion_counts'],
                        license=spec['license'], scope=spec['scope'])
        write_json(folder / 'manifest.json', manifest)
        manifests[name] = manifest
    write_json(out / 'manifest.json', dict(
        selection=spec, selection_sha256=sha256(args.spec),
        exclusions_sha256=sha256(exclusions_path), splits=manifests,
        cross_split_disjoint=['source_title', 'id', 'normalized context', 'normalized question'],
        model_outputs_read=False))
    write_json(out / 'command.json', dict(
        argv=[sys.executable, *sys.argv], source_sha256=sha256(args.raw),
        selector_sha256=sha256(Path(__file__)), selection_sha256=sha256(args.spec),
        exclusions_sha256=sha256(exclusions_path)))
    print(json.dumps({name: {'n': value['n'], 'families': value['families'], 'titles': value['titles']}
                      for name, value in manifests.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
