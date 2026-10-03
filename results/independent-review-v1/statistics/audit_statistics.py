"""Independent, standard-library-only audit of frozen confirmation predictions.

No import from the project being audited. This reimplements the *frozen* scoring
and sampling definitions, not a new evaluation or tuning procedure. Output paths
must not exist; historic evidence is read only. --raw enables independent cohort
reconstruction from the pinned public SQuAD source file. Git is optional.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import subprocess
import sys


def load(path):
    return json.loads(path.read_text())


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def norm_hash(value):
    return sha_bytes(' '.join(value.lower().split()).encode())


def tokens(value):
    text = value.lower().translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\b(?:a|an|the)\b', ' ', text).split()


def pair(prediction, answer):
    p, a = tokens(prediction), tokens(answer)
    exact = int(p == a)
    if not p or not a:
        return exact, float(exact)
    overlap = sum(min(p.count(t), a.count(t)) for t in set(p))
    return exact, 2.0 * overlap / (len(p) + len(a))


def score(row, prediction):
    text = prediction.strip()
    abstain = text == 'NO_ANSWER'
    valid = bool(text) and (abstain or text in row['context'])
    if abstain:
        em, f1 = int(row['is_impossible']), float(row['is_impossible'])
    elif not text or row['is_impossible']:
        em, f1 = 0, 0.0
    else:
        scores = [pair(text, answer) for answer in row['answers']]
        em, f1 = max(s[0] for s in scores), max(s[1] for s in scores)
    category = ('format_or_nonextractive' if not valid else
                'unsupported_answer' if row['is_impossible'] and not abstain else
                'over_abstention' if not row['is_impossible'] and abstain else
                'wrong_or_partial_span' if not em else 'correct')
    return {'id': row['id'], 'em': em, 'f1': f1, 'format_valid': valid,
            'abstain': abstain, 'is_impossible': row['is_impossible'],
            'family_id': row['family_id'], 'category': category,
            'root_cause': None if em else 'unverified'}


def close(actual, expected):
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(close(v, expected[k]) for k, v in actual.items())
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(close(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    return actual == expected


def metrics(scored):
    def average(group):
        return {'n': len(group), 'em': sum(r['em'] for r in group) / len(group),
                'f1': math.fsum(r['f1'] for r in group) / len(group),
                'format_valid_rate': sum(r['format_valid'] for r in group) / len(group)}
    yes = [r for r in scored if not r['is_impossible']]
    no = [r for r in scored if r['is_impossible']]
    families = sorted({r['family_id'] for r in scored})
    categories = {c: sum(r['category'] == c for r in scored) for c in {r['category'] for r in scored}}
    return {'overall': average(scored), 'answerable': average(yes), 'unanswerable': average(no),
            'unsupported_answer_rate': sum(not r['abstain'] for r in no) / len(no),
            'over_abstention_rate': sum(r['abstain'] for r in yes) / len(yes),
            'categories': categories,
            'family_macro_em': math.fsum(average([r for r in scored if r['family_id'] == f])['em'] for f in families) / len(families),
            'always_abstain_em': len(no) / len(scored)}


def bootstrap(data, baseline, candidate, spec):
    # Resample complete families within each fixed article. Different family
    # sizes mean a replicate's denominator is not necessarily the original n.
    a = {r['id']: r['em'] for r in baseline}
    b = {r['id']: r['em'] for r in candidate}
    strata = []
    for title in sorted({r['source_title'] for r in data}):
        rows = [r for r in data if r['source_title'] == title]
        strata.append([[r['id'] for r in rows if r['family_id'] == family]
                       for family in sorted({r['family_id'] for r in rows})])
    rng = random.Random(spec['seed'])
    differences, denominators = [], []
    for _ in range(spec['replicates']):
        ids = []
        for families in strata:
            for _ in range(len(families)):
                ids.extend(families[rng.randrange(len(families))])
        differences.append(sum(b[i] - a[i] for i in ids) / len(ids))
        denominators.append(len(ids))
    ordered = sorted(differences)
    result = {'lost': sorted(i for i in a if a[i] == 1 and b[i] == 0),
              'gained': sorted(i for i in a if a[i] == 0 and b[i] == 1),
              'em_difference': sum(b[i] - a[i] for i in a) / len(a),
              'ci95': [ordered[math.floor((len(ordered)-1)*q)] for q in (0.025, 0.975)],
              'families': sum(len(s) for s in strata), 'articles': len(strata)}
    detail = {'replicates': differences, 'replicate_row_counts': denominators,
              'percentile_indices': [math.floor((len(ordered)-1)*q) for q in (0.025, 0.975)]}
    return result, detail


def reconstruct(raw, spec, denied):
    # Start from question records and validate gold spans before rank selection.
    eligible = {}
    for article in raw['data']:
        title = article['title']
        if title in denied['titles']:
            continue
        candidates = {False: [], True: []}
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if len(context) > spec['max_context_chars'] or norm_hash(context) in denied['context_hashes']:
                continue
            for item in paragraph['qas']:
                if item['id'] in denied['ids'] or norm_hash(item['question']) in denied['question_hashes']:
                    continue
                impossible = item['is_impossible']
                assert bool(item['answers']) != impossible
                assert all(context[a['answer_start']:a['answer_start']+len(a['text'])] == a['text'] for a in item['answers'])
                candidates[impossible].append({'id': item['id'], 'context': context,
                    'question': item['question'], 'answers': list(dict.fromkeys(a['text'] for a in item['answers'])),
                    'is_impossible': impossible, 'source_title': title,
                    'family_id': norm_hash(context), 'split': 'confirmation'})
        chosen, questions = [], set()
        for impossible in (False, True):
            families, selected = set(), []
            ranked = sorted(candidates[impossible], key=lambda r: sha_bytes((str(spec['seed'])+r['id']).encode()))
            for row in ranked:
                family, question = row['family_id'], norm_hash(row['question'])
                if family in families or question in questions:
                    continue
                selected.append(row)
                families.add(family)
                questions.add(question)
                if len(selected) == spec['per_class_per_article']:
                    break
            chosen.extend(selected)
        if len(chosen) == 2 * spec['per_class_per_article']:
            eligible[title] = chosen
    titles = sorted(eligible, key=lambda t: sha_bytes((str(spec['seed'])+t).encode()))[:spec['articles']]
    rows = sorted([r for t in titles for r in eligible[t]], key=lambda r: r['id'])
    assert len({norm_hash(r['question']) for r in rows}) == len(rows)
    return rows, titles, len(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--folder', default='results/qwen-confirmation-v1')
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace existing audit output')
    root, folder = args.root.resolve(), (args.root / args.folder).resolve()
    data = [json.loads(line) for line in (folder/'data.jsonl').read_text().splitlines()]
    spec, summary, selection = (load(folder/name) for name in ('protocol.json', 'summary.json', 'selection.json'))
    denied, manifest, run = (load(folder/name) for name in ('exclusions.json', 'dataset-manifest.json', 'run.json'))
    inputs = {str(p.relative_to(root)): sha_bytes(p.read_bytes()) for p in [
        folder/'data.jsonl', folder/'protocol.json', folder/'selection.json', folder/'exclusions.json',
        folder/'dataset-manifest.json', folder/'summary.json', folder/'run.json',
        *[folder/f'{v}-quality.json' for v in spec['variants']]]}
    checks = {}
    checks['exact_dataset_identity'] = sha_bytes((folder/'data.jsonl').read_bytes()) == spec['data_sha256'] == manifest['data_sha256']
    checks['unique_ids'] = len({r['id'] for r in data}) == len(data)
    checks['rows_128_and_balanced'] = len(data) == spec['quality']['expected_n'] == 128 and sum(r['is_impossible'] for r in data) == 64
    checks['families_85'] = len({r['family_id'] for r in data}) == 85
    checks['correct_families'] = all(r['family_id'] == norm_hash(r['context']) for r in data)
    checks['known_exclusions_disjoint'] = all(r['id'] not in denied['ids'] and r['source_title'] not in denied['titles'] and norm_hash(r['context']) not in denied['context_hashes'] and norm_hash(r['question']) not in denied['question_hashes'] for r in data)
    checks['article_class_unique_contexts'] = all(len(group) == len({r['family_id'] for r in group}) == 16 for title in manifest['titles'] for impossible in (False, True) for group in [[r for r in data if r['source_title']==title and r['is_impossible']==impossible]])
    scored, aggregates, by_variant, signatures = {}, {}, {}, []
    for variant in spec['variants']:
        result = load(folder/f'{variant}-quality.json')
        predictions = result['predictions']
        by_variant[variant] = {r['id']: r for r in predictions}
        checks[variant+'_exact_prediction_coverage'] = len(predictions) == len(by_variant[variant]) == len(data) and set(by_variant[variant]) == {r['id'] for r in data}
        scored[variant] = [score(row, by_variant[variant][row['id']]['prediction']) for row in data]
        aggregates[variant] = metrics(scored[variant])
        checks[variant+'_all_row_scores_match'] = close(scored[variant], result['scored'])
        checks[variant+'_all_metrics_match'] = close(aggregates[variant], result['metrics'])
        checks[variant+'_summary_metrics_match'] = close(aggregates[variant], summary['variants'][variant]['metrics'])
        signatures.append([(p['id'], p['prompt_sha256'], p['input_token_ids_sha256'], p['input_tokens']) for p in predictions])
    checks['identical_recorded_prompt_signatures'] = all(s == signatures[0] for s in signatures)
    comparisons, bootstrap_raw = {}, {}
    for baseline in ('q4', 'control', 'fp16'):
        comparisons[baseline], bootstrap_raw[baseline] = bootstrap(data, scored[baseline], scored['selected'], spec['bootstrap'])
        checks[baseline+'_paired_and_interval_match'] = close(comparisons[baseline], summary['selected_vs'][baseline])
    f1_drop = aggregates['fp16']['overall']['f1'] - aggregates['selected']['overall']['f1']
    criteria = {'em_gain_over_q4': comparisons['q4']['ci95'][0] > 0,
                'em_gain_over_control': comparisons['control']['ci95'][0] > 0,
                'f1_drop_vs_fp16_at_most_0_02': f1_drop <= .02}
    checks['gate_matches'] = criteria == summary['gate']['criteria'] and all(criteria.values()) == summary['gate']['passed']
    checks['f1_drop_matches'] = math.isclose(f1_drop, summary['f1_drop_vs_fp16'], abs_tol=1e-12)
    reconstruction = {'status': 'not_requested', 'limitation': 'Pass --raw with the pinned SQuAD source to audit exact cohort selection.'}
    if args.raw:
        raw_hash = sha_bytes(args.raw.read_bytes())
        checks['pinned_raw_identity'] = raw_hash == selection['source_sha256']
        rebuilt, titles, eligible = reconstruct(load(args.raw), selection, denied)
        checks['independent_cohort_reconstruction'] = rebuilt == data and titles == manifest['titles']
        reconstruction = {'status': 'complete', 'raw_sha256': raw_hash, 'rows_identical': rebuilt == data,
                          'title_order_identical': titles == manifest['titles'], 'eligible_articles': eligible}
    freeze = '5ca4191ec0b25aeece4bbc9736d87175ce98c1cd'
    chronology = {'freeze_commit': freeze, 'recorded_started': run['started'],
                  'caveat': 'Local Git and run timestamps support ordering; not independent timestamp attestation or proof that no unrecorded run occurred.'}
    try:
        timestamp = subprocess.check_output(['git','show','-s','--format=%cI',freeze], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        chronology['commit_timestamp'] = timestamp
        chronology['seconds_before_recorded_inference'] = (datetime.datetime.fromisoformat(run['started']) - datetime.datetime.fromisoformat(timestamp)).total_seconds()
        checks['recorded_freeze_precedes_inference'] = chronology['seconds_before_recorded_inference'] > 0
        matches = {}
        for original, recorded in [('configs/qwen-confirmation/study.json','protocol.json'), ('configs/qwen-confirmation/selection.json','selection.json'), ('configs/qwen-confirmation/exclusions.json','exclusions.json'), ('configs/qwen-confirmation/dataset/data.jsonl','data.jsonl'), ('lab/qa_metrics.py','source/lab/qa_metrics.py'), ('lab/confirmation.py','source/lab/confirmation.py'), ('experiments/qwen_confirmation.py','source/experiments/qwen_confirmation.py')]:
            frozen = subprocess.check_output(['git','show',freeze+':'+original], cwd=root, stderr=subprocess.DEVNULL)
            matches[original] = frozen == (folder/recorded).read_bytes()
        chronology['freeze_file_matches'] = matches
        checks['frozen_protocol_and_runner_match_archive'] = all(matches.values())
    except subprocess.CalledProcessError:
        chronology['git_status'] = 'unavailable; content/score checks remain runnable from an archive'
    # Preserve a concrete distinction: normalized EM need not satisfy the
    # experiment's exact-substring output contract.
    nonextractive_em = {v: [s['id'] for s in scored[v] if s['em'] and not s['format_valid']] for v in scored}
    hand_cases = []
    rows_by_id = {r['id']:r for r in data}
    for i in comparisons['q4']['gained']:
        r = rows_by_id[i]
        hand_cases.append({'id': i, 'question': r['question'], 'gold': r['answers'],
                           'q4': by_variant['q4'][i]['prediction'], 'selected': by_variant['selected'][i]['prediction'],
                           'q4_scores': score(r, by_variant['q4'][i]['prediction']),
                           'selected_scores': score(r, by_variant['selected'][i]['prediction'])})
    report = {'schema': 1, 'audit': 'Independent code path; no project metric/statistics imports; no model inference',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version, 'command': sys.argv, 'input_sha256': inputs,
        'all_checks_passed': all(checks.values()), 'checks': checks, 'reconstruction': reconstruction,
        'n_predictions': sum(len(v) for v in scored.values()),
        'variants': {v: {'em_count': sum(s['em'] for s in scored[v]), 'metrics': aggregates[v]} for v in scored},
        'comparisons': comparisons, 'f1_drop_vs_fp16': f1_drop,
        'gate': {'criteria': criteria, 'passed': all(criteria.values())}, 'chronology': chronology,
        'normalized_em_but_nonextractive_ids': nonextractive_em, 'gain_examples': hand_cases,
        'boundaries': [
            'All 640 existing predictions rescored; this is reproduction of disclosed data, not a fresh confirmation.',
            'Seeded family-bootstrap arithmetic matches; interval is conditional on four fixed articles and does not resample articles.',
            'Exclusion checks are against the recorded known-data inventory; unrecorded project exposure, pretraining contamination and semantic near-duplicates are not excluded.',
            'All five variants have zero correct abstentions on 64 unanswerable items and underperform the 50% always-abstain EM baseline overall.',
            'Two of three selected-vs-Q4 EM gains remove explanation text around an answer already present in the Q4 output; this is not three uniformly recovered factual answers.',
            'Normalized SQuAD-style EM and the exact-substring output contract are distinct; keep both metrics.',
            'Observed +3/128 and a zero lower confidence bound fail the prespecified gate; failure is not proof of zero benefit.',
            'Hash and timestamp consistency does not attest source authenticity or user mastery.']}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('audit.json', report), ('independently-scored.json', scored), ('bootstrap-replicates.json', bootstrap_raw)]:
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    (args.output/'sha256.json').write_text(json.dumps({p.name: sha_bytes(p.read_bytes()) for p in sorted(args.output.iterdir())}, indent=2)+'\n')
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'predictions': report['n_predictions'],
        'em_counts': {v: report['variants'][v]['em_count'] for v in scored}, 'gate': report['gate'],
        'selected_vs_q4': comparisons['q4'], 'f1_drop_vs_fp16': f1_drop}, indent=2))
    if not report['all_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
