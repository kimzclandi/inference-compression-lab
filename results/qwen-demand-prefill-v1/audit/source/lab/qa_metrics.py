"""SQuAD-style normalized EM/token F1; invalid/empty generations never earn abstention credit."""
from collections import Counter
import json
from pathlib import Path
import re
import string


def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def pair_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    em = float(p == g)
    if not p or not g:
        return em, float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    return em, 2 * common / (len(p) + len(g))


def score(row, prediction):
    raw = prediction['prediction'].strip()
    abstain = raw == 'NO_ANSWER'
    valid = bool(raw) and (abstain or raw in row['context'])
    if abstain:
        em = f1 = float(row['is_impossible'])
    elif not raw or row['is_impossible']:
        em = f1 = 0.0
    else:
        values = [pair_score(raw, a) for a in row['answers']]
        em, f1 = max(v[0] for v in values), max(v[1] for v in values)
    if not valid:
        category = 'format_or_nonextractive'
    elif row['is_impossible'] and not abstain:
        category = 'unsupported_answer'
    elif not row['is_impossible'] and abstain:
        category = 'over_abstention'
    elif em < 1:
        category = 'wrong_or_partial_span'
    else:
        category = 'correct'
    return dict(id=row['id'], em=em, f1=f1, format_valid=valid, abstain=abstain,
                is_impossible=row['is_impossible'], family_id=row['family_id'], category=category,
                root_cause='unverified' if em < 1 else None)


def evaluate(rows, predictions):
    ids = [p['id'] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in rows} or len(rows) != len({r['id'] for r in rows}):
        raise ValueError('Duplicate, missing or extra IDs: metrics require exact coverage.')
    by_id = {p['id']:p for p in predictions}
    scored = [score(r, by_id[r['id']]) for r in rows]
    def summary(items):
        if not items:
            return dict(n=0, em=None, f1=None, format_valid_rate=None)
        return dict(n=len(items), em=sum(x['em'] for x in items)/len(items),
                    f1=sum(x['f1'] for x in items)/len(items),
                    format_valid_rate=sum(x['format_valid'] for x in items)/len(items))
    no = [s for s in scored if s['is_impossible']]
    yes = [s for s in scored if not s['is_impossible']]
    result = dict(overall=summary(scored), answerable=summary(yes), unanswerable=summary(no),
                  unsupported_answer_rate=sum(not s['abstain'] for s in no)/len(no) if no else None,
                  over_abstention_rate=sum(s['abstain'] for s in yes)/len(yes) if yes else None,
                  categories=dict(Counter(s['category'] for s in scored)),
                  family_macro_em=sum(summary([s for s in scored if s['family_id']==f])['em']
                      for f in {s['family_id'] for s in scored}) / len({s['family_id'] for s in scored}),
                  always_abstain_em=len(no)/len(scored))
    return result, scored

