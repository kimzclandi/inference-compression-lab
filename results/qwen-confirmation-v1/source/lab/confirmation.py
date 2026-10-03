"""Paired, article-stratified family bootstrap for a frozen confirmation cohort."""
import random
from lab.quantization_diagnostics import index_by_id, paired


def comparison(data, baseline, candidate, config):
    a, b = index_by_id(baseline), index_by_id(candidate)
    if set(a) != set(b) or set(a) != set(index_by_id(data)):
        raise ValueError('Paired comparison requires exact ID coverage')
    strata = {}
    for row in data:
        family = strata.setdefault(row['source_title'], {}).setdefault(row['family_id'], [])
        family.append(b[row['id']]['em'] - a[row['id']]['em'])
    groups = [[(sum(items), len(items)) for _, items in sorted(families.items())]
              for _, families in sorted(strata.items())]
    rng = random.Random(config['seed']); values = []
    if config['replicates'] < 100:
        raise ValueError('Insufficient bootstrap replicates')
    for _ in range(config['replicates']):
        total, n = 0, 0
        for families in groups:
            for _ in families:
                delta, count = rng.choice(families); total += delta; n += count
        values.append(total / n)
    values.sort(); last = len(values) - 1
    return {**paired(baseline, candidate),
            'em_difference': sum(b[i]['em'] - a[i]['em'] for i in a) / len(a),
            'ci95': [values[int(last * .025)], values[int(last * .975)]],
            'families': sum(len(g) for g in groups), 'articles': len(groups)}


def summarize(data, metrics, scored, spec):
    comparisons = {name: comparison(data, scored[name], scored['selected'], spec['bootstrap'])
                   for name in ['q4', 'control', 'fp16']}
    f1_drop = metrics['fp16']['overall']['f1'] - metrics['selected']['overall']['f1']
    criteria = {'em_gain_over_q4': comparisons['q4']['ci95'][0] > 0,
                'em_gain_over_control': comparisons['control']['ci95'][0] > 0,
                'f1_drop_vs_fp16_at_most_0_02': f1_drop <= .02}
    return {'n': len(data), 'variants': {v: {'metrics': metrics[v],
             'em_count': int(sum(r['em'] for r in scored[v])),
             'format_valid_count': sum(r['format_valid'] for r in scored[v])} for v in spec['variants']},
            'selected_vs': comparisons, 'f1_drop_vs_fp16': f1_drop,
            'gate': {'criteria': criteria, 'passed': all(criteria.values())},
            'scope': spec['dataset_boundary'], 'bootstrap': spec['bootstrap']}
