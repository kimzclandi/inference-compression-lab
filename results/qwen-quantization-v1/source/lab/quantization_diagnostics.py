"""Dependency-free analysis of first-token interventions and paired QA outcomes."""
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    # Result directories are reserved by the caller. Individual artifacts cannot clobber.
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_by_id(records):
    indexed = {r['id']: r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate sample ID')
    return indexed


def paired(base, candidate):
    b, c = index_by_id(base), index_by_id(candidate)
    if set(b) != set(c):
        raise ValueError('Paired comparison requires identical IDs')
    return {
        'lost': [i for i in b if b[i]['em'] == 1 and c[i]['em'] == 0],
        'gained': [i for i in b if b[i]['em'] == 0 and c[i]['em'] == 1],
    }


def rank_blocks(records, spec):
    expected = set(spec['diagnosis_ids'])
    ranking = []
    for block in spec['screen_blocks']:
        group = [r for r in records if r['block'] == block]
        if len(group) != len(expected) or set(index_by_id(group)) != expected:
            raise ValueError('Missing/duplicate screen cell')
        ranking.append({
            'block': block,
            'restored_top1': sum(r['top1'] == r['reference_top1'] for r in group),
            'mean_abs_pair_margin_change': statistics.mean(
                abs(r['reference_pair_margin'] - r['fp16_margin']) for r in group),
        })
    if len(records) != len(expected) * len(spec['screen_blocks']):
        raise ValueError('Unexpected screen cell')
    return sorted(ranking, key=lambda r: (-r['restored_top1'],
                                         r['mean_abs_pair_margin_change'], r['block']))


def performance(timings):
    if not timings or any(r['generated_tokens'] != 32 or len(r['token_ids']) != 32
                          or r['decode_seconds'] <= 0 or r['ttft_seconds'] <= 0
                          for r in timings):
        raise ValueError('Expected nonempty forced32 timing records')
    return {'mean_ttft_seconds': statistics.mean(r['ttft_seconds'] for r in timings),
            'decode_tokens_per_second': sum(r['generated_tokens'] - 1 for r in timings)
                / sum(r['decode_seconds'] for r in timings),
            'mean_total_seconds': statistics.mean(r['total_seconds'] for r in timings)}


def gate(variants, limits):
    b, q, s, c = (variants[v] for v in ['fp16', 'q4', 'selected', 'control'])
    em = lambda x: x['em_count']
    checks = {
        'em_gain_vs_q4': em(s) - em(q) >= limits['min_net_em_gain_vs_q4'],
        'em_gain_vs_control': em(s) - em(c) >= limits['min_net_em_gain_vs_control'],
        'em_vs_fp16': em(s) - em(b) >= limits['min_em_vs_fp16'],
        'f1_vs_fp16': s['quality']['overall']['f1'] + limits['max_f1_drop_vs_fp16']
                      >= b['quality']['overall']['f1'],
        'weight_vs_fp16': s['weight_bytes'] <= b['weight_bytes'] * limits['max_weight_ratio_vs_fp16'],
        'rss_vs_fp16': s['performance']['rss_peak_sampled_bytes'] <=
                      b['performance']['rss_peak_sampled_bytes'] * limits['max_rss_ratio_vs_fp16'],
        'ttft_vs_q4': s['performance']['mean_ttft_seconds'] <=
                     q['performance']['mean_ttft_seconds'] * limits['max_ttft_ratio_vs_q4'],
        'decode_vs_q4': s['performance']['decode_tokens_per_second'] >=
                       q['performance']['decode_tokens_per_second'] * limits['min_decode_ratio_vs_q4'],
    }
    return {'checks': checks, 'pass_all': all(checks.values()),
            'scope': 'Exploratory dev-only candidate gate; not independent confirmation or deployment readiness.'}
