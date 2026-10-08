"""CPU audit of archived native-generation arrivals, schedules and arithmetic.

This verifies receipts, not the unpublished score arrays or device performance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from lab.request_scheduling import AdmissionQueue, Request

# Post-run boundary hardening; archived timing source is unchanged.
CURRENT_POLICY_SHA256 = '44f28343fa2688e71696da106052fb84721db05aa85b5822a676fa182bfe99e0'
CURRENT_RUNNER_SHA256 = 'db41ef9d584bf26419c36d469fdc9a1014b1be8031f66b8bd57d552ce5c59aca'
RESOURCE_HELPER_SHA256 = 'e693686dddedfb4d69636bc3ac92e7257468fe6027eecefcd86dc387953ea137'
EXPECTED_RUN_SHA256 = '4de1218bbcc19059d4dfb53ca4d0c86560df4b66960051b12f5f6f5319d15843'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right):
    assert math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def verify(root=Path('results/qwen-request-scheduling-v1')):
    if not __debug__:
        raise RuntimeError('Evidence verification requires assertions; optimized Python is unsupported')
    root = Path(root)
    assert sha(root / 'run.json') == EXPECTED_RUN_SHA256, 'frozen receipt drift'
    run = json.loads((root / 'run.json').read_text())
    assert run['status'] == 'complete'
    spec = run['spec']
    assert len(run['protocol_commit']) == 40
    for name, expected in run['source_sha256'].items():
        assert sha(root / 'source' / name) == expected, 'archived source modified'
        current_expected = {'lab/request_scheduling.py': CURRENT_POLICY_SHA256,
                            'experiments/qwen_request_scheduling.py': CURRENT_RUNNER_SHA256}.get(name, expected)
        assert sha(Path(name)) == current_expected, 'current execution source drift'
    assert sha(Path('lab/request_resources.py')) == RESOURCE_HELPER_SHA256, 'resource helper drift'
    assert json.loads((root / 'source/configs/qwen-request-scheduling-v1.json').read_text()) == spec
    for name, expected in run['artifacts'].items():
        assert sha(root / name) == expected, 'artifact modified'
    requests = {r['id']: r for r in spec['requests']}
    assert len(requests) == 12
    checks = json.loads((root / 'correctness.json').read_text())
    assert {x['id'] for x in checks} == set(requests) and len(checks) == len(requests)
    expected = {row['id']: row['tokens'] for row in checks}
    for row in checks:
        assert row['tokens_equal'] and row['finite'] and row['logprobs_allclose']
        assert row['shape'][0] == requests[row['id']]['output_tokens'] and row['shape'][1] == 151936
        assert math.isfinite(row['max_abs']) and row['max_abs'] >= 0
        assert len(row['tokens']) == row['shape'][0]
        assert all(len(row[key]) == 64 for key in ('left_sha256', 'right_sha256'))
    lifecycle = json.loads((root / 'lifecycle.json').read_text())
    assert [x['kind'] for x in lifecycle] == ['cancel', 'failure']
    assert [x['outcome'] for x in lifecycle] == ['cancelled', 'injected_failure']
    assert all(x['delivered_count'] == 3 and x['recovery_tokens_equal'] for x in lifecycle)
    for row in lifecycle:
        assert row['delivered_tokens'] == expected[spec['requests'][0]['id']][:3]
        assert len(row['token_times_service_s']) == 3
        assert 0 <= row['token_times_service_s'][0] <= row['token_times_service_s'][-1] <= row['close_complete_service_s']
    trials = json.loads((root / 'trials.json').read_text())
    assert len(trials) == len(spec['traces']) * spec['rounds'] * len(spec['policies'])
    expected_order = []
    rng = random.Random(spec['seed'])
    for trace in spec['traces']:
        for rnd in range(spec['rounds']):
            policies = list(spec['policies']); rng.shuffle(policies)
            expected_order += [(trace['name'], rnd, order, policy) for order, policy in enumerate(policies)]
    assert [(x['trace'], x['round'], x['arm_order'], x['policy']) for x in trials] == expected_order
    traces = {x['name']: x for x in spec['traces']}
    for trial in trials:
        trace = traces[trial['trace']]
        arrivals = trial['arrivals']
        assert [a['id'] for a in arrivals] == trace['order']
        for i, arrival in enumerate(arrivals):
            close(arrival['scheduled_s'], i * trace['interval_s'])
            assert arrival['actual_s'] >= arrival['scheduled_s']
        actual = {a['id']: a['actual_s'] for a in arrivals}
        assert len(trial['rows']) == len(requests)
        assert {r['id'] for r in trial['rows']} == set(requests)
        queue = AdmissionQueue([Request(a['id'], a['actual_s'], requests[a['id']]['prompt_tokens'],
                                       requests[a['id']]['output_tokens']) for a in arrivals],
                               trial['policy'], spec['max_bypasses'], spec['decode_weight'])
        previous_end = 0
        for row in trial['rows']:
            req = queue.pop_ready(row['dispatch_s'])
            assert req and req.request_id == row['id'], 'policy schedule inconsistent with actual arrivals'
            assert row['outcome'] == 'complete' and row['tokens'] == expected[row['id']]
            close(row['arrival_s'], actual[row['id']])
            assert previous_end <= row['dispatch_s'] <= row['start'] <= row['token_times'][0]
            assert row['token_times'] == sorted(row['token_times']) and row['token_times'][-1] <= row['end']
            assert len(row['token_times']) == len(row['tokens'])
            close(row['queue_s'], row['dispatch_s']-row['arrival_s'])
            close(row['ttft_s'], row['token_times'][0]-row['arrival_s'])
            close(row['total_s'], row['end']-row['arrival_s'])
            close(row['service_s'], row['end']-row['dispatch_s'])
            close(row['tpot_s'], (row['token_times'][-1]-row['token_times'][0])/(len(row['tokens'])-1))
            assert row['allocated_kv_bytes'] > 0 and len(row['layer_offsets']) == 24
            previous_end = row['end']
        assert queue.bypasses == trial['bypasses'] and max(queue.bypasses.values()) <= spec['max_bypasses']
        assert previous_end <= trial['elapsed_s']
        assert 0 < trial['peak_mlx_bytes'] <= spec['peak_memory_budget_bytes']
        # Recompute directly from raw rows, without the runner's summarize helper.
        metrics = trial['metrics']
        assert metrics['requests'] == len(trial['rows'])
        assert metrics['tokens'] == sum(len(row['tokens']) for row in trial['rows'])
        close(metrics['elapsed_s'], trial['elapsed_s'])
        close(metrics['throughput_tokens_s'], metrics['tokens']/trial['elapsed_s'])
        for key in ('queue_s', 'ttft_s', 'tpot_s', 'total_s', 'service_s'):
            values = sorted(row[key] for row in trial['rows'])
            assert all(math.isfinite(value) and value >= 0 for value in values)
            close(metrics[key]['mean'], statistics.mean(values))
            for quantile, label in ((.5, 'p50'), (.95, 'p95')):
                pos = quantile*(len(values)-1)
                index = math.floor(pos)
                interpolated = values[index]*(1-(pos-index)) + values[min(index+1, len(values)-1)]*(pos-index)
                close(metrics[key][label], interpolated)
    result = {'evidence_valid': True, 'timed_requests': sum(len(t['rows']) for t in trials),
              'full_score_comparisons': len(checks), 'lifecycle_checks': len(lifecycle), 'traces': {},
              'scope': 'CPU verifies archived records, schedule, token identity and arithmetic; not GPU rerun or independent score-array replay.'}
    gates = spec['acceptance']
    for trace in spec['traces']:
        groups = {p: [t for t in trials if t['trace'] == trace['name'] and t['policy'] == p] for p in spec['policies']}
        stats = {}
        for p, rows in groups.items():
            stats[p] = {name: {agg: statistics.median(r['metrics'][name][agg] for r in rows)
                               for agg in ('mean', 'p50', 'p95')}
                        for name in ('queue_s', 'ttft_s', 'tpot_s', 'total_s', 'service_s')}
            stats[p]['throughput_tokens_s'] = statistics.median(r['metrics']['throughput_tokens_s'] for r in rows)
            stats[p]['peak_mlx_bytes'] = statistics.median(r['peak_mlx_bytes'] for r in rows)
        base, candidate = stats['fcfs'], stats['short_budget']
        ratios = dict(mean_ttft_speedup=base['ttft_s']['mean']/candidate['ttft_s']['mean'],
                      p95_total_ratio=candidate['total_s']['p95']/base['total_s']['p95'],
                      throughput_ratio=candidate['throughput_tokens_s']/base['throughput_tokens_s'],
                      peak_memory_ratio=candidate['peak_mlx_bytes']/base['peak_mlx_bytes'])
        faster = sum(b['metrics']['ttft_s']['mean'] > c['metrics']['ttft_s']['mean']
                     for b, c in zip(groups['fcfs'], groups['short_budget']))
        accepted = dict(mean_ttft=ratios['mean_ttft_speedup'] >= gates['min_mean_ttft_ratio'],
                        faster_rounds=faster >= gates['min_faster_rounds'],
                        p95_total=ratios['p95_total_ratio'] <= gates['max_p95_total_ratio'],
                        throughput=ratios['throughput_ratio'] >= gates['min_throughput_ratio'],
                        peak_memory=ratios['peak_memory_ratio'] <= gates['max_peak_memory_ratio'])
        result['traces'][trace['name']] = dict(stats=stats, ratios=ratios, faster_rounds=faster,
                                              gates=accepted, accepted=all(accepted.values()))
    result['accepted'] = all(row['accepted'] for row in result['traces'].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/qwen-request-scheduling-v1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(serialized)
    print(serialized)

if __name__ == '__main__':
    main()
