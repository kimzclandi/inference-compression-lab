"""Independent standard-library audit of archived performance, not a model rerun.

Run from the repository root: python3 results/independent-review-v1/performance/recompute.py
No lab/ or experiments/ implementation or verifier imports. Never changes evidence.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[3]
READS = {}


def read(name):
    path = ROOT / name
    data = path.read_bytes()
    READS[str(path.relative_to(ROOT))] = sha256(data).hexdigest()
    return json.loads(data)


def close(a, b, label='number'):
    assert math.isfinite(a) and math.isfinite(b), label
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (label, a, b)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [None] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j + 1) / 2
        i = j
    return result


def spearman(gold, pred):
    x, y = ranks(gold), ranks(pred)
    a, b = st.mean(x), st.mean(y)
    return sum((u-a)*(v-b) for u, v in zip(x, y)) / math.sqrt(
        sum((u-a)**2 for u in x)*sum((v-b)**2 for v in y))


def complete(folder, field):
    manifest = read(f'{folder}/complete.json')[field]
    present = {p.name for p in (ROOT/folder).glob('*.json')} - {'complete.json'}
    assert set(manifest) == present and len(manifest) > 0, ('manifest coverage', folder)
    for name, digest in manifest.items():
        read(f'{folder}/{name}')
        assert READS[f'{folder}/{name}'] == digest
    return len(manifest)


def mini_runtime():
    result = {}
    for folder in ['minilm-runtime-study-v1', 'minilm-short-request-check-v2']:
        groups = defaultdict(list)
        all_records = []
        for path in sorted((ROOT/f'results/{folder}/confirm/raw').glob('*.json')):
            r = read(str(path.relative_to(ROOT)))
            samples = r['timing']['repeats'][0]['samples_ms']
            assert len(samples) == 200 and all(math.isfinite(x) and x > 0 for x in samples)
            close(st.median(samples), r['timing']['repeats'][0]['median_ms'], 'raw median')
            groups[r['scope'], r['name']].append((r['round'], st.median(samples)))
            all_records.append(r)
        assert len(all_records) == 64
        medians = {}
        for (scope, name), vals in groups.items():
            assert len(vals) == 8 and {r for r, _ in vals} == set(range(8))
            medians[f'{scope}/{name}'] = st.median(v for _, v in vals)
        predictions = {}
        for path in sorted((ROOT/f'results/{folder}/quality').glob('*-predictions.json')):
            rows = read(str(path.relative_to(ROOT)))
            assert len(rows) in (1379, 973)
            predictions[path.name] = {'n': len(rows), 'spearman': spearman(
                [r['gold'] for r in rows], [r['cosine'] for r in rows])}
        result[folder] = {'raw_files': len(all_records), 'raw_timing_samples': 200*len(all_records),
                          'median_of_round_medians_ms': medians, 'quality': predictions}
    primary = result['minilm-runtime-study-v1']['median_of_round_medians_ms']
    result['same_int8_4_to_8_threads_reduction'] = {
        s: 1-primary[f'{s}/quantized-tuned']/primary[f'{s}/quantized-original']
        for s in ['session', 'pipeline']}
    result['int8_vs_tuned_fp32_latency_ratio'] = {
        s: primary[f'{s}/quantized-tuned']/primary[f'{s}/fp32-tuned'] for s in ['session', 'pipeline']}
    return result


def mini_quantization():
    folder = 'results/minilm-mac-m4max-layer-errors-optimized'
    raw = read(folder+'/raw-errors.json')
    groups = defaultdict(list)
    for r in raw:
        groups[r['kind'], r['variant'], r['node']].append(r)
    assert len(groups) == 120 and len(raw) == 960
    saved = {(r['kind'], r['variant'], r['node']): r for r in read(folder+'/summary.json')}
    local = []
    for key, records in groups.items():
        assert len(records) == 8 and {r['batch_start'] for r in records} == set(range(0,128,16))
        total = {k: sum(r[k] for r in records) for k in ['n', 'sse', 'reference_energy', 'candidate_energy', 'dot']}
        total['max_abs'] = max(r['max_abs'] for r in records)
        total['mse'] = total['sse']/total['n']
        total['nmse'] = total['sse']/total['reference_energy']
        total['cosine'] = total['dot']/math.sqrt(total['reference_energy']*total['candidate_energy'])
        for k, v in total.items():
            close(v, saved[key][k], 'local/cumulative error sums')
        if key[0] == 'local':
            local.append({'node': key[2], 'nmse': total['nmse']})
    chosen = sorted(local, key=lambda r: (-r['nmse'], r['node']))[0]
    assert read(folder+'/exclusion.json')['nodes_to_exclude'] == [chosen['node']]
    graphs = {}
    for variant in ['fp32','int8_per_channel','int8_per_channel_excluded']:
        execution = read(f'results/minilm-mac-m4max-ablation/{variant}-execution.json')
        events = Counter(r['args']['op_name'] for r in execution['profile_node_events']
                         if r.get('args', {}).get('provider') == 'CPUExecutionProvider')
        integer_events = sum(events[name] for name in ['DynamicQuantizeMatMul','MatMulIntegerToFloat'])
        expected = {'fp32':0, 'int8_per_channel':36, 'int8_per_channel_excluded':35}[variant]
        assert len(execution['integer_matmul_nodes']) == integer_events == expected
        graphs[variant] = {'integer_related_profile_events': integer_events,
                           'remaining_fp_matmul_nodes': len(execution['remaining_float_matmul_nodes'])}
    ablation = read('results/minilm-mac-m4max-ablation/summary.json')
    return {'raw_error_rows': len(raw), 'groups': len(groups), 'selected': chosen,
            'executed_cpu_integer_events': graphs,
            'historical_ablation_summary_only': [{k:r[k] for k in ['variant','spearman','model_bytes','embedding_mse_vs_fp32']} for r in ablation],
            'limits': 'Recomputed stored sums, not original hidden tensors. Instrumented graphs are not performance benchmarks; local NMSE alone is not task sensitivity.'}


def bucketing():
    folder = 'results/minilm-bucketing-confirm-v1'
    hashes = complete(folder, 'files_sha256')
    timings, summary = read(folder+'/timings.json'), read(folder+'/summary.json')
    assert len(timings) == 28
    expected = set(product(range(7), ['fp32', 'int8_per_channel'], [False, True]))
    assert {(r['round'], r['precision'], r['bucket']) for r in timings} == expected
    output = {}
    for precision, bucket in product(['fp32', 'int8_per_channel'], [False, True]):
        subset = [r for r in timings if (r['precision'], r['bucket']) == (precision, bucket)]
        assert all(r['stats'] == subset[0]['stats'] for r in subset)
        stats = subset[0]['stats']
        assert (stats['requests'], stats['batches'], stats['valid_tokens']) == (2758, 345, 39028)
        assert stats['padded_tokens'] == (42072 if bucket else 54416)
        predictions = read(f'{folder}/predictions-{precision}-{int(bucket)}.json')
        assert len(predictions) == 1379 and [r['row'] for r in predictions] == list(range(1379))
        rho = spearman([r['label'] for r in predictions], [r['cosine'] for r in predictions])
        saved = next(r for r in summary if (r['precision'], r['bucket']) == (precision, bucket))
        close(rho, saved['spearman'], 'independent rank correlation')
        seconds = st.median(r['seconds'] for r in subset)
        close(seconds, saved['median_seconds'], 'bucketing median')
        output[f'{precision}/{int(bucket)}'] = {'seconds': seconds, 'spearman': rho, 'stats': stats}
    return {'complete_files': hashes, 'rows': output, 'time_reduction': {
        p: 1-output[f'{p}/1']['seconds']/output[f'{p}/0']['seconds'] for p in ['fp32', 'int8_per_channel']},
        'padding_slot_reduction': 1-42072/54416}


def check_decode(o):
    assert o['ttft_seconds'] > 0 and o['decode_seconds'] > 0
    close(o['ttft_seconds']+o['decode_seconds'], o['total_seconds'], 'TTFT plus decode')
    if 'decode_tokens_per_second' in o:
        close((len(o['token_ids'])-1)/o['decode_seconds'], o['decode_tokens_per_second'], 'N minus one')


def fair():
    folder = 'results/qwen-prefix-fair-v1'
    hashes = complete(folder, 'sha256')
    records = read(folder+'/timings.json')
    modes = ['full', 'legacy_segmented', 'direct_segmented', 'cached_workload', 'warm_hit']
    expected = set(product(['historical', 'confirmation'], [64, 2048], [False, True], range(5), modes))
    index = {(r['dataset'], r['prefix_tokens'], r['profile'], r['round'], r['mode']): r for r in records}
    assert len(records) == len(index) == 200 and set(index) == expected
    for key, r in index.items():
        baseline = index[key[:2]+(False, key[3], 'full')]
        assert len(r['outputs']) == 4
        assert r['seconds'] >= sum(o['total_seconds'] for o in r['outputs'])
        for i, o in enumerate(r['outputs']):
            check_decode(o)
            assert len(o['token_ids']) == 32 and o['token_ids'] == baseline['outputs'][i]['token_ids']
            if r['profile']:
                assert all(math.isfinite(x) and x >= 0 for x in o['phases'].values())
                close(sum(o['phases'].values()), o['total_seconds'], 'phase sum')
    output = []
    for dataset, length in product(['historical', 'confirmation'], [64, 2048]):
        cell = [r for r in records if (r['dataset'], r['prefix_tokens'], r['profile']) == (dataset, length, False)]
        medians = {m: st.median(r['seconds'] for r in cell if r['mode'] == m) for m in modes}
        rates = {m: st.median((len(o['token_ids'])-1)/o['decode_seconds']
                 for r in cell if r['mode'] == m for o in r['outputs']) for m in modes}
        output.append({'dataset': dataset, 'prefix_tokens': length, 'seconds': medians,
                       'cache_reduction_vs_direct': 1-medians['cached_workload']/medians['direct_segmented'],
                       'decode_tps_median': rates})
    quality = read('results/qwen-prefix-v2/quality-predictions.json')
    correct = {m: 0 for m in ['cold', 'cached']}
    for r in quality:
        assert r['cold']['token_ids'] == r['cached']['token_ids']
        for m in correct:
            answer = r[m]['text'].strip()
            valid = answer == 'NO_ANSWER' if r['is_impossible'] else answer in r['references']
            assert valid == r[m]['strict_correct']
            correct[m] += valid
    return {'complete_files': hashes, 'all_800_saved_requests_match_full': True,
            'uninstrumented_requests': 400, 'instrumented_requests': 400, 'performance': output,
            'historical_quality': {'n': len(quality), 'strict_correct': correct,
             'always_abstain_correct': sum(r['is_impossible'] for r in quality)}}


def lifecycle():
    folder = 'results/qwen-cache-lifecycle-gpu-v1'
    hashes = complete(folder, 'sha256')
    records = read(folder+'/timings.json')
    modes = ['direct', 'entries2', 'entries3', 'entries3_bytes2']
    index = {(r['length'], r['round'], r['mode']): r for r in records}
    assert len(records) == len(index) == 40
    assert set(index) == set(product([64, 1024], range(5), modes))
    result = []
    for key, r in index.items():
        residents = []
        counters = dict(hits=0, misses=0, evictions=0, bypasses=0, failures=0)
        assert len(r['outputs']) == 12
        for i, o in enumerate(r['outputs']):
            check_decode(o)
            assert o['token_ids'] == index[(key[0], key[1], 'direct')]['outputs'][i]['token_ids']
            assert len(o['token_ids']) == 8
            if key[2] != 'direct':
                token_family = i % 3
                capacity = min(r['max_entries'], r['max_bytes']//(12288*key[0]))
                if token_family in residents:
                    residents.remove(token_family)
                    status = 'hit'
                    counters['hits'] += 1
                else:
                    status = 'miss'
                    counters['misses'] += 1
                    if len(residents) == capacity:
                        residents.pop(0)
                        counters['evictions'] += 1
                residents.append(token_family)
                assert o['status'] == status
                assert o['cache']['stored_tensor_bytes'] == len(residents)*12288*key[0]
        assert counters == r['cache_delta']
    for length in [64, 1024]:
        medians = {m: st.median(r['seconds'] for r in records if (r['length'], r['mode']) == (length, m)) for m in modes}
        result.append({'prefix_tokens': length, 'seconds': medians,
                       'three_entry_reduction': 1-medians['entries3']/medians['direct']})
    contracts = read(folder+'/contracts.json')
    assert len(contracts['faults']) == 5
    for r in contracts['faults']:
        assert r['before']['resident_tokens'] == r['after']['resident_tokens']
        for k, v in r['before']['stats'].items():
            assert r['after']['stats'][k] == v+int(k == 'failures')
        assert [o['token_ids'] for o in r['replay']] == r['reference_token_ids']
    return {'complete_files': hashes, 'request_count': 480, 'lru_and_byte_simulation_match': True,
            'five_injected_faults_preserve_state_and_replay': True, 'performance': result}


def quantization():
    folder = 'results/qwen-quantization-v1'
    variants = ['fp16', 'q4', 'q8', 'selected', 'control']
    outputs, pids = {}, []
    saved = read(folder+'/summary.json')['variants']
    prompt_ids = None
    for v in variants:
        runs = []
        tokens = None
        for r in range(5):
            data = read(f'{folder}/bench-{r}-{v}/run.json')
            assert data['status'] == 'complete' and data['variant'] == v
            pids.append(data['pid'])
            rows = data['timings']
            assert len(rows) == 8
            ids = [(o['id'], o['input_token_ids_sha256'], o['input_tokens']) for o in rows]
            prompt_ids = ids if prompt_ids is None else prompt_ids
            assert ids == prompt_ids
            generated = [o['token_ids'] for o in rows]
            tokens = generated if tokens is None else tokens
            assert generated == tokens
            for o in rows:
                check_decode(o)
                assert o['generated_tokens'] == len(o['token_ids']) == 32
            stats = {'mean_ttft_seconds': st.mean(o['ttft_seconds'] for o in rows),
                     'mean_total_seconds': st.mean(o['total_seconds'] for o in rows),
                     'decode_tokens_per_second': 8*31/sum(o['decode_seconds'] for o in rows)}
            for k, x in stats.items():
                close(x, data['performance'][k], 'run aggregate')
            runs.append({**stats, **data['memory']})
        medians = {k: st.median(r[k] for r in runs) for k in runs[0]}
        for k, x in medians.items():
            close(x, saved[v]['performance'][k], 'five-process median')
        outputs[v] = medians
    assert len(set(pids)) == 25
    return {'isolated_pids': 25, 'requests': 200, 'input_identities_match_all_variants': True,
            'tokens_match_across_rounds_within_variant': True, 'performance': outputs,
            'memory_scope': 'RSS sampled at 10 ms includes model load/warmup; MLX peak active and end-of-run allocator cache are distinct, not additive; none is logical saved-prefix KV.'}


def main():
    result = {'audit_kind': 'fresh independent arithmetic over historical raw records; no model inference',
              'run_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'minilm_runtime': mini_runtime(), 'minilm_quantization': mini_quantization(),
              'bucketing': bucketing(), 'prefix_fair': fair(),
              'lifecycle': lifecycle(), 'quantization': quantization()}
    result['read_sha256'] = READS
    result['status'] = 'pass'
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
