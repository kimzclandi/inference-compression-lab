"""Independently verify frozen prefill evidence without MLX or model weights.

This checks archived identities, internal consistency and acceptance arithmetic;
hash receipts are not signatures or a substitute for independently executing a
model. A valid negative study exits successfully and reports accepted=False.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import string


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/qwen-demand-prefill/study.json'
PROTOCOL_SHA256 = 'a087cc0a5ab44e8faa7dc38c56cb2a049f9f4403ccc954596619ac06db89c0f2'
SOURCES = ('experiments/qwen_demand_prefill.py', 'experiments/__init__.py',
           'lab/qwen_demand_prefill.py', 'lab/qa_metrics.py', 'lab/__init__.py')
ARMS = ('full', 'head_only', 'split_last', 'final_query')
FIELDS = ('ttft_seconds', 'decode_seconds', 'total_seconds',
          'active_before_bytes', 'peak_active_bytes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def _finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Non-finite JSON number')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'),
                       object_pairs_hook=_pairs, parse_constant=_nonfinite)
    _finite(value)
    return value


def hashed(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def number(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid nonnegative numeric measurement')
    return value


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), 'Missing or symlink receipt directory')
    result = {}
    for path in folder.rglob('*'):
        require(not path.is_symlink(), 'Symlinks are forbidden in evidence')
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = sha(path)
    return result


def receipt(folder, expected_files):
    actual = inventory(folder)
    require(set(actual) == set(expected_files), 'Receipt file coverage differs: ' + str(folder))
    hashes = read(folder / 'checksums.json')
    require(isinstance(hashes, dict) and all(hashed(v) for v in hashes.values()),
            'Invalid checksum manifest')
    # The runner excludes every checksums.json, including nested worker receipts.
    calculated = {k: v for k, v in actual.items() if Path(k).name != 'checksums.json'}
    require(hashes == calculated, 'Receipt file identity mismatch: ' + str(folder))


def check_protocol(root):
    path = root / PROTOCOL
    require(sha(path) == PROTOCOL_SHA256, 'Frozen protocol changed')
    spec = read(path)
    require(tuple(spec['arms']) == ARMS, 'Arm contract changed')
    for key, hashkey in [('path', 'sha256'), ('system_path', 'system_sha256')]:
        require(sha(root / spec['quality'][key]) == spec['quality'][hashkey], 'Frozen quality input changed')
    return spec


def check_state(folder, root, spec, reference=None):
    run = read(folder / 'run.json')
    require(run['status'] == 'complete', 'Incomplete worker or audit')
    require(run['spec_sha256'] == PROTOCOL_SHA256
            and sha(folder / 'protocol.json') == PROTOCOL_SHA256,
            'Archived protocol differs')
    require(read(folder / 'protocol.json') == spec, 'Archived protocol content differs')
    source = {name: sha(root / name) for name in SOURCES}
    require(run['source_sha256'] == source, 'Current source differs from archived source identity')
    for name in SOURCES:
        require(sha(folder / 'source' / name) == source[name], 'Source snapshot differs: ' + name)
    for field, expected in [('model_files_sha256', spec['model_files_sha256']),
                            ('versions', spec['required_versions']),
                            ('upstream_source_sha256', spec['upstream_source_sha256'])]:
        require(run[field] == expected, 'Frozen identity mismatch: ' + field)
    require(isinstance(run['git_head'], str) and re.fullmatch('[0-9a-f]{40}', run['git_head'])
            and isinstance(run['git_status'], str), 'Invalid Git source receipt')
    require(type(run['pid']) is int and run['pid'] > 0, 'Invalid process identifier')
    require(all(isinstance(run[field], str) and run[field] for field in ['platform', 'python'])
            and isinstance(run['device'], dict) and run['device'], 'Missing machine provenance')
    require(run['default_device'] == 'Device(gpu, 0)', 'Metal GPU execution device required')
    config = run['model_config']
    require(config['model_type'] == 'qwen2' and config['num_hidden_layers'] == 24
            and config['hidden_size'] == 896 and config['num_attention_heads'] == 14
            and config['num_key_value_heads'] == 2 and config['vocab_size'] == 151936
            and config['tie_word_embeddings'] is True and not config.get('use_sliding_window'),
            'Unsupported frozen model architecture')
    if reference is not None:
        for field in ['model_config', 'source_sha256', 'versions', 'upstream_source_sha256',
                      'model_files_sha256', 'platform', 'python', 'device', 'default_device']:
            require(run[field] == reference[field], 'Process provenance differs: ' + field)
    return run


def token_ids(value, vocab, minimum=1, maximum=None):
    require(isinstance(value, list) and len(value) >= minimum
            and (maximum is None or len(value) <= maximum)
            and all(type(t) is int and 0 <= t < vocab for t in value), 'Invalid token sequence')


def check_workloads(folder, spec, vocab):
    rows = read(folder / 'workloads.json')
    order = [(n, i) for n in spec['benchmark']['lengths']
             for i in range(spec['benchmark']['prompts_per_length'])]
    require(len(rows) == len(order), 'Workload coverage differs')
    for row, (n, i) in zip(rows, order):
        require(set(row) == {'case_id', 'length', 'prompt_index', 'token_ids', 'input_sha256'}
                and row['case_id'] == f'{n}-{i}' and row['length'] == n
                and row['prompt_index'] == i, 'Workload order/shape differs')
        token_ids(row['token_ids'], vocab, n, n)
        require(row['input_sha256'] == digest(row['token_ids']), 'Workload token digest differs')
    longest = {r['prompt_index']: r['token_ids'] for r in rows if r['length'] == max(spec['benchmark']['lengths'])}
    require(all(row['token_ids'] == longest[row['prompt_index']][:row['length']] for row in rows),
            'Synthetic workload prefix contract differs')
    return rows, longest


def check_cache(cache, offset, config):
    require(isinstance(cache, list) and len(cache) == config['num_hidden_layers'], 'Wrong KV layer count')
    shape = [1, config['num_key_value_heads'], offset,
             config['hidden_size'] // config['num_attention_heads']]
    for layer in cache:
        require(set(layer) == {'offset', 'tensors'} and type(layer['offset']) is int
                and layer['offset'] == offset and len(layer['tensors']) == 2, 'Wrong cache offset or K/V count')
        for tensor in layer['tensors']:
            require(set(tensor) == {'shape', 'dtype', 'sha256'} and tensor['shape'] == shape
                    and all(type(value) is int for value in tensor['shape'])
                    and re.fullmatch(r'(?:mlx\.core\.|mlx\.)?(?:float16|float32|bfloat16)', tensor['dtype'])
                    and hashed(tensor['sha256']), 'Invalid logical cache tensor')
        require(layer['tensors'][0]['dtype'] == layer['tensors'][1]['dtype'], 'K/V dtype mismatch')


def check_trace(trace, mode, n, prior, config):
    expected = {'mode': mode, 'input_shape': [1, n], 'prior_cache_tokens': prior,
                'output_shape': [1, 1, config['vocab_size']],
                'cache_offsets_after': [prior + n] * config['num_hidden_layers'], 'shapes': {}}
    if n > 1:
        h, heads, kv = config['hidden_size'], config['num_attention_heads'], config['num_key_value_heads']
        if mode == 'head_only':
            expected['shapes'] = {'head_input_shape': [1, 1, h]}
        elif mode == 'split_last':
            expected['shapes'] = {'model_call_input_shapes': [[1, n-1], [1, 1]]}
        elif mode == 'final_query':
            expected['shapes'] = dict(query_input_shape=[1, 1, h], key_value_input_shape=[1, n, h],
                attention_query_shape=[1, heads, 1, h // heads],
                attention_key_shape=[1, kv, prior+n, h // heads], head_input_shape=[1, 1, h],
                query_rope_offset=prior+n-1, key_rope_offset=prior)
    require(digest(trace) == digest(expected), 'Demand-dependency trace differs')


def check_logits(record, reference, vocab):
    require(set(record) == {'top1', 'reference_top1', 'reference_top2', 'reference_margin',
            'max_abs_error', 'rms_error', 'relative_l2_error', 'sha256_float64'}, 'Logit fields differ')
    for field in ['top1', 'reference_top1', 'reference_top2']:
        require(type(record[field]) is int and 0 <= record[field] < vocab, 'Invalid logit top index')
    require(record['reference_top1'] != record['reference_top2'] and hashed(record['sha256_float64']),
            'Invalid logit identity')
    for field in ['reference_margin', 'max_abs_error', 'rms_error', 'relative_l2_error']:
        number(record[field])
    require(record['rms_error'] <= record['max_abs_error'] + 1e-12, 'Impossible logit RMS')
    require(all(record[k] == reference[k] for k in ['reference_top1', 'reference_top2', 'reference_margin'])
            and reference['top1'] == reference['reference_top1'], 'Reference logit context changed')
    if record['sha256_float64'] == reference['sha256_float64']:
        require(record['top1'] == reference['top1'] and all(record[k] == 0 for k in
                ['max_abs_error', 'rms_error', 'relative_l2_error']), 'Equal-logit digest has unequal errors')


def check_mechanism(folder, spec, config, longest):
    records = read(folder / 'mechanism.json')
    order = [(n, p, s) for n in spec['audit']['shape_lengths'] for p in spec['audit']['prior_cache_tokens']
             for s in range(spec['audit']['continuation_steps'] + 1)]
    require(len(records) == len(order), 'Mechanism shape/step coverage differs')
    summary = {mode: dict(rows=0, kv_equal_full=0, top1_equal_full=0, followup_top1_equal_full=0,
                         logits_bitwise_equal_full=0, max_abs_error=0.0, max_rms_error=0.0,
                         max_relative_l2_error=0.0) for mode in ARMS}
    previous = None
    for row, (n, prior, step) in zip(records, order):
        require(set(row) == {'length', 'prior_tokens', 'step', 'input_sha256', 'arms'}
                and all(type(row[k]) is int for k in ['length', 'prior_tokens', 'step'])
                and (row['length'], row['prior_tokens'], row['step']) == (n, prior, step)
                and set(row['arms']) == set(ARMS), 'Mechanism record order/arms differ')
        ids = longest[0][:n] if step == 0 else [previous]
        require(row['input_sha256'] == digest(ids), 'Mechanism input or forced continuation differs')
        full = row['arms']['full']
        for mode in ARMS:
            arm = row['arms'][mode]
            require(set(arm) == {'logits', 'cache', 'cache_equal_full', 'trace'}, 'Mechanism arm fields differ')
            check_cache(arm['cache'], prior+n+step, config)
            equal = arm['cache'] == full['cache']
            require(type(arm['cache_equal_full']) is bool and arm['cache_equal_full'] == equal,
                    'Reported KV equality contradicts element-byte digests/shapes/offsets')
            if step == 0:
                check_trace(arm['trace'], mode, n, prior, config)
            else:
                require(arm['trace'] == {}, 'Followup must use standard one-token model')
            logit = arm['logits']
            check_logits(logit, full['logits'], config['vocab_size'])
            stats = summary[mode]
            stats['rows'] += 1
            stats['kv_equal_full'] += int(equal)
            stats['top1_equal_full'] += int(logit['top1'] == full['logits']['top1'])
            stats['followup_top1_equal_full'] += int(step > 0 and logit['top1'] == full['logits']['top1'])
            stats['logits_bitwise_equal_full'] += int(logit['sha256_float64'] == full['logits']['sha256_float64'])
            for field, dest in [('max_abs_error', 'max_abs_error'), ('rms_error', 'max_rms_error'),
                                ('relative_l2_error', 'max_relative_l2_error')]:
                stats[dest] = max(stats[dest], logit[field])
        previous = full['logits']['top1']
    return summary


def pair_score(prediction, gold):
    def normalized(value):
        value = ''.join(char for char in value.lower() if char not in string.punctuation)
        return re.sub(r'\b(a|an|the)\b', ' ', value).split()
    p, g = normalized(prediction), normalized(gold)
    em = float(p == g)
    if not p or not g:
        return em, em
    return em, 2 * sum((Counter(p) & Counter(g)).values()) / (len(p) + len(g))


def quality_score(row, prediction):
    answer = prediction.strip()
    if row['is_impossible']:
        return (float(answer == 'NO_ANSWER'),) * 2
    if not answer or answer == 'NO_ANSWER':
        return 0.0, 0.0
    scores = [pair_score(answer, gold) for gold in row['answers']]
    return max(s[0] for s in scores), max(s[1] for s in scores)


def check_quality(folder, root, spec, config):
    rows = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_nonfinite)
            for line in (root / spec['quality']['path']).read_text().splitlines()]
    records = read(folder / 'quality.json')
    require(len(rows) == spec['quality']['count'] == len(records)
            and len({r['id'] for r in rows}) == len(rows), 'Quality ID coverage differs')
    result = {mode: dict(count=len(rows), em=0.0, f1=0.0, sequence_equal_full=0,
                        prediction_equal_full=0, stop_equal_full=0, different_ids=[]) for mode in ARMS}
    eos = config.get('eos_token_id')
    eos = {eos} if type(eos) is int else set(eos or [])
    for row, record in zip(rows, records):
        require(set(record) == {'id', 'input_token_ids', 'input_sha256', 'arms'}
                and record['id'] == row['id'] and set(record['arms']) == set(ARMS), 'Quality record order/arms differ')
        token_ids(record['input_token_ids'], config['vocab_size'], 1, spec['quality']['max_input_tokens'])
        require(digest(record['input_token_ids']) == record['input_sha256'], 'Quality input digest differs')
        full = record['arms']['full']
        for mode, arm in record['arms'].items():
            require(set(arm) == {'token_ids', 'stop_reason', 'prediction', 'em', 'f1'}
                    and isinstance(arm['prediction'], str), 'Quality prediction fields differ')
            token_ids(arm['token_ids'], config['vocab_size'], 1, spec['quality']['max_new_tokens'])
            require(arm['stop_reason'] in ('eos', 'token_limit')
                    and (arm['stop_reason'] != 'token_limit' or len(arm['token_ids']) == spec['quality']['max_new_tokens']),
                    'Quality termination length differs')
            if eos:
                require(not any(token in eos for token in arm['token_ids'][:-1]), 'Generation continued after EOS')
                require(arm['stop_reason'] != 'token_limit' or arm['token_ids'][-1] not in eos,
                        'Known EOS marked token_limit')
            # Tokenizer chat EOS may include extra IDs not present in model config.
            em, f1 = quality_score(row, arm['prediction'])
            require(type(arm['em']) in (int, float) and type(arm['f1']) in (int, float)
                    and close(arm['em'], em) and close(arm['f1'], f1), 'Archived EM/F1 does not reproduce')
            stat = result[mode]
            stat['em'] += em / len(rows)
            stat['f1'] += f1 / len(rows)
            same = arm['token_ids'] == full['token_ids']
            stat['sequence_equal_full'] += int(same)
            stat['prediction_equal_full'] += int(arm['prediction'] == full['prediction'])
            stat['stop_equal_full'] += int(arm['stop_reason'] == full['stop_reason'])
            if same:
                require(arm['prediction'] == full['prediction'] and arm['stop_reason'] == full['stop_reason'],
                        'Same tokens have inconsistent decoding/termination')
            else:
                stat['different_ids'].append(row['id'])
    return result


def check_timing(row, count):
    for field in FIELDS:
        number(row[field], positive=True)
    for field in ['active_before_bytes', 'peak_active_bytes']:
        require(type(row[field]) is int, 'Memory counters must be integer bytes')
    require(row['peak_active_bytes'] >= row['active_before_bytes'], 'Peak below pre-call active memory')
    elapsed = row['token_elapsed_seconds']
    require(isinstance(elapsed, list) and len(elapsed) == count, 'Missing synchronized token timestamps')
    for value in elapsed:
        number(value, positive=True)
    require(all(a < b for a, b in zip(elapsed, elapsed[1:])), 'Nonincreasing token timestamps')
    require(close(row['ttft_seconds'], elapsed[0])
            and close(row['decode_seconds'], elapsed[-1] - elapsed[0])
            and close(row['total_seconds'], elapsed[-1])
            and close(row['total_seconds'], row['ttft_seconds'] + row['decode_seconds']),
            'Timing components do not reproduce synchronized intervals')


def aggregate(records, spec):
    result = {}
    for n in spec['benchmark']['lengths']:
        by_arm = {}
        for mode in ARMS:
            fields = {}
            for field in FIELDS:
                rounds = [[r[field] for r in records if r['length'] == n and r['mode'] == mode
                           and r['round'] == i] for i in range(spec['benchmark']['rounds'])]
                require(all(len(values) == spec['benchmark']['prompts_per_length'] for values in rounds),
                        'Timing aggregation coverage differs')
                medians = [statistics.median(values) for values in rounds]
                fields[field] = dict(request_values_by_round=rounds, round_medians=medians,
                                     median_of_round_medians=statistics.median(medians))
            by_arm[mode] = fields
        result[str(n)] = by_arm
    return result


def acceptance(spec, mechanism, quality, parity, measurements):
    limits = spec['acceptance']
    candidate = limits['candidate']
    def val(n, arm, field):
        return measurements[str(n)][arm][field]['median_of_round_medians']
    comparisons, gates = {}, {}
    n = limits['primary_length']
    for control in ARMS[:-1]:
        speedup = val(n, control, 'ttft_seconds') / val(n, candidate, 'ttft_seconds')
        baseline_rounds = measurements[str(n)][control]['ttft_seconds']['round_medians']
        candidate_rounds = measurements[str(n)][candidate]['ttft_seconds']['round_medians']
        faster = sum(c < b for b, c in zip(baseline_rounds, candidate_rounds))
        threshold = limits['min_ttft_speedup_vs_full'] if control == 'full' else limits['min_ttft_speedup_vs_each_strong_control']
        comparisons[control] = dict(ttft_speedup=speedup, faster_rounds=faster,
                                    required_speedup=threshold,
                                    required_faster_rounds=limits['min_faster_rounds_vs_each_control'])
        gates['primary_ttft_vs_' + control] = speedup >= threshold
        gates['faster_rounds_vs_' + control] = faster >= limits['min_faster_rounds_vs_each_control']
    short_ratio = val(64, candidate, 'ttft_seconds') / val(64, 'full', 'ttft_seconds')
    gates['short_prompt_ttft'] = short_ratio <= limits['max_64_token_ttft_ratio_vs_full']
    decode_ratios = {str(n): val(n, candidate, 'decode_seconds') / val(n, 'full', 'decode_seconds')
                     for n in spec['benchmark']['lengths']}
    for n, ratio in decode_ratios.items():
        gates['decode_time_' + n] = ratio <= limits['max_decode_time_ratio_vs_full_each_length']
    expected_mechanism = len(spec['audit']['shape_lengths']) * len(spec['audit']['prior_cache_tokens'])
    gates['all_quality_sequences_equal_full'] = quality[candidate]['sequence_equal_full'] == spec['quality']['count']
    gates['all_benchmark_sequences_equal_full'] = parity[candidate]['sequence_equal_full'] == parity[candidate]['count']
    gates['audit_kv_equal_full'] = mechanism[candidate]['kv_equal_full'] == mechanism[candidate]['rows']
    gates['audit_all_top1_equal_full'] = mechanism[candidate]['top1_equal_full'] == mechanism[candidate]['rows']
    gates['audit_followup_top1_equal_full'] = mechanism[candidate]['followup_top1_equal_full'] == expected_mechanism * spec['audit']['continuation_steps']
    gates['audit_shapes_offsets_valid'] = True  # malformed mechanism records already raise
    return dict(candidate=candidate, accepted=all(gates.values()), gates=gates,
                primary_comparisons=comparisons, short_prompt_ttft_ratio=short_ratio,
                decode_ratios=decode_ratios,
                failure_behavior=limits['failure_behavior'])


def verify(audit_root, benchmark_root, *, root=ROOT):
    root, audit_root, benchmark_root = map(Path, (root, audit_root, benchmark_root))
    spec = check_protocol(root)
    common = {'run.json', 'protocol.json', 'checksums.json'} | {'source/' + s for s in SOURCES}
    receipt(audit_root, common | {'workloads.json', 'mechanism.json', 'quality.json'})
    audit = check_state(audit_root, root, spec)
    require(audit['scope'] == 'Consumed public functional/numerical reproduction; no timing or new quality claim',
            'Audit scope changed')
    config = audit['model_config']
    workloads, longest = check_workloads(audit_root, spec, config['vocab_size'])
    mechanism = check_mechanism(audit_root, spec, config, longest)
    quality = check_quality(audit_root, root, spec, config)
    bench = read(benchmark_root / 'run.json')
    require(bench['status'] == 'complete' and bench['protocol'] == spec
            and bench['spec_sha256'] == PROTOCOL_SHA256 and bench['source_sha256'] == audit['source_sha256']
            and bench['audit_checksums_sha256'] == sha(audit_root / 'checksums.json'), 'Benchmark audit/protocol binding differs')
    number(bench['wall_seconds'], positive=True)
    require(bench['wall_seconds'] <= spec['benchmark']['budget_seconds'], 'Frozen study budget exceeded')
    order = [(i, mode) for i, modes in enumerate(spec['benchmark']['order']) for mode in modes]
    expected_files = {'run.json', 'checksums.json'}
    for i, mode in order:
        folder = f'{i}-{mode}'
        expected_files.add(folder + '.log')
        expected_files.update(folder + '/' + name for name in common)
    receipt(benchmark_root, expected_files)
    require(len(bench['workers']) == len(order), 'Worker coverage differs')
    pids, records = set(), []
    all_tokens = {}
    record_fields = {'case_id', 'length', 'prompt_index', 'input_sha256', 'token_ids', 'stop_reason',
                     'token_elapsed_seconds'} | set(FIELDS)
    for worker, (round_id, mode) in zip(bench['workers'], order):
        folder = f'{round_id}-{mode}'
        require(worker['round'] == round_id and worker['mode'] == mode and worker['folder'] == folder
                and type(worker['round']) is int and type(worker['exit_code']) is int
                and worker['exit_code'] == 0 and worker['status'] == 'complete', 'Process order or outcome differs')
        worker_root = benchmark_root / folder
        receipt(worker_root, common)
        run = check_state(worker_root, root, spec, audit)
        require(run['round'] == round_id and run['mode'] == mode, 'Worker run identity differs')
        require(run['pid'] not in pids and run['pid'] != audit['pid'], 'Fresh-process identities reused')
        pids.add(run['pid'])
        require(len(run['records']) == len(workloads), 'Benchmark request coverage differs')
        for row, expected in zip(run['records'], workloads):
            require(set(row) == record_fields and all(row[k] == expected[k] for k in
                    ['case_id', 'length', 'prompt_index', 'input_sha256']), 'Benchmark workload order/identity differs')
            count = spec['benchmark']['forced_new_tokens']
            token_ids(row['token_ids'], config['vocab_size'], count, count)
            require(row['stop_reason'] == 'token_limit', 'Benchmark must ignore EOS and generate fixed token count')
            check_timing(row, count)
            records.append(dict(row, round=round_id, mode=mode))
            all_tokens[(round_id, mode, row['case_id'])] = row['token_ids']
    require(sum(r['total_seconds'] for r in records) <= bench['wall_seconds'] + 1e-6,
            'Measured request durations exceed sequential process budget')
    parity = {mode: dict(count=0, sequence_equal_full=0, different_cases=[]) for mode in ARMS}
    for row in records:
        mode, round_id, case_id = row['mode'], row['round'], row['case_id']
        same = row['token_ids'] == all_tokens[(round_id, 'full', case_id)]
        parity[mode]['count'] += 1
        parity[mode]['sequence_equal_full'] += int(same)
        if not same:
            parity[mode]['different_cases'].append(dict(round=round_id, case_id=case_id))
    measurements = aggregate(records, spec)
    decision = acceptance(spec, mechanism, quality, parity, measurements)
    return dict(schema='qwen-demand-prefill-independent-verification-v1', evidence_valid=True,
        protocol_sha256=PROTOCOL_SHA256, source_sha256=audit['source_sha256'],
        audit_checksums_sha256=sha(audit_root / 'checksums.json'),
        benchmark_checksums_sha256=sha(benchmark_root / 'checksums.json'),
        model_files_sha256=audit['model_files_sha256'], versions=audit['versions'],
        upstream_source_sha256=audit['upstream_source_sha256'],
        process_count=len(pids), wall_seconds=bench['wall_seconds'],
        mechanism=mechanism, quality=quality, benchmark_sequence_parity=parity,
        measurements=measurements, acceptance=decision,
        scope='Existing consumed cohort and one fixed local compute study; no new quality confirmation, production default, or original kernel claim.',
        limitations=['Checksums establish byte equality and record consistency, not authenticity against coordinated evidence replacement.',
                     'No weights or logits tensors are loaded; numeric logit error statistics and tokenizer decoding are checked for consistency, not independently regenerated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.output:
            require(not args.output.exists(), 'Output already exists; historical verification must not be overwritten')
            require(not any(args.output.resolve().is_relative_to(folder.resolve())
                            for folder in [args.audit_root, args.benchmark_root]),
                    'Verification output must be outside the immutable evidence directories')
        result = verify(args.audit_root, args.benchmark_root)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        parser.exit(1, 'Invalid evidence: ' + str(exc) + '\n')
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
