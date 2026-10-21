"""Verify fixed Metal fusion evidence using only the Python standard library.

A valid negative study exits successfully with accepted=False. Receipt checks
establish byte identity and record consistency, not execution authenticity.
No model weights, MLX runtime, or GPU are needed for this offline verification.
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
PROTOCOL = 'configs/metal-residual-rmsnorm/study.json'
PROTOCOL_SHA256 = '9e46fc69c65f631cc69919b61e820eaf191ddd4817a2aa4f58a8cbe5b42e304e'
SOURCES = (
    'experiments/metal_residual_rmsnorm.py', 'experiments/qwen_demand_prefill.py',
    'experiments/__init__.py', 'lab/metal_residual_rmsnorm.py',
    'lab/kernels/residual_rmsnorm.metal', 'third_party/MLX-MIT.txt',
    'lab/qwen_demand_prefill.py', 'lab/qa_metrics.py', 'lab/__init__.py',
)
ARMS = ('native', 'compiled', 'metal')
FIELDS = ('ttft_seconds', 'decode_seconds', 'total_seconds',
          'active_before_bytes', 'peak_active_bytes')
AUDIT_EXTRA_FILES = {'native.dot', 'compiled.dot', 'metal.dot'}

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
    binaries = run['runtime_binary_sha256']
    require(isinstance(binaries, dict) and binaries
            and all(isinstance(path, str) and not Path(path).is_absolute()
                    and '..' not in Path(path).parts
                    and Path(path).suffix in ('.so', '.dylib', '.metallib')
                    and hashed(value) for path, value in binaries.items()),
            'Missing or invalid runtime binary identity')
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
                      'model_files_sha256', 'platform', 'python', 'device', 'default_device',
                      'runtime_binary_sha256']:
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
    result = {mode: dict(count=len(rows), em=0.0, f1=0.0, sequence_equal_native=0,
                        prediction_equal_native=0, stop_equal_native=0, different_ids=[]) for mode in ARMS}
    eos = config.get('eos_token_id')
    eos = {eos} if type(eos) is int else set(eos or [])
    for row, record in zip(rows, records):
        require(set(record) == {'id', 'input_token_ids', 'input_sha256', 'arms'}
                and record['id'] == row['id'] and set(record['arms']) == set(ARMS), 'Quality record order/arms differ')
        token_ids(record['input_token_ids'], config['vocab_size'], 1, spec['quality']['max_input_tokens'])
        require(digest(record['input_token_ids']) == record['input_sha256'], 'Quality input digest differs')
        native = record['arms']['native']
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
            same = arm['token_ids'] == native['token_ids']
            stat['sequence_equal_native'] += int(same)
            stat['prediction_equal_native'] += int(arm['prediction'] == native['prediction'])
            stat['stop_equal_native'] += int(arm['stop_reason'] == native['stop_reason'])
            if same:
                require(arm['prediction'] == native['prediction'] and arm['stop_reason'] == native['stop_reason'],
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


def boolean(value):
    require(type(value) is bool, 'Expected JSON boolean')
    return value


def input_hashes(value):
    return isinstance(value, list) and len(value) == 3 and all(hashed(v) for v in value)


def check_numerical(folder, spec):
    rows = read(folder / 'numerical.json')
    audit = spec['audit']
    expected = {(dtype, n, width, pattern) for dtype in audit['dtypes']
                for n in audit['rows'] for width in audit['widths']
                for pattern in audit['patterns']}
    seen = set()
    summary = {arm: dict(cases=0, finite=0, residual_exact=0, within_tolerance=0,
                        max_abs_error=0.0, max_ulp_error=0, max_tolerance_ratio=0.0)
               for arm in ARMS}
    require(isinstance(rows, list) and len(rows) == len(expected),
            'Numerical case coverage differs')
    for row in rows:
        require(set(row) == {'dtype', 'rows', 'width', 'pattern', 'input_sha256', 'arms'}
                and type(row['rows']) is int and type(row['width']) is int,
                'Numerical case fields differ')
        key = tuple(row[k] for k in ('dtype', 'rows', 'width', 'pattern'))
        require(key in expected and key not in seen and input_hashes(row['input_sha256']),
                'Numerical case identity or coverage differs')
        seen.add(key)
        require(set(row['arms']) == set(ARMS), 'Numerical arm coverage differs')
        for mode, arm in row['arms'].items():
            require(set(arm) == {'finite', 'residual_exact', 'within_tolerance',
                                'max_abs_error', 'max_ulp_error', 'max_tolerance_ratio',
                                'residual_sha256', 'output_sha256'},
                    'Numerical measurement fields differ')
            finite = boolean(arm['finite'])
            exact = boolean(arm['residual_exact'])
            within = boolean(arm['within_tolerance'])
            for field in ('max_abs_error', 'max_ulp_error', 'max_tolerance_ratio'):
                number(arm[field])
            require(type(arm['max_ulp_error']) is int, 'ULP distance must be integer')
            require(hashed(arm['residual_sha256']) and hashed(arm['output_sha256']),
                    'Invalid numerical output identity')
            native = row['arms']['native']
            require(exact == (arm['residual_sha256'] == native['residual_sha256']),
                    'Residual equality contradicts output identities')
            if arm['output_sha256'] == native['output_sha256']:
                require(all(arm[field] == 0 for field in
                            ('max_abs_error', 'max_ulp_error', 'max_tolerance_ratio')),
                        'Equal output digest has unequal numerical errors')
            require(within == (finite and arm['max_tolerance_ratio'] <= 1),
                    'Numerical tolerance boolean contradicts reported ratio')
            if mode == 'native':
                require(exact and all(arm[field] == 0 for field in
                        ('max_abs_error', 'max_ulp_error', 'max_tolerance_ratio')),
                        'Native numerical reference cannot differ from itself')
            require((arm['max_abs_error'] == 0) == (arm['max_tolerance_ratio'] == 0),
                    'Numerical absolute and tolerance errors contradict')
            stats = summary[mode]
            stats['cases'] += 1
            for field in ('finite', 'residual_exact', 'within_tolerance'):
                stats[field] += int(arm[field])
            for field in ('max_abs_error', 'max_ulp_error', 'max_tolerance_ratio'):
                stats[field] = max(stats[field], arm[field])
    require(seen == expected, 'Numerical coverage differs')
    return summary


def check_mechanism(folder, spec, config, longest):
    records = read(folder / 'mechanism.json')
    order = [(n, prior, step) for n in spec['audit']['model_lengths']
             for prior in spec['audit']['prior_cache_tokens']
             for step in range(spec['audit']['continuation_steps'] + 1)]
    require(len(records) == len(order), 'Mechanism shape/step coverage differs')
    summary = {mode: dict(rows=0, kv_equal_native=0, top1_equal_native=0,
                         logits_bitwise_equal_native=0, max_abs_error=0.0,
                         max_rms_error=0.0, max_relative_l2_error=0.0)
               for mode in ARMS}
    upstream_equal = 0
    previous = None
    for row, (n, prior, step) in zip(records, order):
        require(set(row) == {'length', 'prior_tokens', 'step', 'input_token_ids',
                             'input_sha256', 'arms', 'upstream', 'upstream_native_equal'}
                and all(type(row[k]) is int for k in ('length', 'prior_tokens', 'step'))
                and (row['length'], row['prior_tokens'], row['step']) == (n, prior, step)
                and set(row['arms']) == set(ARMS), 'Mechanism record order/arms differ')
        ids = longest[0][:n] if step == 0 else [previous]
        require(row['input_token_ids'] == ids and row['input_sha256'] == digest(ids),
                'Mechanism input or teacher-forced continuation differs')
        native = row['arms']['native']
        for mode, arm in row['arms'].items():
            require(set(arm) == {'logits', 'cache', 'cache_equal_native', 'trace'},
                    'Mechanism arm fields differ')
            check_cache(arm['cache'], prior+n+step, config)
            equal = arm['cache'] == native['cache']
            require(boolean(arm['cache_equal_native']) == equal,
                    'Reported KV equality contradicts tensor identities')
            check_trace(arm['trace'], mode, n if step == 0 else 1,
                        prior if step == 0 else prior+n+step-1, config)
            check_logits(arm['logits'], native['logits'], config['vocab_size'])
            logit, stats = arm['logits'], summary[mode]
            stats['rows'] += 1
            stats['kv_equal_native'] += int(equal)
            stats['top1_equal_native'] += int(logit['top1'] == native['logits']['top1'])
            stats['logits_bitwise_equal_native'] += int(
                logit['sha256_float64'] == native['logits']['sha256_float64'])
            for source, target in [('max_abs_error', 'max_abs_error'),
                                   ('rms_error', 'max_rms_error'),
                                   ('relative_l2_error', 'max_relative_l2_error')]:
                stats[target] = max(stats[target], logit[source])
        upstream = row['upstream']
        require(set(upstream) == {'logits', 'cache'}, 'Upstream reference fields differ')
        check_cache(upstream['cache'], prior+n+step, config)
        check_logits(upstream['logits'], native['logits'], config['vocab_size'])
        equal = (upstream['cache'] == native['cache'] and
                 upstream['logits']['sha256_float64'] == native['logits']['sha256_float64'])
        require(boolean(row['upstream_native_equal']) == equal,
                'Upstream equality contradicts logit/cache identities')
        upstream_equal += int(equal)
        previous = native['logits']['top1']
    return dict(arms=summary, rows=len(order), upstream_native_equal=upstream_equal)


def check_trace(trace, mode, length, prior, config):
    require(set(trace) == {'mode', 'input_shape', 'prior_cache_tokens', 'output_shape',
                          'cache_offsets_after', 'residual_norm_pairs',
                          'residual_norm_pair_calls', 'gpu_profiler'},
            'Mechanism trace fields differ')
    expected = dict(mode=mode, input_shape=[1, length], prior_cache_tokens=prior,
                    output_shape=[1, 1, config['vocab_size']],
                    cache_offsets_after=[prior+length] * config['num_hidden_layers'],
                    residual_norm_pair_calls=config['num_hidden_layers'], gpu_profiler=False)
    require(all(trace[k] == value and type(trace[k]) is type(value)
                for k, value in expected.items()), 'Mechanism trace shape/offset differs')
    pairs = trace['residual_norm_pairs']
    require(isinstance(pairs, list) and len(pairs) == config['num_hidden_layers'],
            'Wrong residual-norm pair count')
    for i, pair in enumerate(pairs):
        require(set(pair) == {'layer', 'shape', 'dtype', 'mode'} and type(pair['layer']) is int
                and pair['layer'] == i and pair['shape'] == [1, length, config['hidden_size']]
                and pair['mode'] == mode
                and re.fullmatch(r'(?:mlx\.core\.|mlx\.)?float16', pair['dtype']),
                'Residual-norm layer trace differs')


def check_micro(run, spec, reference_inputs):
    bench = spec['benchmark']
    rows = run['micro']
    require(isinstance(rows, list) and len(rows) == len(bench['micro_rows']),
            'Micro shape coverage differs')
    result = []
    for row, expected in zip(rows, bench['micro_rows']):
        require(set(row) == {'rows', 'width', 'input_sha256', 'samples', 'batch_calls'}
                and type(row['rows']) is int and row['rows'] == expected
                and type(row['width']) is int and row['width'] == bench['micro_width']
                and type(row['batch_calls']) is int and row['batch_calls'] == bench['micro_batch_calls']
                and input_hashes(row['input_sha256']), 'Micro shape or batch identity differs')
        require(reference_inputs.setdefault(expected, row['input_sha256']) == row['input_sha256'],
                'Micro input identity differs across workers')
        samples = row['samples']
        require(isinstance(samples, list) and len(samples) == bench['micro_samples'],
                'Micro sample coverage differs')
        for sample in samples:
            require(set(sample) == {'total_seconds', 'per_call_seconds'},
                    'Micro sample fields differ')
            number(sample['total_seconds'], positive=True)
            number(sample['per_call_seconds'], positive=True)
            require(close(sample['per_call_seconds'], sample['total_seconds'] / row['batch_calls']),
                    'Micro per-call arithmetic differs')
        result.append(dict(row, mode=run['mode'], round=run['round']))
    return result


def aggregate_micro(records, spec):
    measurements = {}
    for n in spec['benchmark']['micro_rows']:
        measurements[str(n)] = {}
        for mode in ARMS:
            matches = [r for r in records if r['rows'] == n and r['mode'] == mode]
            require(len(matches) == spec['benchmark']['rounds']
                    and {r['round'] for r in matches} == set(range(spec['benchmark']['rounds'])),
                    'Micro aggregation round coverage differs')
            matches.sort(key=lambda r: r['round'])
            values = [[s['total_seconds'] / spec['benchmark']['micro_batch_calls']
                       for s in r['samples']] for r in matches]
            medians = [statistics.median(samples) for samples in values]
            measurements[str(n)][mode] = dict(sample_values_by_round=values,
                round_medians=medians, median_of_round_medians=statistics.median(medians))
    return measurements


def acceptance(spec, numerical, mechanism, quality, parity, micro, model):
    limits = spec['acceptance']
    gates, comparisons = {}, {}
    primary_micro = str(limits['micro_primary_rows'])
    primary_model = str(limits['model_primary_length'])
    def value(arm, metric, n=primary_model):
        return model[n][arm][metric]['median_of_round_medians']
    model_speedups = {metric: {} for metric in limits['regression_metrics']}
    regressions = {}
    for control in ARMS[:-1]:
        control_micro, candidate_micro = micro[primary_micro][control], micro[primary_micro]['metal']
        speedup = control_micro['median_of_round_medians'] / candidate_micro['median_of_round_medians']
        faster = sum(c < b for b, c in zip(control_micro['round_medians'], candidate_micro['round_medians']))
        comparisons[control] = dict(micro_speedup=speedup, faster_micro_rounds=faster,
            required_micro_speedup=limits['min_micro_speedup_vs_each_control'],
            required_faster_micro_rounds=limits['min_faster_micro_rounds_vs_each_control'])
        gates['micro_speedup_vs_' + control] = speedup >= limits['min_micro_speedup_vs_each_control']
        gates['micro_faster_rounds_vs_' + control] = faster >= limits['min_faster_micro_rounds_vs_each_control']
        for metric in limits['regression_metrics']:
            model_speedups[metric][control] = value(control, metric) / value('metal', metric)
        regressions[control] = {}
        for n in spec['benchmark']['lengths']:
            regressions[control][str(n)] = {}
            for metric in limits['regression_metrics']:
                ratio = value('metal', metric, str(n)) / value(control, metric, str(n))
                regressions[control][str(n)][metric] = ratio
                gates[f'no_regression_{control}_{n}_{metric}'] = (
                    ratio <= limits['max_model_time_ratio_vs_each_control_each_length'])
    metric_passes = {metric: all(s >= limits['min_model_speedup_vs_each_control']
                                for s in ratios.values()) for metric, ratios in model_speedups.items()}
    gates['same_model_metric_beats_both_controls'] = any(metric_passes.values())
    gates['all_outputs_finite'] = all(s['finite'] == s['cases'] for s in numerical.values())
    gates['all_residual_outputs_exact'] = all(s['residual_exact'] == s['cases'] for s in numerical.values())
    gates['all_micro_numerical_within_tolerance'] = all(
        s['within_tolerance'] == s['cases'] for s in numerical.values())
    gates['all_mechanism_top1_equal_native'] = all(
        s['top1_equal_native'] == s['rows'] for s in mechanism['arms'].values())
    gates['model_logits_within_bound'] = all(
        s['max_abs_error'] <= limits['model_logits_max_abs'] for s in mechanism['arms'].values())
    gates['upstream_native_exact'] = mechanism['upstream_native_equal'] == mechanism['rows']
    gates['all_quality_sequences_equal_native'] = all(
        s['sequence_equal_native'] == s['count'] for s in quality.values())
    gates['all_benchmark_sequences_equal_native'] = all(
        s['sequence_equal_native'] == s['count'] for s in parity.values())
    return dict(candidate='metal', accepted=all(gates.values()), gates=gates,
                micro_primary_comparisons=comparisons, model_primary_speedups=model_speedups,
                model_metric_passes=metric_passes, model_time_ratios=regressions,
                failure_behavior=limits['failure_behavior'])


def verify(audit_root, benchmark_root, *, root=ROOT):
    root, audit_root, benchmark_root = map(Path, (root, audit_root, benchmark_root))
    spec = check_protocol(root)
    common = {'run.json', 'protocol.json', 'checksums.json'} | {'source/' + name for name in SOURCES}
    receipt(audit_root, common | {'numerical.json', 'workloads.json', 'mechanism.json', 'quality.json'}
            | AUDIT_EXTRA_FILES)
    audit = check_state(audit_root, root, spec)
    config = audit['model_config']
    workloads, longest = check_workloads(audit_root, spec, config['vocab_size'])
    numerical = check_numerical(audit_root, spec)
    mechanism = check_mechanism(audit_root, spec, config, longest)
    quality = check_quality(audit_root, root, spec, config)
    bench = read(benchmark_root / 'run.json')
    require(bench['status'] == 'complete' and bench['protocol'] == spec
            and bench['spec_sha256'] == PROTOCOL_SHA256
            and bench['source_sha256'] == audit['source_sha256']
            and bench['audit_checksums_sha256'] == sha(audit_root / 'checksums.json'),
            'Benchmark audit/protocol binding differs')
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
    pids, records, micro_records = set(), [], []
    all_tokens, micro_inputs = {}, {}
    record_fields = {'case_id', 'length', 'prompt_index', 'input_sha256', 'token_ids',
                     'stop_reason', 'token_elapsed_seconds'} | set(FIELDS)
    for worker, (round_id, mode) in zip(bench['workers'], order):
        folder = f'{round_id}-{mode}'
        require(worker['round'] == round_id and worker['mode'] == mode and worker['folder'] == folder
                and type(worker['round']) is int and type(worker['exit_code']) is int
                and worker['exit_code'] == 0 and worker['status'] == 'complete',
                'Process order or outcome differs')
        worker_root = benchmark_root / folder
        receipt(worker_root, common)
        run = check_state(worker_root, root, spec, audit)
        require(type(run['round']) is int and run['round'] == round_id and run['mode'] == mode,
                'Worker run identity differs')
        require(run['pid'] not in pids and run['pid'] != audit['pid'], 'Fresh-process identities reused')
        pids.add(run['pid'])
        micro_records.extend(check_micro(run, spec, micro_inputs))
        require(len(run['records']) == len(workloads), 'Benchmark request coverage differs')
        for row, expected in zip(run['records'], workloads):
            require(set(row) == record_fields and all(row[k] == expected[k] for k in
                    ['case_id', 'length', 'prompt_index', 'input_sha256']),
                    'Benchmark workload order/identity differs')
            count = spec['benchmark']['forced_new_tokens']
            token_ids(row['token_ids'], config['vocab_size'], count, count)
            require(row['stop_reason'] == 'token_limit', 'Benchmark must ignore EOS and generate fixed token count')
            check_timing(row, count)
            records.append(dict(row, round=round_id, mode=mode))
            all_tokens[(round_id, mode, row['case_id'])] = row['token_ids']
    measured = (sum(row['total_seconds'] for row in records) +
                sum(sample['total_seconds'] for row in micro_records for sample in row['samples']))
    require(measured <= bench['wall_seconds'] + 1e-6,
            'Measured micro and model durations exceed sequential study wall time')
    parity = {mode: dict(count=0, sequence_equal_native=0, different_cases=[]) for mode in ARMS}
    for row in records:
        mode, round_id, case_id = row['mode'], row['round'], row['case_id']
        same = row['token_ids'] == all_tokens[(round_id, 'native', case_id)]
        parity[mode]['count'] += 1
        parity[mode]['sequence_equal_native'] += int(same)
        if not same:
            parity[mode]['different_cases'].append(dict(round=round_id, case_id=case_id))
    model, micro = aggregate(records, spec), aggregate_micro(micro_records, spec)
    decision = acceptance(spec, numerical, mechanism, quality, parity, micro, model)
    return dict(schema='metal-residual-rmsnorm-independent-verification-v1', evidence_valid=True,
        protocol_sha256=PROTOCOL_SHA256, source_sha256=audit['source_sha256'],
        audit_checksums_sha256=sha(audit_root / 'checksums.json'),
        benchmark_checksums_sha256=sha(benchmark_root / 'checksums.json'),
        model_files_sha256=audit['model_files_sha256'], versions=audit['versions'],
        upstream_source_sha256=audit['upstream_source_sha256'],
        runtime_binary_sha256=audit['runtime_binary_sha256'],
        process_count=len(pids), wall_seconds=bench['wall_seconds'],
        numerical=numerical, mechanism=mechanism, quality=quality,
        benchmark_sequence_parity=parity, micro_measurements=micro,
        model_measurements=model, acceptance=decision,
        scope='Existing consumed public cohort and one fixed local Metal fusion study. '
              'Micro latency includes host enqueue/synchronization; compiled controls compile the pair only.',
        limitations=['Checksums establish byte equality and record consistency, not authenticity against coordinated evidence replacement.',
                     'No weights or complete output tensors are loaded. Error summaries and tokenizer decoding are checked for consistency, not independently regenerated.',
                     'MLX DOT graphs are graph evidence, not hardware dispatch traces or sustained memory-bandwidth measurements.',
                     'A passing local study does not establish new QA quality or cross-device performance.'])


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
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
