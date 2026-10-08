"""CPU-only archival replay of the bounded diagnostic, never GPU acceptance.

The numerical reference and metric reconstruction below do not call the runner,
its error_metrics helper, MLX, or a Metal kernel. A completed diagnostic may have
failed every old tolerance gate; that is preserved rather than turned into a
successful v1 experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PAIRS = [('native_a', 'native_b'), ('native_a', 'adapter_native'),
         ('adapter_native', 'candidate'), ('native_a', 'candidate')]
ARMS = ['native_a', 'native_b', 'adapter_native', 'candidate']
PARTS = ['q', 'k_new', 'v_new', 'out']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_digest(array):
    return digest(array.tobytes())


def load_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, name):
    path = Path(name)
    assert isinstance(name, str) and not path.is_absolute() and '..' not in path.parts
    result = Path(root) / path
    assert result.resolve().is_relative_to(Path(root).resolve()), 'escaping artifact'
    return result


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def reference(q, k, v):
    """Independent per-head FP64 sum/product, without einsum or BLAS."""
    result = np.empty((1, 14, 1, 64), dtype=np.float64)
    for head in range(14):
        keys = k[0, head // 7].astype(np.float64)
        values = v[0, head // 7].astype(np.float64)
        query = q[0, head, 0].astype(np.float64)
        scores = np.sum(keys * query, axis=1, dtype=np.float64) / 8
        weights = np.exp(scores - np.max(scores))
        weights /= np.sum(weights, dtype=np.float64)
        result[0, head, 0] = np.sum(values * weights[:, None], axis=0, dtype=np.float64)
    return result


def metrics(a, b, atol, rtol):
    """Independent reconstruction of every recorded scalar/coordinate field."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.size > 0
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite metric input'
    left, right = a.astype(np.float64), b.astype(np.float64)
    delta = np.abs(left - right)
    original_fail = np.logical_not(np.isclose(a, b, atol=atol, rtol=rtol))
    mathematical_fail = delta > atol + rtol * np.abs(right)
    changed = a != b

    def first(mask):
        flat = np.flatnonzero(mask)
        return list(np.unravel_index(int(flat[0]), mask.shape)) if len(flat) else None

    maximum = np.unravel_index(int(delta.argmax()), a.shape)
    result = dict(shape=list(a.shape), left_sha256=array_digest(a), right_sha256=array_digest(b),
        bitwise_equal=bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()),
        equal_values=bool(np.array_equal(a, b)), unequal_elements=int(changed.sum()), first_unequal=first(changed),
        allclose=bool(not original_fail.any()), failed_elements=int(original_fail.sum()), first_failure=first(original_fail),
        float64_gate_allclose=bool(not mathematical_fail.any()),
        float64_failed_elements=int(mathematical_fail.sum()), float64_first_failure=first(mathematical_fail),
        gate_method='numpy.isclose on original dtypes; float64 gate separately descriptive',
        max_abs=float(delta[maximum]), max_coordinate=list(maximum),
        left_at_max=float(left[maximum]), right_at_max=float(right[maximum]),
        abs_percentiles={f'p{p}': float(np.percentile(delta, p)) for p in [50, 95, 99]},
        atol=atol, rtol=rtol, relative_reference='right')
    if a.dtype == b.dtype == np.float16:
        # Sign/magnitude mapping differs from the runner's biased ordered code.
        def signed_code(x):
            magnitude = np.abs(x).view(np.uint16).astype(np.int64)
            return np.where(np.signbit(x), -magnitude, magnitude)
        ulps = np.abs(signed_code(a) - signed_code(b))
        result.update(ulp_max=int(ulps.max()), ulp_p99=float(np.percentile(ulps, 99)))
    return result


def assert_metrics(record, left, right, atol, rtol):
    expected = metrics(left, right, atol, rtol)
    assert record == expected, 'metric mismatch'
    return expected


def check_array(array, shape, dtype=np.float16):
    assert array.shape == shape and array.dtype == dtype
    assert np.isfinite(array).all(), 'nonfinite archived array'


def verify(root):
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    root = Path(root)
    run = load_json(root / 'run.json')
    spec = run['spec']
    assert run['status'] in ('complete', 'failed')
    assert len(run['protocol_commit']) == 40 and all(c in '0123456789abcdef' for c in run['protocol_commit'])
    actual = {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*')
              if p.is_file() and p != root / 'run.json'}
    assert actual == run['artifacts'], 'artifact modified'
    assert set(run['source_sha256']) == {'configs/gqa-shared-diagnostic-v1.json',
        'experiments/gqa_shared_diagnostic.py', 'lab/gqa_diagnostic.py',
        'lab/kernels/gqa_diagnostic_layout.metal'}
    for name, expected in run['source_sha256'].items():
        assert digest(safe_path(root / 'source', name).read_bytes()) == expected
        assert digest(safe_path(Path.cwd(), name).read_bytes()) == expected, 'diagnostic source changed'
    for name, expected in spec['protected_source_sha256'].items():
        assert digest(safe_path(Path.cwd(), name).read_bytes()) == expected, 'protected v1 source/evidence changed'
    assert spec == load_json(root / 'source/configs/gqa-shared-diagnostic-v1.json')
    assert spec == load_json('configs/gqa-shared-diagnostic-v1.json')
    assert spec['trajectory_order'] == ARMS
    assert (len(spec['prompt_ids']), len(spec['forced_decode_tokens'])) == (128, 16)
    old_run = load_json('results/gqa-shared-decode-v1/run.json')
    old = load_json('results/gqa-shared-decode-v1/model-correctness.json')[0]
    assert spec['prompt_ids'] == old_run['prompt_token_ids']['128']
    assert spec['forced_decode_tokens'] == old['native']['tokens']
    assert (spec['operator_atol'], spec['operator_rtol'], spec['kv_atol'], spec['kv_rtol'],
            spec['logprobs_atol'], spec['logprobs_rtol']) == (.003, .003, .01, .01, .02, .01)
    assert run['performance_trials'] == 0
    assert not (root / 'operator-samples.json').exists() and not (root / 'model-samples.json').exists()
    if run['status'] == 'failed':
        assert run['diagnostic_complete'] is False and run.get('error_type')
        return dict(archival_integrity_valid=True, evidence_valid=False, diagnostic_complete=False,
            numerical_replay_complete=False, phase=run['phase'], error_type=run['error_type'],
            artifact_count=len(actual), performance_trials=0,
            scope='Partial failure manifest/hashes only; not a completed numerical replay or successful experiment.')
    assert run['diagnostic_complete'] and run['phase'] == 'summarize'
    assert run['metadata_helper_calls'] == spec['metadata_helper_calls'] == 2400
    assert run['peak_mlx_bytes'] <= spec['budget']['peak_mlx_soft_bytes']
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= spec['budget']['disk_bytes']
    captures = load_json(root / 'captures.json')
    assert len(captures) == 4 * 17 * 24
    expected_order = [(arm, step, layer) for arm in ARMS for step in range(17) for layer in range(24)]
    assert [(x['arm'], x['step'], x['layer']) for x in captures] == expected_order
    indexed = {(x['arm'], x['step'], x['layer']): x for x in captures}
    for c in captures:
        step = c['step']
        assert c['offset'] == 128 + step and c['capacity'] == 256
        assert c['native_route'] == (step == 0 or c['arm'] in ('native_a', 'native_b'))
        layout = c['layout']
        assert layout['q_shape'] == [1, 14, 128 if step == 0 else 1, 64]
        assert layout['k_shape'] == layout['v_shape'] == [1, 2, 128 + step, 64]
        assert layout['k_strides'][1:] == layout['v_strides'][1:] == [16384, 64, 1]
        for prefix in ['q', 'k', 'v']:
            strides, shape = layout[prefix + '_strides'], layout[prefix + '_shape']
            assert len(strides) == 4 and all(type(x) is int and
                (x >= 0 if size == 1 else x > 0) for x, size in zip(strides, shape))
    trajectories = {}
    prefix_retention = {}
    compact_keys = [f's{s:02d}-l{l:02d}-{kind}' for s in range(1, 17) for l in range(24) for kind in PARTS]
    final_keys = [f'l{l:02d}-{kind}' for l in range(24) for kind in ['K', 'V']]
    for arm in ARMS:
        folder = root / arm
        pred, compact, final = (load_npz(folder / name) for name in ['predictions.npz', 'compact.npz', 'final-kv.npz'])
        assert set(pred) == {'logits', 'logprobs'}
        check_array(pred['logits'], (17, 151936)); check_array(pred['logprobs'], (17, 151936))
        assert list(compact) == compact_keys and list(final) == final_keys
        for key, a in compact.items():
            check_array(a, (1, 2 if key.endswith(('k_new', 'v_new')) else 14, 1, 64))
        for value in final.values():
            check_array(value, (1, 2, 144, 64))
        trajectory = load_json(folder / 'trajectory.json')
        assert (trajectory['prefill_calls'], trajectory['decode_calls'], trajectory['prediction_rows']) == (24, 384, 17)
        assert trajectory['cache_offsets'] == [144] * 24
        assert trajectory['routes'] == ({} if arm in ('native_a', 'native_b') else dict(prefill=24, decode=384))
        assert trajectory['greedy_prediction_ids'] == np.argmax(pred['logits'], axis=-1).tolist()
        prefill = [indexed[(arm, 0, layer)] for layer in range(24)]
        assert trajectory['prefill_hashes'] == prefill
        for step in range(1, 17):
            for layer in range(24):
                key = f's{step:02d}-l{layer:02d}'
                cap = indexed[(arm, step, layer)]
                assert array_digest(compact[key + '-q']) == cap['q_sha256']
                assert array_digest(compact[key + '-out']) == cap['out_sha256']
                for kind, name in [('K', 'k_new'), ('V', 'v_new')]:
                    full = final[f'l{layer:02d}-{kind}']
                    assert np.array_equal(full[:, :, 127 + step:128 + step], compact[key + '-' + name])
                    assert array_digest(full[:, :, :128 + step]) == cap[kind.lower() + '_sha256'], 'Cache prefix changed after capture'
        prefix_retention[arm] = {key: array_digest(final[key][:, :, :128]) ==
            prefill[int(key[1:3])][key[-1].lower() + '_sha256'] for key in final_keys}
        # Cache writes should only append. This checks all archived prefix values.
        assert all(prefix_retention[arm].values()), 'prefill Cache prefix changed'
        trajectories[arm] = dict(**pred, compact=compact, final=final, prefill=prefill)
    shadow = load_json(root / 'shadow.json')
    assert [(r['step'], r['layer']) for r in shadow] == [(s, l) for s in range(1, 17) for l in range(24)]
    largest_reference_delta = 0.0
    shadow_failures = {key: 0 for key in ['native_reference', 'candidate_reference', 'native_candidate']}
    for row in shadow:
        s, l = row['step'], row['layer']; n = 128 + s
        key = f's{s:02d}-l{l:02d}'
        arrays = load_npz(root / 'native_a' / (key + '.npz'))
        assert set(arrays) == {'q', 'k_storage', 'v_storage', 'native', 'candidate', 'reference'}
        for name in ['q', 'native', 'candidate']:
            check_array(arrays[name], (1, 14, 1, 64))
        for name in ['k_storage', 'v_storage']:
            # The pinned KVCache allocates with mx.zeros. Only its visible
            # prefix participates in Attention; unused capacity must remain zero.
            assert arrays[name].shape == (1, 2, 256, 64) and arrays[name].dtype == np.float16
            assert np.isfinite(arrays[name]).all()
            assert np.all(arrays[name][:, :, n:] == 0), 'unused Cache capacity is not zero'
        q, k, v = arrays['q'], arrays['k_storage'][:, :, :n], arrays['v_storage'][:, :, :n]
        cap = indexed[('native_a', s, l)]
        for a, hash_name in [(q, 'q_sha256'), (k, 'k_sha256'), (v, 'v_sha256'), (arrays['native'], 'out_sha256')]:
            assert array_digest(a) == cap[hash_name]
        assert row['inputs_unchanged'] and row['layout'] == cap['layout']
        ref = reference(q, k, v)
        check_array(arrays['reference'], ref.shape, np.float64)
        np.testing.assert_allclose(ref, arrays['reference'], atol=1e-12, rtol=1e-12)
        largest_reference_delta = max(largest_reference_delta, float(np.max(np.abs(ref - arrays['reference']))))
        for name, left, right in [('native_reference', arrays['native'], arrays['reference']),
                                 ('candidate_reference', arrays['candidate'], arrays['reference']),
                                 ('native_candidate', arrays['native'], arrays['candidate'])]:
            metric = assert_metrics(row[name], left, right, spec['operator_atol'], spec['operator_rtol'])
            shadow_failures[name] += not metric['allclose']
    summary = load_json(root / 'summary.json')
    assert 'original_gate_pass' not in summary
    assert summary['diagnostic_complete'] and summary['performance_trials'] == 0
    assert summary['original_v1_model_gate_pass'] is False
    assert summary['original_kv_gate_scope'] == 'new diagnostic trajectory using original thresholds; does not replace v1 failure'
    assert set(summary['comparisons']) == {a + '__' + b for a, b in PAIRS}
    prefix_comparisons = {}
    for left, right in PAIRS:
        a, b = trajectories[left], trajectories[right]
        name = left + '__' + right; records = summary['comparisons'][name]
        assert [r['key'] for r in records['compact']] == compact_keys
        assert [r['key'] for r in records['final_kv']] == final_keys
        for category, keys in [('compact', compact_keys), ('final_kv', final_keys)]:
            for row, key in zip(records[category], keys):
                kind = 'kv' if category == 'final_kv' or key.endswith(('k_new', 'v_new')) else 'operator'
                assert_metrics({k: v for k, v in row.items() if k != 'key'}, a[category if category == 'compact' else 'final'][key],
                    b[category if category == 'compact' else 'final'][key], spec[kind + '_atol'], spec[kind + '_rtol'])
        assert_metrics(records['logprobs'], a['logprobs'], b['logprobs'], spec['logprobs_atol'], spec['logprobs_rtol'])
        assert_metrics(records['raw_logits'], a['logits'], b['logits'], 0, 0)
        assert records['raw_logits_scope'] == 'diagnostic exact comparison, no original logits tolerance'
        assert records['prefill_hash_equal'] == [all(x[k] == y[k] for k in ['q_sha256', 'k_sha256', 'v_sha256', 'out_sha256']) for x, y in zip(a['prefill'], b['prefill'])]
        assert records['prefill_layout_equal'] == [x['layout'] == y['layout'] for x, y in zip(a['prefill'], b['prefill'])]
        prefix_comparisons[name] = {key: array_digest(a['final'][key][:, :, :128]) == array_digest(b['final'][key][:, :, :128]) for key in final_keys}
    old_arrays = load_npz('results/gqa-shared-decode-v1/model-logprobs-128.npz')
    assert set(summary['fidelity']) == set(ARMS)
    for arm in ARMS:
        old_arm = 'candidate' if arm == 'candidate' else 'native'
        t = trajectories[arm]; f = summary['fidelity'][arm]
        assert_metrics(f['logprobs_first16'], old_arrays[old_arm], t['logprobs'][:16], spec['logprobs_atol'], spec['logprobs_rtol'])
        matches = {f"l{r['layer']:02d}-{r['kind']}": array_digest(t['final'][f"l{r['layer']:02d}-{r['kind']}"]) == r[old_arm + '_sha256'] for r in old['kv']}
        assert f['final_kv_hash_matches'] == matches and f['all_final_kv_exact'] == all(matches.values())
        assert f['greedy_first16_match_forced'] == bool(np.array_equal(np.argmax(t['logits'][:16], axis=-1), spec['forced_decode_tokens']))
    failed = [r['key'] for r in summary['comparisons']['native_a__candidate']['final_kv'] if not r['allclose']]
    previous = [f"l{r['layer']:02d}-{r['kind']}" for r in old['kv'] if not r['allclose']]
    assert summary['original_failed_kv'] == failed and summary['old_failed_kv'] == previous
    assert summary['original_kv_gate_pass'] == (not failed) and summary['failed_set_reproduced'] == (failed == previous)
    first16 = assert_metrics(summary['first16_native_candidate_logprobs'], trajectories['native_a']['logprobs'][:16],
        trajectories['candidate']['logprobs'][:16], spec['logprobs_atol'], spec['logprobs_rtol'])
    greedy_equal = bool(np.array_equal(np.argmax(trajectories['native_a']['logits'][:16], axis=-1), np.argmax(trajectories['candidate']['logits'][:16], axis=-1)))
    return dict(evidence_valid=True, diagnostic_complete=True, protocol_commit=run['protocol_commit'],
        captures_checked=len(captures), shadow_probes_checked=len(shadow), final_kv_arrays_replayed=192,
        prediction_rows_per_arm=17, compared_delivered_rows=16, artifact_count=len(actual),
        independent_reference_max_abs=largest_reference_delta, shadow_failed_probes=shadow_failures,
        new_trajectory_model_conditions=dict(first16_greedy_equal=greedy_equal, first16_logprobs_gate=first16['allclose'],
            final_kv_gate=not failed, conjunction=greedy_equal and first16['allclose'] and not failed),
        original_v1_model_gate_pass=False, prefix_retention=prefix_retention, prefix_comparisons=prefix_comparisons,
        performance_trials=0, scope='Full CPU array/metric replay. Archived GPU-layout and input-immutability receipts are checked for consistency, not re-executed. New diagnostic conditions never replace the failed v1 experiment or identify a unique cause.')


def main():
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/gqa-shared-diagnostic-v1')
    print(json.dumps(verify(parser.parse_args().root), indent=2))


if __name__ == '__main__':
    main()
