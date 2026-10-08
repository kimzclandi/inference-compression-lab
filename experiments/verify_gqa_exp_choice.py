"""Independent CPU replay of the single-factor, model-free exp diagnostic.

No call to the experiment runner, exp-choice library, MLX, or Metal is used.
Hash integrity is necessary but insufficient: arrays, fixed-input fidelity,
tap casts, every mechanism statistic, and the aggregate are recomputed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.verify_gqa_shared_diagnostic import reference

CONFIG = 'configs/gqa-exp-choice-v1.json'
SOURCES = {CONFIG, 'experiments/gqa_exp_choice.py', 'lab/gqa_exp_choice.py'}
OUTPUTS = {'native': np.float16, 'original': np.float16, 'standard_half': np.float16,
           'standard_float': np.float32, 'fast_half': np.float16, 'fast_float': np.float32}
REFERENCE_ARMS = ('native', 'standard_half', 'fast_half', 'standard_float', 'fast_float')
COUNTS = ('elements', 'standard_native_unequal', 'fast_native_unequal', 'resolved',
          'persistent', 'introduced', 'half_changed', 'float_changed')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def members(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*')
            if p.is_file() and p != root / 'run.json'}


def safe(root, name):
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    out = Path(root) / path
    assert out.resolve().is_relative_to(Path(root).resolve()), 'escaping source path'
    return out


def npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def expected_sources():
    """Independently allow only two added stores and the three exp substitutions."""
    partial = Path('lab/kernels/gqa_shared_partial.metal').read_text()
    original_merge = Path('lab/kernels/gqa_shared_merge.metal').read_text()
    assert partial.count('metal::exp(') == 2 and original_merge.count('metal::exp(') == 1
    lines = []
    taps = 0
    for line in original_merge.splitlines(keepends=True):
        lines.append(line)
        for target, numerator in [('head * 64 + 2 * lane', 'numerator0'),
                                  ('head * 64 + 2 * lane + 1', 'numerator1')]:
            if line.strip() == f'out[{target}] = T({numerator} / denominator);':
                lines.append(f'fp32[{target}] = {numerator} / denominator;\n')
                taps += 1
    assert taps == 2
    merge = ''.join(lines)
    return dict(standard_partial_metal=partial, standard_merge_metal=merge,
        fast_partial_metal=partial.replace('metal::exp(', 'metal::fast::exp('),
        fast_merge_metal=merge.replace('metal::exp(', 'metal::fast::exp('))


def check_generated(root):
    expected = {key.replace('_metal', '.metal'): value for key, value in expected_sources().items()}
    assert {p.name for p in (Path(root) / 'generated').iterdir()} == set(expected)
    for name, value in expected.items():
        assert (Path(root) / 'generated' / name).read_text() == value, 'generated arithmetic changed'


def recompute_metrics(arrays, ref):
    """Reconstruct numerical and midpoint records without the producer helper."""
    assert set(arrays) == set(OUTPUTS)
    assert ref.shape == (1, 14, 1, 64) and ref.dtype == np.float64 and np.isfinite(ref).all()
    for key, dtype in OUTPUTS.items():
        a = arrays[key]
        assert a.shape == ref.shape and a.dtype == dtype and np.isfinite(a).all(), 'invalid output'
    native, standard, fast = [arrays[k] for k in ['native', 'standard_half', 'fast_half']]
    sf, ff = arrays['standard_float'], arrays['fast_float']
    old_diff, new_diff = standard != native, fast != native
    # These four mutually exclusive cases partition every output element.
    resolved = np.logical_and(old_diff, np.logical_not(new_diff))
    persistent = np.logical_and(old_diff, new_diff)
    introduced = np.logical_and(np.logical_not(old_diff), new_diff)
    equal = np.logical_and(np.logical_not(old_diff), np.logical_not(new_diff))
    assert int(resolved.sum() + persistent.sum() + introduced.sum() + equal.sum()) == native.size
    row = dict(elements=int(native.size), standard_native_unequal=int(old_diff.sum()),
        fast_native_unequal=int(new_diff.sum()), resolved=int(resolved.sum()), persistent=int(persistent.sum()),
        introduced=int(introduced.sum()), half_changed=int(np.count_nonzero(standard != fast)),
        float_changed=int(np.count_nonzero(sf != ff)))
    for arm in REFERENCE_ARMS:
        error = np.abs(arrays[arm].astype(np.float64) - ref)
        row[arm + '_reference'] = dict(max_abs=float(np.max(error)), mean_abs=float(np.mean(error)),
            failed_elements=int(np.count_nonzero(~np.isclose(arrays[arm], ref, atol=.003, rtol=.003))))
    for kind in ('half', 'float'):
        original_error = np.abs(arrays['standard_' + kind].astype(np.float64) - ref)
        changed_error = np.abs(arrays['fast_' + kind].astype(np.float64) - ref)
        row[kind + '_reference_change'] = dict(improved=int(np.count_nonzero(changed_error < original_error)),
            worsened=int(np.count_nonzero(changed_error > original_error)),
            equal=int(np.count_nonzero(changed_error == original_error)))
    rounded = ref.astype(np.float16)
    row['reference_rounding_disagreement'] = {arm: int(np.count_nonzero(arrays[arm] != rounded))
                                              for arm in ['native', 'standard_half', 'fast_half']}
    boundaries = []
    for flat in np.flatnonzero(old_diff):
        index = tuple(int(i) for i in np.unravel_index(int(flat), ref.shape))
        lo = min(float(standard[index]), float(native[index]))
        hi = max(float(standard[index]), float(native[index]))
        midpoint = (lo + hi) * .5
        # The denominator is the endpoint gap. It is a single FP16 spacing
        # only when the separately recorded adjacency check is true.
        boundaries.append(dict(coordinate=list(index), native=float(native[index]), standard=float(standard[index]),
            fast=float(fast[index]), standard_float=float(sf[index]), fast_float=float(ff[index]),
            reference=float(ref[index]), midpoint=midpoint,
            adjacent=bool(np.nextafter(np.float16(lo), np.float16(np.inf)) == np.float16(hi)),
            reference_distance_gap_units=float((ref[index] - midpoint) / (hi - lo)),
            standard_distance_gap_units=float((float(sf[index]) - midpoint) / (hi - lo)),
            fast_distance_gap_units=float((float(ff[index]) - midpoint) / (hi - lo)),
            midpoint_ties_to_even=float(np.float16(midpoint))))
    row['original_disagreement_boundaries'] = boundaries
    return row


def recompute_summary(rows):
    assert len(rows) > 0
    result = {key: sum(r[key] for r in rows) for key in COUNTS}
    for arm in REFERENCE_ARMS:
        entries = [r[arm + '_reference'] for r in rows]
        result[arm + '_reference'] = dict(max_abs=max(x['max_abs'] for x in entries),
            mean_abs=sum(x['mean_abs'] for x in entries) / len(entries),
            failed_elements=sum(x['failed_elements'] for x in entries))
    for kind in ('half', 'float'):
        result[kind + '_reference_change'] = {metric: sum(r[kind + '_reference_change'][metric] for r in rows)
                                              for metric in ['improved', 'worsened', 'equal']}
    result['reference_rounding_disagreement'] = {arm: sum(r['reference_rounding_disagreement'][arm] for r in rows)
                                                for arm in ['native', 'standard_half', 'fast_half']}
    result['all_native_disagreements_eliminated'] = result['fast_native_unequal'] == 0
    result['operator_tolerance_pass'] = all(result[arm + '_reference']['failed_elements'] == 0
                                          for arm in ['native', 'standard_half', 'fast_half'])
    result['hypothesis_supported'] = result['all_native_disagreements_eliminated'] and result['operator_tolerance_pass']
    result['scope'] = 'Exp-family intervention on instrumented operators only; not native arithmetic identity, unique root cause, model repair, quality or speed.'
    return result


def verify_probe(row, arrays, archived, old):
    step, layer = row['step'], row['layer']; n = 128 + step
    assert row['key'] == f's{step:02d}-l{layer:02d}'
    assert (step, layer) == (old['step'], old['layer'])
    assert row['layout_before'] == row['layout_after'] == old['layout']
    assert archived['q'].shape == (1, 14, 1, 64) and archived['q'].dtype == np.float16
    for name in ['k_storage', 'v_storage']:
        assert archived[name].shape == (1, 2, 256, 64) and archived[name].dtype == np.float16
        assert np.isfinite(archived[name]).all() and np.all(archived[name][:, :, n:] == 0)
    q, k, v = archived['q'], archived['k_storage'][:, :, :n], archived['v_storage'][:, :, :n]
    hashes = [digest(x.tobytes()) for x in (q, k, v)]
    assert row['input_sha256_before'] == row['input_sha256_after'] == hashes, 'input identity changed'
    independent = reference(q, k, v)
    assert archived['reference'].shape == independent.shape and archived['reference'].dtype == np.float64
    np.testing.assert_allclose(archived['reference'], independent, atol=1e-12, rtol=1e-12)
    assert exact(arrays['native'], archived['native']), 'native fidelity changed'
    assert exact(arrays['original'], archived['candidate']), 'original fidelity changed'
    assert exact(arrays['standard_half'], arrays['original']), 'tap fidelity changed'
    for mode in ['standard', 'fast']:
        assert exact(arrays[mode + '_float'].astype(np.float16), arrays[mode + '_half']), 'tap cast changed'
    computed = recompute_metrics(arrays, archived['reference'])
    assert row['metrics'] == computed, 'mechanism metrics changed'
    return computed, float(np.max(np.abs(independent - archived['reference'])))


def verify_receipt(receipt, row):
    gates = ['layout_reconstructed', 'values_reconstructed', 'finite', 'inputs_unchanged',
             'native_reproduced', 'original_reproduced', 'tap_reproduced', 'standard_cast', 'fast_cast']
    expected = dict(key=row['key'], stage='after', expected_layout=row['layout_before'],
        expected_input_sha256=row['input_sha256_before'], layout_before=row['layout_before'],
        layout_after=row['layout_after'], input_sha256_before=row['input_sha256_before'],
        input_sha256_after=row['input_sha256_after'], gates={key: True for key in gates})
    assert receipt == expected, 'probe receipt changed'
    assert all(type(value) is bool for value in receipt['gates'].values())


def verify(root):
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    root = Path(root); run = read(root / 'run.json'); spec = run['spec']
    assert run['status'] in ('complete', 'failed')
    assert len(run['protocol_commit']) == 40 and all(c in '0123456789abcdef' for c in run['protocol_commit'])
    assert members(root) == run['artifacts'], 'artifact modified'
    assert set(run['source_sha256']) == SOURCES
    for name, value in run['source_sha256'].items():
        assert digest(safe(root / 'source', name).read_bytes()) == value
        assert digest(safe(Path.cwd(), name).read_bytes()) == value, 'source changed'
    assert spec == read(root / 'source' / CONFIG) == read(CONFIG)
    assert spec['input_count'] == 384 and spec['arm_order'] == ['native', 'original', 'standard', 'fast']
    assert (spec['operator_atol'], spec['operator_rtol']) == (.003, .003)
    assert spec['repetition'] == dict(primary_runs=1, warmup=0, repeats=1, seed=None,
        reason='No RNG, no timing. Fixed archived tensors; one diagnostic sweep only.')
    for name, value in spec['protected_sha256'].items():
        assert digest(safe(Path.cwd(), name).read_bytes()) == value, 'protected artifact changed'
    oldroot = Path(spec['input_archive'])
    assert str(oldroot / 'run.json') in spec['protected_sha256']
    oldrun = read(oldroot / 'run.json')
    assert oldrun['status'] == 'complete' and oldrun['diagnostic_complete']
    assert members(oldroot) == oldrun['artifacts'], 'input archive modified'
    oldsummary = read(oldroot / 'summary.json')
    assert oldsummary['original_v1_model_gate_pass'] is False
    assert len(oldsummary['original_failed_kv']) == 10 and oldsummary['failed_set_reproduced']
    assert read('results/gqa-shared-decode-v1/run.json')['status'] == 'failed'
    assert run['model_runs'] == run['performance_trials'] == 0
    assert not any((root / x).exists() for x in ['model-samples.json', 'operator-samples.json'])
    if run['status'] == 'failed':
        assert not run['diagnostic_complete'] and run.get('error_type')
        return dict(archival_integrity_valid=True, evidence_valid=False, diagnostic_complete=False,
            numerical_replay_complete=False, probes_completed=run['probes_completed'], model_runs=0,
            performance_trials=0, original_model_gate_pass=False,
            scope='Partial manifest integrity only. This is not a complete experiment or a numerical replay.')
    assert run['phase'] == 'complete' and run['diagnostic_complete']
    assert run['device'] == oldrun['device'], 'recorded device changed'
    assert run['probes_completed'] == 384
    assert run['peak_mlx_bytes'] <= spec['budget']['peak_mlx_bytes']
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= spec['budget']['disk_bytes']
    check_generated(root)
    records, old_records = read(root / 'records.json'), read(oldroot / 'shadow.json')
    order = [(step, layer) for step in range(1, 17) for layer in range(24)]
    assert [(r['step'], r['layer']) for r in records] == order
    assert [(r['step'], r['layer']) for r in old_records] == order
    probe_names = {f's{s:02d}-l{l:02d}.{ext}' for s, l in order for ext in ['npz', 'json']}
    assert {p.name for p in (root / 'probes').iterdir()} == probe_names
    expected_members = {'last-probe.json', 'records.json', 'summary.json'} | {'source/' + x for x in SOURCES}
    expected_members |= {'probes/' + x for x in probe_names}
    expected_members |= {'generated/' + key.replace('_metal', '.metal') for key in expected_sources()}
    assert set(run['artifacts']) == expected_members, 'unexpected or missing archive member'
    assert read(root / 'last-probe.json') == dict(key='s16-l23', step=16, layer=23)
    rebuilt = []; max_reference_delta = 0.
    for row, old in zip(records, old_records):
        arrays = npz(root / 'probes' / (row['key'] + '.npz'))
        archived = npz(oldroot / 'native_a' / (row['key'] + '.npz'))
        result, delta = verify_probe(row, arrays, archived, old)
        verify_receipt(read(root / 'probes' / (row['key'] + '.json')), row)
        rebuilt.append(result); max_reference_delta = max(max_reference_delta, delta)
    summary = recompute_summary(rebuilt)
    assert summary == read(root / 'summary.json'), 'aggregate changed'
    assert summary['elements'] == 384 * 896
    assert sum(len(x['original_disagreement_boundaries']) for x in rebuilt) == summary['standard_native_unequal']
    return dict(evidence_valid=True, diagnostic_complete=True, protocol_commit=run['protocol_commit'],
        probes_replayed=384, full_output_elements=summary['elements'], input_artifacts_checked=len(oldrun['artifacts']),
        independent_reference_max_abs=max_reference_delta, generated_sources_exact=True,
        mechanism_summary=summary, midpoint_records_replayed=summary['standard_native_unequal'],
        model_runs=0, performance_trials=0, original_model_gate_pass=False, original_failed_kv=10,
        scope='CPU recomputes every saved array metric and fixed-input fidelity claim. GPU execution, physical layout and before/after immutability remain hash-bound recorded observations. This does not repair the model or prove a unique numerical cause.')


def main():
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    p = argparse.ArgumentParser(); p.add_argument('--root', default='results/gqa-exp-choice-v1')
    print(json.dumps(verify(p.parse_args().root), indent=2))


if __name__ == '__main__':
    main()
