"""Diagnostic-only exp-family intervention; no model adapter or performance API."""
from functools import lru_cache
from pathlib import Path


def sources():
    folder = Path(__file__).parent / 'kernels'
    partial = (folder / 'gqa_shared_partial.metal').read_text()
    merge = (folder / 'gqa_shared_merge.metal').read_text()
    assert partial.count('metal::exp(') == 2 and merge.count('metal::exp(') == 1
    for index, numerator in [('2 * lane', 'numerator0'), ('2 * lane + 1', 'numerator1')]:
        old = f'out[head * 64 + {index}] = T({numerator} / denominator);'
        assert merge.count(old) == 1
        merge = merge.replace(old, old + f'\nfp32[head * 64 + {index}] = {numerator} / denominator;')
    return {'standard_partial.metal': partial, 'standard_merge.metal': merge,
            'fast_partial.metal': partial.replace('metal::exp(', 'metal::fast::exp('),
            'fast_merge.metal': merge.replace('metal::exp(', 'metal::fast::exp(')}


@lru_cache(maxsize=2)
def compiled(mode):
    import mlx.core as mx
    if mode not in ('standard', 'fast'):
        raise ValueError('unknown exp mode')
    src = sources()
    first = mx.fast.metal_kernel(name=f'icl_gqa_exp_{mode}_partial_v1',
        input_names=['q', 'k', 'v'], output_names=['partial'],
        source=src[f'{mode}_partial.metal'], header='#include <metal_simdgroup>\n',
        ensure_row_contiguous=False)
    second = mx.fast.metal_kernel(name=f'icl_gqa_exp_{mode}_merge_v1',
        input_names=['partial'], output_names=['out', 'fp32'],
        source=src[f'{mode}_merge.metal'], ensure_row_contiguous=False)

    def execute(q, k, v):
        parts = (k.shape[2] + 127) // 128
        partial = first(inputs=[q, k, v], template=[('T', mx.float16)],
            grid=(parts * 256, 2, 1), threadgroup=(256, 1, 1),
            output_shapes=[(14, parts, 66)], output_dtypes=[mx.float32])[0]
        return tuple(second(inputs=[partial], template=[('T', mx.float16)],
            grid=(14 * 32, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[q.shape, q.shape], output_dtypes=[mx.float16, mx.float32]))
    return mx.compile(execute)


def metrics(arrays, reference):
    """Full-tensor diagnostics; native agreement and FP64 error are separate."""
    import numpy as np
    n = arrays['native']; s = arrays['standard_half']; f = arrays['fast_half']
    sf = arrays['standard_float']; ff = arrays['fast_float']
    if reference.shape != (1, 14, 1, 64) or not np.isfinite(reference).all():
        raise ValueError('invalid reference')
    expected = {'native':np.float16, 'original':np.float16, 'standard_half':np.float16,
                'fast_half':np.float16, 'standard_float':np.float32, 'fast_float':np.float32}
    if set(arrays) != set(expected):
        raise ValueError('invalid output keys')
    if any(x.shape != (1, 14, 1, 64) or x.dtype != expected[name] or not np.isfinite(x).all()
           for name,x in arrays.items()):
        raise ValueError('invalid or nonfinite output')
    old = s != n; new = f != n
    result = dict(elements=int(n.size), standard_native_unequal=int(old.sum()),
        fast_native_unequal=int(new.sum()), resolved=int((old & ~new).sum()),
        persistent=int((old & new).sum()), introduced=int((~old & new).sum()),
        half_changed=int((s != f).sum()), float_changed=int((sf != ff).sum()))
    for name in ('native', 'standard_half', 'fast_half', 'standard_float', 'fast_float'):
        delta = np.abs(arrays[name].astype(np.float64) - reference)
        result[name + '_reference'] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
            failed_elements=int((~np.isclose(arrays[name], reference, atol=.003, rtol=.003)).sum()))
    for suffix in ('half', 'float'):
        a = np.abs(arrays['standard_' + suffix].astype(np.float64) - reference)
        b = np.abs(arrays['fast_' + suffix].astype(np.float64) - reference)
        result[suffix + '_reference_change'] = dict(improved=int((b < a).sum()),
            worsened=int((b > a).sum()), equal=int((b == a).sum()))
    rounded = reference.astype(np.float16)
    result['reference_rounding_disagreement'] = {name: int((arrays[name] != rounded).sum())
        for name in ('native', 'standard_half', 'fast_half')}
    boundaries = []
    for c in np.argwhere(old):
        index = tuple(int(x) for x in c)
        low, high = sorted((float(s[index]), float(n[index])))
        midpoint = (low + high) / 2
        boundaries.append(dict(coordinate=list(index), native=float(n[index]), standard=float(s[index]),
            fast=float(f[index]), standard_float=float(sf[index]), fast_float=float(ff[index]),
            reference=float(reference[index]), midpoint=midpoint,
            adjacent=bool(np.nextafter(np.float16(low), np.float16(np.inf), dtype=np.float16) == np.float16(high)),
            reference_distance_gap_units=float((reference[index] - midpoint) / (high - low)),
            standard_distance_gap_units=float((float(sf[index]) - midpoint) / (high - low)),
            fast_distance_gap_units=float((float(ff[index]) - midpoint) / (high - low)),
            midpoint_ties_to_even=float(np.float16(midpoint))))
    result['original_disagreement_boundaries'] = boundaries
    return result


def aggregate(records):
    rows = [r['metrics'] for r in records]
    counts = ('elements', 'standard_native_unequal', 'fast_native_unequal', 'resolved',
              'persistent', 'introduced', 'half_changed', 'float_changed')
    result = {key: sum(r[key] for r in rows) for key in counts}
    for name in ('native', 'standard_half', 'fast_half', 'standard_float', 'fast_float'):
        result[name + '_reference'] = {
            'max_abs': max(r[name + '_reference']['max_abs'] for r in rows),
            'mean_abs': sum(r[name + '_reference']['mean_abs'] for r in rows) / len(rows),
            'failed_elements': sum(r[name + '_reference']['failed_elements'] for r in rows)}
    for suffix in ('half', 'float'):
        result[suffix + '_reference_change'] = {k: sum(r[suffix + '_reference_change'][k] for r in rows)
                                               for k in ('improved', 'worsened', 'equal')}
    result['reference_rounding_disagreement'] = {name: sum(r['reference_rounding_disagreement'][name] for r in rows)
        for name in ('native', 'standard_half', 'fast_half')}
    result['all_native_disagreements_eliminated'] = result['fast_native_unequal'] == 0
    result['operator_tolerance_pass'] = all(result[n + '_reference']['failed_elements'] == 0
        for n in ('native', 'standard_half', 'fast_half'))
    result['hypothesis_supported'] = result['all_native_disagreements_eliminated'] and result['operator_tolerance_pass']
    result['scope'] = 'Exp-family intervention on instrumented operators only; not native arithmetic identity, unique root cause, model repair, quality or speed.'
    return result
