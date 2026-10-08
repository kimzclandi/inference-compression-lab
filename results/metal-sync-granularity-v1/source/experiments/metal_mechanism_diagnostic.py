"""One fixed, additive Metal reduction experiment; existing kernels stay unchanged."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

SPEC = Path('configs/mechanism-diagnostics-v1.json')
SOURCES = [__file__, 'lab/kernels/residual_rmsnorm.metal',
           'lab/kernels/residual_rmsnorm_masked.metal', str(SPEC)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(records, spec):
    """Validate fixed coverage and recompute separate control comparisons."""
    expected = {(r, n, a) for r in spec['timing_rows']
                for n in range(spec['rounds']) for a in spec['arms']}
    found = set()
    cells = {}
    import math
    for row in records:
        key = (row['rows'], row['round'], row['arm'])
        if key not in expected or key in found:
            raise ValueError('Unexpected or duplicate timing cell')
        found.add(key)
        samples = row['seconds']
        if len(samples) != spec['repeats'] or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError('Invalid timing samples')
        cells[key] = statistics.median(samples)
    if found != expected:
        raise ValueError('Incomplete timing coverage')
    output = {}
    for r in spec['timing_rows']:
        medians = {a: statistics.median(cells[r, n, a] for n in range(spec['rounds']))
                   for a in spec['arms']}
        comparisons = {}
        for control in ('original', 'compiled'):
            speedup = medians[control] / medians['masked']
            faster = sum(cells[r, n, 'masked'] < cells[r, n, control]
                         for n in range(spec['rounds']))
            comparisons[control] = dict(speedup=speedup, faster_rounds=faster,
                diagnostic_speed_gate=speedup >= spec['minimum_speedup'] and
                faster >= spec['minimum_faster_rounds'])
        output[str(r)] = dict(median_seconds=medians, masked_vs=comparisons)
    return dict(scope=spec['scope'], timing_scope=spec['timing_scope'], shapes=output,
                primary_rows=spec['primary_rows'], model_integration=False,
                causal_limit='Intervention changes zero-fill, synchronization and compiled code together; does not measure isolated barrier cost.')


def run(output):
    spec = json.loads(SPEC.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit protocol and implementation before execution')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for value in SOURCES:
        path = Path(value).resolve().relative_to(Path.cwd())
        dest = output / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        source_hashes[str(path)] = sha(path)
    save(output / 'protocol.json', spec)
    manifest = dict(status='running', git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=source_hashes,
        platform=platform.platform(), started_utc=__import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat())
    save(output / 'run.json', manifest)
    try:
        import mlx.core as mx
        import numpy as np
        if importlib.metadata.version('mlx') != spec['mlx_version'] or not mx.metal.is_available():
            raise RuntimeError('Require pinned MLX and Metal GPU')
        mx.set_default_device(mx.gpu)
        manifest.update(device=mx.metal.device_info(), mlx=importlib.metadata.version('mlx'),
                        numpy=importlib.metadata.version('numpy'))
        rng = np.random.default_rng(spec['seed'])
        eps = mx.array([spec['epsilon']], dtype=mx.float32)
        mx.eval(eps)
        def native(x, residual, weight):
            h = x + residual
            return h, mx.fast.rms_norm(h, weight, spec['epsilon'])
        arms = {'compiled': mx.compile(native, shapeless=False)}
        def make_arm(name, file):
            kernel = mx.fast.metal_kernel(name='diagnostic_' + name,
                input_names=['x', 'residual', 'weight', 'epsilon'], output_names=['h', 'y'],
                source=Path(file).read_text(), header='#include <metal_simdgroup>\n',
                ensure_row_contiguous=True)
            def call(x, residual, weight):
                threads = 32 * ((x.shape[-1] + 127) // 128)
                return kernel(inputs=[x, residual, weight, eps], template=[('T', x.dtype)],
                    grid=(x.size // x.shape[-1] * threads, 1, 1),
                    threadgroup=(threads, 1, 1), output_shapes=[x.shape, x.shape],
                    output_dtypes=[x.dtype, x.dtype], stream=mx.gpu)
            return mx.compile(call, shapeless=False)
        for arm, suffix in [('original', ''), ('masked', '_masked')]:
            arms[arm] = make_arm(arm, f'lab/kernels/residual_rmsnorm{suffix}.metal')
        checks = []
        for dtype in spec['correctness_dtypes']:
            for width in spec['correctness_widths']:
                for rows in spec['correctness_rows']:
                    for pattern in spec['patterns']:
                        shape = (rows, width * (2 if pattern == 'strided' else 1))
                        x = mx.array(rng.normal(size=shape).astype(dtype))
                        residual = mx.array(rng.normal(size=shape).astype(dtype))
                        if pattern == 'zeros':
                            x, residual = mx.zeros(shape, dtype=getattr(mx, dtype)), mx.zeros(shape, dtype=getattr(mx, dtype))
                        elif pattern == 'cancellation':
                            residual = -x
                        elif pattern == 'strided':
                            x, residual = x[:, ::2], residual[:, ::2]
                        weight = mx.array(rng.uniform(0.5, 1.5, width).astype(dtype))
                        values = {a: arms[a](x, residual, weight) for a in spec['arms']}
                        mx.eval(*[v for pair in values.values() for v in pair])
                        values = {a: [np.asarray(v).astype(np.float64) for v in pair] for a, pair in values.items()}
                        for arm in ('original', 'masked'):
                            if not np.array_equal(values[arm][0], values['compiled'][0]):
                                raise ValueError('Residual rounding mismatch')
                            np.testing.assert_allclose(values[arm][1], values['compiled'][1], **spec['tolerances'][dtype])
                        # Removing zero-fill must retain exact custom-kernel results.
                        if not all(np.array_equal(a, b) for a, b in zip(values['masked'], values['original'])):
                            raise ValueError('Candidate differs from original custom kernel')
                        checks.append(dict(dtype=dtype, rows=rows, width=width, pattern=pattern,
                                           original_masked_exact=True, compiled_tolerance_pass=True))
        save(output / 'correctness.json', checks)
        records = []
        order_rng = random.Random(spec['seed'])
        for rows in spec['timing_rows']:
            shape = (rows, spec['timing_width'])
            args = [mx.array(rng.normal(size=shape).astype(spec['timing_dtype'])) for _ in range(2)]
            args.append(mx.ones((shape[-1],), dtype=getattr(mx, spec['timing_dtype'])))
            mx.eval(*args)
            for a in spec['arms']:
                for _ in range(spec['warmups']):
                    mx.eval(*arms[a](*args)); mx.synchronize()
            for round_id in range(spec['rounds']):
                order = list(spec['arms']); order_rng.shuffle(order)
                for a in order:
                    samples = []
                    for _ in range(spec['repeats']):
                        mx.synchronize(); start = time.perf_counter()
                        mx.eval(*arms[a](*args)); mx.synchronize()
                        samples.append(time.perf_counter() - start)
                    records.append(dict(rows=rows, round=round_id, arm=a, seconds=samples, arm_order=order))
                    save(output / 'timings.json', records)
        save(output / 'summary.json', summarize(records, spec))
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save(output / 'run.json', manifest)
        save(output / 'checksums.json', {str(p.relative_to(output)): sha(p)
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'checksums.json'})


def verify(output):
    expected = json.loads((output / 'checksums.json').read_text())
    actual = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
              if p.is_file() and p.name != 'checksums.json'}
    if actual != expected:
        raise ValueError('Evidence bytes changed')
    spec = json.loads((output / 'protocol.json').read_text())
    if spec != json.loads(SPEC.read_text()):
        raise ValueError('Protocol differs')
    run_record = json.loads((output / 'run.json').read_text())
    if run_record['status'] != 'complete':
        raise ValueError('Incomplete experiment')
    for path, digest in run_record['source_sha256'].items():
        if sha(output / 'source' / path) != digest:
            raise ValueError('Archived source identity mismatch')
    expected_cases = {(d, w, r, p) for d in spec['correctness_dtypes']
                      for w in spec['correctness_widths'] for r in spec['correctness_rows']
                      for p in spec['patterns']}
    checks = json.loads((output / 'correctness.json').read_text())
    actual_cases = {(c['dtype'], c['width'], c['rows'], c['pattern']) for c in checks}
    if len(checks) != len(expected_cases) or actual_cases != expected_cases or not all(
            c['original_masked_exact'] is True and c['compiled_tolerance_pass'] is True for c in checks):
        raise ValueError('Incomplete correctness coverage')
    computed = summarize(json.loads((output / 'timings.json').read_text()), spec)
    if computed != json.loads((output / 'summary.json').read_text()):
        raise ValueError('Summary does not reproduce')
    return computed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.output)
    else:
        print(json.dumps(verify(args.output), indent=2))
