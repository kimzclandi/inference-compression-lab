"""One frozen, model-free GPU sweep of archived GQA inputs; never a benchmark."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

CONFIG = Path('configs/gqa-exp-choice-v1.json')
SOURCES = [str(CONFIG), 'experiments/gqa_exp_choice.py', 'lab/gqa_exp_choice.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def finish_manifest(root, run):
    run['artifacts'] = {str(p.relative_to(root)): sha(p) for p in root.rglob('*')
                        if p.is_file() and p != root / 'run.json'}
    save(root / 'run.json', run)


def worker():
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    spec = json.loads(CONFIG.read_text())
    root = Path(spec['output'])
    if root.exists():
        raise FileExistsError('new output directory required; rerun forbidden')
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'clean frozen checkout required'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    root.mkdir()
    run = dict(status='running', phase='identity', protocol_commit=commit, spec=spec,
        source_sha256={p: sha(p) for p in SOURCES}, probes_completed=0,
        diagnostic_complete=False, performance_trials=0, model_runs=0)
    records = []
    save(root / 'run.json', run)
    try:
        for name in SOURCES:
            target = root / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
        for name, digest in spec['protected_sha256'].items():
            assert sha(name) == digest, 'protected identity changed'
        oldroot = Path(spec['input_archive'])
        oldrun = json.loads((oldroot / 'run.json').read_text())
        assert {str(p.relative_to(oldroot)): sha(p) for p in oldroot.rglob('*')
            if p.is_file() and p != oldroot / 'run.json'} == oldrun['artifacts']
        import mlx.core as mx
        import mlx_lm
        import numpy as np
        from mlx_lm.generate import generation_stream
        from lab.gqa_shared_decode import attention
        from lab.gqa_diagnostic import array_sha
        from lab.gqa_exp_choice import sources, compiled, metrics, aggregate
        for name, version in spec['versions'].items():
            assert importlib.metadata.version(name) == version, 'version changed'
        site = Path(mlx_lm.__file__).parent.parent
        for name, digest in spec['upstream_sha256'].items():
            assert sha(site / name) == digest, 'upstream changed'
        assert mx.default_device() == mx.gpu and mx.metal.is_available(), 'GPU required'
        run['device'] = mx.metal.device_info()
        assert run['device'] == oldrun['device'], 'device properties changed'
        assert subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip() == spec['host_os_build'], 'OS build changed'
        mx.reset_peak_memory()
        (root / 'generated').mkdir()
        for name, src in sources().items():
            (root / 'generated' / name).write_text(src)
        (root / 'probes').mkdir()
        metadata = mx.fast.metal_kernel(name='icl_exp_layout_v1', input_names=['q','k','v'],
            output_names=['meta'], source=Path('lab/kernels/gqa_diagnostic_layout.metal').read_text(),
            ensure_row_contiguous=False)

        def layout(q, k, v):
            a = np.array(metadata(inputs=[q,k,v], grid=(32,1,1), threadgroup=(32,1,1),
                output_shapes=[(24,)], output_dtypes=[mx.int32])[0])
            return {key: [int(x) for x in a[i:i+4]] for key, i in [('q_shape',0),('q_strides',4),
                ('k_shape',8),('k_strides',12),('v_shape',16),('v_strides',20)]}

        shadow = json.loads((oldroot / 'shadow.json').read_text())
        assert [(r['step'], r['layer']) for r in shadow] == [(s,l) for s in range(1,17) for l in range(24)]
        run['phase'] = 'fixed_sweep'
        for old in shadow:
            step, layer = old['step'], old['layer']
            key = f's{step:02d}-l{layer:02d}'
            save(root / 'last-probe.json', dict(key=key, step=step, layer=layer))
            with np.load(oldroot / 'native_a' / (key + '.npz'), allow_pickle=False) as values:
                a = {k: values[k] for k in values.files}
            n = 128 + step
            with mx.stream(generation_stream):
                q = mx.array(a['q']).reshape(1,1,14,64).transpose(0,2,1,3)
                ks, vs = mx.array(a['k_storage']), mx.array(a['v_storage'])
                k, v = ks[:,:,:n,:], vs[:,:,:n,:]
                before = layout(q,k,v)
                inputs_before = [array_sha(np.array(t)) for t in (q,k,v)]
                expected_hashes = [array_sha(x) for x in (a['q'],a['k_storage'][:,:,:n],a['v_storage'][:,:,:n])]
                receipt = dict(key=key, stage='before', expected_layout=old['layout'],
                    expected_input_sha256=expected_hashes, layout_before=before,
                    input_sha256_before=inputs_before,
                    gates=dict(layout_reconstructed=before == old['layout'], values_reconstructed=inputs_before == expected_hashes))
                save(root / 'probes' / (key + '.json'), receipt)
                if not all(receipt['gates'].values()):
                    np.savez_compressed(root / 'input-failure.npz', q=np.array(q), k=np.array(k), v=np.array(v))
                    raise ValueError('reconstructed layout or input fidelity failure')
                # Fixed order; no warmup/repeats because no time measurements.
                native = attention(q,k,v,mode='native')
                mx.eval(native)
                original = attention(q,k,v,mode='shared_compiled')
                mx.eval(original)
                sh,sf = compiled('standard')(q,k,v)
                mx.eval(sh,sf)
                fh,ff = compiled('fast')(q,k,v)
                mx.eval(fh,ff)
                arrays = {name: np.array(value).copy() for name,value in zip(
                    ['native','original','standard_half','standard_float','fast_half','fast_float'],
                    [native,original,sh,sf,fh,ff])}
                after = layout(q,k,v)
                inputs_after = [array_sha(np.array(t)) for t in (q,k,v)]
            np.savez_compressed(root / 'probes' / (key + '.npz'), **arrays)
            def exact(x,y):
                return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            receipt.update(stage='after',layout_after=after,input_sha256_after=inputs_after)
            receipt['gates'].update(finite=all(np.isfinite(x).all() for x in arrays.values()),
                inputs_unchanged=before == after and inputs_before == inputs_after,
                native_reproduced=exact(arrays['native'],a['native']),
                original_reproduced=exact(arrays['original'],a['candidate']),
                tap_reproduced=exact(arrays['standard_half'],arrays['original']))
            for mode in ('standard','fast'):
                receipt['gates'][mode+'_cast'] = exact(arrays[mode+'_float'].astype(np.float16),arrays[mode+'_half'])
            save(root / 'probes' / (key + '.json'), receipt)
            if not all(receipt['gates'].values()):
                if not receipt['gates']['inputs_unchanged']:
                    np.savez_compressed(root / 'input-failure.npz',q=np.array(q),k=np.array(k),v=np.array(v))
                raise ValueError('fidelity failure; see current probe receipt')
            records.append(dict(key=key,step=step,layer=layer,layout_before=before,layout_after=after,
                input_sha256_before=inputs_before,input_sha256_after=inputs_after,
                metrics=metrics(arrays,a['reference'])))
            save(root / 'records.json', records)
            run['probes_completed'] = len(records)
            run['peak_mlx_bytes'] = max(run.get('peak_mlx_bytes',0), int(mx.get_peak_memory()))
            save(root / 'run.json', run)
            assert run['peak_mlx_bytes'] <= spec['budget']['peak_mlx_bytes'], 'memory budget'
            assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= spec['budget']['disk_bytes'], 'disk budget'
        assert len(records) == 384
        save(root / 'summary.json', aggregate(records))
        run.update(status='complete',phase='complete',diagnostic_complete=True)
    except Exception as error:
        run.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        finish_manifest(root, run)


def main():
    if not __debug__:
        raise RuntimeError('optimized mode unsupported')
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    spec = json.loads(CONFIG.read_text())
    root = Path(spec['output'])
    if root.exists():
        raise FileExistsError('new output directory required; rerun forbidden')
    try:
        completed = subprocess.run([sys.executable, '-m', 'experiments.gqa_exp_choice', '--worker'],
                                   timeout=spec['budget']['hard_seconds'])
    except subprocess.TimeoutExpired:
        run = json.loads((root / 'run.json').read_text())
        run.update(status='failed', diagnostic_complete=False, error_type='TimeoutExpired', error='hard budget')
        finish_manifest(root, run)
        raise
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
