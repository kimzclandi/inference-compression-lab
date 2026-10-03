"""Rebuild FP16/Q4/Q8 and fixed block controls from a pinned local upstream model."""
import argparse
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import platform
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, write, sha
from lab.model_identity import tensor_identity, verify_tensor_identity
from lab.artifact_integrity import file_hashes, git_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--evidence-dir', type=Path, required=True)
    p.add_argument('--identities', type=Path, default=Path('configs/qwen-quantization/tensor-identities.json'))
    a = p.parse_args()
    if not a.source.is_dir():
        p.error('--source must be an existing local directory')
    upstream = read('configs/qwen-quantization/upstream-identity.json')
    for name, meta in upstream.items():
        if sha(a.source / name) != meta['sha256']:
            raise ValueError('Pinned upstream source differs: ' + name)
    for out in [a.output_dir, a.evidence_dir]:
        if out.exists():
            raise FileExistsError(out)
    evidence = reserve_directory(a.evidence_dir); out = reserve_directory(a.output_dir)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.utils import fetch_from_hub, quantize_model, save
    from experiments.qwen_quantization import export_block, model_files, utc
    source_files = ['experiments/prepare_qwen_quantization_assets.py', 'experiments/qwen_quantization.py',
                    'lab/model_identity.py', 'lab/artifact_integrity.py', 'lab/quantization_diagnostics.py',
                    'lab/evidence.py', 'lab/qa_metrics.py']
    for name in source_files:
        dest = evidence / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, dest)
    status = {'status': 'running', 'started': utc(), **git_identity(Path.cwd()), 'models': {},
              'source_sha256': {name: sha(name) for name in source_files},
              'expected_identities_sha256': sha(a.identities), 'upstream_identity': upstream,
              'python': platform.python_version(),
              'packages': {n: importlib.metadata.version(n) for n in ['mlx','mlx-lm','numpy','transformers']}}
    try:
        expected = read(a.identities)
        for variant in ['fp16', 'q4', 'q8']:
            source = a.source if variant == 'fp16' else out / 'student-fp16'
            destination = out / f'student-{variant}'
            model, config, tok = fetch_from_hub(source, lazy=True, trust_remote_code=False)
            if variant == 'fp16':
                model.update(tree_map(lambda x: x.astype(mx.float16) if mx.issubdtype(x.dtype, mx.floating) else x,
                                      model.parameters()))
                config['torch_dtype'] = 'float16'
            else:
                model, config = quantize_model(model, config, 64, int(variant[1:]))
            save(destination, source, model, tok, config, hf_repo=None)
            for name in ['LICENSE', 'README.md']:
                shutil.copy2(source / name, destination / name)
            del model, tok; gc.collect(); mx.clear_cache()
            actual = tensor_identity(destination); verify_tensor_identity(actual, expected[variant])
            status['models'][variant] = {'tensor_identity_verified': True, 'files': model_files(destination)}
        for variant, block in [('selected', 10), ('control', 22)]:
            export = export_block(out / 'student-fp16', out / 'student-q4', block, out / variant)
            verify_tensor_identity(tensor_identity(out / variant), expected[variant])
            status['models'][variant] = {'tensor_identity_verified': True, 'files': export['model_files']}
        status['status'] = 'complete'
    except BaseException as e:
        status.update(status='failed', error=repr(e)); raise
    finally:
        status['finished'] = utc(); write(evidence / 'run.json', status)
        write(evidence / 'checksums.json', file_hashes(evidence))


if __name__ == '__main__':
    main()
