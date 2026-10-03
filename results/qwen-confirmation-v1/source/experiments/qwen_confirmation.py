"""One-shot evaluation of fixed variants; never selects or changes modules."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
from lab.artifact_integrity import file_hashes, git_identity
from lab.evidence import reserve_directory
from lab.model_identity import tensor_identity, verify_tensor_identity
from lab.quantization_diagnostics import read, write, sha, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=Path('configs/qwen-confirmation/study.json'))
    a = p.parse_args(); spec = read(a.spec)
    for name, expected in [(spec['data'], spec['data_sha256']), (spec['prompt'], spec['prompt_sha256']),
                           (spec['quality']['unchanged_scorer'], spec['scorer_sha256']),
                           ('results/qwen-quantization-v1/selection.json', spec['selection_frozen_sha256'])]:
        if sha(name) != expected:
            raise ValueError('Frozen protocol input changed: ' + name)
    if spec['variants'] != ['fp16', 'q4', 'q8', 'selected', 'control'] or spec['blocks'] != {'selected': 10, 'control': 22}:
        raise ValueError('Confirmation may not reselect variants or blocks')
    if len(rows(spec['data'])) != spec['quality']['expected_n']:
        raise ValueError('Confirmation dataset count')
    paths = {v: a.model_root / (f'student-{v}' if v in ['fp16','q4','q8'] else v) for v in spec['variants']}
    if not all(path.is_dir() for path in paths.values()):
        raise ValueError('All five existing local models are required')
    out = reserve_directory(a.output_dir)
    for label, path in [('protocol.json', a.spec), ('data.jsonl', spec['data']), ('prompt.json', spec['prompt']),
                        ('tensor-identities.json', 'configs/qwen-quantization/tensor-identities.json'),
                        ('dataset-manifest.json', 'configs/qwen-confirmation/dataset/manifest.json'),
                        ('exclusions.json', 'configs/qwen-confirmation/exclusions.json'),
                        ('selection.json', 'configs/qwen-confirmation/selection.json')]:
        shutil.copy2(path, out / label)
    source = sorted([*Path('lab').glob('*.py'), *Path('experiments').glob('*qwen*.py')])
    for path in source:
        dest = out / 'source' / path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from experiments.qwen_quantization import quality, utc
    status = {'status': 'running', 'started': utc(), **git_identity(Path.cwd()), 'argv': sys.argv,
              'protocol_sha256': sha(a.spec), 'source_sha256': {str(path): sha(path) for path in source},
              'python': platform.python_version(), 'os': platform.platform(),
              'packages': {n: importlib.metadata.version(n) for n in ['mlx','mlx-lm','numpy','transformers','psutil']},
              'model_identity_verified': {}, 'completed_variants': []}
    write(out / 'started.json', status)
    try:
        import mlx.core as mx
        mx.random.seed(spec['seed']); status['device'] = mx.metal.device_info()
        identities = read(out / 'tensor-identities.json')
        for v in spec['variants']:
            verify_tensor_identity(tensor_identity(paths[v]), identities[v])
            status['model_identity_verified'][v] = True
        ref = None
        for v in spec['variants']:
            result = quality(paths[v], v, out, spec, ref, spec['blocks'].get(v))
            if v == 'fp16': ref = result
            status['completed_variants'].append(v)
        status['status'] = 'complete'
    except BaseException as e:
        status.update(status='failed', error=repr(e)); raise
    finally:
        status['finished'] = utc(); write(out / 'run.json', status)
        write(out / 'checksums.json', file_hashes(out))


if __name__ == '__main__':
    main()
