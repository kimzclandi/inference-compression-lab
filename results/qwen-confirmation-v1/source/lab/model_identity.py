"""Tensor identity survives container-header ordering; file hashes remain recorded."""
from pathlib import Path
import hashlib
from lab.quantization_diagnostics import read, sha

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')


def tensor_identity(path):
    import mlx.core as mx
    import numpy as np
    if not Path(path).is_dir():
        raise ValueError('Existing local model directory required')
    tensors = {}
    for file in sorted(Path(path).glob('*.safetensors')):
        for name, value in mx.load(str(file)).items():
            if name in tensors:
                raise ValueError('Duplicate tensor across shards: ' + name)
            tensors[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                             'bytes': int(value.nbytes),
                             'sha256': hashlib.sha256(np.array(value).tobytes()).hexdigest()}
    if not tensors:
        raise ValueError('No local safetensors weights')
    config = read(Path(path) / 'config.json')
    # Upstream may rewrite JSON spacing; compare complete parsed configuration.
    return {'config': config, 'tensors': tensors,
            'tokenizer_files': {name: sha(Path(path) / name) for name in TOKENIZER_FILES}}


def verify_tensor_identity(actual, expected):
    if actual != expected:
        changed = [k for k in set(actual['tensors']) | set(expected['tensors'])
                   if actual['tensors'].get(k) != expected['tensors'].get(k)]
        raise ValueError('Model semantic identity mismatch: ' + repr(changed[:8]))
