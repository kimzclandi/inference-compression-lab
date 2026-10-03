"""Tensor identity survives container-header ordering; file hashes remain recorded."""
from pathlib import Path
import hashlib
from lab.quantization_diagnostics import read, sha

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
INFERENCE_FILES = (*TOKENIZER_FILES, 'added_tokens.json', 'special_tokens_map.json',
                   'chat_template.jinja', 'generation_config.json')


def verify_inference_files(actual, expected):
    """Also bind loader sidecars: an external chat template can override tokenizer config."""
    for name in INFERENCE_FILES:
        if actual.get(name) != expected.get(name) or name not in expected:
            raise ValueError('Inference sidecar identity mismatch: ' + name)


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
