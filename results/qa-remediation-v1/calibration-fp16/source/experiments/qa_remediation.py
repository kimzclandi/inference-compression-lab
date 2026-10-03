"""Bounded local QA runner with frozen identity checks and retained failures.

Locked specs bind label, prompt mode, precision, complete source model file
hashes and data hash. Development specs may be unlocked. Existing output
directories are never reused. A reserved output retains all subsequent errors.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes, git_identity
from lab.quantization_diagnostics import read, write, sha, rows

SOURCE_FILES = [
    'experiments/qa_remediation.py', 'lab/grounded_qa.py', 'lab/qa_metrics.py',
    'experiments/qwen_quantization.py', 'lab/evidence.py',
    'lab/artifact_integrity.py', 'lab/quantization_diagnostics.py',
]
RUN_FIELDS = {'label', 'mode', 'bits', 'model_files_sha256', 'data_sha256'}


def validate_spec(spec):
    if not isinstance(spec, dict) or type(spec.get('locked', False)) is not bool:
        raise ValueError('Spec must be an object with an optional boolean locked field')
    if type(spec.get('seed')) is not int or not 0 <= spec['seed'] < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for field in ('max_new_tokens', 'max_input_tokens'):
        if type(spec.get(field)) is not int or spec[field] <= 0:
            raise ValueError(field + ' must be a positive integer')


def validate_locked_run(spec, *, label, mode, bits, model_files_sha256, data_sha256):
    """Require one exact registered run; no subset or best-effort matching."""
    validate_spec(spec)
    if not spec.get('locked', False):
        return {'locked': False, 'scope': 'development; no registered-run identity assertion'}
    allowed = spec.get('allowed_runs')
    if not isinstance(allowed, list) or not allowed:
        raise ValueError('Locked spec requires a nonempty allowed_runs list')
    for candidate in allowed:
        if not isinstance(candidate, dict) or set(candidate) != RUN_FIELDS:
            raise ValueError('Each allowed run must contain exactly the registered identity fields')
        if not isinstance(candidate['label'], str) or not candidate['label']:
            raise ValueError('Registered label must be a nonempty string')
        if candidate['mode'] not in {'legacy', 'grounded'}:
            raise ValueError('Invalid registered prompt mode')
        if candidate['bits'] is not None and (type(candidate['bits']) is not int or candidate['bits'] not in (4, 8)):
            raise ValueError('Registered bits must be null, 4 or 8')
        manifest = candidate['model_files_sha256']
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError('Registered model manifest cannot be empty')
        if any(not isinstance(name, str) or not name or name in {'.', '..'} or '/' in name or '\\' in name
               for name in manifest):
            raise ValueError('Registered model files must be canonical top-level names')
        if any(not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None
               for digest in [candidate['data_sha256'], *manifest.values()]):
            raise ValueError('Registered identities must be lowercase SHA256 digests')
    actual = dict(label=label, mode=mode, bits=bits,
                  model_files_sha256=model_files_sha256, data_sha256=data_sha256)
    matches = [index for index, candidate in enumerate(allowed) if candidate == actual]
    if len(matches) != 1:
        raise ValueError('Run identity must match exactly one entry in the locked allowed_runs')
    return {'locked': True, 'allowed_run_index': matches[0], 'matched_identity': actual}


def source_model_files(model_path):
    if not model_path.is_dir():
        raise ValueError('Source model must be an existing local directory')
    result = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
              for p in sorted(model_path.iterdir()) if p.is_file()}
    if not result or 'config.json' not in result:
        raise ValueError('Source model requires config.json and a nonempty file manifest')
    config = read(model_path / 'config.json')
    if not isinstance(config, dict) or any(key in config for key in ('quantization', 'quantization_config')):
        raise ValueError('Source model must be original floating weights, never a prequantized export')
    return result


def model_layout(model, mx, tree_flatten, nn):
    """Physical parameter storage, separately from file size and runtime memory."""
    parameters = tree_flatten(model.parameters())
    dtypes = {}
    norms = []
    for name, value in parameters:
        kind = str(value.dtype)
        record = dtypes.setdefault(kind, {'tensors': 0, 'tensor_bytes': 0})
        record['tensors'] += 1
        record['tensor_bytes'] += int(value.nbytes)
        if 'norm' in name.lower():
            norms.append(dict(name=name, dtype=kind, tensor_bytes=int(value.nbytes),
                              is_fp16=bool(value.dtype == mx.float16)))
    counts = Counter()
    quantizers = Counter()
    for _, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if isinstance(module, nn.QuantizedLinear):
            counts['quantized_linear'] += 1
        elif isinstance(module, nn.Linear):
            counts['floating_linear'] += 1
        elif isinstance(module, nn.QuantizedEmbedding):
            counts['quantized_embedding'] += 1
        elif isinstance(module, nn.Embedding):
            counts['floating_embedding'] += 1
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            quantizers[f'bits={module.bits},group_size={module.group_size}'] += 1
    return dict(parameter_tensor_bytes=sum(int(value.nbytes) for _, value in parameters),
                parameter_tensor_count=len(parameters), dtype_distribution=dtypes,
                module_counts={key: counts[key] for key in ('quantized_linear', 'floating_linear',
                                                            'quantized_embedding', 'floating_embedding')},
                quantizer_counts=dict(quantizers), normalization_tensors=norms,
                normalizations_all_fp16=all(row['is_fp16'] for row in norms) if norms else None,
                scope='Sum of physical parameter tensor storage by parameter path; excludes activations, KV, RSS and allocator reservations. Not peak runtime memory or source weight-file bytes.')


def _load_runtime():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import fetch_from_hub, quantize_model
    from mlx.utils import tree_map, tree_flatten
    return mx, fetch_from_hub, quantize_model, tree_map, tree_flatten, nn


def run(a):
    out = reserve_directory(a.output_dir)
    state = dict(status='running', stage='preflight',
                 started=datetime.now(timezone.utc).isoformat(), predictions=[],
                 argv=[sys.executable, *sys.argv],
                 arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()},
                 mode=a.mode, model_label=a.label, bits=a.bits,
                 python=platform.python_version())
    try:
        spec = read(a.spec)
        validate_spec(spec)
        data = rows(a.data)
        if not data or len({r['id'] for r in data}) != len(data):
            raise ValueError('Dataset must be nonempty with unique IDs')
        for row in data:
            if (not isinstance(row['id'], str) or not row['id'] or
                    not isinstance(row['context'], str) or not row['context'].strip() or
                    not isinstance(row['question'], str) or not row['question'].strip()):
                raise ValueError('Each dataset row requires nonempty string id, context and question')
        state.update(**git_identity(Path.cwd()), dataset_sha256=sha(a.data), spec_sha256=sha(a.spec),
                     source_model_files=source_model_files(a.model))
        state['locked_spec_validation'] = validate_locked_run(
            spec, label=a.label, mode=a.mode, bits=a.bits,
            model_files_sha256={name: info['sha256'] for name, info in state['source_model_files'].items()},
            data_sha256=state['dataset_sha256'])
        for name in SOURCE_FILES:
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, dest)
        state['source_sha256'] = {name: sha(out / 'source' / name) for name in SOURCE_FILES}
        shutil.copy2(a.spec, out / 'protocol.json')
        shutil.copy2(a.data, out / 'data.jsonl')
        write(out / 'started.json', {k: v for k, v in state.items() if k != 'predictions'})
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        state['stage'] = 'runtime_import'
        mx, fetch_from_hub, quantize_model, tree_map, tree_flatten, nn = _load_runtime()
        from lab.grounded_qa import predict
        from lab.qa_metrics import evaluate
        state['packages'] = {name: importlib.metadata.version(name)
                             for name in ('mlx', 'mlx-lm', 'numpy', 'transformers')}
        state['stage'] = 'model_load'
        model, config, tok = fetch_from_hub(a.model, lazy=False, trust_remote_code=False)
        if any(key in config for key in ('quantization', 'quantization_config')):
            raise ValueError('Loaded source config unexpectedly contains quantization')
        model.update(tree_map(lambda x: x.astype(mx.float16) if mx.issubdtype(x.dtype, mx.floating) else x,
                              model.parameters()))
        config['torch_dtype'] = 'float16'
        if a.bits is not None:
            model, config = quantize_model(model, config, 64, a.bits)
        mx.eval(model.parameters())
        mx.random.seed(spec['seed'])
        state['device'] = mx.metal.device_info()
        state['loaded_config'] = config
        state['model_layout'] = model_layout(model, mx, tree_flatten, nn)
        state['stage'] = 'inference'
        for row in data:
            state['current_id'] = row['id']
            pred = predict(model, tok, row['context'], row['question'], a.mode,
                           spec['max_new_tokens'], spec['max_input_tokens'])
            pred['id'] = row['id']
            serialized = json.dumps(pred, allow_nan=False)
            state['predictions'].append(pred)
            with (out / 'predictions.jsonl').open('a') as log:
                log.write(serialized + '\n')
        state['stage'] = 'scoring'
        state['metrics'], state['scored'] = evaluate(data, state['predictions'])
        state.update(status='complete', stage='complete')
        state.pop('current_id', None)
        print(json.dumps({'label': a.label, 'mode': a.mode, 'n': len(data), 'raw_metrics': state['metrics']}, indent=2))
    except BaseException as error:
        state.update(status='failed', error=repr(error), error_type=type(error).__name__)
        raise
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat()
        write(out / 'run.json', state)
        write(out / 'checksums.json', file_hashes(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--bits', type=int, choices=[4, 8])
    parser.add_argument('--mode', choices=['legacy', 'grounded'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
