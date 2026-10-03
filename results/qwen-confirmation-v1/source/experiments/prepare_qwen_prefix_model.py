"""Convert a local Qwen source to MLX FP16-based Q8, without implicit downloads."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
from lab.evidence import reserve_directory, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.source.is_dir(): raise ValueError('Local source model is required')
    config = json.loads((args.source/'config.json').read_text())
    if config.get('model_type') != 'qwen2' or config.get('hidden_size') != 896 or config.get('quantization'):
        raise ValueError('Expected an unquantized Qwen2.5-0.5B source')
    out = reserve_directory(args.output_dir)
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.utils import fetch_from_hub, quantize_model, save
    source_hashes = {f.name: sha256(f) for f in sorted(args.source.iterdir()) if f.is_file()}
    model, config, tokenizer = fetch_from_hub(args.source, lazy=True, trust_remote_code=False)
    model.update(tree_map(lambda a: a.astype(mx.float16) if mx.issubdtype(a.dtype, mx.floating) else a,
                          model.parameters()))
    config['torch_dtype'] = 'float16'
    model, config = quantize_model(model, config, 64, 8)
    save(out, args.source, model, tokenizer, config, hf_repo=None)
    for name in ['LICENSE', 'README.md']:
        if (args.source/name).exists(): shutil.copy2(args.source/name, out/name)
    manifest = {'source_files_sha256': source_hashes,
                'conversion': 'MLX FP16 cast followed by affine Q8 weight quantization, group size64',
                'kv_quantization': False,
                'versions': {k: importlib.metadata.version(k) for k in ['mlx','mlx-lm','transformers']},
                'output_files_sha256': {f.name: sha256(f) for f in sorted(out.iterdir()) if f.is_file()}}
    (out/'conversion-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['output_files_sha256'], indent=2))


if __name__ == '__main__': main()
