"""Prepare bucketing assets without running the older thread-tuning study."""
import argparse
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from onnxruntime.quantization import quantize_dynamic, QuantType
from lab.evidence import reserve_directory, sha256
from experiments.prepare_minilm import DATASET_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(Path('results/minilm-bucketing-confirm-v1/manifest.json').read_text())
    expected = {'models/minilm/onnx/model.onnx': reference['models']['fp32']['sha256'],
                'models/minilm/tokenizer.json': reference['tokenizer_sha256'],
                'data/stsb-validation.parquet': reference['data_sha256']['stsb-validation.parquet']}
    for name, digest in expected.items():
        if sha256(name) != digest:
            raise ValueError(f'Input hash differs from frozen protocol: {name}')
    out = reserve_directory(args.assets_dir)
    test = Path('data/stsb-test.parquet')
    test_hash = reference['data_sha256']['stsb-test.parquet']
    if not test.exists():
        url = ('https://huggingface.co/datasets/sentence-transformers/stsb/resolve/'
               f'{DATASET_REVISION}/data/test-00000-of-00001.parquet')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=test.parent, delete=False) as handle:
                temporary = Path(handle.name)
                with urlopen(url, timeout=180) as response:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if sha256(temporary) != test_hash:
                raise ValueError('Downloaded test data hash mismatch')
            # Atomic no-clobber publication: a racing writer is never replaced.
            test.hardlink_to(temporary)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if sha256(test) != test_hash:
        raise ValueError('Existing test data hash mismatch')
    quantized = out / 'int8_per_channel.onnx'
    quantize_dynamic('models/minilm/onnx/model.onnx', str(quantized),
                     op_types_to_quantize=['MatMul'], per_channel=True, reduce_range=False,
                     weight_type=QuantType.QInt8, extra_options={'MatMulConstBOnly': True})
    manifest = {'input_sha256': expected, 'test_sha256': test_hash,
                'quantized_sha256': sha256(quantized),
                'matches_historical_quantized_bytes': sha256(quantized) ==
                    reference['models']['int8_per_channel']['sha256']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
