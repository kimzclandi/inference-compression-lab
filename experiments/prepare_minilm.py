"""Download a pinned public pretrained model and the STS benchmark dataset."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DATASET_REVISION = 'ab7a5ac0e35aa22088bdcf23e7fd99b220e53308'
ROOT = Path('models/minilm')


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=180) as response, path.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    revision = MODEL_REVISION
    manifest = {'model': MODEL, 'revision': revision, 'files': {}}
    for filename in ['onnx/model.onnx', 'tokenizer.json', 'config.json']:
        print('Downloading', filename, flush=True)
        url = f'https://huggingface.co/{MODEL}/resolve/{revision}/{filename}'
        manifest['files'][filename] = {'url': url, 'sha256': download(url, ROOT / filename)}
    dataset = 'sentence-transformers/stsb'
    dataset_revision = DATASET_REVISION
    url = f'https://huggingface.co/datasets/{dataset}/resolve/{dataset_revision}/data/validation-00000-of-00001.parquet'
    path = Path('data/stsb-validation.parquet')
    manifest['dataset'] = {'url': url, 'revision': dataset_revision,
                           'split': 'validation', 'sha256': download(url, path)}
    ROOT.joinpath('manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
