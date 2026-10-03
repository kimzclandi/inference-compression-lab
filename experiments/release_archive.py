"""Create/verify a deterministic source+evidence zip; model weights are forbidden."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

MANIFEST='release-manifest.json'


def validate_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or '\\' in name:
        raise ValueError('Unsafe archive path: '+name)
    if any(x in p.parts for x in ['.git','.venv','runs','models']) or p.suffix in ['.safetensors','.onnx','.pt','.pth','.engine','.plan','.bin']:
        raise ValueError('Local asset forbidden in release: '+name)


def verify(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        for name in names:validate_name(name)
        manifest=json.loads(z.read(MANIFEST))
        if set(names)-{MANIFEST}!=set(manifest['sha256']):raise ValueError('Incomplete archive manifest')
        if not manifest['sha256']:raise ValueError('Empty archive')
        for name,digest in manifest['sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Archive checksum: '+name)
    return manifest


def build(output):
    if output.exists():raise FileExistsError(output)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Commit the reviewed source and evidence before packaging')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files={}
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',head]))) as archive:
        for item in archive:
            if item.isdir():continue
            if not item.isfile():raise ValueError('Non-regular archive member')
            validate_name(item.name);data=archive.extractfile(item).read()
            # Scoped token signatures, not generic variable names such as token=.
            if re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',data):
                raise ValueError('Potential secret requires review: '+item.name)
            files[item.name]=data
    manifest={'schema':1,'commit':head,'sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
              'scope':'Source and frozen research evidence. No model weights. Does not publish or change repository visibility.'}
    files[MANIFEST]=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;z.writestr(info,data)
    return verify(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','verify'])
    p.add_argument('archive',type=Path);a=p.parse_args()
    result=build(a.archive) if a.mode=='build' else verify(a.archive)
    print(json.dumps({'commit':result['commit'],'files':len(result['sha256']),
                      'archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
