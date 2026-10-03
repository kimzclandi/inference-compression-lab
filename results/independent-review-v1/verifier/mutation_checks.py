"""Mutation probes against disposable copies; never rewrites frozen evidence.

Re-signing a manifest simulates a consistent but semantically invalid record,
not authentication against an attacker controlling the whole repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from lab.artifact_integrity import file_hashes
from lab.quantization_diagnostics import sha
from experiments.verify_release import verify as release
from experiments.verify_qwen_quantization import verify as exploration
from experiments.verify_qwen_confirmation import verify as confirmation
from experiments.release_archive import verify as archive

ROOT = Path(__file__).resolve().parents[3]

def read(p): return json.loads(p.read_text())
def write(p, v): p.write_text(json.dumps(v, indent=2) + '\n')
def resign(p): write(p/'checksums.json', file_hashes(p, exclude=('checksums.json', 'summary.json')))

def mutate_release(p, kind):
    f=p/'configs/release/protected-results.json'; value=read(f)
    if kind=='empty': value['sha256']={}
    elif kind=='partial': value['sha256'].pop(next(iter(value['sha256'])))
    write(f,value)

def mutate_evidence(p, kind):
    if kind in ['empty_source','partial_source','source_escape']:
        f=p/'run.json';value=read(f)
        if kind=='empty_source':value['source_sha256']={}
        elif kind=='partial_source':value['source_sha256'].pop(next(iter(value['source_sha256'])))
        else:value['source_sha256']={'../protocol.json':sha(p/'protocol.json')}
        write(f,value);resign(p)
    elif kind=='wrong_tokenizer':
        f=p/'q4-quality.json';value=read(f);value['model_files']['tokenizer.json']['sha256']='0'*64;write(f,value);resign(p)
    elif kind=='wrong_score':
        f=p/'q4-quality.json';value=read(f);value['metrics']['overall']['em']=0.99;write(f,value);resign(p)
    elif kind=='missing_data':(p/'data.jsonl').unlink()
    elif kind=='empty_checksums':write(p/'checksums.json',{})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    probes=[]
    cases=[('release','empty'),('release','partial'),('exploration','empty_source'),('exploration','source_escape'),
           ('confirmation','partial_source'),('confirmation','wrong_tokenizer'),('confirmation','wrong_score'),
           ('confirmation','missing_data'),('confirmation','empty_checksums')]
    for target,kind in cases:
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'copy'
            if target=='release':
                shutil.copytree(ROOT,p,ignore=shutil.ignore_patterns('.git','__pycache__','independent-review-v1'))
                mutate_release(p,kind);fn=release
            else:
                source='qwen-quantization-v1' if target=='exploration' else 'qwen-confirmation-v1'
                shutil.copytree(ROOT/'results'/source,p);mutate_evidence(p,kind)
                fn=exploration if target=='exploration' else confirmation
            try:fn(p);status='ACCEPTED_INVALID';error=None
            except Exception as exc:status='REJECTED';error=repr(exc)
            probes.append(dict(target=target,mutation=kind,status=status,error=error))
    for name,mode in [('link',0o120777),('.',0o100644)]:
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'invalid.zip';payload=b'outside';manifest={'schema':1,'commit':'a'*40,'sha256':{name:hashlib.sha256(payload).hexdigest()}}
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('release-manifest.json',json.dumps(manifest));info=zipfile.ZipInfo(name);info.external_attr=mode<<16;z.writestr(info,payload)
            try:archive(path);status='ACCEPTED_INVALID';error=None
            except Exception as exc:status='REJECTED';error=repr(exc)
            probes.append(dict(target='archive',mutation='symlink' if name=='link' else 'dot_path',status=status,error=error))
    result={'scope':'semantic consistency and coverage after local manifest update; hashes do not authenticate producer or execution', 'probes':probes}
    write(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
