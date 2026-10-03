"""Bounded local QA development/calibration/confirmation runner; no training/API."""
import argparse
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes,git_identity
from lab.quantization_diagnostics import read,write,sha,rows


def run(a):
    spec=read(a.spec)
    data=rows(a.data)
    if len({r['id'] for r in data})!=len(data):raise ValueError('Duplicate dataset IDs')
    out=reserve_directory(a.output_dir)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import fetch_from_hub,quantize_model
    from mlx.utils import tree_map
    from lab.grounded_qa import predict
    from lab.qa_metrics import evaluate
    files=['experiments/qa_remediation.py','lab/grounded_qa.py','lab/qa_metrics.py','experiments/qwen_quantization.py']
    for f in files:
        dest=out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    shutil.copy2(a.spec,out/'protocol.json');shutil.copy2(a.data,out/'data.jsonl')
    state={'status':'running','started':datetime.now(timezone.utc).isoformat(),**git_identity(Path.cwd()),
           'dataset_sha256':sha(a.data),'spec_sha256':sha(a.spec),'source_sha256':{f:sha(f) for f in files},
           'mode':a.mode,'model_label':a.label,'bits':a.bits,
           'source_model_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.model.iterdir()) if p.is_file()},
           'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','numpy','transformers']},
           'predictions':[]}
    write(out/'started.json',{k:v for k,v in state.items() if k!='predictions'})
    try:
        model,config,tok=fetch_from_hub(a.model,lazy=False,trust_remote_code=False)
        model.update(tree_map(lambda x:x.astype(mx.float16) if mx.issubdtype(x.dtype,mx.floating) else x,model.parameters()))
        config['torch_dtype']='float16'
        if a.bits is not None:
            if 'quantization' in config:raise ValueError('Quantize from original floating source, not an existing quantized export')
            model,config=quantize_model(model,config,64,a.bits)
        mx.eval(model.parameters());mx.random.seed(spec['seed'])
        state['device']=mx.metal.device_info()
        state['loaded_config']=config
        for row in data:
            pred=predict(model,tok,row['context'],row['question'],a.mode,spec['max_new_tokens'],spec['max_input_tokens'])
            pred['id']=row['id'];state['predictions'].append(pred)
            with (out/'predictions.jsonl').open('a') as log:
                log.write(json.dumps(pred)+'\n')
        state['metrics'],state['scored']=evaluate(data,state['predictions'])
        state['status']='complete'
        print(json.dumps({'label':a.label,'mode':a.mode,'n':len(data),'raw_metrics':state['metrics']},indent=2))
    except BaseException as e:
        state.update(status='failed',error=repr(e));raise
    finally:
        state['finished']=datetime.now(timezone.utc).isoformat();write(out/'run.json',state)
        write(out/'checksums.json',file_hashes(out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--bits',type=int,choices=[4,8]);p.add_argument('--mode',choices=['legacy','grounded'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args())

if __name__=='__main__':main()
