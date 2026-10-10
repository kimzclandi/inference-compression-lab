"""Run one locked public-benchmark partition with a local extractive QA model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from lab.evidence import reserve_directory
from lab.artifact_integrity import git_identity,file_hashes
from lab.quantization_diagnostics import read,write,rows,sha

SOURCE_FILES=('experiments/qa_specialist.py','lab/qa_specialist_runtime.py','lab/extractive_qa.py',
              'lab/selective_qa.py','lab/qa_metrics.py','lab/qa_gate.py')


def validate_run(spec,data_path,variant,source_root):
    if spec.get('locked') is not True or variant not in ('fp32','int8'):
        raise ValueError('Study and model precision must be locked')
    data_hash=sha(data_path)
    matching=[key for key,digest in spec['data_sha256'].items() if digest==data_hash]
    if len(matching)!=1:
        raise ValueError('Dataset is not exactly one frozen partition')
    if set(spec['source_sha256'])!=set(SOURCE_FILES):
        raise ValueError('Incomplete inference/scoring source identity')
    for name,digest in spec['source_sha256'].items():
        if sha(source_root/name)!=digest:raise ValueError('Frozen source changed: '+name)
    return matching[0]


def run(args):
    out=reserve_directory(args.output_dir)
    state=dict(status='running',stage='preflight',variant=args.variant,predictions=[],
               python=platform.python_version())
    try:
        state.update(**git_identity(Path.cwd()))
        spec=read(args.spec);split=validate_run(spec,args.data,args.variant,Path.cwd())
        data=rows(args.data)
        if not data or len({r['id'] for r in data})!=len(data) or any(r['split']!=split for r in data):
            raise ValueError('Nonempty exact partition ID coverage required')
        if split=='evaluation':
            if args.selection is None:raise ValueError('Evaluation requires frozen passing calibration selection')
            selection=read(args.selection)
            from experiments.verify_qa_specialist import calibration_selection
            if selection != calibration_selection(args.selection.parent,spec,sha(args.spec)):
                raise ValueError('Selection does not reproduce from complete calibration evidence')
            if selection['protocol_sha256']!=sha(args.spec) or selection['calibration_feasible'] is not True:
                raise ValueError('Calibration did not authorize evaluation')
            state['selection_sha256']=sha(args.selection)
            shutil.copy2(args.selection,out/'selection.json')
        state.update(split=split,dataset_sha256=sha(args.data),protocol_sha256=sha(args.spec),
                     source_sha256=spec['source_sha256'])
        for name in SOURCE_FILES:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
        shutil.copy2(args.spec,out/'protocol.json');shutil.copy2(args.data,out/'data.jsonl')
        state['stage']='load'
        from lab.qa_specialist_runtime import ExtractiveRuntime,RUNTIME_PACKAGES,LIMITS
        if spec['input_limits']!=LIMITS:raise ValueError('Frozen runtime input contract changed')
        state['packages']={name:importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
        before=time.perf_counter()
        runtime=ExtractiveRuntime(args.asset_root,args.variant,spec['asset_manifest_sha256'])
        shutil.copy2(args.asset_root/'manifest.json',out/'assets.json')
        state.update(model_load_seconds=time.perf_counter()-before,assets=runtime.manifest,
                     provider='CPUExecutionProvider',intra_op_threads=4,inter_op_threads=1)
        write(out/'started.json',{k:v for k,v in state.items() if k!='predictions'})
        state['stage']='inference'
        for row in data:
            state['current_id']=row['id']
            prediction=runtime.predict(row['context'],row['question'])
            prediction['id']=row['id']
            text=json.dumps(prediction,ensure_ascii=False,allow_nan=False)
            with (out/'predictions.jsonl').open('a') as f:f.write(text+'\n')
            state['predictions'].append(prediction)
        state.update(status='complete',stage='complete');state.pop('current_id',None)
        print(json.dumps(dict(status='complete',split=split,variant=args.variant,n=len(data))))
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=repr(exc));raise
    finally:
        # The append-only raw file is the single prediction store; do not duplicate logits.
        state['completed_predictions']=len(state.pop('predictions'))
        write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--variant',choices=['fp32','int8'],required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
