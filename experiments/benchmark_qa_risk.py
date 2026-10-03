"""Three isolated INT8 processes measuring the complete fixed risk-gated request."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from experiments.benchmark_qa_specialist import peak_bytes
from experiments.qa_risk import rebuild_head
from lab.artifact_integrity import file_hashes,git_identity
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read,write,rows,sha
from lab.qa_risk_calibration import extract_features,predict_probability
from lab.selective_qa import parse_output,apply_threshold


def request(runtime,head,threshold,row):
    prediction=runtime.predict(row['context'],row['question'])
    feature=extract_features(row['context'],prediction)
    score=predict_probability(head,feature['features'])
    decision=apply_threshold(parse_output(row['context'],prediction['prediction']),score,threshold)
    return decision


def worker(args):
    out=reserve_directory(args.output_dir);state=dict(status='running',python=platform.python_version(),
        platform=platform.platform(),records=[],pid=__import__('os').getpid())
    try:
        protocol=read(args.protocol)
        for name,digest in protocol['source_sha256'].items():
            if sha(name)!=digest:raise ValueError('Benchmark source changed')
        if sha(args.training_dir/'selection.json')!=protocol['selection_sha256']:raise ValueError('Selection changed')
        selection=read(args.training_dir/'selection.json')['variants']['int8']
        if selection['threshold']!=.7 or not selection['eligible']:raise ValueError('Frozen policy not eligible')
        data=rows('configs/qa-specialist/dataset/calibration/data.jsonl')
        inputs=sorted(data,key=lambda r:hashlib.sha256(('2026100407'+r['id']).encode()).hexdigest())[:8]
        if [r['id'] for r in inputs]!=protocol['sample_ids']:raise ValueError('Benchmark inputs changed')
        from lab.qa_specialist_runtime import ExtractiveRuntime
        started=time.perf_counter()
        head=rebuild_head(args.training_dir,'int8',selection['model_sha256'])
        runtime=ExtractiveRuntime(args.asset_root,'int8',protocol['asset_manifest_sha256'])
        state['initialization_seconds']=time.perf_counter()-started
        for row in inputs:request(runtime,head,.7,row)
        for rep in range(5):
            for row in inputs:
                started=time.perf_counter();decision=request(runtime,head,.7,row);elapsed=time.perf_counter()-started
                state['records'].append(dict(id=row['id'],repetition=rep,seconds=elapsed,status=decision['status'],
                    decision_sha256=hashlib.sha256(json.dumps(decision,sort_keys=True).encode()).hexdigest()))
        state.update(status='complete',median_seconds=statistics.median(r['seconds'] for r in state['records']),
            peak_rss_bytes=peak_bytes(),protocol_sha256=sha(args.protocol))
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('asset-root','training-dir','protocol','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:return worker(args)
    out=reserve_directory(args.output_dir);shutil.copy2(args.protocol,out/'protocol.json')
    shutil.copy2(__file__,out/'benchmark_source.py');state=dict(status='running',**git_identity(Path.cwd()),commands=[])
    try:
        for i in range(3):
            cmd=[sys.executable,'-B','-m','experiments.benchmark_qa_risk','--worker','--asset-root',str(args.asset_root),
                 '--training-dir',str(args.training_dir),'--protocol',str(args.protocol),'--output-dir',str(out/str(i))]
            state['commands'].append(cmd)
            with (out/(str(i)+'.log')).open('x') as log:
                done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:raise RuntimeError('Worker failed; keep raw process record')
        runs=[read(out/str(i)/'run.json') for i in range(3)]
        write(out/'summary.json',dict(median_of_process_medians_seconds=statistics.median(r['median_seconds'] for r in runs),
            process_medians_seconds=[r['median_seconds'] for r in runs],peak_rss_bytes=[r['peak_rss_bytes'] for r in runs],
            initialization_seconds=[r['initialization_seconds'] for r in runs],processes=3,measurements=120,
            scope=read(args.protocol)['timing_scope'],quality='Single INT8 fixed-policy performance, no FP32 noninferiority or compression-quality claim'))
        state['status']='complete'
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))

if __name__=='__main__':main()
