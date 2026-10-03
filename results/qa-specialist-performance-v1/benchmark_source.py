"""Fixed-order isolated CPU QA latency and process peak-RSS experiment.

This is single-caller performance on published calibration inputs. It neither
chooses quality thresholds nor evaluates the held-out local partition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes,git_identity
from lab.quantization_diagnostics import read,write,rows,sha


def selected_inputs(spec):
    p=Path('configs/qa-specialist/dataset/calibration/data.jsonl')
    if sha(p)!=spec['data_sha256']['calibration']:raise ValueError('Calibration input identity changed')
    return sorted(rows(p),key=lambda r:hashlib.sha256(('2026100407'+r['id']).encode()).hexdigest())[:8]


def peak_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform=='darwin':return int(value)
    if sys.platform.startswith('linux'):return int(value*1024)
    raise ValueError('Unknown ru_maxrss units for this platform')


def worker(args):
    out=reserve_directory(args.output_dir);state=dict(status='running',variant=args.variant,
        python=platform.python_version(),platform=platform.platform(),records=[])
    try:
        spec=read(args.spec);inputs=selected_inputs(spec)
        state.update(spec_sha256=sha(args.spec),sample_ids=[r['id'] for r in inputs],
                     process_peak_before_load_bytes=peak_bytes())
        from lab.qa_specialist_runtime import ExtractiveRuntime
        started=time.perf_counter();runtime=ExtractiveRuntime(args.asset_root,args.variant,spec['asset_manifest_sha256'])
        state['initialization_seconds']=time.perf_counter()-started
        for row in inputs:runtime.predict(row['context'],row['question'],include_logits=False)
        for rep in range(spec['performance']['repetitions_per_sample']):
            for row in inputs:
                started=time.perf_counter()
                pred=runtime.predict(row['context'],row['question'],include_logits=False)
                # Synchronous CPU Session.run and all request-side preprocessing/decoding.
                elapsed=time.perf_counter()-started
                state['records'].append(dict(id=row['id'],repetition=rep,seconds=elapsed,
                    tokenization_seconds=pred['tokenization_seconds'],inference_seconds=pred['inference_seconds'],
                    decode_seconds=pred['decode_seconds'],input_tokens=pred['input_tokens'],
                    feature_count=pred['feature_count'],prediction_sha256=hashlib.sha256(pred['prediction'].encode()).hexdigest()))
        state.update(status='complete',process_peak_rss_bytes=peak_bytes(),
            median_seconds=statistics.median(r['seconds'] for r in state['records']),
            memory_scope='Process lifetime peak RSS, includes initialization/tokenizer/ORT/model/inference. Not file size or isolated model memory.')
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def summarize(root,spec):
    runs=[]
    for index,variant in enumerate(spec['performance']['precision_order']):
        record=read(root/f'{index:02d}-{variant}'/'run.json')
        if record['status']!='complete' or record['variant']!=variant or len(record['records'])!=40:
            raise ValueError('Incomplete or changed process coverage')
        runs.append(record)
    medians={v:[r['median_seconds'] for r in runs if r['variant']==v] for v in ('fp32','int8')}
    median={v:statistics.median(values) for v,values in medians.items()}
    speedup=median['fp32']/median['int8']
    return dict(process_median_seconds=medians,median_of_process_medians_seconds=median,
        fp32_over_int8_speedup=speedup,
        speedup_point_gate_passed=speedup>=spec['performance']['minimum_median_speedup_for_claim'],
        process_peak_rss_bytes={v:[r['process_peak_rss_bytes'] for r in runs if r['variant']==v] for v in medians},
        initialization_seconds={v:[r['initialization_seconds'] for r in runs if r['variant']==v] for v in medians},
        samples_per_process=40,processes=len(runs),scope=spec['performance']['timing'],
        limit='Three processes per precision on one machine; no concurrency, tail-latency SLO, energy, or cross-device claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=['fp32','int8']);args=p.parse_args()
    if args.worker:return worker(args)
    out=reserve_directory(args.output_dir);spec=read(args.spec)
    shutil.copy2(args.spec,out/'protocol.json');shutil.copy2(__file__,out/'benchmark_source.py')
    source={'experiments/benchmark_qa_specialist.py':sha(__file__)}
    state=dict(status='running',**git_identity(Path.cwd()),source_sha256=source,commands=[])
    try:
        for index,variant in enumerate(spec['performance']['precision_order']):
            folder=out/f'{index:02d}-{variant}'
            cmd=[sys.executable,'-B','-m','experiments.benchmark_qa_specialist','--worker','--variant',variant,
                 '--asset-root',str(args.asset_root),'--spec',str(args.spec),'--output-dir',str(folder)]
            state['commands'].append(cmd)
            with (out/f'{index:02d}-{variant}.log').open('x') as log:
                done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:raise RuntimeError(f'Worker {index} exited {done.returncode}')
        write(out/'summary.json',summarize(out,spec));state['status']='complete'
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


if __name__=='__main__':main()
