"""Four-position crossover; fresh CPU worker per variant/round, no quality workload."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from experiments import minilm_ptq as ptq
from lab.benchmark import measure
from lab.evidence import reserve_directory, source_record, sha256

VARIANTS = ['fp32','int8_per_tensor','int8_per_channel','int8_per_channel_excluded']


def model_path(args, variant):
    if variant == 'fp32':
        return Path('models/minilm/onnx/model.onnx')
    root = args.ablation_work if variant.endswith('excluded') else args.baseline_work
    return root/(variant+'.onnx')


def worker(args):
    tokenizer = Tokenizer.from_file('models/minilm/tokenizer.json')
    tokenizer.enable_truncation(max_length=64)
    tokenizer.enable_padding(length=64,pad_id=0,pad_token='[PAD]')
    text = pq.read_table('data/stsb-validation.parquet').to_pylist()[0]['sentence1']
    feed = ptq.feeds(tokenizer,[text])
    sess = ptq.session(model_path(args,args.worker))
    timing = measure(lambda:sess.run(None,feed),warmup=50,samples=200,repeats=1,
                     scope='ORT session.run only; batch=1 padded sequence=64; excludes tokenization/pooling/loading')
    ptq.save(args.output_dir/f'{args.round}-{args.worker}.json',
             {'variant':args.worker,'round':args.round,'actual_providers':sess.get_providers(),'latency':timing})


def main(args):
    reserve_directory(args.output_dir)
    order = [VARIANTS[i:]+VARIANTS[:i] for i in range(4)]
    ptq.save(args.output_dir/'protocol.json',{'source':source_record(),
      'timestamp_utc':datetime.now(timezone.utc).isoformat(),'order':order,
      'model_sha256':{v:sha256(model_path(args,v)) for v in VARIANTS},
      'threads':4,'inter_op_threads':1,'provider':'CPUExecutionProvider',
      'warmup':50,'samples':200,'rounds':4,'batch':1,'sequence':64,
      'process':'fresh worker each measurement; session creation excluded; one model resident',
      'design':'deterministic cyclic Latin square, each variant occupies each position once; not randomized',
      'limitations':'one device, uncontrolled background/thermals/frequency; samples are not independent machines'})
    for i, variants in enumerate(order):
        for variant in variants:
            subprocess.run([sys.executable,'-m','experiments.minilm_latency_crossover',
              '--output-dir',str(args.output_dir),'--baseline-work',str(args.baseline_work),
              '--ablation-work',str(args.ablation_work),'--worker',variant,'--round',str(i)],check=True)
    summary=[]
    for v in VARIANTS:
        records=[json.loads((args.output_dir/f'{i}-{v}.json').read_text()) for i in range(4)]
        medians=[r['latency']['repeats'][0]['median_ms'] for r in records]
        summary.append({'variant':v,'round_medians_ms':medians,'median_of_round_medians_ms':float(np.median(medians))})
    ptq.save(args.output_dir/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-work',type=Path,required=True)
    p.add_argument('--ablation-work',type=Path,required=True)
    p.add_argument('--worker',choices=VARIANTS)
    p.add_argument('--round',type=int)
    args=p.parse_args()
    worker(args) if args.worker else main(args)
