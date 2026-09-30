"""Confirm a discovered short-request crossover without retuning model/threads."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from tokenizers import Tokenizer
from lab.evidence import reserve_directory,sha256,source_record
from lab.minilm_runtime import MiniLMRuntime
from experiments import minilm_runtime_study as study
from experiments.minilm_runtime_report import ratio_interval


def short_pair_indices(rows,tokenizer,limit):
    """Count real tokens; serialized tokenizer padding/truncation must not bias selection."""
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return [i for i,r in enumerate(rows) if max(len(tokenizer.encode(r['sentence1']).ids),
            len(tokenizer.encode(r['sentence2']).ids))<=limit]


def main(args):
    root=reserve_directory(args.output_dir)
    prior=study.read(args.prior/'manifest.json');selection=study.read(args.prior/'selection.json')
    for v,h in prior['model_sha256'].items():
        if sha256(study.model_path(args.assets_dir,v))!=h:raise ValueError('Frozen model changed')
    study.SPEC=Path('configs/minilm-short-request-check.json')
    spec=study.read(study.SPEC)
    study.save(root/'manifest.json',{'environment':study.environment(),'spec':spec,'spec_sha256':sha256(study.SPEC),
      'parent_study':str(args.prior),'parent_selection_sha256':sha256(args.prior/'selection.json'),
      'model_sha256':prior['model_sha256'],'note':'After exploratory shape discovery; frozen variants and threads; no retuning'})
    shutil.copyfile(args.prior/'selection.json',root/'selection.json')
    if args.reuse_confirmation_from:
        old=study.read(args.reuse_confirmation_from/'manifest.json')
        if old['spec_sha256']!=sha256(study.SPEC) or old['model_sha256']!=prior['model_sha256']:
            raise ValueError('Cannot reuse measurements from a different protocol/model')
        if (args.reuse_confirmation_from/'selection.json').read_bytes()!=(root/'selection.json').read_bytes():
            raise ValueError('Cannot reuse measurements from different frozen selection')
        shutil.copytree(args.reuse_confirmation_from/'confirm',root/'confirm')
        study.save(root/'latency-reuse.json',{'from':str(args.reuse_confirmation_from),
            'reason':'Only quality-length filtering was corrected; timing code, inputs and models unchanged',
            'sha256':{str(p.relative_to(args.reuse_confirmation_from)):sha256(p)
                      for p in sorted((args.reuse_confirmation_from/'confirm').rglob('*.json'))}})
    else:
        study.execute(args,'confirm',study.confirmation_configs(selection),[spec['primary_shape']],spec['confirm_rounds'],['session','pipeline'])
    # Freeze the label-independent length selection before model evaluation.
    tok=Tokenizer.from_file('models/minilm/tokenizer.json')
    rows=pq.read_table('data/stsb-test.parquet').to_pylist()
    indices=short_pair_indices(rows,tok,16)
    if len(indices)<3:raise ValueError('Insufficient untruncated short pairs; refusing to report NaN quality')
    qdir=reserve_directory(root/'quality')
    study.save(qdir/'protocol.json',{'source':source_record(),'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'selected_rows':indices,'pairs':len(indices),'rule':spec['short_quality_rule'],
        'dataset_sha256':sha256('data/stsb-test.parquet'),'batch':1,'sequence':16,
        'length_filter':'padding and truncation explicitly disabled before counting tokens'})
    gold=np.array([rows[i]['score'] for i in indices]);predictions={};quality=[]
    for conf in study.confirmation_configs(selection)[2:]:
        rt=MiniLMRuntime(study.model_path(args.assets_dir,conf['variant']),threads=conf['threads'],max_length=16,fixed_padding=True)
        values=[]
        for i in indices:
            # Batch=1 exactly as timed. Do not quantize both sentences together.
            a=rt.encode([rows[i]['sentence1']])[0];b=rt.encode([rows[i]['sentence2']])[0]
            values.append(float(a@b))
        predictions[conf['name']]=np.array(values)
        study.save(qdir/(conf['name']+'-predictions.json'),[{'row':i,'gold':float(g),'cosine':v} for i,g,v in zip(indices,gold,values)])
        quality.append({**conf,'pairs':len(indices),'spearman':float(spearmanr(gold,values).statistic)})
        del rt
    rng=np.random.default_rng(20260930);deltas=[]
    for _ in range(2000):
        idx=rng.integers(0,len(indices),len(indices))
        deltas.append(float(spearmanr(gold[idx],predictions['quantized-tuned'][idx]).statistic-spearmanr(gold[idx],predictions['fp32-tuned'][idx]).statistic))
    study.save(qdir/'summary.json',quality)
    study.save(qdir/'bootstrap.json',{'deltas':deltas,'seed':20260930,'unit':'paired short-test rows, repeated-sentence dependence ignored',
                                     'percentile_95_interval':np.quantile(deltas,[.025,.975]).tolist()})
    comparisons={}
    for scope in ['session','pipeline']:
        records={r['name']:r for r in study.read(root/'confirm/summary.json') if r['scope']==scope}
        comparisons[scope]=ratio_interval(records['fp32-tuned']['round_medians_ms'],records['quantized-tuned']['round_medians_ms'])
    study.save(root/'audit.json',{'source':source_record(),'comparisons':comparisons,'quality':quality,
         'quality_spearman_drop':quality[0]['spearman']-quality[1]['spearman'],
         'quality_interval':np.quantile(deltas,[.025,.975]).tolist(),
         'limits':spec['limitations']})
    print('Short request check',comparisons,quality,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--reuse-confirmation-from',type=Path)
    main(p.parse_args())
