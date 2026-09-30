"""Exploratory paired uncertainty and output-drift cases, without raw sentences."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiments.minilm_ptq import save
from lab.evidence import reserve_directory, source_record, sha256


def main(args):
    reserve_directory(args.output_dir)
    variants=['fp32','int8_per_channel','int8_per_channel_excluded']
    paths={v:args.ablation_results/(v+'-predictions.json') for v in variants}
    raw={v:json.loads(p.read_text()) for v,p in paths.items()}
    gold=np.array([r['gold'] for r in raw['fp32']])
    pred={v:np.array([r['cosine'] for r in rows]) for v,rows in raw.items()}
    rng=np.random.default_rng(20260930)
    deltas=[]
    for _ in range(1000):
        idx=rng.integers(0,len(gold),len(gold))
        deltas.append(float(spearmanr(gold[idx],pred[variants[2]][idx]).statistic-
                            spearmanr(gold[idx],pred[variants[1]][idx]).statistic))
    cases=[]
    for i in np.argsort(-np.abs(pred[variants[1]]-pred['fp32']))[:20]:
        cases.append({'row':int(i),'gold':float(gold[i]),
            **{v:float(p[i]) for v,p in pred.items()},
            'abs_drift_before':float(abs(pred[variants[1]][i]-pred['fp32'][i])),
            'abs_drift_after':float(abs(pred[variants[2]][i]-pred['fp32'][i]))})
    summary={'source':source_record(),'prediction_sha256':{v:sha256(p) for v,p in paths.items()},
      'seed':20260930,'bootstrap_replicates':1000,
      'bootstrap_unit':'paired validation rows with replacement; no training/selection rerun',
      'delta_spearman_excluded_minus_channel':float(spearmanr(gold,pred[variants[2]]).statistic-spearmanr(gold,pred[variants[1]]).statistic),
      'percentile_95_interval':np.quantile(deltas,[.025,.975]).tolist(),
      'limitations':'conditional exploratory interval; ignores repeated-sentence dependence and selection uncertainty; not evidence of generalization',
      'subsets':{name:{v:float(spearmanr(gold[sl],p[sl]).statistic) for v,p in pred.items()}
                 for name,sl in [('probe_rows_0_63',slice(0,64)),('remaining_rows_64_1499',slice(64,None))]}}
    save(args.output_dir/'summary.json',summary)
    save(args.output_dir/'bootstrap-deltas.json',deltas)
    save(args.output_dir/'largest-output-drift.json',cases)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--ablation-results',type=Path,required=True)
    main(p.parse_args())
