"""Audit frozen study results and emit a portable deployment specification."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from lab.evidence import sha256, source_record
from experiments.minilm_runtime_study import read, save, model_path


def ratio_interval(baseline,candidate):
    """Paired resampling of round medians, not thousands of correlated timings."""
    a=np.asarray(baseline);b=np.asarray(candidate)
    if len(a)!=len(b) or len(a)<2:raise ValueError('Paired rounds required')
    rng=np.random.default_rng(20260930)
    idx=rng.integers(0,len(a),size=(5000,len(a)))
    ratios=np.median(a[idx],axis=1)/np.median(b[idx],axis=1)
    return {'speedup_ratio':float(np.median(a)/np.median(b)),
            'paired_round_percentile_95_interval':np.quantile(ratios,[.025,.975]).tolist(),
            'bootstrap_seed':20260930,'bootstrap_replicates':5000,
            'limitation':'within-host round uncertainty only; not device/population-level inference'}


def main(args):
    root=args.output_dir
    if (root/'audit.json').exists() or (root/'deployment.json').exists():raise FileExistsError('Audit already exists')
    manifest=read(root/'manifest.json');selection=read(root/'selection.json')
    spec=manifest['spec']; quality={r['name']:r for r in read(root/'quality/summary.json')}
    records=read(root/'confirm/summary.json');comparisons={}
    for scope in ['session','pipeline']:
        rows={r['name']:r for r in records if r['scope']==scope}
        comparisons[scope]={}
        for label,base,other in [('same_quantized_thread_change','quantized-original','quantized-tuned'),
                                ('same_fp32_thread_change','fp32-original','fp32-tuned'),
                                ('fair_tuned_precision_tradeoff','fp32-tuned','quantized-tuned'),
                                ('original_fp32_to_final','fp32-original','quantized-tuned')]:
            comparisons[scope][label]=ratio_interval(rows[base]['round_medians_ms'],rows[other]['round_medians_ms'])
    fp=quality['fp32-tuned'];qt=quality['quantized-tuned']
    drop=fp['spearman']-qt['spearman'];reduction=1-qt['model_bytes']/fp['model_bytes']
    speed=comparisons['session']['same_quantized_thread_change']['speedup_ratio']
    profile=read(root/'profile/quantized-tuned.json')
    integer_events=sum(r['events'] for r in profile['summary'] if 'Integer' in r['op'] or r['op']=='DynamicQuantizeMatMul')
    expected_nodes=35 if selection['selected_quantized']['variant'].endswith('excluded') else 36
    checks={'test_quality_point_drop_within_budget':drop<=spec['quality_acceptance_max_absolute_spearman_drop'],
            'same_precision_primary_latency_target':1-1/speed>=spec['primary_latency_improvement_target'],
            'file_reduction_target':reduction>=spec['file_reduction_target'],
            'integer_execution_observed':integer_events==10*expected_nodes,
            'CPU_provider_only':all(r['provider']=='CPUExecutionProvider' for r in profile['summary'])}
    audit={'source':source_record(),'selection_sha256':sha256(root/'selection.json'),
           'comparisons':comparisons,'test_spearman_drop':drop,'file_reduction_fraction':reduction,
           'quality_bootstrap_interval':read(root/'quality/paired-bootstrap.json')['percentile_95_interval'],
           'engineering_acceptance':checks,'all_engineering_targets_pass':all(checks.values()),
           'limits':'Project-level targets, not hiring certification. Resume attribution and user understanding require separate review.'}
    save(root/'audit.json',audit)
    deployment={'status':'accepted_for_local_demo' if all(checks.values()) else 'candidate_requires_review',
       'target_hardware':manifest['environment'].get('cpu'),'architecture':manifest['environment']['architecture'],
       'provider':'CPUExecutionProvider','model_path':str(model_path(args.assets_dir,qt['variant'])),
       'model_sha256':qt['model_sha256'],'tokenizer_path':'models/minilm/tokenizer.json',
       'tokenizer_sha256':sha256('models/minilm/tokenizer.json'),'threads':qt['threads'],'eval_max_length':256,
       'fixed_padding':False,'selection_sha256':sha256(root/'selection.json'),
       'quality_result_sha256':sha256(root/'quality/summary.json'),
       'note':'Local CPU demo; not a production service or portable recommendation for other CPUs/shapes'}
    save(root/'deployment.json',deployment)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    main(p.parse_args())
