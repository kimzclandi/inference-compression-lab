"""Frozen descriptive aggregation and acceptance gates (no MLX dependency)."""
from statistics import median


def summarize(spec, operator, model):
    op=[];models=[]
    for n in spec['lengths']:
        rows=[r for r in operator if r['length']==n]
        med={m:median(r['per_call_s'] for r in rows if r['mode']==m) for m in spec['operator_modes']}
        ratios={m:med[m]/med['shared_compiled'] for m in ('native','native_compiled')}
        faster={m:sum(median(r['per_call_s'] for r in rows if r['mode']==m and r['round']==rnd)>
                         median(r['per_call_s'] for r in rows if r['mode']=='shared_compiled' and r['round']==rnd)
                         for rnd in range(spec['rounds'])) for m in ratios}
        op.append(dict(length=n,median_s=med,native_over_shared=ratios,faster_rounds=faster,
            accepted=all(ratios[m]>=spec['acceptance']['operator_min_ratio_against_each_native'] and
                         faster[m]>=spec['acceptance']['min_faster_rounds'] for m in ratios)))
    for n in spec['prompt_lengths']:
        rows=[r for r in model if r['prompt']==n]
        med={m:{key:median(r[key] for r in rows if r['mode']==m) for key in ('total_s','ttft_s','tpot_s')} for m in spec['model_modes']}
        peaks={m:max(r['peak_mlx_bytes'] for r in rows if r['mode']==m) for m in spec['model_modes']}
        ratio=med['native']['total_s']/med['shared_compiled']['total_s']
        faster=sum(median(r['total_s'] for r in rows if r['mode']=='native' and r['round']==rnd)>
                   median(r['total_s'] for r in rows if r['mode']=='shared_compiled' and r['round']==rnd)
                   for rnd in range(spec['rounds']))
        memory_ratio=peaks['shared_compiled']/peaks['native']
        models.append(dict(prompt=n,median_s=med,native_over_shared=ratio,faster_rounds=faster,
            max_peak_mlx_bytes=peaks,candidate_over_native_peak=memory_ratio,
            nonregression_pass=1/ratio<=spec['acceptance']['model_max_candidate_over_native'] and memory_ratio<=spec['acceptance']['model_max_peak_candidate_over_native'],
            speed_claim_pass=ratio>=spec['acceptance']['model_speed_claim_min_ratio'] and faster>=spec['acceptance']['min_faster_rounds']))
    return dict(operator=op,model=models,accepted=all(r['accepted'] for r in op) and all(r['nonregression_pass'] for r in models),
                whole_model_speed_claim=all(r['speed_claim_pass'] for r in models))
