"""Offline integrity/arithmetic audit; does not rerun MLX or establish quality."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path('results/qwen-cache-reservation-v1')
def verify(root=ROOT):
    info=json.loads((root/'run.json').read_text());spec=info['spec']
    assert info['status']=='complete'
    expected={'configs/qwen-cache-reservation-v1.json','lab/mlx_cache_reservation.py','experiments/qwen_cache_reservation.py'}
    assert set(info['source_sha256'])==expected
    for name,h in info['source_sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==h
    assert spec==json.loads((root/'source/configs/qwen-cache-reservation-v1.json').read_text())
    assert set(info['artifacts'])=={'correctness.json','samples.json'}
    for name,h in info['artifacts'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
    checks=json.loads((root/'correctness.json').read_text());samples=json.loads((root/'samples.json').read_text())
    assert len(checks)==len(spec['prefix_lengths'])==3
    assert len(samples)==3*2*spec['rounds']*spec['repeats']==60
    keys={(r['prefix'],r['arm'],r['round'],r['repeat']) for r in samples}
    assert keys=={(p,a,r,t) for p in spec['prefix_lengths'] for a in spec['arms'] for r in range(spec['rounds']) for t in range(spec['repeats'])}
    summary=[]
    for p,c in zip(spec['prefix_lengths'],checks):
        assert c['prefix']==p and c['layers']==24
        assert all(c[k] is True for k in ['finite','tokens_equal','logits_allclose','kv_exact'])
        assert c['logits_max_abs']==0
        expected_tokens=c['native']['tokens']
        assert len(expected_tokens)==32 and c['reserved']['tokens']==expected_tokens
        for arm in spec['arms']:
            row=c[arm];assert row['offsets']==[p+31]*24
            assert len(row['capacities'])==32
            caps=[((p+i+255)//256)*256 if arm=='native' else p+32 for i in range(32)]
            assert row['capacities']==[[n]*24 for n in caps]
            assert row['allocated_kv_bytes']==caps[-1]*12288
            assert row['active_kv_bytes']==(p+31)*12288
        rows={a:[r for r in samples if r['prefix']==p and r['arm']==a] for a in spec['arms']}
        for r in rows['native']+rows['reserved']:
            assert r['tokens']==expected_tokens
            assert all(math.isfinite(r[k]) and r[k]>0 for k in ['total_s','ttft_s'])
            assert r['ttft_s']<=r['total_s']
        for rnd in range(spec['rounds']):
            for rep in range(spec['repeats']):
                assert {r['order'] for r in samples if r['prefix']==p and r['round']==rnd and r['repeat']==rep}=={0,1}
        values={a:{k:median(r[k] for r in rows[a]) for k in ['total_s','ttft_s']} for a in spec['arms']}
        speedup=values['native']['total_s']/values['reserved']['total_s']
        faster=sum(median(r['total_s'] for r in rows['native'] if r['round']==i)>median(r['total_s'] for r in rows['reserved'] if r['round']==i) for i in range(spec['rounds']))
        summary.append(dict(prefix=p,medians=values,speedup=speedup,faster_rounds=faster,performance_accepted=speedup>=spec['min_speedup'] and faster>=spec['min_faster_rounds'],allocated_kv_reduction=1-c['reserved']['allocated_kv_bytes']/c['native']['allocated_kv_bytes']))
    return dict(evidence_valid=True,gpu_rerun=False,quality_evaluated=False,cases=summary)

if __name__=='__main__':print(json.dumps(verify(),indent=2,allow_nan=False))
