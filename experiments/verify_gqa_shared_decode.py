"""CPU audit of the preserved v1 model-KV rejection; never a GPU replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from lab.gqa_reference import dense_reference


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    if not __debug__:raise RuntimeError('optimized Python mode unsupported')
    root=Path(root);run=json.loads((root/'run.json').read_text());spec=run['spec']
    assert run['status']=='failed' and run['phase']=='model_correctness' and run['error_type']=='AssertionError'
    assert run['protocol_commit']=='ce482ab22ba89dad0069d8eb62181478685fb953'
    actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='run.json'}
    assert actual==run['artifacts'], 'artifact modified'
    for name,expected in run['source_sha256'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(root/'source'/name)==expected
        assert digest(Path(name))==expected,'execution source changed; add explicit version distinction before changing source'
    assert spec==json.loads((root/'source/configs/gqa-shared-decode-v1.json').read_text())
    assert not (root/'operator-samples.json').exists() and not (root/'model-samples.json').exists()
    assert 'summary' not in run,'no performance conclusion on correctness failure'
    checks=json.loads((root/'operator-correctness.json').read_text());assert len(checks)==24
    for n in spec['lengths']:
        for family in spec['families']:
            with np.load(root/f'operator-{n}-{family}.npz',allow_pickle=False) as z:
                q,k,v=z['q'],z['k_storage'],z['v_storage']
                assert q.shape==(1,14,1,64) and k.shape==v.shape and k.shape[:2]==(1,2)
                capacity=((n+255)//256)*256+256
                assert k.shape==(1,2,capacity,64)
                assert (k[:,:,n:]==60000).all() and (v[:,:,n:]==-60000).all()
                ref=dense_reference(q,k[:,:,:n],v[:,:,:n])
                np.testing.assert_allclose(z['reference'],ref,atol=1e-12,rtol=1e-12)
                for mode in spec['operator_modes']:
                    value=z[mode];row=[c for c in checks if (c['length'],c['family'],c['mode'])==(n,family,mode)]
                    assert len(row)==1;row=row[0]
                    assert value.shape==ref.shape and row['finite'] and row['allclose'] and np.isfinite(value).all()
                    assert np.allclose(value,ref,atol=spec['operator_atol'],rtol=spec['operator_rtol'])
                    assert abs(row['max_abs']-float(np.max(abs(value.astype('float64')-ref))))<1e-12
            receipt=json.loads((root/f'operator-{n}-{family}-input-check.json').read_text())
            assert receipt==dict(unchanged=True,capacity=capacity,visible=n,sentinel_k=60000,sentinel_v=-60000)
    models=json.loads((root/'model-correctness.json').read_text());assert len(models)==1
    model=models[0];assert model['prompt']==128 and model['tokens_equal'] and model['logprobs_finite'] and model['logprobs_allclose']
    assert model['native']['tokens']==model['candidate']['tokens'] and len(model['native']['tokens'])==16
    for mode in ('native','candidate'):
        assert model[mode]['cache_offsets']==[144]*24 and model[mode]['status']=='complete'
    assert model['candidate']['routing_counts']==dict(prefill=24,decode=384)
    with np.load(root/'model-logprobs-128.npz',allow_pickle=False) as z:
        a,b=z['native'],z['candidate'];assert a.shape==b.shape==(16,151936)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.allclose(a,b,atol=spec['model_logprobs_atol'],rtol=spec['model_logprobs_rtol'])
        assert model['logprobs_max_abs']==float(np.max(abs(a.astype('float64')-b.astype('float64'))))
    kv=model['kv'];assert len(kv)==48
    for i,row in enumerate(kv):
        assert row['layer']==i//2 and row['kind']==('K','V')[i%2]
        assert row['shape']==[1,2,144,64] and row['finite'] and np.isfinite(row['max_abs'])
        for key in ('native_sha256','candidate_sha256'):
            assert len(row[key])==64 and all(c in '0123456789abcdef' for c in row[key])
    failed=[dict(layer=r['layer'],kind=r['kind']) for r in kv if not r['allclose']]
    expected=[(9,'K'),(11,'K'),(11,'V'),(12,'K'),(12,'V'),(13,'K'),(13,'V'),(16,'K'),(21,'V'),(23,'V')]
    assert [(r['layer'],r['kind']) for r in failed]==expected
    attempts=json.loads((root/'request-attempts.json').read_text());assert len(attempts)==2
    for row in attempts:
        assert row['audit'] and row['prompt']==128 and row['status']=='complete'
        t=row['token_times_s'];assert len(t)==16 and all(np.isfinite(t)) and 0<t[0]<=t[-1]<=row['total_s']
        assert all(x<=y for x,y in zip(t,t[1:]))
        assert row['ttft_s']==t[0] and abs(row['tpot_s']-(t[-1]-t[0])/15)<1e-12
        assert row['peak_mlx_bytes']<=spec['budget']['peak_mlx_soft_bytes']
    assert run['peak_mlx_bytes']<=spec['budget']['peak_mlx_soft_bytes']
    return dict(evidence_valid=True,accepted=False,operator_full_checks=24,model_logprobs_checks=1,
                model_kv_record_checks=48,model_kv_failed_records=failed,performance_trials=0,
                unexecuted_prompt=4096,model_correctness_pass=False,
                scope='CPU replays archived operator arrays and complete returned logprobs, verifies source/records. Final KV arrays not archived; its failure records/hashes are audited, not numerical replay. No GPU performance result.')


def main():
    if not __debug__:raise RuntimeError('optimized Python mode unsupported')
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/gqa-shared-decode-v1')
    print(json.dumps(verify(parser.parse_args().root),indent=2))

if __name__=='__main__':main()
