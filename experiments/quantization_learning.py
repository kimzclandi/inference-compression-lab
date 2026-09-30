"""Numerical teaching lab. Fake quantization; NOT optimized INT8 inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from lab.evidence import reserve_directory, source_record


def weight_qdq(weight, per_channel):
    """Input/output FP32 [K,N]; optional scale [1,N], along output channels."""
    bound = np.max(np.abs(weight), axis=0, keepdims=True) if per_channel else np.max(np.abs(weight))
    scale = np.where(bound == 0, 1.0, bound / 127.0)
    quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    return quantized.astype(np.float32) * scale, np.asarray(scale)


def activation_qdq(x):
    """Input/output FP32 [B,S,K]; one dynamic asymmetric U8 range per tensor."""
    low, high = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    scale = (high-low)/255 if high > low else 1.0
    zero = np.clip(np.rint(-low/scale), 0, 255)
    quantized = np.clip(np.rint(x/scale)+zero, 0, 255).astype(np.uint8)
    return (quantized.astype(np.float32)-zero)*scale, scale


def comparison(x, weight, per_channel, quantize_activation):
    baseline = x @ weight  # [1,32,8] @ [8,4] -> [1,32,4]
    w_hat, scales = weight_qdq(weight, per_channel)
    x_hat, activation_scale = activation_qdq(x) if quantize_activation else (x, None)
    candidate = x_hat @ w_hat  # Float arithmetic on dequantized values: fake quant.
    delta = baseline.astype(np.float64)-candidate.astype(np.float64)
    channel_sse = np.sum(delta**2, axis=(0,1))
    channel_energy = np.sum(baseline.astype(np.float64)**2, axis=(0,1))
    return {'weight_granularity':'per_channel' if per_channel else 'per_tensor',
            'activation_quantized':quantize_activation,
            'weight_scales':scales.tolist(),'activation_scale':activation_scale,
            'weight_mse':float(np.mean((weight.astype(np.float64)-w_hat)**2)),
            'output_nmse':float(np.sum(delta**2)/max(np.sum(baseline.astype(np.float64)**2),1e-30)),
            'output_nmse_each_channel':(channel_sse/np.maximum(channel_energy,1e-30)).tolist()}


def run(output_dir):
    out=reserve_directory(output_dir)
    rng=np.random.default_rng(20260930)
    x=rng.normal(size=(1,32,8)).astype(np.float32)
    w=rng.normal(size=(8,4)).astype(np.float32)
    w[:,0] *= 40  # Outlier output channel; damages shared weight resolution.
    x_outlier=x.copy(); x_outlier[0,0,0]=100
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source':source_record(),
            'seed':20260930,'kind':'synthetic fake-quant teaching lab, not model-performance evidence',
            'shapes':{'x':[1,32,8],'weight':[8,4],'output':[1,32,4]},
            'inputs':{'x':x.tolist(),'weight':w.tolist(),'x_activation_outlier':x_outlier.tolist()},
            'experiments':{name:[comparison(a,w,pc,aq) for pc in [False,True] for aq in [False,True]]
                           for name,a in [('weight_outlier_only',x),('weight_and_activation_outliers',x_outlier)]}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SIMULATION ONLY: FP32 matrix multiplication after quantize/dequantize')
    for name, rows in result['experiments'].items():
        print(name)
        for row in rows:print(row['weight_granularity'],'activation QDQ:',row['activation_quantized'],
                              'output NMSE:',format(row['output_nmse'],'.6g'))
    print('Saved',out/'result.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
