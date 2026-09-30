"""Real pretrained MiniLM dynamic quantization on ORT CPU; no GPU/Jetson claims."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import psutil
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from tokenizers import Tokenizer
from lab.benchmark import measure

ROOT = Path('models/minilm')
OUT = Path('results/minilm-cpu-dynamic-int8')
VARIANTS = ['fp32', 'int8_per_tensor', 'int8_per_channel']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def model_path(variant):
    return ROOT / ('onnx/model.onnx' if variant == 'fp32' else variant + '.onnx')


def feeds(tokenizer, texts):
    batch = tokenizer.encode_batch(texts)
    return {key: np.array([getattr(x, attr) for x in batch], dtype=np.int64)
            for key, attr in [('input_ids', 'ids'), ('attention_mask', 'attention_mask'),
                              ('token_type_ids', 'type_ids')]}


def session(path, profile=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profile
    if profile:
        options.profile_file_prefix = str(Path('runs') / path.stem)
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=['CPUExecutionProvider'])


def worker(variant):
    rows = pq.read_table('data/stsb-validation.parquet').to_pylist()
    tok = Tokenizer.from_file(str(ROOT / 'tokenizer.json'))
    tok.enable_truncation(max_length=256)
    tok.enable_padding(pad_id=0, pad_token='[PAD]')
    sess = session(model_path(variant))
    texts = [r['sentence1'] for r in rows] + [r['sentence2'] for r in rows]
    embeddings = []
    for start in range(0, len(texts), 16):
        feed = feeds(tok, texts[start:start+16])
        hidden = sess.run(None, feed)[0]
        mask = feed['attention_mask'][..., None]
        pooled = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        embeddings.append(pooled)
    embeddings = np.concatenate(embeddings)
    n = len(rows)
    similarities = (embeddings[:n] * embeddings[n:]).sum(1)
    scores = [float(r['score']) for r in rows]
    np.save(Path('runs') / (variant + '-embeddings.npy'), embeddings)
    predictions = [{'row': i, 'gold': scores[i], 'cosine': float(similarities[i])}
                   for i in range(n)]
    save(OUT / (variant + '-predictions.json'), predictions)
    print(variant, 'quality complete', n, flush=True)
    tok.enable_truncation(max_length=64)
    tok.enable_padding(length=64, pad_id=0, pad_token='[PAD]')
    fixed_feed = feeds(tok, [texts[0]])
    timing = measure(lambda: sess.run(None, fixed_feed), warmup=50, samples=200,
                     repeats=3, scope='ORT session.run only; batch=1 padded sequence=64; excludes tokenization/pooling')
    memory = psutil.Process().memory_info()
    graph = onnx.load(model_path(variant))
    graph_ops = dict(Counter(x.op_type for x in graph.graph.node))
    quantized_nodes = [x.name for x in graph.graph.node if x.op_type == 'MatMulInteger']
    fp_nodes = [x.name for x in graph.graph.node if x.op_type == 'MatMul']
    del sess
    prof = session(model_path(variant), profile=True)
    prof.run(None, fixed_feed)
    profile_path = Path(prof.end_profiling())
    events = json.loads(profile_path.read_text())
    execution = Counter((e.get('args', {}).get('op_name'), e.get('args', {}).get('provider'))
                        for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
    evidence = [{'op': op, 'provider': provider, 'events': count}
                for (op, provider), count in sorted(execution.items())]
    save(OUT / (variant + '-execution.json'), {'graph_ops': graph_ops,
         'integer_matmul_nodes': quantized_nodes, 'remaining_float_matmul_nodes': fp_nodes,
         'profile_executed_ops': evidence,
         'profile_node_events': [e for e in events if e.get('cat') == 'Node']})
    result = {'variant': variant, 'pairs': n, 'spearman': float(spearmanr(scores, similarities).statistic),
              'model_bytes': model_path(variant).stat().st_size,
              'model_sha256': hashlib.sha256(model_path(variant).read_bytes()).hexdigest(),
              'latency': timing, 'rss_after_evaluation_and_timing_bytes': memory.rss,
              'process_lifetime_peak_working_set_bytes': getattr(memory, 'peak_wset', None),
              'memory_scope': 'Isolated worker process; includes Python, dataset, tokenizer, model loading and inference. Not tensor-only memory.',
              'execution_summary': evidence}
    save(OUT / (variant + '.json'), result)
    print(variant, 'finished', result['spearman'], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Path('runs').mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, info in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == info['sha256']
    assert hashlib.sha256(Path('data/stsb-validation.parquet').read_bytes()).hexdigest() == manifest['dataset']['sha256']
    save(OUT / 'provenance.json', manifest)
    environment = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'os': platform.system(), 'os_release': platform.release(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'cpu_model': None, 'ort_providers': ort.get_available_providers(),
        'versions': {p: importlib.metadata.version(p) for p in ['onnxruntime','onnx','tokenizers','numpy','scipy','pyarrow','psutil']},
        'protocol': {'provider':'CPUExecutionProvider','threads':4,'inter_op_threads':1,
          'eval_max_length':256,'eval_batch':16,'benchmark_batch':1,'benchmark_sequence':64,
          'order':VARIANTS,'warmup':50,'samples':200,'rounds':3,
          'quantization':'dynamic U8 activations / signed INT8 weights, constant-weight MatMul only',
          'calibration':'none; activation ranges computed dynamically',
          'weight_reduce_range':False, 'tuning':'none; granularity comparison predefined',
          'data':'full STS-B validation split; not a new held-out test; pretrained data overlap not audited'}}
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            environment['cpu_model'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    save(OUT / 'environment.json', environment)
    for variant in VARIANTS[1:]:
        quantize_dynamic(str(model_path('fp32')), str(model_path(variant)),
            op_types_to_quantize=['MatMul'], per_channel=variant.endswith('channel'),
            reduce_range=False, weight_type=QuantType.QInt8,
            extra_options={'MatMulConstBOnly':True})
        onnx.checker.check_model(str(model_path(variant)))
    for variant in VARIANTS:
        subprocess.run([sys.executable, '-m', 'experiments.minilm_ptq', '--worker', variant], check=True)
    baseline = np.load('runs/fp32-embeddings.npy')
    summaries = []
    for variant in VARIANTS:
        result = json.loads((OUT / (variant + '.json')).read_text())
        candidate = np.load('runs/' + variant + '-embeddings.npy')
        result['embedding_mse_vs_fp32'] = float(np.mean((candidate-baseline)**2))
        result['embedding_cosine_vs_fp32_mean'] = float(np.mean((candidate*baseline).sum(1)))
        result['latency_median_of_round_medians_ms'] = float(np.median([r['median_ms'] for r in result['latency']['repeats']]))
        save(OUT / (variant + '.json'), result)
        summaries.append({k:v for k,v in result.items() if k != 'latency'})
    save(OUT / 'summary.json', summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=VARIANTS)
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
