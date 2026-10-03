"""Real-model offline bucketing comparison with quality and raw timing evidence."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from lab.evidence import reserve_directory, sha256
from lab.minilm_runtime import MiniLMRuntime
from experiments.minilm_runtime_study import environment, save, read


def flattened(rows):
    return [text for row in rows for text in (row['sentence1'], row['sentence2'])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/minilm-bucketing.json'))
    args = parser.parse_args()
    spec = read(args.spec)
    out = reserve_directory(args.output_dir)
    paths = {'fp32': Path('models/minilm/onnx/model.onnx'),
             'int8_per_channel': Path('runs/minilm-runtime-assets-v1/int8_per_channel.onnx')}
    save(out / 'manifest.json', {'environment': environment(), 'spec': spec,
         'spec_sha256': sha256(args.spec),
         'models': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in paths.items()},
         'tokenizer_sha256': sha256('models/minilm/tokenizer.json'),
         'data_sha256': {p: sha256('data/' + p) for p in
                         ['stsb-validation.parquet', 'stsb-test.parquet']}})
    runtimes = {k: MiniLMRuntime(v, threads=spec['threads'], max_length=spec['max_length'])
                for k, v in paths.items()}
    benchmark_split = spec.get('benchmark_split', 'validation')
    if benchmark_split not in ('validation', 'test'):
        raise ValueError('Unsupported benchmark split')
    benchmark_pairs = spec.get('benchmark_pairs', spec.get('benchmark_validation_pairs'))
    benchmark = flattened(pq.read_table(f'data/stsb-{benchmark_split}.parquet').to_pylist()
                          [:benchmark_pairs])
    kwargs = dict(batch_size=spec['batch_size'], window_size=spec['window_size'])
    cells = [(precision, bucket) for precision in paths for bucket in [False, True]]
    for precision, bucket in cells:
        for _ in range(spec['warmup_passes']):
            runtimes[precision].encode_batched(benchmark, bucket=bucket, **kwargs)
    timings = []
    rng = np.random.default_rng(spec['seed'])
    for round_id in range(spec['rounds']):
        for cell in rng.permutation(len(cells)):
            precision, bucket = cells[int(cell)]
            start = time.perf_counter()
            vectors, stats = runtimes[precision].encode_batched(benchmark, bucket=bucket, **kwargs)
            elapsed = time.perf_counter() - start
            timings.append({'round': round_id, 'precision': precision, 'bucket': bucket,
                            'seconds': elapsed, 'stats': stats})
        save(out / 'timings.json', timings)
        print('benchmark round', round_id + 1, flush=True)
    rows = pq.read_table('data/stsb-test.parquet').to_pylist()
    texts = flattened(rows)
    labels = np.array([r['score'] for r in rows])
    summary = []
    baselines = {}
    for precision, bucket in cells:
        vectors, stats = runtimes[precision].encode_batched(texts, bucket=bucket, **kwargs)
        scores = np.sum(vectors[0::2] * vectors[1::2], axis=1)
        rho = float(spearmanr(labels, scores).statistic)
        if not np.isfinite(rho):
            raise ValueError('Nonfinite quality metric')
        if not bucket:
            baselines[precision] = (vectors, rho)
        baseline_vectors, baseline_rho = baselines[precision]
        cosine = np.sum(vectors * baseline_vectors, axis=1)
        elapsed = [r['seconds'] for r in timings
                   if r['precision'] == precision and r['bucket'] == bucket]
        item = {'precision': precision, 'bucket': bucket, 'pairs': len(rows),
                'spearman': rho, 'spearman_drop_vs_same_precision_consecutive': baseline_rho-rho,
                'passes_quality_gate': bool(baseline_rho-rho <= spec['quality_max_spearman_drop']),
                'max_embedding_cosine_distance_vs_consecutive': float(np.max(np.abs(1-cosine))),
                'round_seconds': elapsed, 'median_seconds': float(np.median(elapsed)),
                'sentences_per_second': len(benchmark) / float(np.median(elapsed)),
                'benchmark_stats': next(r['stats'] for r in timings
                                       if r['precision'] == precision and r['bucket'] == bucket),
                'quality_stats': stats}
        save(out / f'predictions-{precision}-{int(bucket)}.json',
             [{'row': i, 'label': float(labels[i]), 'cosine': float(scores[i])} for i in range(len(rows))])
        summary.append(item)
        save(out / 'summary.json', summary)
        print('quality', precision, bucket, rho, flush=True)
    for item in summary:
        if item['bucket']:
            base = next(r for r in summary if r['precision'] == item['precision'] and not r['bucket'])
            ratios = 1 - np.array(item['round_seconds']) / np.array(base['round_seconds'])
            item['paired_round_time_reduction'] = ratios.tolist()
            item['median_time_reduction'] = 1 - item['median_seconds']/base['median_seconds']
            boot = np.random.default_rng(spec['seed']).choice(ratios, (10000, len(ratios))).mean(axis=1)
            item['paired_mean_reduction_bootstrap_95_interval'] = np.quantile(boot, [.025,.975]).tolist()
    save(out / 'summary.json', summary)
    save(out / 'complete.json', {'files_sha256': {p.name: sha256(p) for p in sorted(out.glob('*.json'))}})


if __name__ == '__main__':
    main()
