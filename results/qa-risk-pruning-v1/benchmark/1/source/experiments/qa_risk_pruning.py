"""Fixed exactness replay and one bounded CPU latency comparison; no new QA confirmation."""
import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import shutil
import statistics
import subprocess
import sys
import time

from experiments.qa_risk import rebuild_head
from lab.artifact_integrity import file_hashes, git_identity
from lab.evidence import reserve_directory
from lab.qa_risk_calibration import extract_features as reference, predict_probability
from lab.qa_risk_pruning import extract_features as pruned
from lab.quantization_diagnostics import read, rows, sha, write
from lab.selective_qa import apply_threshold, evaluate_selective, parse_output

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/qa-risk-pruning/study.json'
EXTRACTORS = {'reference': reference, 'pruned': pruned}
SOURCES = ('experiments/qa_risk_pruning.py', 'lab/qa_risk_pruning.py',
           'lab/qa_risk_calibration.py', 'lab/qa_metrics.py', 'lab/extractive_qa.py',
           'lab/qa_specialist_runtime.py', 'lab/selective_qa.py', 'experiments/qa_risk.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def protocol():
    result = read(PROTOCOL)
    for name, expected in result['inputs_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input/reference changed: ' + name)
    return result


def sources():
    return {name: sha(ROOT / name) for name in SOURCES}


def head(variant):
    folder = ROOT / 'results/qa-risk-v2/training'
    selected = read(folder / 'selection.json')['variants'][variant]
    return rebuild_head(folder, variant, selected['model_sha256'])


def samples(spec):
    data = {r['id']: r for r in rows(ROOT / 'configs/qa-specialist/dataset/calibration/data.jsonl')}
    raw = {r['id']: r for r in rows(ROOT / 'results/qa-specialist-v1/calibration-int8/predictions.jsonl')}
    return [(data[key], raw[key]) for key in spec['performance']['sample_ids']]


def decision(context, pred, feature, model):
    score = predict_probability(model, feature['features'])
    return apply_threshold(parse_output(context, pred['prediction']), score, .7)


def replay(spec, out):
    records = []
    development = {r['id']: r for split in ('train', 'calibration')
                   for r in rows(ROOT / f'configs/qa-risk/dataset/{split}/data.jsonl')}
    evaluation = rows(ROOT / 'configs/qa-risk/dataset/evaluation/data.jsonl')
    heads = {variant: head(variant) for variant in ('fp32', 'int8')}
    predictions = []
    for variant in ('fp32', 'int8'):
        cohorts = [('development', development, [r for split in ('calibration', 'evaluation')
                    for r in rows(ROOT / f'results/qa-specialist-v1/{split}-{variant}/predictions.jsonl')])]
        if variant == 'int8':
            cohorts.append(('historical_evaluation', {r['id']: r for r in evaluation},
                            rows(ROOT / 'results/qa-risk-v2/evaluation-int8/predictions.jsonl')))
        for role, data, raw in cohorts:
            if len(raw) != len(data) or {p['id'] for p in raw} != set(data):
                raise ValueError('Incomplete or duplicate replay cohort')
            for pred in raw:
                context = data[pred['id']]['context']
                a, b = reference(context, pred), pruned(context, pred)
                if canonical(a) != canonical(b):
                    write(out / 'mismatch.json', dict(id=pred['id'], role=role, reference=a, pruned=b))
                    raise ValueError('Exact feature/diagnostic mismatch: ' + pred['id'])
                score_a = predict_probability(heads[variant], a['features'])
                score_b = predict_probability(heads[variant], b['features'])
                da = decision(context, pred, a, heads[variant])
                db = decision(context, pred, b, heads[variant])
                if score_a != score_b or canonical(da) != canonical(db):
                    raise ValueError('Score/decision mismatch: ' + pred['id'])
                records.append(dict(id=pred['id'], variant=variant, role=role,
                    windows=len(pred['raw_windows']), feature_sha256=digest(a),
                    score=score_a, decision_sha256=digest(da)))
                if role == 'historical_evaluation':
                    predictions.append(dict(id=pred['id'], prediction=pred['prediction'], confidence=score_b))
    counts = {role: sum(r['role'] == role for r in records)
              for role in ('development', 'historical_evaluation')}
    if counts != {'development': spec['correctness']['development_rows'],
                  'historical_evaluation': spec['correctness']['historical_evaluation_rows']}:
        raise ValueError('Replay size differs from fixed protocol')
    quality, _ = evaluate_selective(evaluation, predictions, .7)
    mechanism = {}
    for name, extractor in EXTRACTORS.items():
        profiler = cProfile.Profile()
        profiler.enable()
        for row, pred in samples(spec):
            extractor(row['context'], pred)
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        mechanism[name] = dict(normalize_calls=sum(v[1] for k, v in stats.stats.items()
                              if k[2] == 'normalize' and k[0].endswith('qa_metrics.py')))
        stats.strip_dirs().sort_stats('cumulative').print_stats(15)
        (out / (name + '-profile.txt')).write_text(stream.getvalue())
    write(out / 'records.json', records)
    result = dict(status='pass', counts=counts, exact_feature_score_decision_differences=0,
                  mechanism=mechanism, historical_quality_replayed=quality,
                  scope='Consumed public samples; exact implementation replay, not unseen quality confirmation.')
    write(out / 'summary.json', result)
    return result


def worker(spec, out, arm, asset_root):
    from lab.qa_specialist_runtime import ExtractiveRuntime, RUNTIME_PACKAGES
    from experiments.benchmark_qa_specialist import peak_bytes
    extractor = EXTRACTORS[arm]
    inputs = samples(spec)
    start = time.perf_counter()
    model = head('int8')
    manifest = read(ROOT / 'results/qa-risk-performance-v1/protocol.json')['asset_manifest_sha256']
    runtime = ExtractiveRuntime(asset_root, 'int8', manifest, threads=spec['performance']['threads'])
    initialization = time.perf_counter() - start
    expected = {}
    for row, raw in inputs:
        ref = reference(row['context'], raw)
        expected[row['id']] = dict(feature_sha256=digest(ref),
                                  decision_sha256=digest(decision(row['context'], raw, ref, model)))
        live = runtime.predict(row['context'], row['question'])
        actual = extractor(row['context'], live)
        if canonical(actual) != canonical(ref):
            raise ValueError('Live warmup features differ from pinned logits')
    records = []
    for kind in ('full_request', 'feature_only'):
        for repetition in range(spec['performance']['repetitions']):
            for row, raw in inputs:
                context = row['context']
                start = time.perf_counter()
                if kind == 'full_request':
                    pred = runtime.predict(context, row['question'])
                    feature = extractor(context, pred)
                    answer = decision(context, pred, feature, model)
                else:
                    feature = extractor(context, raw)
                elapsed = time.perf_counter() - start
                hashes = dict(feature_sha256=digest(feature))
                if kind == 'full_request':
                    hashes['decision_sha256'] = digest(answer)
                if any(value != expected[row['id']][key] for key, value in hashes.items()):
                    raise ValueError('Timed output differs from frozen reference')
                records.append(dict(kind=kind, id=row['id'], repetition=repetition,
                                    seconds=elapsed, **hashes))
    result = dict(arm=arm, records=records,
        median_seconds={kind: statistics.median(r['seconds'] for r in records if r['kind'] == kind)
                        for kind in ('full_request', 'feature_only')},
        initialization_seconds=initialization,
        initialization_scope='Archived head reconstruction and validated asset/model loading; excludes full quality-evidence verification.',
        peak_rss_bytes=peak_bytes(), packages={p: importlib.metadata.version(p) for p in RUNTIME_PACKAGES})
    write(out / 'measurements.json', result)
    return result


def benchmark(spec, out, asset_root):
    runs = []
    started = time.monotonic()
    original_sources = sources()
    for i, arm in enumerate(spec['performance']['process_arms']):
        remaining = spec['limits']['compute_budget_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Fixed benchmark compute budget exhausted')
        cmd = [sys.executable, '-B', '-m', 'experiments.qa_risk_pruning', 'worker',
               '--arm', arm, '--asset-root', str(asset_root), '--output-dir', str(out / str(i))]
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
        with (out / (str(i) + '.log')).open('x') as log:
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        if done.returncode:
            raise RuntimeError('Benchmark worker failed; retained log: ' + str(i))
        state = read(out / str(i) / 'run.json')
        if state['source_sha256'] != original_sources or state['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('Benchmark implementation/protocol changed between processes')
        runs.append(read(out / str(i) / 'measurements.json'))
        print('completed process', i, arm, flush=True)
    metrics = {}
    for kind in ('feature_only', 'full_request'):
        medians = {arm: [r['median_seconds'][kind] for r in runs if r['arm'] == arm] for arm in EXTRACTORS}
        paired = []
        for i in range(0, len(runs), 2):
            pair = {r['arm']: r['median_seconds'][kind] for r in runs[i:i+2]}
            paired.append(pair['reference'] / pair['pruned'])
        metrics[kind] = dict(process_medians_seconds=medians,
            median_seconds={arm: statistics.median(v) for arm, v in medians.items()},
            speedup=statistics.median(medians['reference']) / statistics.median(medians['pruned']),
            paired_speedups=paired, pairs_faster=sum(x > 1 for x in paired))
    perf = spec['performance']
    passed = (metrics['feature_only']['speedup'] >= perf['feature_speedup_min'] and
              metrics['full_request']['speedup'] >= perf['full_request_speedup_min'] and
              all(m['pairs_faster'] >= perf['required_pairs_faster'] for m in metrics.values()))
    result = dict(status='pass' if passed else 'gate_failed', metrics=metrics,
                  measured_requests=240, feature_measurements=240, warmup_requests=48,
                  scope=perf['timing'], statistical_claim='Descriptive fixed-load measurements; no population latency guarantee.')
    write(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'benchmark', 'worker'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--arm', choices=tuple(EXTRACTORS))
    args = parser.parse_args()
    out = reserve_directory(args.output_dir).resolve()
    state = dict(status='running', mode=args.mode, python=platform.python_version(),
                 platform=platform.platform(), machine=platform.machine(), **git_identity(ROOT),
                 source_sha256=sources(), protocol_sha256=sha(PROTOCOL),
                 model_inference_run=args.mode != 'audit', new_quality_confirmation=False,
                 parameters_distributed=False)
    try:
        spec = protocol()
        shutil.copy2(PROTOCOL, out / 'protocol.json')
        for name in SOURCES:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if args.mode == 'audit':
            result = replay(spec, out)
        else:
            if args.asset_root is None or (args.mode == 'worker' and args.arm is None):
                raise ValueError('Local assets and worker arm are required')
            if args.mode == 'worker':
                result = worker(spec, out, args.arm, args.asset_root.resolve())
            else:
                result = benchmark(spec, out, args.asset_root.resolve())
        protocol()
        if sources() != state['source_sha256']:
            raise ValueError('Source changed during run')
        state['status'] = result.get('status', 'pass')
        return 0 if state['status'] == 'pass' else 1
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        write(out / 'run.json', state)
        write(out / 'checksums.json', file_hashes(out))


if __name__ == '__main__':
    raise SystemExit(main())
