"""One frozen real-model request-admission study, using native MLX generation."""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import subprocess
import threading
import time

from lab.request_scheduling import AdmissionQueue, Request, consume_stream, summarize

SPEC = Path('configs/qwen-request-scheduling-v1.json')
SOURCES = [str(SPEC), 'lab/request_scheduling.py', 'experiments/qwen_request_scheduling.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', protocol_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256={}, artifacts={}, spec=spec)
    for name in SOURCES:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        info['source_sha256'][name] = digest(target)
    save(args.output / 'run.json', info)
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        import mlx_lm.models.cache as cache_module
        assert mx.default_device() == mx.gpu, 'GPU required'
        for package, version in spec['versions'].items():
            assert importlib.metadata.version(package) == version, 'dependency version drift'
        package = Path(inspect.getfile(cache_module)).parent.parent
        for name, expected in spec['upstream_sha256'].items():
            assert digest(package / name) == expected, 'upstream source drift: ' + name
        for name, expected in spec['model_files_sha256'].items():
            assert digest(args.model / name) == expected, 'model identity drift: ' + name
        model, tokenizer = load(str(args.model))
        mx.eval(model.parameters()); mx.synchronize()
        info['hardware'] = {'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                            'physical_memory_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                            'macos': subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip()}
        requests = spec['requests']
        token_ids = {}
        for req in requests:
            tokens = tokenizer.encode((spec['prompt_text'] + str(req['id']) + '. ') * 200)
            assert len(tokens) >= req['prompt_tokens']
            token_ids[req['id']] = tokens[:req['prompt_tokens']]
        info['prompt_token_ids'] = token_ids
        save(args.output / 'run.json', info)
        deadline = time.monotonic() + spec['wall_budget_s']

        def request(req, audit=False, cancel_after=None, fail_after=None):
            start = time.perf_counter()
            cache = make_prompt_cache(model)
            tokens, stamps, logprobs = [], [], []
            iterator = generate_step(mx.array(token_ids[req['id']]), model,
                                     max_tokens=req['output_tokens'], prompt_cache=cache)
            outcome = 'running'
            try:
                def on_token(item):
                    token, scores = item
                    tokens.append(int(token)); stamps.append(time.perf_counter())
                    if audit:
                        logprobs.append(np.array(scores).copy())
                outcome = consume_stream(iterator, on_token, cancel_after=cancel_after, fail_after=fail_after)
            except RuntimeError as error:
                if str(error) != 'injected consumer failure':
                    raise
                outcome = 'injected_failure'
            finally:
                # A native generator may prefetch work before yielding. Close does
                # not abort that device work; sync before dropping private caches.
                mx.synchronize()
                kv_bytes = sum(c.keys.nbytes + c.values.nbytes for c in cache)
                offset = [int(c.offset) for c in cache]
                del iterator, cache
            end = time.perf_counter()
            row = dict(id=req['id'], outcome=outcome, tokens=tokens, token_times=stamps,
                       start=start, end=end, allocated_kv_bytes=kv_bytes, layer_offsets=offset)
            if audit:
                return row, np.stack(logprobs)
            return row

        # Same native code with fresh cache is a reference. Full-vocabulary scores
        # are compared in memory and bound by digests, not published as arrays.
        checks = []
        expected = {}
        for req in requests:
            left, a = request(req, True); right, b = request(req, True)
            check = dict(id=req['id'], tokens_equal=left['tokens'] == right['tokens'],
                         finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                         logprobs_allclose=bool(np.allclose(a, b, atol=spec['atol'], rtol=spec['rtol'])),
                         max_abs=float(np.max(np.abs(a.astype('float64')-b.astype('float64')))),
                         shape=list(a.shape), left_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                         right_sha256=hashlib.sha256(b.tobytes()).hexdigest(), tokens=left['tokens'])
            checks.append(check); save(args.output / 'correctness.json', checks)
            assert check['tokens_equal'] and check['finite'] and check['logprobs_allclose'], 'correctness failed'
            expected[req['id']] = left['tokens']
        safety = []
        for kind in ('cancel', 'failure'):
            req = requests[0]
            interrupted = request(req, cancel_after=3 if kind == 'cancel' else None,
                                  fail_after=3 if kind == 'failure' else None)
            resumed = request(req)
            safety.append(dict(kind=kind, outcome=interrupted['outcome'], delivered_tokens=interrupted['tokens'],
                               delivered_count=len(interrupted['tokens']), layer_offsets=interrupted['layer_offsets'],
                               token_times_service_s=[t-interrupted['start'] for t in interrupted['token_times']],
                               close_complete_service_s=interrupted['end']-interrupted['start'],
                               recovery_tokens_equal=resumed['tokens'] == expected[req['id']]))
        save(args.output / 'lifecycle.json', safety)
        assert all(x['delivered_count'] == 3 and x['recovery_tokens_equal'] for x in safety)
        for req in requests:
            for _ in range(spec['warmups_per_request']):
                assert request(req)['tokens'] == expected[req['id']]

        trials = []
        rng = random.Random(spec['seed'])
        for trace in spec['traces']:
            for round_id in range(spec['rounds']):
                policies = list(spec['policies']); rng.shuffle(policies)
                for arm_order, policy in enumerate(policies):
                    assert time.monotonic() < deadline, 'wall budget exceeded'
                    gc.collect(); mx.synchronize(); mx.clear_cache(); mx.reset_peak_memory()
                    queue = AdmissionQueue([], policy, spec['max_bypasses'], spec['decode_weight'])
                    condition = threading.Condition()
                    arrivals, producer_errors = [], []
                    by_id = {req['id']: req for req in requests}
                    origin = time.perf_counter()
                    def produce():
                        try:
                            for position, request_id in enumerate(trace['order']):
                                scheduled = position * trace['interval_s']
                                delay = origin + scheduled - time.perf_counter()
                                if delay > 0:
                                    time.sleep(delay)
                                with condition:
                                    actual = time.perf_counter() - origin
                                    req = by_id[request_id]
                                    queue.submit(Request(request_id, actual, req['prompt_tokens'], req['output_tokens']))
                                    arrivals.append(dict(id=request_id, scheduled_s=scheduled, actual_s=actual))
                                    condition.notify_all()
                        except BaseException as error:
                            with condition:
                                producer_errors.append(type(error).__name__)
                                condition.notify_all()
                    producer = threading.Thread(target=produce)
                    producer.start()
                    rows = []
                    try:
                        for _ in requests:
                            with condition:
                                dispatch_s = time.perf_counter() - origin
                                admitted = queue.pop_ready(dispatch_s)
                                while admitted is None:
                                    assert not producer_errors, 'arrival producer failed'
                                    assert time.monotonic() < deadline, 'wall budget exceeded'
                                    condition.wait(timeout=.5)
                                    dispatch_s = time.perf_counter() - origin
                                    admitted = queue.pop_ready(dispatch_s)
                            row = request(by_id[admitted.request_id])
                            assert row['tokens'] == expected[admitted.request_id], 'timed output drift'
                            row['start'] -= origin; row['end'] -= origin
                            row['token_times'] = [stamp-origin for stamp in row['token_times']]
                            row.update(arrival_s=admitted.arrival_s, dispatch_s=dispatch_s,
                                       queue_s=dispatch_s-admitted.arrival_s,
                                       ttft_s=row['token_times'][0]-admitted.arrival_s,
                                       total_s=row['end']-admitted.arrival_s,
                                       service_s=row['end']-dispatch_s,
                                       tpot_s=(row['token_times'][-1]-row['token_times'][0])/(len(row['tokens'])-1))
                            rows.append(row)
                    finally:
                        producer.join(timeout=1)
                        elapsed = time.perf_counter()-origin
                        save(args.output / 'in-progress.json', dict(trace=trace['name'], round=round_id,
                             policy=policy, arrivals=arrivals, rows=rows))
                    assert not producer.is_alive() and not producer_errors
                    peak = int(mx.get_peak_memory())
                    assert peak <= spec['peak_memory_budget_bytes'], 'memory budget exceeded'
                    trial = dict(trace=trace['name'], round=round_id, arm_order=arm_order, policy=policy,
                                 arrivals=arrivals, rows=rows, elapsed_s=elapsed, peak_mlx_bytes=peak,
                                 bypasses=queue.bypasses, metrics=summarize(rows, elapsed))
                    trials.append(trial); save(args.output / 'trials.json', trials)
                    assert sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) <= spec['disk_budget_bytes'], 'disk budget exceeded'
                print('completed', trace['name'], round_id, flush=True)
        info['status'] = 'complete'
    except BaseException as error:
        info['status'] = 'failed'
        info['error_type'] = type(error).__name__
        # Keep private paths and traceback local, not in the public receipt.
        raise
    finally:
        for name in ('correctness.json', 'lifecycle.json', 'trials.json', 'in-progress.json'):
            if (args.output / name).exists():
                info['artifacts'][name] = digest(args.output / name)
        save(args.output / 'run.json', info)

if __name__ == '__main__':
    main()
