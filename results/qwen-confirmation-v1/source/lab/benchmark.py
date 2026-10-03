"""Synchronous wall-clock samples. GPU workloads require explicit synchronization."""
import math
import statistics
import time


def percentile(values, p):
    if not values or not 0 <= p <= 100:
        raise ValueError('nonempty values and percentile in [0, 100] required')
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def measure(fn, *, warmup=50, samples=200, repeats=3, synchronize=None,
            device='cpu', scope='unspecified'):
    if warmup < 0 or samples < 1 or repeats < 1:
        raise ValueError('invalid measurement counts')
    if device != 'cpu' and synchronize is None:
        raise ValueError('non-CPU workload requires a device synchronization callback')
    sync = synchronize if synchronize is not None else lambda: None
    rounds = []
    for repeat in range(repeats):
        for _ in range(warmup):
            fn()
        sync()
        timings = []
        for _ in range(samples):
            sync()
            start = time.perf_counter_ns()
            fn()
            sync()
            timings.append((time.perf_counter_ns() - start) / 1e6)
        rounds.append({'repeat': repeat, 'samples_ms': timings,
                       'median_ms': statistics.median(timings),
                       'p95_ms': percentile(timings, 95)})
    return {'device': device, 'scope': scope, 'warmup_per_repeat': warmup,
            'samples_per_repeat': samples, 'repeats': rounds,
            'note': 'Serialized latency, not concurrent throughput. Memory not measured.'}
