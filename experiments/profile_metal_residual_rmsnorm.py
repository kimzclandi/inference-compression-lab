"""Long-running deterministic workload for external Metal profiling.

This command is not a benchmark and records no latency.  It exists so tools
such as Instruments/xctrace can collect enough repeated native or custom
residual-RMSNorm work to diagnose dispatch and GPU behavior.  Historical
performance evidence and acceptance remain unchanged.
"""

import argparse
import json
import os
import time

from lab.metal_residual_rmsnorm import residual_rmsnorm


def validate(rows, width, warmup_batches, batches, chain):
    values = {
        "rows": rows,
        "width": width,
        "warmup_batches": warmup_batches,
        "batches": batches,
        "chain": chain,
    }
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("profile dimensions and counts must be positive integers")
    if width > 4096:
        raise ValueError("width exceeds the kernel contract")
    if rows * width > 64 * 1024 * 1024:
        raise ValueError("profile tensor is unreasonably large")
    if (warmup_batches + batches) * chain > 2_000_000:
        raise ValueError("profile call budget is too large")
    return values


def _progress(stage, mode, **details):
    event = {
        "schema": "metal-residual-rmsnorm-profiler-progress-v1",
        "stage": stage,
        "mode": mode,
        **details,
    }
    print(
        json.dumps(event, sort_keys=True),
        flush=True,
    )


def run(
    mode,
    rows,
    width,
    warmup_batches,
    batches,
    chain,
    emit_progress=False,
    startup_delay=0,
):
    if mode not in ("native", "compiled", "metal"):
        raise ValueError("mode must be native, compiled or metal")
    counts = validate(rows, width, warmup_batches, batches, chain)
    if type(startup_delay) is not int or not 0 <= startup_delay <= 60:
        raise ValueError("startup_delay must be an integer from 0 to 60 seconds")
    if emit_progress:
        _progress("python_started", mode, pid=os.getpid(), startup_delay=startup_delay)
    if startup_delay:
        time.sleep(startup_delay)
    import mlx.core as mx

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("profiling requires the Metal GPU")
    x0 = mx.full((rows, width), 0.125, dtype=mx.float16)
    residual = mx.full((rows, width), 0.01, dtype=mx.float16)
    weight = mx.full((width,), 1.0, dtype=mx.float16)
    mx.eval(x0, residual, weight)
    if emit_progress:
        _progress("inputs_ready", mode)

    def batch(x):
        h = None
        for _ in range(chain):
            h, x = residual_rmsnorm(x, residual, weight, 1e-6, mode=mode)
        mx.eval(h, x)
        return x

    x = x0
    for _ in range(warmup_batches):
        x = batch(x)
    if emit_progress:
        _progress("warmup_complete", mode)
    for _ in range(batches):
        x = batch(x)
    # A small host-visible checksum prevents dead-work ambiguity without
    # treating this profiler harness as a numerical or performance study.
    # Accumulate in FP32: a full 2048x896 normalized FP16 tensor can exceed
    # the FP16 scalar range even when every element is finite.
    checksum = float(mx.sum(x.astype(mx.float32)).item())
    result = {
        "schema": "metal-residual-rmsnorm-profiler-workload-v1",
        "status": "trace_workload_only_not_benchmark",
        "mode": mode,
        **counts,
        "profiled_calls": batches * chain,
        "dtype": "float16",
        "checksum": checksum,
        "startup_delay": startup_delay,
        "changes_fixed_acceptance": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "compiled", "metal"), required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--warmup-batches", type=int, default=5)
    # Keep each evaluated graph short.  Long chains spend the trace window in
    # Python graph construction/compilation instead of producing attributable
    # GPU work for the external profiler.
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--chain", type=int, default=2)
    parser.add_argument("--emit-progress", action="store_true")
    parser.add_argument("--startup-delay", type=int, default=0)
    args = parser.parse_args(argv)
    run(
        args.mode,
        args.rows,
        args.width,
        args.warmup_batches,
        args.batches,
        args.chain,
        emit_progress=args.emit_progress,
        startup_delay=args.startup_delay,
    )


if __name__ == "__main__":
    main()
