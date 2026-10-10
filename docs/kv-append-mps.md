# Fixed-capacity KV append versus concatenation on MPS

## Question and scope

The prior Attention study starts with ready-made K/V. This separate experiment
asks whether avoiding repeated concatenation improves a synthetic decoding loop
once cache writes, Python/dispatch and Attention are all included. It does not
change the old performance records or claim model generation performance.

[AppendOnlyKV](../lab/append_only_kv.py) preallocates `[B,H,capacity,D]` K and V,
copies the initial prefix, and writes subsequent tokens into slices. `append`
returns borrowed prefix views; callers must treat them as read-only. The storage
does not grow or reset, and the implementation is inference-only, single-writer
and same-stream. There is no paging, eviction, concurrency, offload or KV quantization.
Preallocation reserves maximum capacity up front, and prefix views can be
noncontiguous; hidden framework copies or layout effects can offset savings.

Shape, dtype, device and capacity validation happens before writing. Visible
length advances after both copy submissions succeed. A synchronous second-copy
failure can dirty unused tail storage but cannot expose that tail or change the
old prefix; a retry overwrites it. Async device faults are not recoverable by this
contract. CPU fault-injection tests cover this bounded guarantee.

## Preregistered experiment

[v2 protocol](../configs/kv-append-mps-v2.json) and [runner/verifier](../experiments/kv_append_mps.py).
MPS, PyTorch 2.8.0, NumPy 2.2.6, FP16, H=8, D=64; B=1/4, initial prefix=128/1024,
64 one-token append steps. Queries and all K/V are generated beforehand from a
fixed seed; queries are independent random tensors, not autoregressive model output.
All inputs and input views are prepared outside timing. CPU fallback, fast math
and prefer-Metal overrides are disabled.

Two arms: `torch.cat` K/V growth and fixed-capacity append. Two measured scopes:
append-only and append plus explicit FP32-intermediate Attention with FP16 output
(the single query sees the complete available prefix). Both cache arms use the
same Attention implementation. Each sample uses a fresh cache.
Initial allocation/prefix copy is timed separately; the full 64-step chain waits
for MPS completion at each step. Chain timers include Python, validation, copying,
dispatch and synchronization, not just GPU instructions. Final cache cleanup and
input generation/transfers are excluded. Report initialization, chain and their
sample-paired sum; this sum is not a contiguous whole-model benchmark.

One warmup chain per arm/scope, five rounds, three samples per cell. Case and
arm/scope order are seeded and randomized. The primary per-case gate requires
append+attention speedup ≥1.05× and 4/5 faster rounds. An append-only win cannot
pass this gate. All failures remain recorded, without post-result retuning.
The shared laptop's thermal/frequency/concurrent workloads are not controlled.

Before timing, all 64 visible K/V prefixes per arm must match source tensors
exactly; all 64 complete Attention outputs must pass non-BLAS float64 reference
checks with `atol=rtol=0.01`. This gives 512 full-output comparisons across four
cases and two arms. Full-tensor checks are run receipts; 24 deterministic output
rows are archived for independent CPU replay, not full GPU re-execution.

Logical write model: with prefix P, T append steps and U bytes per K/V token,
concatenation writes `U * sum(P+t, t=1..T)` output bytes; preallocation writes
`U*T` new bytes, in addition to the same `U*P` initial prefix copy. Storage is
`U*(P+T)` for the preallocated K/V tensors. This counts tensor payload writes,
not measured DRAM traffic, reserved allocator memory or peak device memory.

## Reproduction

Use a clean committed checkout, actual Apple GPU access and a new output folder.
No existing result directory may be overwritten.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 \
  python -m experiments.kv_append_mps run --output runs/kv-append-mps
python -m experiments.kv_append_mps verify --output runs/kv-append-mps
python -m unittest discover -s tests -p test_append_only_kv.py -v
```

PyTorch supplies [copy_](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.copy_.html),
[cat](https://docs.pytorch.org/docs/2.8/generated/torch.cat.html) and tensor kernels.
Preallocation is a standard engineering technique. Cache integration, protocol,
failure tests and execution are AI-assisted; there is no original kernel,
CUDA/Ascend, production cache or model-quality claim.

## v1 stopped at correctness, not a discarded timing result

The [v1 protocol](../configs/kv-append-mps-v1.json), committed as `8b89bea`, used
framework SDPA. It stopped at the correctness guard before recording any timings.
Its [failed receipt and source](../results/kv-append-mps-v1/run.json) are unchanged.
This is a failure to validate this input path, not a preallocation speed failure.

A first-step diagnostic reconstructed case B=4, prefix=1024: K/V matched their
input arrays exactly, but SDPA failed the tolerance with both the original and
contiguous query layouts. A further correctness-only diagnostic on six prefix
lengths found errors/nonfinite outputs with automatic/forced-math SDPA while
explicit FP32 computation passed. No cause is assigned to PyTorch, Metal or the
cache allocator without a stronger investigation. The earlier MPS Attention
results remain limited to their own archived inputs and checked outputs.

[First-step records](../results/kv-append-mps-diagnostic-v1/first-step-diagnostic.json)
and [backend diagnostic](../results/kv-append-mps-diagnostic-v1/backend-length-diagnostic.json)
are retained. The original backend diagnostic serialized NaN as a nonstandard
JSON constant; its unchanged bytes are stored as `.json.txt`, and strict JSON
uses the string `"NaN"`, not a finite substitute.

v2 switches **both** cache arms to the same explicit FP32 Attention, keeping
input seeds, shapes, rounds, tolerance and speed gates unchanged. This is a
correctness repair before performance measurement, not tuning after a speed
result. It is committed separately before its one performance run. It cannot
be compared directly with the older SDPA timing study or described as SDPA
cache acceleration.


## v2 observed results

The corrected protocol was committed as `ab2ed35` before its single fixed run.
All 512 full-output checks passed, and every visible K/V prefix matched the
source arrays exactly. All four primary speed gates passed in 5/5 rounds.
Median-of-round-median times below are milliseconds per **64-step chain**,
not per-token model latency.

| B | Prefix | Cat append ms | Prealloc append ms | Cat append+Attention ms | Prealloc append+Attention ms | Primary speedup |
|---|---|---|---|---|---|---|
| 1 | 128 | 55.486 | 10.072 | 251.147 | 169.582 | 1.481× |
| 1 | 1024 | 55.073 | 9.709 | 261.355 | 173.968 | 1.502× |
| 4 | 128 | 53.802 | 9.422 | 272.336 | 173.359 | 1.571× |
| 4 | 1024 | 57.539 | 9.636 | 270.519 | 181.598 | 1.490× |

Initialization is not silently charged to just one arm: separate initialization
and sample-paired initialization+chain medians are in the summary.

| B | Prefix | Cat init ms | Prealloc init ms | Cat init+chain ms | Prealloc init+chain ms |
|---|---|---|---|---|---|
| 1 | 128 | 0.205 | 0.236 | 251.301 | 169.864 |
| 1 | 1024 | 0.261 | 0.276 | 261.602 | 174.244 |
| 4 | 128 | 0.257 | 0.247 | 272.624 | 173.630 |
| 4 | 1024 | 0.459 | 0.453 | 271.308 | 182.251 |

[Raw timings](../results/kv-append-mps-v2/timings.json),
[full-output/KV check receipts](../results/kv-append-mps-v2/correctness.json),
[24 output-row witnesses](../results/kv-append-mps-v2/witnesses.json),
[source/environment receipt](../results/kv-append-mps-v2/run.json),
and [summary including logical writes](../results/kv-append-mps-v2/summary.json).

The append-only gains are not evidence of the same whole-loop or model gains.
This compares a bounded preallocation implementation with a naive repeated-cat
control, not with a tuned cache in vLLM/MLX. No existing model runtime is switched
to this cache. CPU CI replays witnesses and lifecycle tests; only the original
M4 Max run measured performance.

## 后续强对照

[真实 Qwen / MLX 原生 Cache 研究](qwen-cache-reservation.md)保留了本文全部结果，并补充成熟框架对照：原生已按 256 token 扩容，按请求预留容量在 3 个长度均未达到加速门槛。本文相对逐步 `cat` 的收益不能外推为整模型提速。
