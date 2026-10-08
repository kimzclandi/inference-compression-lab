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

[Protocol](../configs/kv-append-mps-v1.json) and [runner/verifier](../experiments/kv_append_mps.py).
MPS, PyTorch 2.8.0, NumPy 2.2.6, FP16, H=8, D=64; B=1/4, initial prefix=128/1024,
64 one-token append steps. Queries and all K/V are generated beforehand from a
fixed seed; queries are independent random tensors, not autoregressive model output.
All inputs and input views are prepared outside timing. CPU fallback, fast math
and prefer-Metal overrides are disabled.

Two arms: `torch.cat` K/V growth and fixed-capacity append. Two measured scopes:
append-only and append plus framework SDPA (`is_causal=False`, because the single
query sees the complete available prefix). Each sample uses a fresh cache.
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
[cat](https://docs.pytorch.org/docs/2.8/generated/torch.cat.html) and SDPA kernels.
Preallocation is a standard engineering technique. Cache integration, protocol,
failure tests and execution are AI-assisted; there is no original kernel,
CUDA/Ascend, production cache or model-quality claim.
