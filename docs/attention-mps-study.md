# Apple GPU Attention: fixed API-level performance study

This separate study uses the available Apple GPU, not a substitute measurement
for the pending CUDA protocol. PyTorch supplies all GPU kernels. Implementation
and execution are AI-assisted; no original kernel or model/serving speedup is claimed.

## Preregistered design

[Protocol](../configs/attention-mps-v1.json), [runner and verifier](../experiments/attention_mps_study.py).
PyTorch 2.8.0, NumPy 2.2.6, FP16 Q/K/V; H=8, D=64, B=1/4;
square prefill lengths 512/1024 and single-token decode KV lengths 1024/4096.
There are eight shape cells, all using deterministic random normal inputs.
Each input's bytes are hashed. No model weights or external data are required.

Three arms: explicit FP16 computation, explicit FP32 intermediate computation
with FP16 output, and framework SDPA automatic dispatch. Explicit causal masks
are prebuilt outside timing. Prefill is causal; decode sees all available keys.
The SDPA implementation is not called FlashAttention: exact kernels require
independent profiler evidence. CPU fallback, fast math and prefer-Metal overrides
are disabled. Unsupported GPU execution must fail rather than run on CPU.

All three full outputs for every shape must match the independent NumPy float64
reference at fixed `atol=rtol=0.01` before timing that shape. This is numerical
tolerance, not bitwise equality or exhaustive correctness. Full-output error
receipts and hashes are retained. Deterministic output rows from each arm are
also archived so Linux CI can reconstruct the inputs and independently replay
those rows without a GPU. Selected-row replay does not revalidate every tensor
element; full-tensor checks are measurements from the original MPS run.

Ten warmups, five rounds, twenty samples per arm per round; seeded randomized
shape order and arm order. Each sample starts after `torch.mps.synchronize()` and
ends after one call and another synchronization. The output remains alive until
the timer ends. Inputs and masks are reused; tensor transfers, oracle computation,
output copies and correctness checks are outside timing. Internal casts, temporary
allocations, dispatch and synchronization remain inside. Report the median of
five round medians. This measures synchronized API latency, not pure GPU duration,
serving p95, TTFT/TPOT or a model's tokens/s. The host is not a dedicated isolated
benchmark machine; competing work, frequency and thermal state are uncontrolled.

Speed acceptance is per shape: SDPA must beat **both** explicit controls by at
least 1.05× in median latency and be faster in at least 4/5 rounds versus each.
Keep all eight shapes and failures. No global promotion, tuning after results or
changes to historical studies. No peak-memory, occupancy or DRAM-traffic metric
is inferred from PyTorch MPS allocator APIs.

## Run and verify

Run from a clean committed checkout with actual Metal access. Output must be new.
Failure receipts and partial records remain in place; do not overwrite a failed run.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 \
  python -m experiments.attention_mps_study run --output runs/attention-mps
python -m experiments.attention_mps_study verify --output runs/attention-mps
```

References: [PyTorch MPS synchronization and memory APIs](https://docs.pytorch.org/docs/2.8/mps.html),
[MPS environment controls](https://docs.pytorch.org/docs/2.8/mps_environment_variables.html),
and the [causal Attention semantic study](attention-backend-study.md).

This protocol is committed before performance measurement. Results will be linked
after a single fixed run; availability of Apple GPU results does not close the
CUDA/Ascend hardware-experience gap.
