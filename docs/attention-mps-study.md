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

The protocol was committed as `25aff9f` before the single fixed GPU run.
Availability of Apple GPU results does not close the CUDA/Ascend hardware-experience gap.


## Observed result on M4 Max

Apple M4 Max, 40 GPU cores, 48 GB unified memory. All 24 full-output
comparisons passed; maximum absolute error across arms/shapes was
`0.0017897032282219172`, maximum normalized tolerance ratio `0.09422278480278447`.
All eight shape cells passed both speed controls. Below are milliseconds per
synchronized API call, using the median of round medians.

| B | L | S | Explicit FP16 ms | Explicit FP32 ms | SDPA ms | FP16/SDPA |
|---|---|---|---|---|---|---|
| 1 | 512 | 512 | 0.3491 | 0.4340 | 0.2455 | 1.422× |
| 1 | 1024 | 1024 | 0.8290 | 1.0401 | 0.4804 | 1.726× |
| 1 | 1 | 1024 | 0.1840 | 0.2493 | 0.1142 | 1.611× |
| 1 | 1 | 4096 | 0.2160 | 0.3053 | 0.1233 | 1.752× |
| 4 | 512 | 512 | 0.8537 | 1.0510 | 0.4953 | 1.724× |
| 4 | 1024 | 1024 | 2.5031 | 3.1482 | 1.4828 | 1.688× |
| 4 | 1 | 1024 | 0.1997 | 0.2856 | 0.1250 | 1.597× |
| 4 | 1 | 4096 | 0.3091 | 0.6126 | 0.1745 | 1.771× |

SDPA was faster in all five rounds versus each control for every shape.
FP16 was the faster explicit control in all cells, so the strongest-control range
is 1.422–1.771×. This is a framework implementation comparison on synthetic
inputs. It does not prove the kernel identity, reduced DRAM traffic, peak-memory
savings, an original optimization, or model/production acceleration.

Subsequent [KV append inputs](kv-append-mps.md#v1-stopped-at-correctness-not-a-discarded-timing-result)
failed an MPS SDPA correctness guard. That failure is retained and its cause is
unresolved. The successful results above therefore remain explicitly limited
to the original eight archived input cells; they do not establish general
correctness of SDPA on dynamic/cache-derived tensors.

Evidence: [raw timings](../results/attention-mps-v1/timings.json),
[full-output error receipts](../results/attention-mps-v1/correctness.json),
[output-row witnesses](../results/attention-mps-v1/witnesses.json),
[environment/source receipt](../results/attention-mps-v1/run.json),
and [summary](../results/attention-mps-v1/summary.json).

### Reference warnings and independent check

NumPy matmul emitted divide-by-zero/overflow/invalid warnings on this host during
the original reference computation and selected-row verification. They were not
suppressed, the original performance run was not repeated, and the underlying
library cause has not been established.

A separate CPU-only audit reconstructed the exact input bytes, compared all eight
full float64 reference tensors against non-BLAS einsum with explicit error
checking, and replayed all 48 saved GPU output rows using that independent
calculation. All values were finite, full-reference maximum absolute difference
was `4.718447854656915e-16`, and all output rows passed the unchanged GPU tolerance. This
checks the reference arithmetic; it does not turn selected-row replay into a
second full GPU execution.

[Non-BLAS audit code](../experiments/verify_attention_mps_reference.py) and
[audit receipt](../results/attention-mps-reference-audit-v1.json) preserve this
limitation. Linux CI performs hash/structure/timing checks plus output-row replay;
it does not rerun Apple GPU performance.
