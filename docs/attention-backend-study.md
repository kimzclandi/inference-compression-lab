# Attention backend study: causal correctness before CUDA performance

## Scope and status

This is an additive, AI-assisted synthetic attention experiment, not an original
Attention kernel, model-quality study, serving benchmark, or CUDA/Ascend success.
The CPU semantic study and CUDA performance study have separate output folders.
CUDA execution is pending access to an NVIDIA GPU. CPU correctness cannot fill
that evidence gap. Existing frozen experiments and negative results are unchanged.
The separate [Apple GPU study](attention-mps-study.md) has now executed on M4 Max;
its results must not be relabeled as CUDA or an original Attention kernel.

## Mechanism and inputs

Inputs are contiguous-prefix MHA tensors: Q `[B,H,L,D]`, K `[B,H,S,D]`,
V `[B,H,S,Dv]`; output is `[B,H,L,Dv]`. No GQA, padding, dropout, sliding-window
cache, cache allocator, paging, model weights, or scheduler is included.
Queries occupy positions `S-L ... S-1`. Key `j` is visible to query `i` iff
`j <= S-L+i`. Square prefill uses ordinary causal attention; a single decode
query can see every available key. Multi-token cached chunks need an offset mask.

With Q/K zero and V `[1,2,3,4]`, a cached single query must return `2.5`.
Non-square PyTorch `is_causal=True` uses upper-left alignment and returns `1.0`
in this counterexample. This is an API-contract distinction, not a PyTorch bug.
The independent [NumPy float64 oracle](../lab/attention_reference.py) and
[study runner](../experiments/attention_backend_study.py) make the distinction explicit.

References: [PyTorch 2.8 SDPA contract](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [backend selection](https://docs.pytorch.org/docs/2.8/generated/torch.nn.attention.sdpa_kernel.html).
PyTorch supplies all optimized kernels. The project contributes the oracle,
masked-prefix integration, controls, protocol, timing harness and evidence audit.

## Fixed CPU semantic study

36 cases: `(L,S)=(7,7),(1,17),(3,17)` × `D=8,32` × contiguous/strided input
× normal/zero/large logits. Batch/head counts are both 2. Both explicit PyTorch
FP32 and forced SDPA math are compared against NumPy float64, with fixed
`atol=rtol=2e-4`. Also check the hand-computable counterexample and two cached
suffix lengths against the corresponding final rows of full causal prefill.
This is forward-only FP32 coverage, not exhaustive correctness or backward testing.

## Frozen CUDA protocol, not yet executed

CPU study completed at protocol commit `a33dfa0fe1b20d93b5f7c2213dd4895b5596b9f3`:
all 36 cases × two arms passed; largest absolute errors were `8.778730345637697e-05`
(explicit FP32) and `2.5588181487012918e-05` (math). The counterexample returned
`2.5` versus `1.0`; both cached-suffix checks passed. Environment: macOS arm64,
PyTorch 2.8.0, NumPy 2.2.6, CPU one thread. No timing was measured.
See [raw semantic records](../results/attention-cpu-semantics-v1/semantics.json)
and [source/environment receipt](../results/attention-cpu-semantics-v1/run.json).

[Protocol](../configs/attention-backend-v1.json): PyTorch 2.8.0 (CUDA wheel suffix
allowed), FP16, H=8, D=64, B=1/4; prefill L=S=512/1024, decode L=1 with
S=1024/4096. There are 8 shape cells and three arms:

- Explicit attention using FP32 intermediate arithmetic and FP16 output.
- Forced PyTorch SDPA math, the primary control.
- Forced PyTorch SDPA Flash Attention; unsupported execution fails, never falls back.

TF32 is disabled. Every arm must match the explicit FP32-computation reference
with `atol=rtol=0.01` for every shape. The FP16 GPU reference comparison is not
bitwise equality or an independent high-precision GPU proof; the separate CPU
study checks semantics against float64 on smaller shapes.

10 warmups/arm/shape, five rounds, randomized arm order, 20 calls per timed block.
Report event elapsed time/call and synchronized host wall time/call separately.
Events include any launch starvation within the block; they are not isolated
kernel instruction durations. No claim of p50/p95 request latency follows from
block averages. Inputs are allocated outside timing; eager casts, score matrix,
mask construction and temporary allocations are inside its call. Backend selection
is outside timing. This is an operator/API comparison, not equal-instruction work.

A shape passes the speed rule only if median math/flash event time is ≥1.05 and
Flash is faster in at least 4/5 rounds. No selected-shape/global promotion follows.
Record all shapes, failures and raw timings; no post-result retuning. Incremental
peak allocated CUDA tensor bytes are measured separately after warmup; they are
not total device memory, allocator-reserved memory, DRAM traffic or KV capacity.
Device identity, CUDA/PyTorch versions, source hashes and protocol commit are recorded.
Profiler traces and hardware-counter root-cause analysis remain follow-up work.

## Run

Run from a clean committed repository. Use an environment with NumPy 2.2.6 and
PyTorch 2.8.0. Choose the CUDA wheel appropriate to the target driver separately;
no target environment or cloud resource is provisioned by these commands.

```bash
python -m unittest discover -s tests -p test_attention_reference.py -v
python -m experiments.attention_backend_study cpu-semantics --output runs/attention-cpu
python -m experiments.attention_backend_study verify --output runs/attention-cpu
# Requires an actual supported NVIDIA CUDA GPU. No CPU/MPS fallback.
python -m experiments.attention_backend_study cuda --output runs/attention-cuda
python -m experiments.attention_backend_study verify --output runs/attention-cuda
```

Directories must not exist. Exceptions leave a failed receipt and partial records;
failed output is not replaced. Structural/hash verification checks archived
evidence without requiring torch or a GPU; CPU CI additionally re-executes semantics.
Neither form of CI verifies pending CUDA timing code on hardware.
