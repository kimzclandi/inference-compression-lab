# Qwen Q8 decode QKV projection packing

This is an opt-in, AI-assisted implementation and one preregistered study. It is
an Attention projection scheduling change using upstream MLX quantized kernels,
not a new Attention architecture or an original low-bit kernel. The previous GQA
and residual/RMSNorm failures remain unchanged.

## Mechanism and controls

The pinned Qwen2.5-0.5B Q8 model applies separate Q/K/V affine 8-bit/group-64
projections to each `[1,1,896]` decode activation. The candidate concatenates
existing packed weight rows, scales, quantization offsets and output bias, runs
one quantized matmul and bias addition, then slices outputs of widths 896/128/128.
It does not dequantize, requantize, retrain or change original parameter arrays.
RoPE, native SDPA, cache update and output projection remain upstream. Prefill is
unchanged. The packing holds additional arrays and must report their memory cost.

Controls include raw native three projections, their compiled graph, and an
adapter-native model route. Native three projections compiled together is the
strong micro control; total model requests compare the packed candidate against
both original native and compiled-three. Context exit restores original Attention
objects, including after errors. Unsupported metadata, masks and cache types fail
explicitly. Only the pinned GPU FP16 decode configuration is supported.

The upstream behavior is read from [MLX v0.29.3 QuantizedLinear](https://github.com/ml-explore/mlx/blob/v0.29.3/python/mlx/nn/layers/quantized.py)
and [MLX-LM v0.26.3 Qwen2](https://github.com/ml-explore/mlx-lm/blob/v0.26.3/mlx_lm/models/qwen2.py).
Algorithmic projection concatenation is established practice; implementation,
contracts and measurements here are project work with AI assistance, not evidence
of the owner's independent mastery.

## Frozen procedure

[Protocol](../configs/qkv-projection-v1.json),
[implementation](../lab/qkv_projection.py),
[runner](../experiments/qkv_projection.py), and
[CPU replay](../experiments/verify_qkv_projection.py).

The protocol and execution code are committed before GPU/model execution.
Two fixed prompt lengths (128/4096), one prefill plus sixteen synchronous greedy
decode forwards, 17 returned tokens, no EOS stop, prefetch or request queue. This
is a fixed direct model loop, not stock `mlx_lm.generate` or a production server.

First, separate synchronized stage measurements record the eight decode stages
for all 24 layers. A >=10% QKV fraction of this instrumented stage sum is only a
preregistered decision to spend the candidate budget. Extra synchronization and
activation copies perturb execution; the fraction is not uninstrumented model
time attribution, a hardware bottleneck proof or an Amdahl upper bound. Native and
profile warmups precede the recorded diagnostic. Audit requests include NumPy
copies and are excluded from performance samples.

Every full-vocabulary last-position logit, token, final visible K/V, and captured
Q/K/V projection output must be bitwise equal to native before timing. Complete
arrays are saved locally before each comparison, including failures. An equality
failure stops performance, does not relax tolerances, and cannot authorize a new
trial. Timing uses fixed orders, warmups and all raw samples; construction and
compilation are excluded. Controls retain the candidate's packed weights too, so
per-request allocator peaks do not measure deployment storage savings. Report the
extra packed bytes and construction time separately.

## Reproduction and evidence boundary

Run only in the pinned local GPU environment, with a new private output directory:

```sh
python -m experiments.qkv_projection --model /path/to/existing/student-q8 --output /path/to/new/private-run
python -m experiments.verify_qkv_projection --root results/qkv-projection-v1 --private-root /path/to/new/private-run --repo-root .
```

Real arrays are local-only. Public JSON can expose raw timings and consistency
receipts after review; public CI cannot independently replay private numerical
arrays. Summary generation and verification share a statistics function; array
reconstruction and hand-calculated tests provide independent checks of the stated
contracts, not an independently implemented performance statistics engine.

No CUDA/Ascend execution, quality improvement, phone deployment or production
claim is made. The implementation and frozen results below are included in this
repository; publishing this research does not mean adopting the candidate.
Existing versioned Releases are separate snapshots and remain unchanged.

## Single execution result

The protocol and all execution sources were frozen in commit
`2e9bdb332242b218cb176c9ef9cb453b4400b37f`. One local M4 Max / 48 GiB execution
completed with MLX 0.29.3, MLX-LM 0.26.3 and NumPy 2.5.3. Both synchronized
profile outputs were bitwise equal to native. QKV occupied 13.50% / 13.04% of the
instrumented stage sums (128/4096 prompts); this passed the budget screen only.

All 48 captured input cases (24 layers × two prompts), compiled-three and packed
projection outputs, and complete model comparisons passed bitwise equality.
Private CPU replay recomputed 104 comparison records across 58 NPZ files and
125,369,480 compared elements, including repeated native references. It verified
all returned full-vocabulary logits, tokens and 48 final K/V arrays per model arm.
The 288 original in-memory QKV parameter arrays retained their hashes. No new QA
quality evaluation was performed.

### Performance gates failed

| Captured-input projection block (24 layers) | Native ms | Compiled-three ms | Packed ms | Compiled-three / packed |
|---|---:|---:|---:|---:|
| From 128-token prompt | 0.836000 | 0.841125 | 0.814958 | 1.032× |
| From 4096-token prompt | 0.827000 | 0.841875 | 0.817208 | 1.030× |

Neither case reached the required 1.05× against both controls. Packed was faster
than compiled-three in only 3/5 and 2/5 rounds, below the required 4/5. Each block
includes API dispatch and completion synchronization; these are not GPU kernel
durations. All 150 raw micro timing samples are retained.

| Whole request (prefill + 16 decode steps) | Native ms | Compiled-three ms | Packed ms | Native / packed | Compiled-three / packed |
|---|---:|---:|---:|---:|---:|
| 128-token prompt | 66.683458 | 67.777771 | 65.657563 | 1.016× | 1.032× |
| 4096-token prompt | 472.702396 | 466.505812 | 469.702605 | 1.006× | 0.993× |

The short request improved 1.54% against original native but missed the 1.02×
threshold. The long request took 0.69% longer than compiled-three. Thus the model
speed gate failed. Nonregression and matched-residency allocator-peak gates
passed. All 80 model samples (including native-adapter controls), per-request
TTFT/decode values and fixed order are retained. Descriptive p95 over 10 requests
per arm is not a production tail-latency estimate. Ratios from different studies
must not be multiplied.

Packing took 62.963 ms in this execution and added 26,376,192 bytes (25.15 MiB)
of packed tensors while retaining the originals. Active MLX allocation immediately
before/after packing was 527,364,360 / 554,160,904 bytes; that observed difference
also includes other preparation allocations. Per-request peak equality among the
timed arms does not erase this extra resident storage. Observed overall MLX peak
was 2,631,478,312 bytes, not process RSS or physical device-wide memory.

**Overall performance acceptance is false.** The candidate stays opt-in; there is
no new accepted Attention or end-to-end acceleration claim. There was no retuning,
second performance execution, threshold change or adoption based on the best case.

[Public run receipt](../results/qkv-projection-v1/run.json),
[raw micro samples](../results/qkv-projection-v1/micro-samples.json),
[raw model samples](../results/qkv-projection-v1/model-samples.json),
[complete statistics](../results/qkv-projection-v1/summary.json),
[correctness receipts](../results/qkv-projection-v1/correctness.json), and
[synchronized profile observations](../results/qkv-projection-v1/profile-samples.json).
These six public JSON files contain no model tensors or weights. Public CI checks
source/record/summary consistency and synthetic fault contracts; real numerical
replay additionally requires the private local arrays.
