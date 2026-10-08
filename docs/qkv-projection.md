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

No CUDA/Ascend execution, quality improvement, phone deployment, release or
production claim is made. A new PR remains separate from the default branch until
owner-approved merge. Results will be recorded below after the single execution.
