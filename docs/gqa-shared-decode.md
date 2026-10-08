# Opt-in GQA shared-load Metal decode study

Status: protocol/implementation prepared; **no GPU result yet**. Default model
execution remains native. This experiment targets operator implementation
evidence, not a previously established whole-model Attention bottleneck.

For the existing Qwen2.5-0.5B Q8 model (14 query heads, 2 KV heads, head dimension
64), the candidate groups seven queries sharing one KV head into a 256-thread
threadgroup. Seven SIMD groups maintain independent FP32 stable online-softmax
state while all eight load shared 32-key K/V tiles. Fixed 128-key partitions
produce `(max, denominator, numerator)` state; a second kernel combines them.
These are existing online-softmax and split-K algorithms, not an original
Attention architecture. Code was developed with AI assistance; its measured
capability and the owner's independent understanding are separate matters.

MLX 0.29.3 already uses stable online softmax and split-K. Its native vector
kernel assigns a query head to each group and maps it to `head // gqa_factor`.
The candidate explicitly shares K/V tile loads within each seven-head group;
this does not prove lower physical DRAM traffic, because native caching and
hardware reuse may already reduce it. Native remains the strong control.

- Frozen [protocol](../configs/gqa-shared-decode-v1.json),
  [runner](../experiments/gqa_shared_decode.py),
  [adapter](../lab/gqa_shared_decode.py),
  [partial](../lab/kernels/gqa_shared_partial.metal) and
  [merge](../lab/kernels/gqa_shared_merge.metal) kernels.
- Independent [float64 reference](../lab/gqa_reference.py) uses NumPy einsum
  with `optimize=False`; [CPU contracts](../tests/test_gqa_shared_decode.py)
  exercise algebra, partial tails, head mapping, restoration and rejection.
- Pinned upstream source identities are in the protocol. Source references:
  [native vector kernel](https://github.com/ml-explore/mlx/blob/v0.29.3/mlx/backend/metal/kernels/sdpa_vector.h),
  [native dispatch](https://github.com/ml-explore/mlx/blob/v0.29.3/mlx/backend/metal/scaled_dot_product_attention.cpp),
  [custom Metal API](https://github.com/ml-explore/mlx/blob/v0.29.3/docs/src/dev/custom_metal_kernels.rst),
  [MLX MIT notice](../third_party/MLX-MIT.txt).

Correctness must pass at all four fixed operator lengths and both value
families before any timing. Capacity-backed KV slices have real strides and
unused finite sentinels; no contiguous conversion is forced. Actual Qwen
requests compare all delivered tokens, complete returned **log probabilities**
(not raw logits), and final visible K/V for all 24 layers. Prefill, projections,
RoPE, Cache updates and native greedy generator stay unchanged. Sixteen
returned tokens entail sixteen decode calls per layer, including an unreturned
prefetch. Final Cache offsets are prompt+16; throughput counts only 16 outputs.
Nested adapters and unsupported Cache offsets are rejected before mutation.

Operator timing uses the normal family only. It includes Python metadata checks,
compiled API dispatch, 32 scheduled evaluations and final synchronization,
reported as amortized API wall time, not pure GPU kernel latency. Model timing
includes fresh Cache creation, generator close, adapter installation/restoration
and synchronization of the native generation stream. Load, tokenization,
result checks, audit copies and Cache destruction are excluded. Per-request MLX
allocator peaks include common resident weights/fixtures, exclude RSS/CPU audit
copies, and are compared by maximum timed-request peak per arm.

Run only after committing protocol and implementation, with an existing matching
model directory and empty output path:

```sh
python -m experiments.gqa_shared_decode --model "$MODEL_DIR" --output results/gqa-shared-decode-v1
```

The parent enforces a 720-second worker timeout. Disk 200 MiB and MLX peak 8 GiB
are checkpoint limits, not a memory watchdog. The single primary run retains
all timing trials, full operator inputs/outputs, full returned model logprobs,
K/V comparison hashes and failure details. Final model K/V arrays are not
published; CPU evidence verification cannot independently repeat GPU numeric
checks. No reruns, shape substitutions or threshold changes seek positive
results. Tested distributions/shapes cannot establish correctness for every
finite FP16 input or stride pattern. No CUDA, Ascend, production, concurrent
service, model quality, or general model acceleration claim follows from this
bounded study.
