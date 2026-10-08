# Opt-in GQA shared-load Metal decode study

Status: **model correctness failed; performance never started**. Default model
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

## Frozen v1 outcome

Protocol/source commit `ce482ab22ba89dad0069d8eb62181478685fb953` preceded the
only GPU execution on M4 Max, 48 GiB, MLX 0.29.3 / mlx-lm 0.26.3.
[Run manifest](../results/gqa-shared-decode-v1/run.json) preserves every source
snapshot and artifact hash. Eight fixtures × three arms yielded 24 full-output
checks; the custom candidate itself has eight checks. All met `atol=0.003,
rtol=0.003` against float64 reference; overall maximum absolute error was
0.000963737. All input/sentinel checks passed.

At prompt 128, native and candidate delivered the same 16 greedy tokens. Full
16×151,936 returned logprobs passed `atol=0.02, rtol=0.01` (max absolute
difference 0.0625; tolerance is element-dependent, not a uniform 0.02 bound).
Candidate routing recorded 24 prefill and 384 decode calls; all 24 final Cache
offsets were 144. However, 10 of 48 final K/V tensors failed the frozen
`atol=0.01, rtol=0.01` gate: K at layers 9, 11, 12, 13, 16; V at layers 11,
12, 13, 21, 23 (zero-based). The first failure was layer 9 K; the largest
absolute difference among failed tensors was 0.033447265625 at layer 11 K.
Across all 48 tensors the maximum was 0.125 at layer 8 K, which passed because
its element-wise relative tolerance allowed that difference. Absolute maxima
alone do not decide the allclose gate. This is a failure
of the predeclared model integration criterion, despite token agreement.

The runner stopped immediately. Prompt 4096 was **not executed at model level**;
operator/model performance contain **zero trials**, so there is no speed or
comparative-memory conclusion. `request-attempts.json` contains correctness
phase diagnostic times affected by full logprob CPU copies; those are not
performance measurements. The observed global MLX allocator peak was
706,097,560 bytes, below the checkpoint budget; it does not compare arms.

The failure mechanism is unresolved. Different reduction order, FP16 rounding
and propagation through subsequent layers are hypotheses, not demonstrated
causes. The archived final K/V records contain per-tensor maxima and hashes,
but not arrays or failing-element counts; this limits independent numerical
localization. No GPU rerun, tolerance relaxation, new input selection or
post-failure timing was performed. The implementation stays opt-in.

CPU audit replays all archived micro inputs/outputs and full returned logprobs,
and audits the 48 K/V records and failed gate. It cannot replay final K/V
without a GPU/model run:

```sh
python -m experiments.verify_gqa_shared_decode
python -m unittest discover -s tests -p test_gqa_shared_decode.py -v
```

This branch is independent of open PRs 20/21/22 and based directly on the
verified default `cd2d1a6`; it does not duplicate their experiments. Default
README/EVIDENCE_MAP/workflow edits can overlap during owner-reviewed merging;
there is no algorithm/code dependency on those PRs. No merge is performed.
