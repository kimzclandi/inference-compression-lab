# GQA model-KV numerical diagnosis (v1)

Status: protocol prepared; no diagnostic GPU execution yet. The original
[GQA study](gqa-shared-decode.md) remains a model-correctness failure with zero
performance trials. This diagnostic cannot replace its frozen conclusion.

The [protocol](../configs/gqa-shared-diagnostic-v1.json) fixes the old 128-token
prompt and its native 16-token sequence, four fresh-Cache trajectories, and
384 same-input shadow probes. The [runner](../experiments/gqa_shared_diagnostic.py)
has no benchmark/timing path. All original kernels, adapter, tolerances and
v1 artifacts remain unchanged.

1. `native_a` and `native_b`: repeated native Attention under identical capture
   and synchronization, to observe repeatability of this instrumented execution.
2. `adapter_native`: unchanged opt-in adapter, with only its decode Attention
   dispatch scoped to native SDPA. This exposes adapter/graph-boundary effects.
3. `candidate`: identical adapter with the frozen custom SDPA. Comparing it to
   `adapter_native` changes only that dispatch; no new algorithm is introduced.

Each arm calls the model on the prompt then teacher-forces the 16 fixed tokens.
There are 17 raw-logit/logprob rows: the first 16 correspond to the original
returned predictions and the last to the original unreturned prefetch. All
24 layer caches finish at offset 144, capacity 256. Raw logits use only a
zero-tolerance descriptive comparison; v1 had no raw-logit tolerance gate.

Scoped diagnostic hooks capture the original native Q/K/V views, actual
metadata, and host snapshots. After all trajectories, the candidate runs on
those same live, strided views, without feedback into model state. A
[metadata-only Metal helper](../lab/kernels/gqa_diagnostic_layout.metal) reads
actual strides with `ensure_row_contiguous=False`; values and layouts must
remain unchanged before/after each probe. The archived full K/V capacity
storage permits CPU reconstruction without pretending that a contiguous NPZ
array is the original GPU layout. Independent float64 reference avoids BLAS.

All trajectories retain per-step Q/new-K/new-V/Attention output; all final K/V
arrays and all 17 complete logits/logprobs are archived. Prefill hashes and
final Cache prefixes are checked explicitly. Error summaries include failing
element counts, first unequal/failing coordinates, signed values at maximum,
percentiles and descriptive FP16 ULP distance (signed zero folded).

The old `np.allclose(native, candidate)` direction is retained: the relative
term references candidate values. Primary counts use original-dtype NumPy
`isclose`; float64 mathematical tolerance counts are separate diagnostics.
Same-input operator reference comparisons use float64 as the right operand.
A first threshold failure is not necessarily the first numerical divergence.

Replay fidelity is measured against v1: first-16 logprob arrays, each final
K/V hash, and failed-tensor sets. Retained views, synchronization, and direct
teacher forcing may change graph/evaluation scheduling. If parity changes,
conclusions apply to the new diagnostic trajectory and cannot establish the
original failure's root cause. Local same-input differences, adapter effects,
and accumulated trajectory differences are reported separately. Reduction,
exponential, FMA and rounding mechanisms remain hypotheses unless supported
by additional controlled evidence; none is presumed here.

Run once only after protocol/code commit and review:

```sh
python -m experiments.gqa_shared_diagnostic --model "$MODEL_DIR" --output results/gqa-shared-diagnostic-v1
```

Hard subprocess timeout is 720 seconds. MLX peak 8 GiB and disk 200 MiB are
checkpoint limits. Original numerical gates are observations, so expected
mismatches do not truncate the fixed diagnosis. Nonfinite values, changed
identities, routing/layout/capture violations or exhausted budgets stop with
partial records preserved. No performance, new optimization, input selection,
tolerance relaxation, or rerun for success is authorized by this protocol.
