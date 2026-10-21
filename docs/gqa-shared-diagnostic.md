# GQA model-KV numerical diagnosis (v1)

Status: **diagnosis completed; original model-KV failure exactly reproduced**. The original
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

## Frozen diagnostic outcome

Protocol and execution code were committed as
`87287fe1a7c882c6b7c4899fc44e04356196c1e6` before the only diagnostic GPU run.
The [manifest](../results/gqa-shared-diagnostic-v1/run.json),
[comparisons](../results/gqa-shared-diagnostic-v1/summary.json), and
[same-input records](../results/gqa-shared-diagnostic-v1/shadow.json) retain
all four trajectories and 384 probes. The machine was the existing M4 Max,
48 GiB, with MLX 0.29.3 / mlx-lm 0.26.3 / NumPy 2.5.3 and the same Q8 model.
No candidate kernel, adapter, old tolerance or old artifact changed.

| Check | Observation | Interpretation |
|---|---|---|
| Native repeat | All captured Q/new-K/new-V/Attention outputs, 17 logprob rows and 48 final K/V arrays were bitwise equal | Repeatability observed within this instrumented execution |
| Adapter using native SDPA | Bitwise equal to native on those same recorded states | The adapter alone did not produce an observed difference on this workload |
| Fidelity to frozen v1 | Every arm's first 16 logprob rows and all 48 final K/V hashes exactly match its corresponding old native/candidate record | The archived output and final-cache failure were reproduced despite instrumentation |
| Same native Q/K/V inputs | Both native and candidate passed the old operator tolerance against float64 on all 384 probes; 146 native/candidate output pairs differ in 197 elements, each by 1 FP16 ULP | Small local numerical differences remain; this is not universal operator correctness or a quality result |
| Final model state | The same 10/48 K/V tensors fail, with 30 failing elements across all 884,736 visible K/V elements | The frozen model condition remains failed, irrespective of the fraction of failing values |

In chronological decode-step/layer order, the first recorded trajectory
difference is **step 1, layer 5 Attention output** (layers zero-based).
Its input Q/K/V is identical; two of 896 output values differ by 1 FP16 ULP.
The maximum absolute difference there is 0.000244140625 at `[0,12,0,11]`:
native -0.26318359375 versus candidate -0.262939453125. The first unequal
coordinate is `[0,8,0,51]`. Later layers and decode steps show additional
state differences. The first *new K/V element* to miss the original K/V
tolerance in chronological order is step 4, layer 23 V, coordinate
`[0,1,0,54]`. This differs from the first failed final tensor when scanning
layers, which is layer 9 K. Neither should be confused with the first
numerical divergence.

These controls localize the first observed change to the replaced Attention
output and show later state divergence under fixed external tokens. They do
not identify reduction association, exponential approximation, FMA or output
rounding as the unique cause. Passing a local tolerance does not guarantee
the final model tolerance. No evidence-backed semantic repair was identified,
so this round does not alter the kernel or relax acceptance. It adds full-array
localization and replay rather than claiming a successful model fix.

The independent CPU verifier reconstructed all 384 float64 references without
einsum/BLAS (maximum difference from the archived reference
1.7763568394002505e-15), all four complete prediction arrays and 192 final K/V
arrays, their metrics, prefixes, source hashes and all 409 artifacts. CPU
replay checks layout/immutability receipts for consistency; it does not rerun
Metal. Final archive size is 88,940,118 bytes; observed global MLX allocator
peak is 625,602,318 bytes. These are budget observations, not comparative
memory or speed measurements. Performance trials remain **zero**.

```sh
python -m experiments.verify_gqa_shared_diagnostic
python -m unittest discover -s tests -p test_gqa_diagnostic.py -v
```

The implementation and diagnosis remain in open PR #23, outside the default
branch. AI-assisted implementation/execution is separate from the owner's
independent understanding. Further work should first isolate a specific
arithmetic mechanism on the archived identical inputs under a new protocol;
it must preserve this diagnosis and the original failed model gate.
