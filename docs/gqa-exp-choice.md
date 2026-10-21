# Same-input GQA exp-family intervention

This is a bounded numerical experiment following the [model-KV diagnosis](gqa-shared-diagnostic.md).
The original model condition remains failed; this experiment never loads a model or records performance.

The frozen candidate has three `metal::exp` call sites (two in the partial stage,
one in the merge). Pinned MLX 0.29.3 `sdpa_vector.h` uses `fast::exp`, but also a
different reduction schedule. The [protocol](../configs/gqa-exp-choice-v1.json)
asks whether changing that function family alone eliminates every native/candidate
value disagreement on all 384 archived inputs. It does not assume faster or more accurate results.

The [implementation](../lab/gqa_exp_choice.py) adds FP32 output taps to the unchanged
FP16 stores. Its standard and fast versions differ only at the three exp calls
and unique kernel names. The [runner](../experiments/gqa_exp_choice.py) reconstructs
the original full-capacity K/V slices and Q transpose, verifies all original shapes,
strides and values, and checks fresh native, frozen candidate and instrumented
candidate outputs for bitwise reproduction. Both tap FP32 values must round to
their own FP16 outputs. A fidelity failure preserves current arrays and stops.

Every input contributes all 896 output elements, including previously equal values.
Native disagreement counts are partitioned into resolved, persistent and introduced;
FP64 reference errors are reported separately. At all original disagreements the
records retain the midpoint and signed distances in units of the gap between the
two FP16 values. That gap is one adjacent FP16 spacing only when `adjacent` is true.
Reference rounding and midpoint ties use NumPy round-to-nearest, ties-to-even.
The float64 reference is not an exact-real arithmetic proof.

Output reproduction validates the instrumented FP16 behavior, but does not prove
its exposed FP32 values equal the uninstrumented kernel's internal values.
Changing all three exp sites cannot identify one unique site. If outputs do not
change, that cannot distinguish equal compiler lowering from input insensitivity,
or exclude an interaction with the different native reduction schedule.
If native agreement improves, FP64 accuracy need not improve. Neither result
establishes a repaired model, model quality, or acceleration.

Run once after protocol and source commit:

```sh
python -m experiments.gqa_exp_choice
```

The output is the new `results/gqa-exp-choice-v1` directory. The runner refuses an
existing directory and optimized Python (`-O`), uses a 600-second hard timeout,
and checks 2 GiB MLX / 25 MiB disk limits. There is no warmup or repeated timing;
the single fixed sweep is diagnostic only. Model/source/version, layout, fidelity,
nonfinite and budget failures stop with a failure manifest. Finite operator
tolerance failures remain observations in the fixed sweep and cannot authorize
performance testing. Old kernels, tolerances, results, tags and Releases are preserved.

Independent CPU replay and focused tests:

```sh
python -m experiments.verify_gqa_exp_choice
python -m unittest discover -s tests -p 'test_gqa_exp_choice*.py' -v
```

CPU replay checks complete arrays and recorded receipts; it cannot rerun GPU
layout checks or prove a Metal instruction sequence. AI-assisted implementation
and execution remain separate from the owner's independent understanding.

## Frozen outcome: substitution is insufficient

Protocol, runner and instrumented operators were committed at
`b87038ed2defe2991d531a119832da0f57056824` before the sole GPU run. All 384
reconstructed-input, native/old-candidate/tap-output and cast fidelity checks
passed. The [manifest](../results/gqa-exp-choice-v1/run.json),
[complete metrics](../results/gqa-exp-choice-v1/records.json),
[aggregate](../results/gqa-exp-choice-v1/summary.json) and all output arrays retain
the unsuccessful hypothesis without changing the original model-KV result.

| Full fixed input set | Result |
|---|---:|
| Output elements per arm | 344,064 |
| Original FP16 native/candidate disagreements | 197 |
| Old disagreements resolved / persistent | 48 / 149 |
| Newly introduced disagreements | 37 |
| Fast-family FP16 native disagreements | 186 |
| FP16 values changed by intervention | 85 |
| FP16 absolute error vs float64 improved / worsened / unchanged | 47 / 38 / 343,979 |
| FP32 tap values changed by intervention | 162,861 |
| FP32 absolute error improved / worsened / unchanged | 78,866 / 83,995 / 181,203 |

Native, standard-half and fast-half all meet the old operator tolerance against
float64 on every element. Standard/fast FP32 maximum absolute errors are
8.241045585499762e-6 / 7.287371269093512e-6; their full-set mean absolute errors
are 6.316642523708049e-8 / 6.339820064375524e-8. Thus a lower maximum and fewer
native disagreements do not mean uniformly better accuracy; the mean error
increased slightly. All three FP16 arms have the same maximum absolute error,
0.0019517061520968326. None is treated as an exact mathematical ground truth.

The source-level exp-family intervention changes outputs under this fixed
schedule, but it does not eliminate the observed native disagreements. The
predeclared hypothesis is **not supported**. There is no evidence to adopt the
variant as a model fix. This experiment cannot attribute the remaining differences
to a unique reduction/FMA mechanism or predict a new variant's final K/V state.
The old candidate's 10/48 model-KV failures remain preserved; this new variant
has **zero model runs and zero performance trials**.

Independent CPU replay checks all 384 probes, 197 original-disagreement midpoint
records, all generated sources, 778 artifact hashes and all metrics. It recomputes
the float64 reference with a separate sum-based implementation. The archive totals
8,033,287 bytes and observed MLX allocator peak is 278,280 bytes; these are budget
observations, not physical device memory measurements or optimization claims.
There was no adaptive follow-up run, tolerance change or input selection.

## CPU replay portability correction

At result commit `50611f2`, both Python 3.11 CI jobs failed the aggregate's exact
comparison while Python 3.12 jobs passed. Every per-probe array metric had already
passed. The failure is retained in the [PR run](https://github.com/kimzclandi/inference-compression-lab/actions/runs/37809049578)
and [push run](https://github.com/kimzclandi/inference-compression-lab/actions/runs/37809037337).

The five means were generated on Python 3.12. Python documents a change to
[floating-point `sum` in 3.12](https://docs.python.org/3/builtins/functions.html#sum).
Replaying these positive per-probe means using a left fold gives different last
bits; `math.fsum` reproduces all five archived means exactly. The CPU verifier
now requests that summation explicitly. Full dictionary equality remains strict,
including rejection of a one-ULP edit to an aggregate mean. No experimental
tolerance, source snapshot, array, protocol, metric, hypothesis or GPU run changed.
This correction makes the observed archive reproducible across the tested Python
versions; it is not a universal cross-platform floating-point equality guarantee.
