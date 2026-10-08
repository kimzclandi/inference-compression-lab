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
