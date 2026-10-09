"""CPU reconstruction of QKV experiment evidence, without MLX or a model.

Public scalar consistency and local full-array numerical replay are distinct.
Neither hashes nor this replay establish that a GPU executed the recorded code.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import median

import numpy as np


class VerificationError(ValueError):
    """An archive does not satisfy its preregistered evidence contract."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise VerificationError(f"nonfinite JSON literal: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def safe_path(root, name):
    require(isinstance(name, str) and bool(name), "invalid artifact name")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "escaping artifact path")
    root = Path(root).resolve()
    target = root / relative
    require(target.resolve().is_relative_to(root), "escaping artifact symlink")
    return target


def compare_arrays(left, right, atol=0, rtol=0):
    """Compare every element using an explicitly FP64 tolerance calculation.

    ``right`` is the reference. Inputs retain dtype/bitwise information, while
    numerical subtraction and the tolerance boundary are evaluated in FP64.
    Nonfinite pairs never pass (even equal infinities are numerical failures).
    """
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.size > 0, "array shape mismatch")
    require(left.dtype == right.dtype, "array dtype mismatch")
    require(left.dtype.kind in "fiu", "unsupported array dtype")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) and x >= 0 for x in (atol, rtol)),
            "invalid tolerance")
    finite = np.isfinite(left) & np.isfinite(right)
    a, b = left.astype(np.float64), right.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(a - b)
        passing = finite & (error <= float(atol) + float(rtol) * np.abs(b))
    return dict(shape=list(left.shape), dtype=str(left.dtype), elements=int(left.size),
                finite=bool(finite.all()),
                bitwise_equal=bool(left.tobytes() == right.tobytes()),
                max_abs=float(error.max()) if finite.all() else None,
                failed_elements=int(left.size - passing.sum()),
                allclose=bool(passing.all()))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate NPZ name")
        return {key: archive[key] for key in archive.files}


MICRO_ARMS = ("native", "compiled_three", "packed")
MODEL_ARMS = ("native", "native_adapter", "compiled_three", "packed")
CONTROLS = ("native", "compiled_three")
PROFILE_STAGES = ("input_norm", "qkv", "rope_cache", "sdpa", "o_projection",
                  "attention_residual", "post_norm", "mlp_residual")
REQUIRED_SOURCES = {"configs/qkv-projection-v1.json", "lab/qkv_projection.py",
                    "experiments/qkv_projection.py", "experiments/verify_qkv_projection.py"}


def expected_correctness(spec):
    rows = {("operator", f"p{p}-l{layer:02d}", arm): 3
            for p in spec["prompts"] for layer in range(spec["layers"])
            for arm in ("compiled_three", "packed")}
    rows.update({("model", f"p{p}", arm): 2 + 2 * spec["layers"]
                 for p in spec["prompts"] for arm in MODEL_ARMS[1:]})
    rows.update({("profile", f"p{p}", "profile"): 2 + 2 * spec["layers"]
                 for p in spec["prompts"]})
    return rows


def validate_spec(spec):
    require(type(spec["seed"]) is int and spec["seed"] == 20261009,
            "changed preregistered order seed")
    require(spec["prompts"] == [128, 4096] and spec["layers"] == 24
            and spec["decode_steps"] == 16, "unexpected study dimensions")
    for key, value in {"micro_rounds": 5, "micro_repeats": 5,
                       "model_rounds": 5, "model_repeats": 2}.items():
        require(type(spec[key]) is int and spec[key] == value,
                f"unexpected preregistered {key}")
    require(spec["gates"] == dict(micro_min_ratio=1.05, micro_min_faster_rounds=4,
            model_min_ratio=1.02, model_min_faster_rounds=4,
            model_max_ratio=1.05, ttft_max_ratio=1.05, memory_max_ratio=1.10),
            "changed preregistered acceptance gates")


def correctness_records(spec, records):
    expected = expected_correctness(spec)
    seen = set()
    passed = True
    for row in records:
        key = (row["kind"], row["case"], row["arm"])
        require(key in expected and key not in seen, "unknown/duplicate correctness case")
        seen.add(key)
        require(type(row["arrays_checked"]) is int
                and row["arrays_checked"] == expected[key], "truncated array count")
        require(type(row["finite"]) is bool and type(row["bitwise"]) is bool,
                "correctness flags must be booleans")
        if row["finite"]:
            require(isinstance(row["max_abs"], (int, float))
                    and not isinstance(row["max_abs"], bool)
                    and math.isfinite(row["max_abs"]) and row["max_abs"] >= 0,
                    "invalid maximum error")
            require(not row["bitwise"] or row["max_abs"] == 0,
                    "bitwise result has nonzero error")
        else:
            require(row["max_abs"] is None, "nonfinite maximum must be null")
        passed = passed and row["finite"] and row["bitwise"]
    return seen == set(expected), passed and seen == set(expected)


def _sample_records(spec, rows, kind):
    arms = MICRO_ARMS if kind == "micro" else MODEL_ARMS
    expected = {(f"p{p}", arm, rnd, rep) for p in spec["prompts"] for arm in arms
                for rnd in range(spec[f"{kind}_rounds"])
                for rep in range(spec[f"{kind}_repeats"])}
    seen, orders = set(), set()
    for row in rows:
        key = (row["case"], row["arm"], row["round"], row["repeat"])
        require(all(type(row[k]) is int for k in ("round", "repeat", "order")),
                "sample indices must be integers")
        require(key in expected and key not in seen, "unknown/duplicate timing sample")
        seen.add(key)
        order_key = (row["case"], row["round"], row["repeat"], row["order"])
        require(0 <= row["order"] < len(arms) and order_key not in orders,
                "duplicate/invalid trial order")
        orders.add(order_key)
        names = ("latency_ms",) if kind == "micro" else ("latency_ms", "ttft_ms", "decode_ms")
        require(all(isinstance(row[name], (int, float))
                    and not isinstance(row[name], bool)
                    and math.isfinite(row[name]) and row[name] > 0 for name in names),
                "nonpositive/nonfinite timing")
        if kind == "micro":
            require(type(row["calls"]) is int and row["calls"] == spec["layers"],
                    "micro call count changed")
        else:
            require(row["ttft_ms"] <= row["latency_ms"]
                    and row["decode_ms"] <= row["latency_ms"], "inconsistent model timing")
            require(type(row["peak_mlx_bytes"]) is int and row["peak_mlx_bytes"] > 0,
                    "invalid allocator peak")
            require(len(row["tokens"]) == spec["decode_steps"] + 1
                    and all(type(t) is int and 0 <= t < 151936 for t in row["tokens"]),
                    "invalid timed token sequence")
            prompt = int(row["case"][1:])
            require(row["cache_offsets"] == [prompt + spec["decode_steps"]] * spec["layers"],
                    "timed cache offsets changed")
    # The same generator is consumed by all micro prompts before any model
    # prompt. Validate the append order too, so a partial archive must be the
    # actual preregistered prefix rather than an arbitrary favourable subset.
    order_rng = random.Random(spec["seed"])
    ordered = {"micro": [], "model": []}
    for phase, phase_arms in (("micro", MICRO_ARMS), ("model", MODEL_ARMS)):
        for prompt in spec["prompts"]:
            for rnd in range(spec[f"{phase}_rounds"]):
                for rep in range(spec[f"{phase}_repeats"]):
                    shuffled = list(phase_arms)
                    order_rng.shuffle(shuffled)
                    ordered[phase].extend((f"p{prompt}", arm, rnd, rep, order)
                                          for order, arm in enumerate(shuffled))
    observed = [(r["case"], r["arm"], r["round"], r["repeat"], r["order"]) for r in rows]
    require(observed == ordered[kind][:len(rows)], "seeded sample order changed")
    return seen == expected


def _timing_case(spec, rows, case, kind):
    arms = MICRO_ARMS if kind == "micro" else MODEL_ARMS
    selected = {arm: [r for r in rows if r["case"] == case and r["arm"] == arm]
                for arm in arms}
    expected_n = spec[f"{kind}_rounds"] * spec[f"{kind}_repeats"]
    if any(len(values) != expected_n for values in selected.values()):
        return dict(case=case, complete=False, samples={a: len(v) for a, v in selected.items()})
    fields = ("latency_ms",) if kind == "micro" else ("latency_ms", "ttft_ms", "decode_ms")
    medians = {a: {name: median(r[name] for r in values) for name in fields}
               for a, values in selected.items()}
    # Descriptive order statistics only: ten requests/arm do not establish an
    # operational tail-latency guarantee. Quantiles are not acceptance gates.
    p95 = {a: {name: float(np.percentile([r[name] for r in values], 95)) for name in fields}
           for a, values in selected.items()}
    comparisons = {}
    gates = spec["gates"]
    for control in CONTROLS:
        faster = sum(median(r["latency_ms"] for r in selected["packed"] if r["round"] == rnd)
                     < median(r["latency_ms"] for r in selected[control] if r["round"] == rnd)
                     for rnd in range(spec[f"{kind}_rounds"]))
        ratio = medians[control]["latency_ms"] / medians["packed"]["latency_ms"]
        item = dict(control_over_packed=ratio, faster_rounds=faster,
                    speed_pass=bool(ratio >= gates[f"{kind}_min_ratio"]
                                    and faster >= gates[f"{kind}_min_faster_rounds"]))
        if kind == "model":
            peak_ratio = max(r["peak_mlx_bytes"] for r in selected["packed"]) / max(
                r["peak_mlx_bytes"] for r in selected[control])
            ttft_ratio = medians["packed"]["ttft_ms"] / medians[control]["ttft_ms"]
            item.update(packed_over_control=1 / ratio, ttft_packed_over_control=ttft_ratio,
                        peak_packed_over_control=peak_ratio,
                        nonregression_pass=bool(1 / ratio <= gates["model_max_ratio"]
                                               and ttft_ratio <= gates["ttft_max_ratio"]),
                        memory_pass=bool(peak_ratio <= gates["memory_max_ratio"]))
        comparisons[control] = item
    return dict(case=case, complete=True, medians=medians, descriptive_p95=p95,
                comparisons=comparisons)


def summarize(spec, correctness, micro_samples, model_samples):
    """Rebuild public statistics; no numerical truth is inferred from scalars."""
    validate_spec(spec)
    correct_complete, correct_pass = correctness_records(spec, correctness)
    micro_complete = _sample_records(spec, micro_samples, "micro")
    model_complete = _sample_records(spec, model_samples, "model")
    require(not micro_samples and not model_samples or correct_pass,
            "timing present before all correctness gates passed")
    micro = [_timing_case(spec, micro_samples, f"p{p}", "micro") for p in spec["prompts"]]
    model = [_timing_case(spec, model_samples, f"p{p}", "model") for p in spec["prompts"]]
    micro_pass = micro_complete and all(c["speed_pass"] for r in micro for c in r["comparisons"].values())
    model_speed = model_complete and all(c["speed_pass"] for r in model for c in r["comparisons"].values())
    nonregression = model_complete and all(c["nonregression_pass"] for r in model for c in r["comparisons"].values())
    memory = model_complete and all(c["memory_pass"] for r in model for c in r["comparisons"].values())
    for prompt in spec["prompts"]:
        tokens = [r["tokens"] for r in model_samples if r["case"] == f"p{prompt}"]
        require(not tokens or all(t == tokens[0] for t in tokens), "timed token identity changed")
    return dict(correctness_complete=correct_complete, recorded_correctness_pass=correct_pass,
                timing_complete=bool(micro_complete and model_complete),
                micro=micro, model=model, micro_pass=bool(micro_pass),
                model_speed_pass=bool(model_speed), nonregression_pass=bool(nonregression),
                memory_pass=bool(memory),
                performance_accepted=bool(correct_pass and micro_pass and model_speed and nonregression and memory))


def _compare_set(left, right, row):
    require(set(left) == set(right), "array set mismatch")
    metrics = [compare_arrays(left[name], right[name]) for name in sorted(left)]
    finite = all(item["finite"] for item in metrics)
    expected = dict(kind=row["kind"], case=row["case"], arm=row["arm"],
                    finite=finite, bitwise=all(item["bitwise_equal"] for item in metrics),
                    arrays_checked=len(metrics),
                    max_abs=max(item["max_abs"] for item in metrics) if finite else None)
    require(row == expected, "full-array correctness record mismatch")
    return sum(item["elements"] for item in metrics)


def _check_shape(array, shape, dtype):
    require(array.shape == shape and array.dtype == np.dtype(dtype),
            "truncated/wrong-dtype archived tensor")


def _check_model(arrays, prompt, spec):
    expected = {"tokens", "logits"} | {f"l{layer:02d}_{kind}" for layer in range(spec["layers"])
                                     for kind in ("K", "V")}
    require(set(arrays) == expected, "incomplete model array set")
    _check_shape(arrays["tokens"], (17,), "int64")
    _check_shape(arrays["logits"], (17, 151936), "float16")
    require(np.array_equal(np.argmax(arrays["logits"], axis=-1), arrays["tokens"]),
            "tokens do not match archived logits")
    for key in expected - {"tokens", "logits"}:
        _check_shape(arrays[key], (1, 2, prompt + 16, 64), "float16")


def verify_profiles(spec, rows, recorded_summary, require_complete):
    """Check perturbed per-stage records without treating them as native costs."""
    require(spec.get("profile_min_qkv_fraction") == 0.1, "changed profile screening gate")
    require(spec.get("profile_stages") == list(PROFILE_STAGES)
            and spec.get("profile_warmups") == 1, "changed profile protocol")
    expected = {(p, step, layer, stage) for p in spec["prompts"] for step in range(1, 17)
                for layer in range(24) for stage in PROFILE_STAGES}
    seen = set()
    for row in rows:
        key = (row["prompt"], row["step"], row["layer"], row["stage"])
        require(all(type(row[k]) is int for k in ("prompt", "step", "layer")),
                "profile indices must be integers")
        require(key in expected and key not in seen, "unknown/duplicate profile sample")
        require(isinstance(row["latency_ms"], (int, float))
                and not isinstance(row["latency_ms"], bool)
                and math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0,
                "nonpositive/nonfinite profile timing")
        seen.add(key)
    complete = seen == expected
    require(not require_complete or complete, "incomplete profile stage coverage")
    if recorded_summary is not None:
        require(complete and set(recorded_summary) == {str(p) for p in spec["prompts"]},
                "profile summary without complete records")
        for prompt in spec["prompts"]:
            totals = {stage: math.fsum(r["latency_ms"] for r in rows
                                       if r["prompt"] == prompt and r["stage"] == stage)
                      for stage in PROFILE_STAGES}
            actual = recorded_summary[str(prompt)]
            require(set(actual["stage_totals_ms"]) == set(PROFILE_STAGES), "profile stage omitted")
            for stage, total in totals.items():
                require(math.isclose(actual["stage_totals_ms"][stage], total,
                                     rel_tol=1e-12, abs_tol=1e-12), "profile sum changed")
            fraction = totals["qkv"] / math.fsum(totals.values())
            require(math.isclose(actual["qkv_fraction_instrumented"], fraction,
                                 rel_tol=1e-12, abs_tol=1e-12), "profile fraction changed")
        return dict(complete=True, recorded_screening_pass=
                    recorded_summary["4096"]["qkv_fraction_instrumented"] >= spec["profile_min_qkv_fraction"])
    require(not require_complete, "complete run missing profile summary")
    return dict(complete=complete, recorded_screening_pass=False)


def replay_private(private_root, run, records, model_samples):
    """Recompute all public comparisons from the complete local output arrays."""
    private_root = Path(private_root)
    manifest_path = private_root / "manifest.json"
    require(digest(manifest_path) == run["private_manifest_sha256"], "private manifest changed")
    manifest = read_json(manifest_path)
    require(set(manifest) == {"files"}, "unexpected private manifest schema")
    files = manifest["files"]
    actual = {str(p.relative_to(private_root)) for p in private_root.rglob("*.npz")}
    require(actual == set(files), "unmanifested/missing private array file")
    operator, model = {}, {}
    for name, entry in files.items():
        path = safe_path(private_root, name)
        require(digest(path) == entry["sha256"], "private artifact modified")
        require(entry["kind"] in ("operator", "model"), "unknown private artifact kind")
        if entry["kind"] == "operator":
            key = entry["case"]
            require(key not in operator, "duplicate operator archive")
            require(("operator", key, "packed") in expected_correctness(run["spec"]),
                    "unexpected operator archive")
            require(name == f"operator-{key}.npz", "operator archive filename changed")
            arrays = load_npz(path)
            allowed = {"x"} | {f"{arm}_{part}" for arm in MICRO_ARMS for part in ("q", "k", "v")}
            require("x" in arrays and set(arrays) <= allowed, "unknown operator array set")
            _check_shape(arrays["x"], (1, 1, 896), "float16")
            for arm in MICRO_ARMS:
                present = {f"{arm}_{part}" for part in ("q", "k", "v")} & set(arrays)
                require(len(present) in (0, 3), "incomplete projection output tuple")
                for part in ("q", "k", "v"):
                    if f"{arm}_{part}" in arrays:
                        _check_shape(arrays[f"{arm}_{part}"], (1, 1, 896 if part == "q" else 128), "float16")
            operator[key] = arrays
        else:
            key = (entry["case"], entry["arm"])
            require(key not in model and entry["arm"] in (*MODEL_ARMS, "profile"),
                    "duplicate/unknown model archive")
            require(entry["case"] in {f"p{p}" for p in run["spec"]["prompts"]},
                    "unknown model prompt")
            require(name == f"model-{key[0]}-{key[1]}.npz", "model archive filename changed")
            arrays = load_npz(path)
            _check_model(arrays, int(entry["case"][1:]), run["spec"])
            model[key] = arrays
    elements = 0
    used_operator, used_model = set(), set()
    for row in records:
        if row["kind"] == "operator":
            require(row["case"] in operator, "missing operator archive")
            arrays = operator[row["case"]]
            require(all(f"{arm}_{part}" in arrays for arm in (row["arm"], "native")
                        for part in ("q", "k", "v")), "missing compared operator output")
            left = {part: arrays[f'{row["arm"]}_{part}'] for part in ("q", "k", "v")}
            right = {part: arrays[f"native_{part}"] for part in ("q", "k", "v")}
            used_operator.add(row["case"])
        else:
            key, native = (row["case"], row["arm"]), (row["case"], "native")
            require(key in model and native in model, "missing model archive")
            left, right = model[key], model[native]
            used_model.update((key, native))
        elements += _compare_set(left, right, row)
    complete = set(operator) == {f"p{p}-l{layer:02d}" for p in run["spec"]["prompts"]
                                for layer in range(run["spec"]["layers"])} and set(model) == {
                                    (f"p{p}", arm) for p in run["spec"]["prompts"]
                                    for arm in (*MODEL_ARMS, "profile")}
    # Files without a completed comparison may be retained after an exception;
    # they do not upgrade a partial run to numerical replay completeness.
    complete = (complete and used_operator == set(operator) and used_model == set(model)
                and correctness_records(run["spec"], records)[0])
    for row in model_samples:
        require(row["tokens"] == model[(row["case"], row["arm"])]["tokens"].tolist(),
                "timed tokens differ from numerical archive")
    return dict(numerical_replay_complete=bool(complete), compared_elements=elements,
                private_array_files=len(files), compared_records=len(records))


def verify(root, private_root=None, repo_root=None):
    root = Path(root)
    run = read_json(root / "run.json")
    require(run["schema_version"] == 1 and run["status"] in
            ("complete", "failed", "profile_gate_failed"), "invalid run state")
    commit = run["protocol_commit"]
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in "0123456789abcdef" for c in commit), "invalid protocol commit")
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    require(isinstance(run["source_sha256"], dict) and REQUIRED_SOURCES <= set(run["source_sha256"]),
            "missing source binding")
    for name, expected in run["source_sha256"].items():
        require(digest(safe_path(repo_root, name)) == expected, "source binding changed")
    configs = [name for name in run["source_sha256"] if name.startswith("configs/") and name.endswith(".json")]
    require(len(configs) == 1 and read_json(safe_path(repo_root, configs[0])) == run["spec"],
            "protocol specification changed")
    correctness = read_json(root / "correctness.json")
    micro, model = (read_json(root / name) for name in ("micro-samples.json", "model-samples.json"))
    summary = summarize(run["spec"], correctness, micro, model)
    require(read_json(root / "summary.json") == summary, "summary arithmetic changed")
    profile = verify_profiles(run["spec"], read_json(root / "profile-samples.json"),
                              run.get("profile_summary"),
                              require_complete=run["status"] in ("complete", "profile_gate_failed"))
    if micro or model or any(r["kind"] == "operator" for r in correctness):
        require(profile["complete"] and profile["recorded_screening_pass"],
                "candidate run before profile screening passed")
    if run["status"] == "profile_gate_failed":
        require(not profile["recorded_screening_pass"] and not micro and not model,
                "inconsistent profile-gate failure")
    if run["status"] == "complete":
        require(summary["correctness_complete"] and summary["recorded_correctness_pass"]
                and summary["timing_complete"], "incomplete experiment marked complete")
    else:
        require(not summary["performance_accepted"], "partial/failed run marked accepted")
    numerical = dict(numerical_replay_complete=False, compared_elements=0,
                     private_array_files=0, compared_records=0)
    if private_root is not None:
        numerical = replay_private(private_root, run, correctness, model)
        if run["status"] == "complete":
            require(numerical["numerical_replay_complete"], "incomplete private replay")
    return dict(public_consistency_valid=True, **numerical,
                recorded_performance_accepted=summary["performance_accepted"],
                accepted=bool(run["status"] == "complete" and summary["performance_accepted"]
                              and numerical["numerical_replay_complete"]),
                gpu_rerun=False, quality_evaluated=False,
                scope="Public scalars/timing consistency; full numerical comparison only when private arrays supplied. No GPU rerun or execution provenance proof.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/qkv-projection-v1"))
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.private_root, args.repo_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
