"""Synthetic CPU contracts; these tests are not target-hardware experiments."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from experiments.verify_qkv_projection import (VerificationError, compare_arrays,
    digest, read_json, replay_private, safe_path, summarize, verify, verify_profiles)


def protocol():
    return dict(seed=20261009, prompts=[128, 4096], layers=24, decode_steps=16,
                micro_rounds=5, micro_repeats=5, model_rounds=5, model_repeats=2,
                profile_min_qkv_fraction=0.1,
                profile_warmups=1,
                profile_stages=["input_norm", "qkv", "rope_cache", "sdpa", "o_projection",
                                "attention_residual", "post_norm", "mlp_residual"],
                gates=dict(micro_min_ratio=1.05, micro_min_faster_rounds=4,
                           model_min_ratio=1.02, model_min_faster_rounds=4,
                           model_max_ratio=1.05, ttft_max_ratio=1.05,
                           memory_max_ratio=1.10))


def recorded_correctness():
    rows = []
    for prompt in (128, 4096):
        for layer in range(24):
            for arm in ("compiled_three", "packed"):
                rows.append(dict(kind="operator", case=f"p{prompt}-l{layer:02d}", arm=arm,
                                 finite=True, bitwise=True, arrays_checked=3, max_abs=0.0))
        for arm in ("native_adapter", "compiled_three", "packed", "profile"):
            rows.append(dict(kind="profile" if arm == "profile" else "model",
                             case=f"p{prompt}", arm=arm, finite=True, bitwise=True,
                             arrays_checked=50, max_abs=0.0))
    return rows


def recorded_samples():
    micro, model = [], []
    for prompt in (128, 4096):
        for rnd in range(5):
            for rep in range(5):
                for order, (arm, latency) in enumerate((("native", 2.0),
                        ("compiled_three", 1.5), ("packed", 1.0))):
                    micro.append(dict(case=f"p{prompt}", arm=arm, round=rnd, repeat=rep,
                                      order=order, latency_ms=latency, calls=24))
            for rep in range(2):
                for order, (arm, latency) in enumerate((("native", 20.0), ("native_adapter", 20.0),
                        ("compiled_three", 18.0), ("packed", 15.0))):
                    model.append(dict(case=f"p{prompt}", arm=arm, round=rnd, repeat=rep,
                                      order=order, latency_ms=latency, ttft_ms=2.0,
                                      decode_ms=latency - 2.0, peak_mlx_bytes=100,
                                      tokens=[0] * 17, cache_offsets=[prompt + 16] * 24))
    # Regroup the fixture in the runner's preregistered single-generator order.
    rng = random.Random(20261009)
    reordered = []
    for rows, repeats, arms in ((micro, 5, ("native", "compiled_three", "packed")),
                               (model, 2, ("native", "native_adapter", "compiled_three", "packed"))):
        lookup = {(r["case"], r["round"], r["repeat"], r["arm"]): r for r in rows}
        ordered = []
        for prompt in (128, 4096):
            for rnd in range(5):
                for rep in range(repeats):
                    order = list(arms)
                    rng.shuffle(order)
                    for position, arm in enumerate(order):
                        row = lookup[(f"p{prompt}", rnd, rep, arm)]
                        row["order"] = position
                        ordered.append(row)
        reordered.append(ordered)
    return tuple(reordered)


def recorded_profiles():
    stages = ("input_norm", "qkv", "rope_cache", "sdpa", "o_projection",
              "attention_residual", "post_norm", "mlp_residual")
    rows = [dict(prompt=p, step=step, layer=layer, stage=stage, latency_ms=1.0)
            for p in (128, 4096) for step in range(1, 17) for layer in range(24) for stage in stages]
    summary = {str(p): dict(stage_totals_ms={stage: 384.0 for stage in stages},
                           qkv_fraction_instrumented=0.125) for p in (128, 4096)}
    return rows, summary


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False))


def public_fixture(root):
    repo = root / "repo"
    folder = root / "public"
    spec = protocol()
    source_names = ["configs/qkv-projection-v1.json", "lab/qkv_projection.py",
                    "experiments/qkv_projection.py", "experiments/verify_qkv_projection.py"]
    for name in source_names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec) if name.endswith(".json") else "# synthetic fixture\n")
    correctness = recorded_correctness()
    micro, model = recorded_samples()
    profiles, profile_summary = recorded_profiles()
    run = dict(schema_version=1, status="complete", phase="complete", protocol_commit="a" * 40,
               spec=spec, source_sha256={name: digest(repo / name) for name in source_names},
               private_manifest_sha256="b" * 64, profile_summary=profile_summary)
    for name, data in (("run.json", run), ("correctness.json", correctness),
                       ("micro-samples.json", micro), ("model-samples.json", model),
                       ("profile-samples.json", profiles),
                       ("summary.json", summarize(spec, correctness, micro, model))):
        write_json(folder / name, data)
    return repo, folder


def private_operator_fixture(root):
    arrays = {"x": np.ones((1, 1, 896), np.float16)}
    for arm in ("native", "compiled_three", "packed"):
        for part, width in (("q", 896), ("k", 128), ("v", 128)):
            arrays[f"{arm}_{part}"] = np.ones((1, 1, width), np.float16)
    path = root / "operator-p128-l00.npz"
    np.savez_compressed(path, **arrays)
    manifest = dict(files={path.name: dict(sha256=digest(path), kind="operator", case="p128-l00")})
    write_json(root / "manifest.json", manifest)
    run = dict(spec=protocol(), private_manifest_sha256=digest(root / "manifest.json"))
    return arrays, path, manifest, run, recorded_correctness()[:2]


class ArrayReplayContracts(unittest.TestCase):
    def test_every_element_and_signed_zero_are_checked(self):
        reference = np.ones((1, 3), np.float16)
        changed = reference.copy()
        changed[0, -1] = np.nextafter(np.float16(1), np.float16(2))
        result = compare_arrays(changed, reference)
        self.assertEqual(result["failed_elements"], 1)
        self.assertEqual(result["max_abs"], 2 ** -10)
        self.assertFalse(result["bitwise_equal"])
        zero = np.array([0.0], np.float16)
        signed = compare_arrays(-zero, zero)
        self.assertTrue(signed["allclose"])
        self.assertFalse(signed["bitwise_equal"])

    def test_nonfinite_cannot_pass_even_when_identical(self):
        for value in (np.nan, np.inf, -np.inf):
            a = np.array([1.0, value], np.float16)
            result = compare_arrays(a, a)
            self.assertFalse(result["finite"])
            self.assertFalse(result["allclose"])
            self.assertEqual(result["failed_elements"], 1)
            self.assertIsNone(result["max_abs"])

    def test_shape_dtype_and_tolerance_contract(self):
        a = np.array([1], np.float16)
        for b in (a.reshape(1, 1), a.astype(np.float32), np.array([], np.float16)):
            with self.assertRaises(VerificationError):
                compare_arrays(a, b)
        for tolerance in (-1, np.inf, True):
            with self.assertRaisesRegex(VerificationError, "tolerance"):
                compare_arrays(a, a, tolerance)

    def test_private_partial_replay_does_not_become_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, run, records = private_operator_fixture(Path(tmp))
            result = replay_private(tmp, run, records, [])
            self.assertFalse(result["numerical_replay_complete"])
            self.assertEqual(result["compared_records"], 2)
            self.assertEqual(result["compared_elements"], 2304)

    def test_fresh_hash_cannot_hide_changed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays, path, manifest, run, records = private_operator_fixture(root)
            arrays["packed_v"].flat[-1] = 9
            np.savez_compressed(path, **arrays)
            manifest["files"][path.name]["sha256"] = digest(path)
            write_json(root / "manifest.json", manifest)
            run["private_manifest_sha256"] = digest(root / "manifest.json")
            with self.assertRaisesRegex(VerificationError, "full-array correctness"):
                replay_private(root, run, records, [])

    def test_cropping_all_outputs_equally_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays, path, manifest, run, records = private_operator_fixture(root)
            arrays = {k: v[..., :1] for k, v in arrays.items()}
            np.savez_compressed(path, **arrays)
            manifest["files"][path.name]["sha256"] = digest(path)
            write_json(root / "manifest.json", manifest)
            run["private_manifest_sha256"] = digest(root / "manifest.json")
            with self.assertRaisesRegex(VerificationError, "truncated"):
                replay_private(root, run, records, [])


class SummaryContracts(unittest.TestCase):
    def test_strong_baseline_hand_calculated_ratios_and_all_rounds(self):
        micro, model = recorded_samples()
        result = summarize(protocol(), recorded_correctness(), micro, model)
        self.assertTrue(result["performance_accepted"])
        self.assertEqual(result["micro"][0]["comparisons"]["native"]["control_over_packed"], 2)
        self.assertEqual(result["micro"][0]["comparisons"]["compiled_three"]["control_over_packed"], 1.5)
        self.assertEqual(result["model"][1]["comparisons"]["compiled_three"]["control_over_packed"], 1.2)
        self.assertEqual(result["model"][1]["comparisons"]["compiled_three"]["faster_rounds"], 5)

    def test_compiled_control_can_fail_despite_native_speedup(self):
        micro, model = recorded_samples()
        for row in micro:
            if row["arm"] == "compiled_three":
                row["latency_ms"] = 0.5
        result = summarize(protocol(), recorded_correctness(), micro, model)
        self.assertFalse(result["micro_pass"])
        self.assertFalse(result["performance_accepted"])

    def test_memory_and_ttft_regressions_are_separate_from_total_speed(self):
        micro, model = recorded_samples()
        for row in model:
            if row["arm"] == "packed":
                row["peak_mlx_bytes"] = 111
                row["ttft_ms"] = 3
        result = summarize(protocol(), recorded_correctness(), micro, model)
        self.assertTrue(result["model_speed_pass"])
        self.assertFalse(result["memory_pass"])
        self.assertFalse(result["nonregression_pass"])
        self.assertFalse(result["performance_accepted"])

    def test_missing_case_round_repeat_or_correctness_never_passes(self):
        micro, model = recorded_samples()
        result = summarize(protocol(), recorded_correctness(), micro[:-1], model)
        self.assertFalse(result["timing_complete"])
        self.assertFalse(result["performance_accepted"])
        with self.assertRaisesRegex(VerificationError, "before all correctness"):
            summarize(protocol(), recorded_correctness()[:-1], micro, model)

    def test_duplicate_order_sample_nan_and_changed_gates_are_rejected(self):
        micro, model = recorded_samples()
        with self.assertRaisesRegex(VerificationError, "duplicate timing"):
            summarize(protocol(), recorded_correctness(), micro + [micro[0]], model)
        duplicate = copy.deepcopy(micro)
        duplicate[1]["order"] = duplicate[0]["order"]
        with self.assertRaisesRegex(VerificationError, "trial order"):
            summarize(protocol(), recorded_correctness(), duplicate, model)
        nonfinite = copy.deepcopy(micro)
        nonfinite[0]["latency_ms"] = float("nan")
        with self.assertRaisesRegex(VerificationError, "nonfinite timing"):
            summarize(protocol(), recorded_correctness(), nonfinite, model)
        spec = protocol()
        spec["gates"]["micro_min_ratio"] = 1.0
        with self.assertRaisesRegex(VerificationError, "acceptance gates"):
            summarize(spec, recorded_correctness(), micro, model)

    def test_seeded_sequence_rejects_reshuffling_and_nonprefix_partial_selection(self):
        micro, model = recorded_samples()
        shuffled = copy.deepcopy(micro)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        shuffled[0]["order"], shuffled[1]["order"] = 0, 1
        with self.assertRaisesRegex(VerificationError, "seeded sample order"):
            summarize(protocol(), recorded_correctness(), shuffled, model)
        with self.assertRaisesRegex(VerificationError, "seeded sample order"):
            summarize(protocol(), recorded_correctness(), micro[3:], [])

    def test_token_and_cache_contracts_not_just_timings(self):
        micro, model = recorded_samples()
        model[0]["tokens"][-1] = 1
        with self.assertRaisesRegex(VerificationError, "token identity"):
            summarize(protocol(), recorded_correctness(), micro, model)
        model[0]["tokens"][-1] = 0
        model[0]["cache_offsets"][-1] -= 1
        with self.assertRaisesRegex(VerificationError, "cache offsets"):
            summarize(protocol(), recorded_correctness(), micro, model)

    def test_profile_is_complete_perturbed_screening_with_all_stages(self):
        rows, recorded = recorded_profiles()
        result = verify_profiles(protocol(), rows, recorded, True)
        self.assertTrue(result["complete"])
        self.assertTrue(result["recorded_screening_pass"])
        with self.assertRaisesRegex(VerificationError, "incomplete profile"):
            verify_profiles(protocol(), rows[:-1], recorded, True)
        recorded["4096"]["qkv_fraction_instrumented"] = 0.9
        with self.assertRaisesRegex(VerificationError, "profile fraction"):
            verify_profiles(protocol(), rows, recorded, True)


class PublicArchiveContracts(unittest.TestCase):
    def test_public_hash_and_scalar_checks_are_not_numerical_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root = public_fixture(Path(tmp))
            result = verify(root, repo_root=repo)
            self.assertTrue(result["public_consistency_valid"])
            self.assertTrue(result["recorded_performance_accepted"])
            self.assertFalse(result["numerical_replay_complete"])
            self.assertFalse(result["accepted"])

    def test_summary_and_source_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root = public_fixture(Path(tmp))
            summary = read_json(root / "summary.json")
            summary["micro"][0]["comparisons"]["native"]["control_over_packed"] += 1
            write_json(root / "summary.json", summary)
            with self.assertRaisesRegex(VerificationError, "summary arithmetic"):
                verify(root, repo_root=repo)
            (repo / "lab/qkv_projection.py").write_text("# changed\n")
            with self.assertRaisesRegex(VerificationError, "source binding"):
                verify(root, repo_root=repo)

    def test_complete_state_requires_complete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root = public_fixture(Path(tmp))
            write_json(root / "micro-samples.json", [])
            write_json(root / "model-samples.json", [])
            write_json(root / "summary.json", summarize(protocol(), recorded_correctness(), [], []))
            with self.assertRaisesRegex(VerificationError, "marked complete"):
                verify(root, repo_root=repo)
            run = read_json(root / "run.json")
            run["status"] = "failed"
            write_json(root / "run.json", run)
            self.assertFalse(verify(root, repo_root=repo)["accepted"])

    def test_unsafe_paths_symlinks_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../outside", "/tmp/outside", ""):
                with self.assertRaises(VerificationError):
                    safe_path(root, name)
            (root / "escape").symlink_to(root.parent)
            with self.assertRaisesRegex(VerificationError, "symlink"):
                safe_path(root, "escape/outside")
            (root / "bad.json").write_text('{"latency": NaN}')
            with self.assertRaisesRegex(VerificationError, "nonfinite JSON"):
                read_json(root / "bad.json")


if __name__ == "__main__":
    unittest.main()
