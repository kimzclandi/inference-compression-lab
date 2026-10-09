"""Independent contracts; actual MLX execution needs RUN_QKV_PROJECTION_TESTS=1.

CPU fixtures exercise rejection, ownership and restoration, not floating-point
equivalence or performance. The opt-in algebra test is synthetic and is not a
substitute for the separately frozen real-model protocol.
"""
import copy
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

from lab import qkv_projection as projection

try:
    import numpy as np
except ImportError:
    np = None


def metadata():
    # Qwen2.5-0.5B has 14 query heads and 2 KV heads, each of width 64.
    return [
        {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            "weight": {"shape": (width, 224), "dtype": "uint32"},
            "scales": {"shape": (width, 14), "dtype": "float16"},
            "biases": {"shape": (width, 14), "dtype": "float16"},
            "bias": {"shape": (width,), "dtype": "float16"},
        }
        for width in (896, 128, 128)
    ]


class ProjectionMetadataTests(unittest.TestCase):
    def test_decode_layout_and_metadata_ownership(self):
        source = metadata()
        saved = copy.deepcopy(source)
        description = projection.validate_projection_metadata(
            source, x_shape=(1, 1, 896), x_dtype="float16"
        )
        self.assertEqual(description["input_dims"], 896)
        self.assertEqual(tuple(description["output_dims"]), (896, 128, 128))
        self.assertEqual(description["packed_output_dims"], 1152)
        self.assertEqual(description["bits"], 8)
        self.assertEqual(description["group_size"], 64)
        self.assertEqual(description["mode"], "affine")
        self.assertEqual(source, saved)

    def test_shape_metadata_accepts_json_lists(self):
        source = metadata()
        for item in source:
            for tensor in ("weight", "scales", "biases", "bias"):
                item[tensor]["shape"] = list(item[tensor]["shape"])
        result = projection.validate_projection_metadata(source)
        self.assertEqual(result["packed_output_dims"], 1152)

    def test_projection_count_and_missing_bias_fail_closed(self):
        for source in ([], metadata()[:2], metadata() + metadata()[:1]):
            with self.subTest(count=len(source)), self.assertRaises(ValueError):
                projection.validate_projection_metadata(source)
        source = metadata()
        source[1].pop("bias")
        with self.assertRaises(ValueError):
            projection.validate_projection_metadata(source)

    def test_packing_and_group_boundaries_cannot_silently_broadcast(self):
        bad_shapes = [
            ("weight", (128, 223)),
            ("weight", (128, 225)),
            ("weight", (1, 128, 224)),
            ("scales", (128, 13)),
            ("scales", (128, 15)),
            ("scales", (1, 14)),
            ("biases", (128, 1)),
            ("biases", (128,)),
            ("bias", (1, 128)),
            ("bias", (1,)),
            ("weight", (0, 224)),
            ("weight", (True, 224)),
            ("weight", (128.0, 224)),
        ]
        for field, shape in bad_shapes:
            source = metadata()
            source[1][field]["shape"] = shape
            saved = copy.deepcopy(source)
            with self.subTest(field=field, shape=shape), self.assertRaises(ValueError):
                projection.validate_projection_metadata(source)
            self.assertEqual(source, saved)

    def test_wrong_head_partition_and_mixed_quantization_fail(self):
        source = metadata()
        source[0], source[1] = source[1], source[0]
        with self.assertRaises(ValueError):
            projection.validate_projection_metadata(source)
        for field, value in [
            ("bits", 4), ("bits", True), ("bits", 8.0),
            ("group_size", 32), ("group_size", True),
            ("group_size", 64.0), ("mode", "mxfp8"),
        ]:
            source = metadata()
            source[2][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                projection.validate_projection_metadata(source)

    def test_dtype_mismatch_is_not_implicitly_cast(self):
        for field, dtype in [
            ("weight", "uint16"), ("weight", "float16"),
            ("scales", "float32"), ("biases", "float32"),
            ("bias", "float32"), ("bias", "bfloat16"),
        ]:
            source = metadata()
            source[2][field]["dtype"] = dtype
            with self.subTest(field=field, dtype=dtype), self.assertRaises(ValueError):
                projection.validate_projection_metadata(source)

    def test_input_must_be_exact_single_token_decode(self):
        for shape in [(), (896,), (1, 896), (2, 1, 896), (1, 2, 896),
                      (1, 1, 895), (1, 1, 897), (True, 1, 896), (1, 1, 896.0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                projection.validate_projection_metadata(
                    metadata(), x_shape=shape, x_dtype="float16"
                )
        for dtype in ("float32", "bfloat16", "uint32"):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                projection.validate_projection_metadata(
                    metadata(), x_shape=(1, 1, 896), x_dtype=dtype
                )
        for kwargs in ({"x_shape": (1, 1, 896)}, {"x_dtype": "float16"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                projection.validate_projection_metadata(metadata(), **kwargs)

    def test_import_does_not_initialize_mlx(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import sys; import lab.qkv_projection; "
             "assert 'mlx.core' not in sys.modules; assert 'mlx.nn' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class BackendAvailabilityTests(unittest.TestCase):
    def test_no_cpu_fallback_when_metal_missing_or_cpu_selected(self):
        for available, device in [(False, "gpu"), (True, "cpu")]:
            modules = {name: types.ModuleType(name) for name in (
                "mlx", "mlx.core", "mlx.nn", "mlx_lm", "mlx_lm.models",
                "mlx_lm.models.base", "mlx_lm.models.cache", "mlx_lm.models.qwen2",
            )}
            modules["mlx.core"].metal = types.SimpleNamespace(is_available=lambda: available)
            modules["mlx.core"].default_device = lambda: device
            modules["mlx.core"].gpu = "gpu"
            modules["mlx_lm.models.base"].scaled_dot_product_attention = object()
            modules["mlx_lm.models.cache"].KVCache = type("KVCache", (), {})
            modules["mlx_lm.models.qwen2"].Attention = type("Attention", (), {})
            modules["mlx_lm.models.qwen2"].Model = type("Model", (), {})
            with self.subTest(available=available, device=device), patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                    projection._backend()


class FakeModule:
    pass


class FakeQuantizedLinear:
    def __init__(self, width, label):
        self.bits, self.group_size, self.mode = 8, 64, "affine"
        self.weight = np.full((width, 224), label, dtype="uint32")
        self.scales = np.full((width, 14), label / 16, dtype="float16")
        self.biases = np.full((width, 14), -label, dtype="float16")
        self.bias = np.arange(width, dtype="float16")


class FakeAttention:
    def __init__(self):
        # Adapter tests replace preparation with a synthetic projection. These
        # unique sentinels detect changes to ownership of learned parameters.
        self.q_proj, self.k_proj, self.v_proj = object(), object(), object()
        self.n_heads, self.n_kv_heads = 14, 2
        self.scale = 0.125
        self.rope = Mock(side_effect=lambda x, **kwargs: x)
        self.o_proj = Mock(side_effect=lambda x: x)
        self.forward = Mock(return_value=object())

    def __call__(self, *args):
        return self.forward(*args)


class FakeModel:
    def __init__(self):
        self.args = types.SimpleNamespace(
            hidden_size=896, num_attention_heads=14,
            num_key_value_heads=2, num_hidden_layers=24,
        )
        self.model = types.SimpleNamespace(layers=[
            types.SimpleNamespace(self_attn=FakeAttention()) for _ in range(24)
        ])


class FakeCache:
    def __init__(self, offset=0):
        self.offset = offset
        self.updates = 0
        self.failure = None
        capacity = offset if type(offset) is int and 0 < offset <= 8192 else 0
        self.keys = (np.zeros((1, 2, capacity, 64), "float16") if capacity else None)
        self.values = (np.zeros((1, 2, capacity, 64), "float16") if capacity else None)

    def update_and_fetch(self, k, v):
        self.updates += 1
        if self.failure is not None:
            raise self.failure
        self.offset += k.shape[2]
        return k, v


class SyntheticProjection:
    def __init__(self, *originals):
        self.originals = originals
        self.packed_nbytes = 1
        self.project = Mock(side_effect=self._project)

    def _project(self, x, mode):
        return tuple(np.full((1, 1, width), label, dtype="float16")
                     for width, label in [(896, 1), (128, 2), (128, 3)])


def fake_backend():
    # No MLX import and no emulation of quantized arithmetic. Unexpected use of
    # a real matmul path is an error in these control-flow/ownership fixtures.
    mx = types.SimpleNamespace(
        array=np.ndarray, all=np.all, isfinite=np.isfinite,
        gpu="gpu", default_device=lambda: "gpu",
        concatenate=Mock(side_effect=np.concatenate), eval=Mock(),
        compile=Mock(side_effect=lambda function, **kwargs: function),
        quantized_matmul=Mock(side_effect=AssertionError("no arithmetic in fixture")),
    )
    nn = types.SimpleNamespace(Module=FakeModule, QuantizedLinear=FakeQuantizedLinear)
    sdpa = Mock(side_effect=lambda q, k, v, **kwargs: q)
    return mx, nn, FakeModel, FakeAttention, FakeCache, sdpa


@unittest.skipIf(np is None, "NumPy needed for synthetic CPU array fixtures")
class PackingOwnershipTests(unittest.TestCase):
    def test_concatenation_preserves_all_rows_and_does_not_alias_originals(self):
        original = [FakeQuantizedLinear(width, label)
                    for width, label in [(896, 11), (128, 22), (128, 33)]]
        snapshots = [{key: getattr(p, key).copy()
                      for key in ("weight", "scales", "biases", "bias")}
                     for p in original]
        ids = [{key: id(getattr(p, key)) for key in snapshot}
               for p, snapshot in zip(original, snapshots)]
        backend = fake_backend()
        with patch.object(projection, "_backend", return_value=backend):
            packed = projection.PackedQKV(*original)
        self.assertEqual(packed.originals, tuple(original))
        # 1152 rows x (224 uint32 + 2 x 14 FP16 + 1 FP16).
        self.assertEqual(packed.packed_nbytes, 1_099_008)
        for key in snapshots[0]:
            start = 0
            for item, snapshot, identity in zip(original, snapshots, ids):
                end = start + snapshot[key].shape[0]
                np.testing.assert_array_equal(getattr(packed, key)[start:end], snapshot[key])
                np.testing.assert_array_equal(getattr(item, key), snapshot[key])
                self.assertEqual(id(getattr(item, key)), identity[key])
                self.assertFalse(np.shares_memory(getattr(packed, key), getattr(item, key)))
                start = end
        backend[0].quantized_matmul.assert_not_called()

    def test_nonfinite_parameters_rejected_before_packing(self):
        for field, value in [("scales", float("nan")), ("biases", float("inf")),
                             ("bias", -float("inf"))]:
            originals = [FakeQuantizedLinear(width, label)
                         for width, label in [(896, 11), (128, 22), (128, 33)]]
            getattr(originals[2], field).flat[-1] = value
            backend = fake_backend()
            with self.subTest(field=field), patch.object(
                    projection, "_backend", return_value=backend):
                with self.assertRaises(ValueError):
                    projection.PackedQKV(*originals)
            backend[0].concatenate.assert_not_called()
            backend[0].compile.assert_not_called()

    def test_bad_metadata_and_derived_modules_rejected_before_array_operations(self):
        class Derived(FakeQuantizedLinear):
            pass
        for bad in (FakeQuantizedLinear(127, 33), Derived(128, 33)):
            backend = fake_backend()
            with patch.object(projection, "_backend", return_value=backend):
                with self.assertRaises(ValueError):
                    projection.PackedQKV(FakeQuantizedLinear(896, 11),
                                         FakeQuantizedLinear(128, 22), bad)
            backend[0].eval.assert_not_called()
            backend[0].concatenate.assert_not_called()

    def test_project_rejection_has_no_dispatch_or_parameter_scan(self):
        backend = fake_backend()
        with patch.object(projection, "_backend", return_value=backend):
            packed = projection.PackedQKV(*[
                FakeQuantizedLinear(width, label)
                for width, label in [(896, 11), (128, 22), (128, 33)]
            ])
            for name in ("_native", "_compiled_three", "_packed"):
                setattr(packed, name, Mock())
            backend[0].eval.reset_mock()
            for x, mode in [(None, "native"), (np.zeros((1, 1, 896), "float32"), "packed"),
                            (np.zeros((1, 2, 896), "float16"), "compiled_three"),
                            (np.zeros((1, 1, 896), "float16"), "autotune")]:
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    packed.project(x, mode)
            for name in ("_native", "_compiled_three", "_packed"):
                getattr(packed, name).assert_not_called()
            backend[0].eval.assert_not_called()


@unittest.skipIf(np is None, "NumPy needed for synthetic CPU array fixtures")
class AdapterSafetyTests(unittest.TestCase):
    def setUp(self):
        self.backend = fake_backend()
        backend_patch = patch.object(projection, "_backend", return_value=self.backend)
        backend_patch.start()
        self.addCleanup(backend_patch.stop)
        packing_patch = patch.object(projection, "PackedQKV", SyntheticProjection)
        packing_patch.start()
        self.addCleanup(packing_patch.stop)
        self.model = FakeModel()
        self.originals = tuple(layer.self_attn for layer in self.model.model.layers)
        self.prepared = projection.prepare_qkv(self.model)
        self.x = np.zeros((1, 1, 896), "float16")

    def assert_restored(self):
        self.assertFalse(self.prepared._active)
        self.assertTrue(all(layer.self_attn is old for layer, old in
                            zip(self.model.model.layers, self.originals)))

    def test_observer_failure_restores_exact_objects_before_cache_changes(self):
        cache = FakeCache()
        counts = {}
        observer = Mock(side_effect=RuntimeError("observer failure"))
        with self.assertRaisesRegex(RuntimeError, "observer failure"):
            with projection.qkv_decode_adapter(self.model, self.prepared, "packed",
                                                counts=counts, observer=observer):
                self.model.model.layers[0].self_attn(self.x, cache=cache)
        self.assert_restored()
        self.assertEqual(cache.updates, 0)
        self.assertEqual(cache.offset, 0)
        self.assertEqual(counts, {})
        self.assertEqual(observer.call_args.args[0], 0)
        self.assertIs(observer.call_args.args[1], self.x)
        self.assertIs(observer.call_args.args[-1], cache)
        self.originals[0].rope.assert_not_called()
        self.backend[-1].assert_not_called()

    def test_nested_adapter_and_prepare_rejected_without_disturbing_outer(self):
        for outer in projection.ADAPTER_MODES:
            with projection.qkv_decode_adapter(self.model, self.prepared, outer):
                installed = tuple(layer.self_attn for layer in self.model.model.layers)
                for inner in projection.ADAPTER_MODES:
                    with self.assertRaises(ValueError):
                        with projection.qkv_decode_adapter(self.model, self.prepared, inner):
                            self.fail("nested context must not be entered")
                with self.assertRaises(ValueError):
                    projection.prepare_qkv(self.model)
                self.assertTrue(all(layer.self_attn is item for layer, item in
                                    zip(self.model.model.layers, installed)))
            self.assert_restored()

    def test_bad_cache_offset_mask_and_decode_shape_never_dispatch_or_mutate(self):
        class DerivedCache(FakeCache):
            pass
        cases = [(self.x, None, FakeCache(offset))
                 for offset in (-1, 8192, True, 1.5)]
        cases += [(self.x, object(), FakeCache()),
                  (self.x, None, DerivedCache()),
                  (np.zeros((2, 1, 896), "float16"), None, FakeCache()),
                  (np.zeros((1, 1, 895), "float16"), None, FakeCache()),
                  (np.zeros((1, 1, 896), "float32"), None, FakeCache())]
        with projection.qkv_decode_adapter(self.model, self.prepared, "packed"):
            for x, mask, cache in cases:
                with self.subTest(shape=x.shape, offset=cache.offset), self.assertRaises(ValueError):
                    self.model.model.layers[0].self_attn(x, mask, cache)
                self.assertEqual(cache.updates, 0)
        self.prepared.projections[0].project.assert_not_called()
        self.assert_restored()

    def test_unsupported_context_arguments_leave_model_unchanged(self):
        for kwargs in ({"mode": "automatic"}, {"counts": []}, {"observer": object()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                with projection.qkv_decode_adapter(self.model, self.prepared, **kwargs):
                    self.fail("invalid context arguments must not yield")
            self.assert_restored()

    def test_wrong_architecture_is_rejected_before_any_projection_preparation(self):
        for mutation in ("hidden_width", "layer_count", "head_count", "layer_type"):
            model = FakeModel()
            if mutation == "hidden_width":
                model.args.hidden_size = 1024
            elif mutation == "layer_count":
                model.model.layers.pop()
            elif mutation == "head_count":
                model.model.layers[-1].self_attn.n_kv_heads = 1
            else:
                model.model.layers[-1].self_attn = object()
            original_ids = [id(layer.self_attn) for layer in model.model.layers]
            with self.subTest(mutation=mutation), patch.object(projection, "PackedQKV") as factory:
                with self.assertRaises(ValueError):
                    projection.prepare_qkv(model)
                factory.assert_not_called()
            self.assertEqual([id(layer.self_attn) for layer in model.model.layers], original_ids)

    def test_prefill_delegates_exact_arguments_to_original(self):
        x = np.zeros((1, 3, 896), "float16")
        mask, cache = object(), object()
        counts = {}
        with projection.qkv_decode_adapter(self.model, self.prepared, "packed", counts=counts):
            result = self.model.model.layers[2].self_attn(x, mask, cache)
        self.assertIs(result, self.originals[2].forward.return_value)
        args = self.originals[2].forward.call_args.args
        self.assertTrue(all(a is b for a, b in zip(args, (x, mask, cache))))
        self.assertEqual(counts, {"prefill": 1})
        self.prepared.projections[2].project.assert_not_called()
        self.assert_restored()

    def test_invalid_existing_cache_storage_rejected_before_projection(self):
        bad_caches = []
        for field in ("keys", "values"):
            cache = FakeCache()
            setattr(cache, field, np.zeros((1, 2, 1, 64), "float16"))
            bad_caches.append(cache)
        cache = FakeCache(2)
        cache.keys = np.zeros((1, 2, 1, 64), "float16")
        cache.values = np.zeros((1, 2, 1, 64), "float16")
        bad_caches.append(cache)
        for shape, dtype in [((1, 2, 2, 64), "float32"), ((1, 1, 2, 64), "float16"),
                             ((1, 2, 2, 63), "float16"), ((1, 2, 3, 64), "float16")]:
            cache = FakeCache(2)
            cache.values = np.zeros(shape, dtype)
            bad_caches.append(cache)
        with projection.qkv_decode_adapter(self.model, self.prepared, "packed"):
            for index, cache in enumerate(bad_caches):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    self.model.model.layers[0].self_attn(self.x, cache=cache)
                self.assertEqual(cache.updates, 0)
        self.prepared.projections[0].project.assert_not_called()
        self.assert_restored()

    def test_decode_keeps_rope_cache_sdpa_and_output_projection(self):
        counts, cache = {}, FakeCache(7)
        with projection.qkv_decode_adapter(self.model, self.prepared, "native_adapter", counts=counts):
            got = self.model.model.layers[0].self_attn(self.x, cache=cache)
        self.prepared.projections[0].project.assert_called_once_with(self.x, "native")
        self.assertEqual(got.shape, (1, 1, 896))
        self.assertEqual(cache.offset, 8)
        self.assertEqual(cache.updates, 1)
        self.assertEqual(counts, {"decode": 1})
        self.assertEqual([c.kwargs["offset"] for c in self.originals[0].rope.call_args_list], [7, 7])
        q, k, v = self.backend[-1].call_args.args
        self.assertEqual(q.shape, (1, 14, 1, 64))
        self.assertEqual(k.shape, (1, 2, 1, 64))
        self.assertEqual(v.shape, (1, 2, 1, 64))
        self.originals[0].o_proj.assert_called_once()
        self.assert_restored()

    def test_projection_and_cache_failures_restore_adapter(self):
        for stage in ("projection", "cache"):
            cache = FakeCache()
            counts = {}
            self.prepared.projections[0].project.side_effect = (
                RuntimeError("projection failure") if stage == "projection"
                else self.prepared.projections[0]._project
            )
            if stage == "cache":
                cache.failure = RuntimeError("cache failure")
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, stage):
                with projection.qkv_decode_adapter(self.model, self.prepared, "packed", counts=counts):
                    self.model.model.layers[0].self_attn(self.x, cache=cache)
            self.assertEqual(cache.updates, 0 if stage == "projection" else 1)
            self.assertEqual(counts, {})
            self.assert_restored()

    def test_setup_failure_halfway_restores_already_replaced_layers(self):
        class FailOnceLayer:
            def __init__(self, original):
                self._value, self.failed = original, False

            @property
            def self_attn(self):
                return self._value

            @self_attn.setter
            def self_attn(self, value):
                if type(value) is not FakeAttention and not self.failed:
                    self.failed = True
                    raise RuntimeError("layer installation failed")
                self._value = value

        self.model.model.layers[12] = FailOnceLayer(self.originals[12])
        with self.assertRaisesRegex(RuntimeError, "installation failed"):
            with projection.qkv_decode_adapter(self.model, self.prepared, "packed"):
                self.fail("failed installation must not yield")
        self.assert_restored()

    def test_wrong_model_or_changed_attention_rejected_before_installation(self):
        other = FakeModel()
        with self.assertRaises(ValueError):
            with projection.qkv_decode_adapter(other, self.prepared, "packed"):
                self.fail("wrong model must not yield")
        self.assert_restored()
        replacement = FakeAttention()
        self.model.model.layers[-1].self_attn = replacement
        with self.assertRaises(ValueError):
            with projection.qkv_decode_adapter(self.model, self.prepared, "packed"):
                self.fail("changed model must not yield")
        self.assertIs(self.model.model.layers[-1].self_attn, replacement)
        self.assertTrue(all(layer.self_attn is old for layer, old in
                            zip(self.model.model.layers[:-1], self.originals[:-1])))
        self.assertFalse(self.prepared._active)


@unittest.skipUnless(os.environ.get("RUN_QKV_PROJECTION_TESTS") == "1",
                     "set RUN_QKV_PROJECTION_TESTS=1 for synthetic Metal algebra tests")
class ProjectionMetalAlgebraTests(unittest.TestCase):
    def test_fixed_projection_matches_dequantized_float32_reference(self):
        import mlx.core as mx
        import mlx.nn as nn
        if not mx.metal.is_available():
            raise RuntimeError("Explicit QKV tests require an available Metal GPU")
        previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)
        try:
            mx.random.seed(20261009)
            originals = [nn.QuantizedLinear(896, width, bias=True, group_size=64, bits=8)
                         for width in (896, 128, 128)]
            for original in originals:
                original.set_dtype(mx.float16)
            packed = projection.PackedQKV(*originals)
            x = mx.random.normal((1, 1, 896)).astype(mx.float16)
            references = []
            for p in originals:
                weight = mx.dequantize(p.weight, p.scales, p.biases,
                                       group_size=64, bits=8, mode="affine").astype(mx.float32)
                references.append(x.astype(mx.float32) @ weight.T + p.bias.astype(mx.float32))
            mx.eval(*references)
            for mode in projection.PROJECTION_MODES:
                actual = packed.project(x, mode)
                mx.eval(*actual)
                self.assertEqual(tuple(a.shape for a in actual),
                                 ((1, 1, 896), (1, 1, 128), (1, 1, 128)))
                for got, expected in zip(actual, references):
                    self.assertTrue(bool(mx.all(mx.isfinite(got)).item()))
                    self.assertTrue(bool(mx.allclose(got.astype(mx.float32), expected,
                                                    atol=0.005, rtol=0.005).item()))
        finally:
            mx.set_default_device(previous_device)


if __name__ == "__main__":
    unittest.main()
