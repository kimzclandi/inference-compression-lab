"""Contract checks; actual Metal execution requires RUN_METAL_TESTS=1."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from lab import metal_residual_rmsnorm as fusion


class ResidualMetadataTests(unittest.TestCase):
    def check(self, **updates):
        args = dict(x_shape=(2, 896), residual_shape=(2, 896), weight_shape=(896,),
                    x_dtype="float16", residual_dtype="float16", weight_dtype="float16",
                    eps=1e-6, mode="metal")
        args.update(updates)
        return fusion.validate_metadata(**args)

    def test_known_fixed_launch_and_single_row(self):
        self.assertEqual(self.check()["threads"], 224)
        self.assertEqual(self.check()["rows"], 2)
        self.assertEqual(self.check(x_shape=(896,), residual_shape=(896,))["rows"], 1)
        for width, threads in [(1, 32), (64, 32), (128, 32), (129, 64), (4096, 1024)]:
            self.assertEqual(self.check(x_shape=(width,), residual_shape=(width,),
                                        weight_shape=(width,))["threads"], threads)

    def test_shape_errors_fail_closed(self):
        for update in [dict(x_shape=()), dict(x_shape=(0, 896)), dict(x_shape=(True, 896)),
                       dict(residual_shape=(1, 896)), dict(weight_shape=(1, 896)),
                       dict(x_shape=(2, 4097), residual_shape=(2, 4097), weight_shape=(4097,)),
                       dict(x_shape=(2**32, 896), residual_shape=(2**32, 896))]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.check(**update)

    def test_dtype_and_mode_rejections(self):
        for update in [dict(x_dtype="bfloat16"), dict(residual_dtype="float32"),
                       dict(weight_dtype="float32"), dict(mode="autotuned")]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.check(**update)

    def test_epsilon_overflow_underflow_and_nonfinite(self):
        for eps in [0, -1, True, "1e-6", float("nan"), float("inf"), 1e300, 1e-100]:
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                self.check(eps=eps)

    def test_no_mlx_import_on_module_import(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import lab.metal_residual_rmsnorm; "
                                 "assert 'mlx.core' not in sys.modules"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_unknown_mode_before_backend_or_cache_mutation(self):
        with patch.object(fusion, "_backend", side_effect=AssertionError("unexpected backend")):
            with self.assertRaises(ValueError):
                fusion.residual_rmsnorm(None, None, None, 1e-6, "unknown")
            with self.assertRaises(ValueError):
                fusion.qwen_forward(None, None, [], "unknown")
            with self.assertRaises(ValueError):
                fusion.qwen_forward(None, None, [], "native", trace=[])


@unittest.skipUnless(os.environ.get("RUN_METAL_TESTS") == "1", "set RUN_METAL_TESTS=1 for real GPU tests")
class ResidualMetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Explicit Metal tests require an available Metal GPU")
        mx.set_default_device(mx.gpu)
        cls.mx = mx

    def test_real_gpu_rounding_boundaries_and_compiled_pair(self):
        mx = self.mx
        mx.random.seed(20261005)
        for dtype in (mx.float16, mx.float32):
            for width in (1, 64, 129, 896, 4096):
                with self.subTest(dtype=dtype, width=width):
                    x = mx.random.normal((2, width)).astype(dtype)
                    r = mx.random.normal((2, width)).astype(dtype)
                    w = mx.random.uniform(low=0.5, high=1.5, shape=(width,)).astype(dtype)
                    reference = fusion.residual_rmsnorm(x, r, w, 1e-6, "native")
                    for mode in ("compiled", "metal"):
                        got = fusion.residual_rmsnorm(x, r, w, 1e-6, mode)
                        mx.eval(reference, got)
                        self.assertTrue(bool(mx.array_equal(reference[0], got[0]).item()))
                        atol, rtol = ((.002, .002) if dtype == mx.float16 else (2e-6, 1e-5))
                        self.assertTrue(bool(mx.allclose(reference[1], got[1], atol=atol, rtol=rtol).item()))

    def test_strided_and_cancellation_inputs(self):
        mx = self.mx
        x = mx.arange(4 * 896, dtype=mx.float32).reshape(2, 1792)[:, ::2].astype(mx.float16)
        w = mx.linspace(.5, 1.5, 1792).astype(mx.float16)[::2]
        for r in (-x, mx.zeros_like(x)):
            native = fusion.residual_rmsnorm(x, r, w, 1e-6, "native")
            metal = fusion.residual_rmsnorm(x, r, w, 1e-6, "metal")
            mx.eval(native, metal)
            self.assertTrue(bool(mx.array_equal(native[0], metal[0]).item()))
            self.assertTrue(bool(mx.allclose(native[1], metal[1], atol=.002, rtol=.002).item()))

    def test_gpu_required_and_invalid_array_types(self):
        mx = self.mx
        with self.assertRaises(ValueError):
            fusion.residual_rmsnorm([], [], [], 1e-6)
        mx.set_default_device(mx.cpu)
        try:
            with self.assertRaises(RuntimeError):
                fusion.residual_rmsnorm(None, None, None, 1e-6)
        finally:
            mx.set_default_device(mx.gpu)


if __name__ == "__main__":
    unittest.main()
