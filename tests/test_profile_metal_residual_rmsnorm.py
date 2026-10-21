import unittest
from unittest.mock import patch

from experiments import profile_metal_residual_rmsnorm as profile


class ProfileWorkloadTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(
            profile.validate(2048, 896, 5, 1000, 2)["batches"],
            1000,
        )
        for values in (
            (0, 896, 5, 1000, 50),
            (1, 4097, 5, 1000, 50),
            (100000, 896, 5, 1000, 50),
            (1, 896, 1, 40001, 50),
        ):
            with self.assertRaises(ValueError):
                profile.validate(*values)

    def test_unknown_mode_rejected_before_mlx_import(self):
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            with self.assertRaisesRegex(ValueError, "mode"):
                profile.run("wrong", 1, 1, 1, 1, 1)

    def test_startup_delay_validation_precedes_mlx_import(self):
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            with self.assertRaisesRegex(ValueError, "startup_delay"):
                profile.run("native", 1, 1, 1, 1, 1, startup_delay=61)


if __name__ == "__main__":
    unittest.main()
