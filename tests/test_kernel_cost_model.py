import json
from pathlib import Path
import tempfile
import unittest

from experiments import analyze_metal_kernel_cost as cli
from lab.kernel_cost_model import analyze_fixed_summary, residual_rmsnorm_cost


class CostModelTests(unittest.TestCase):
    def test_fp16_logical_traffic_and_ceiling(self):
        result = residual_rmsnorm_cost(2, 8, "float16")
        self.assertEqual(result["native_logical_bytes"], 192)
        self.assertEqual(result["fused_logical_bytes"], 160)
        self.assertAlmostEqual(result["logical_byte_reduction_fraction"], 1 / 6)
        self.assertAlmostEqual(result["traffic_only_speedup_ceiling"], 1.2)

    def test_invalid_inputs(self):
        for args in ((0, 8, "float16"), (1, -1, "float16"), (1, 8, "int8")):
            with self.assertRaises(ValueError):
                residual_rmsnorm_cost(*args)

    def test_fixed_negative_summary_is_retained(self):
        summary = json.loads(cli.DEFAULT_SUMMARY.read_text())
        result = analyze_fixed_summary(summary)
        self.assertFalse(result["changes_fixed_acceptance"])
        self.assertFalse(result["fixed_candidate_accepted"])
        self.assertEqual([r["rows"] for r in result["records"]], [1, 64, 512, 2048])
        primary = result["records"][-1]
        self.assertAlmostEqual(primary["observed_speedup_compiled_over_metal"], 0.8075299826912105)

    def test_rejects_changed_acceptance(self):
        summary = json.loads(cli.DEFAULT_SUMMARY.read_text())
        summary["acceptance"]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "rejected-candidate"):
            analyze_fixed_summary(summary)

    def test_cli_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cost.json"
            cli.main(["--output", str(output)])
            with self.assertRaises(FileExistsError):
                cli.main(["--output", str(output)])
            cli.main(["--output", str(output), "--force"])
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "post_hoc_diagnostic_only")


if __name__ == "__main__":
    unittest.main()
