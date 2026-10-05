import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results/metal-residual-rmsnorm-v1/trace-diagnostic.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MetalTraceDiagnosticTests(unittest.TestCase):
    def test_receipt_is_diagnostic_and_preserves_rejection(self):
        receipt = json.loads(RECEIPT.read_text())
        self.assertEqual(receipt["status"], "post_hoc_diagnostic_only")
        self.assertFalse(receipt["changes_fixed_acceptance"])
        self.assertTrue(receipt["interpretation"]["matched_command_buffer_counts"])
        self.assertTrue(receipt["interpretation"]["matched_compute_interval_counts"])
        self.assertEqual(
            receipt["native"]["gpu_compute_interval"]["count"],
            receipt["metal"]["gpu_compute_interval"]["count"],
        )

    def test_receipt_binds_analysis_sources(self):
        receipt = json.loads(RECEIPT.read_text())
        for relative, expected in receipt["source_sha256"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)


if __name__ == "__main__":
    unittest.main()
