import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experiments.verify_metal_trace_rows import EVIDENCE, verify


class PublicTraceRowsTests(unittest.TestCase):
    def test_historical_counts_recompute(self):
        result = verify()
        for arm in ("native", "metal"):
            self.assertEqual(result[arm]["gpu_compute_interval"]["count"], 5007)

    def test_modified_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rows"
            shutil.copytree(EVIDENCE, root)
            path = root / "native-gpu.csv"
            path.write_text(path.read_text() + "0,python (0),Compute,1,1\n")
            with self.assertRaisesRegex(ValueError, "row hash differs"):
                verify(root)

    def test_wrong_source_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rows"
            shutil.copytree(EVIDENCE, root)
            path = root / "manifest.json"
            data = json.loads(path.read_text())
            data["source_xml_sha256"]["native_gpu_xml"] = "0" * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source XML identity"):
                verify(root)
