import tempfile
from pathlib import Path
import unittest
from lab.evidence import reserve_directory


class RunSafetyTests(unittest.TestCase):
    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            dest = reserve_directory(Path(root) / 'result')
            evidence = dest / 'history.json'
            evidence.write_bytes(b'original bytes')
            with self.assertRaises(FileExistsError):
                reserve_directory(dest)
            self.assertEqual(evidence.read_bytes(), b'original bytes')

    def test_empty_existing_directory_is_also_protected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                reserve_directory(root)
