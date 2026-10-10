"""Failure preservation and receipt binding; no GPU or model required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import qwen_demand_prefill as runner


class DemandRunnerTests(unittest.TestCase):
    def test_existing_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'old.json').write_text('original')
            with self.assertRaises(FileExistsError):
                runner.reserve(target)
            self.assertEqual((target / 'old.json').read_text(), 'original')

    def test_nonfinite_save_keeps_previous_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            runner.save(path, {'status': 'running', 'records': [1]})
            with self.assertRaises(ValueError):
                runner.save(path, {'records': [float('nan')]})
            self.assertEqual(json.loads(path.read_text())['records'], [1])

    def test_receipt_rejects_changes_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.save(root / 'run.json', {'status': 'complete'})
            runner.seal(root)
            (root / 'extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'identities'):
                runner.checked_receipt(root)

    def test_frozen_model_identity_rejected_before_mlx_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'model/tokenizer/config'):
                runner.setup(root, root)


if __name__ == '__main__':
    unittest.main()
