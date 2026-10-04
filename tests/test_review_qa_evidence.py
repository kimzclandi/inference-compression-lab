import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import review_qa_evidence as review


class ReviewIntegrityTests(unittest.TestCase):
    def test_invalid_evidence_never_produces_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch.object(review, 'verify_evidence', side_effect=ValueError('tampered evidence')):
                with self.assertRaisesRegex(ValueError, 'tampered evidence'):
                    review.run(output)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')
            self.assertFalse((output / 'review.md').exists())
            self.assertFalse((output / 'acceptance.json').exists())

    def test_existing_directory_is_preserved_without_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'run.json').write_text('prior failure')
            with patch.object(review, 'verify_evidence') as verify:
                with self.assertRaises(FileExistsError):
                    review.run(output)
                verify.assert_not_called()
            self.assertEqual((output / 'run.json').read_text(), 'prior failure')

    def test_changed_input_during_verification_cannot_get_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            with patch.object(review, 'snapshot', side_effect=[{'source': 'before'}, {'source': 'after'}]), \
                 patch.object(review, 'verify_evidence', return_value={'technical_acceptance': 'pass'}), \
                 patch.object(review, 'compact', return_value={}):
                with self.assertRaisesRegex(ValueError, 'changed during verification'):
                    review.run(output)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')
            self.assertFalse((output / 'review.md').exists())


if __name__ == '__main__':
    unittest.main()
