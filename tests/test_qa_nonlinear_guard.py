import unittest

class NonlinearGuardTests(unittest.TestCase):
    def test_failed_candidates_block_before_loading_private_heads(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from experiments import qa_nonlinear,qa_rich
        from lab.quantization_diagnostics import sha
        for module,folder in [(qa_nonlinear,'qa-nonlinear-v2'),(qa_rich,'qa-rich-v1')]:
            training=module.ROOT/'results'/folder/'training'
            with TemporaryDirectory() as temp,patch.object(module,'preflight',return_value={}):
                output=Path(temp)/'evaluation'
                with self.assertRaisesRegex(ValueError,'Eligible frozen selection'):
                    module.evaluate(output,training,Path(temp)/'missing.pkl',Path(temp)/'missing-assets',sha(training/'selection.json'))
                self.assertFalse(output.exists())
