import importlib.util
import tempfile
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'),'experiment dependencies unavailable')
class ShortRequestSelectionTests(unittest.TestCase):
    def test_embedded_padding_and_truncation_do_not_change_eligibility(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        from experiments.minilm_short_request_check import short_pair_indices
        tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'a':1},unk_token='[UNK]'))
        tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer.enable_truncation(max_length=2)
        tokenizer.enable_padding(length=128)
        rows=[{'sentence1':'a a a','sentence2':'a'},{'sentence1':'a','sentence2':'a'}]
        self.assertEqual(short_pair_indices(rows,tokenizer,2),[1])

    def test_nonfinite_results_are_rejected_before_file_creation(self):
        from experiments.minilm_runtime_study import save
        with tempfile.TemporaryDirectory() as root:
            dest=Path(root)/'bad.json'
            with self.assertRaises(ValueError):save(dest,{'spearman':float('nan')})
            self.assertFalse(dest.exists())
