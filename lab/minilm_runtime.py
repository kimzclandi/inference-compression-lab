"""Reusable CPU sentence embedding runtime; one instance per calling thread."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from lab.evidence import sha256


def pool(hidden, mask):
    """[B,S,H] FP32 + [B,S] int64 -> [B,H] normalized FP64, historical parity."""
    expanded = mask[..., None]
    pooled = (hidden * expanded).sum(1) / np.maximum(expanded.sum(1), 1)
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


class MiniLMRuntime:
    def __init__(self, model_path, tokenizer_path='models/minilm/tokenizer.json', *, threads=4,
                 max_length=256, fixed_padding=False, expected_sha256=None, profile_prefix=None):
        if threads < 1 or max_length < 1:
            raise ValueError('threads and max_length must be positive')
        model_path=Path(model_path)
        if expected_sha256 and sha256(model_path) != expected_sha256:
            raise ValueError('Model hash mismatch')
        self.tokenizer=Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(length=max_length if fixed_padding else None,
                                      pad_id=0,pad_token='[PAD]')
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if profile_prefix:
            options.enable_profiling=True
            options.profile_file_prefix=str(profile_prefix)
        self.session=ort.InferenceSession(str(model_path),options,providers=['CPUExecutionProvider'])
        if self.session.get_providers()!=['CPUExecutionProvider']:
            raise RuntimeError('Unexpected execution provider')
        self.output_name=self.session.get_outputs()[0].name

    def tokenize(self,texts):
        if not texts or not all(isinstance(x,str) for x in texts):
            raise ValueError('A nonempty list of strings is required')
        batch=self.tokenizer.encode_batch(texts)
        return {name:np.array([getattr(x,attr) for x in batch],dtype=np.int64)
                for name,attr in [('input_ids','ids'),('attention_mask','attention_mask'),('token_type_ids','type_ids')]}

    def run_tokens(self,feed):
        return self.session.run([self.output_name],feed)[0]

    def encode(self,texts):
        feed=self.tokenize(texts)
        return pool(self.run_tokens(feed),feed['attention_mask'])
