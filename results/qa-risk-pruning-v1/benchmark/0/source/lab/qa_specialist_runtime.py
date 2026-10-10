"""Bounded offline ORT extractive QA runtime; no labels enter prediction."""
import importlib.metadata
import os
from pathlib import Path
import time

from lab.extractive_qa import decode
from lab.quantization_diagnostics import read, sha

RUNTIME_PACKAGES = ('onnxruntime', 'numpy', 'transformers', 'tokenizers')
LIMITS = dict(max_sequence_tokens=384, stride=128, max_question_tokens=64,
              max_context_characters=12000, max_windows=8, max_answer_tokens=30)


def validate_assets(root, expected_manifest_sha256):
    root = Path(root)
    if sha(root/'manifest.json') != expected_manifest_sha256:
        raise ValueError('Asset manifest differs from the frozen protocol')
    manifest = read(root/'manifest.json')
    files = {**manifest['tokenizer_files'], **{r['file']:r['sha256'] for r in manifest['models'].values()}}
    if set(manifest['models']) != {'fp32', 'int8'}:
        raise ValueError('Missing model precision in asset identity')
    for name,digest in files.items():
        if Path(name).name != name or not name or sha(root/name) != digest:
            raise ValueError('Asset file identity mismatch: '+str(name))
    expected = set(files) | {'manifest.json', 'UPSTREAM-README.md'}
    if {p.name for p in root.iterdir()} != expected:
        raise ValueError('Unexpected/missing local asset files')
    if sha(root/'UPSTREAM-README.md') != manifest['source_files']['README.md']['sha256']:
        raise ValueError('Upstream attribution changed')
    current = {name:importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if current != {name:manifest['packages'][name] for name in RUNTIME_PACKAGES}:
        raise ValueError('Runtime dependency versions differ from the prepared assets')
    return manifest


class ExtractiveRuntime:
    def __init__(self, asset_root, variant, manifest_sha256, threads=4):
        if variant not in ('fp32','int8') or type(threads) is not int or threads != 4:
            raise ValueError('Fixed runtime is FP32/INT8 with four CPU threads')
        self.root=Path(asset_root); self.manifest=validate_assets(self.root,manifest_sha256)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError('Fast tokenizer with exact offsets is required')
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(self.root/self.manifest['models'][variant]['file']),
                                          sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_providers()!=['CPUExecutionProvider']:
            raise ValueError('Unexpected execution provider')
        self.variant=variant

    def predict(self, context, question, *, include_logits=True):
        start=time.perf_counter()
        if not isinstance(context,str) or not context.strip() or len(context)>LIMITS['max_context_characters']:
            raise ValueError('Context must be nonempty and within the fixed character limit')
        if not isinstance(question,str) or not question.strip() or len(question)>1000:
            raise ValueError('Question must be nonempty and within the fixed character limit')
        tok=self.tokenizer
        if len(tok.encode(question,add_special_tokens=False))>LIMITS['max_question_tokens']:
            raise ValueError('Question exceeds the frozen token limit; no silent truncation')
        encoded=tok(question,context,truncation='only_second',max_length=LIMITS['max_sequence_tokens'],
                    stride=LIMITS['stride'],return_overflowing_tokens=True,return_offsets_mapping=True,
                    padding=False)
        if not 1<=len(encoded['input_ids'])<=LIMITS['max_windows']:
            raise ValueError('Context exceeds the bounded window count; no silent dropping')
        if any(len(encoded[name])!=len(encoded['input_ids']) for name in ('attention_mask','offset_mapping')):
            raise ValueError('Tokenizer returned inconsistent window counts')
        tokenized=time.perf_counter();windows=[];inference_seconds=0
        import numpy as np
        for index,ids in enumerate(encoded['input_ids']):
            attention=encoded['attention_mask'][index]
            sequence_ids=encoded.sequence_ids(index)
            offsets=[list(p) for p in encoded['offset_mapping'][index]]
            if (not ids or len(ids)>LIMITS['max_sequence_tokens'] or
                    any(type(value) is not int or value<0 for value in ids) or
                    ids[0]!=tok.cls_token_id):
                raise ValueError('Token IDs/CLS/feature length violate the frozen contract')
            if not len(ids)==len(attention)==len(sequence_ids)==len(offsets):
                raise ValueError('Token/mask/offset/sequence identity length mismatch')
            if any(type(value) is not int or value not in (0,1) for value in attention):
                raise ValueError('Attention values must be integer zero or one')
            if any(value is not None and (type(value) is not int or value not in (0,1)) for value in sequence_ids):
                raise ValueError('Sequence IDs must be None, question zero or context one')
            mask=[sid==1 and attention[k]==1 for k,sid in enumerate(sequence_ids)]
            features={'input_ids':np.asarray([ids],dtype=np.int64),
                      'attention_mask':np.asarray([attention],dtype=np.int64)}
            before=time.perf_counter()
            logits=self.session.run(['start_logits','end_logits'],features)
            inference_seconds+=time.perf_counter()-before
            if any(x.shape!=(1,len(ids)) for x in logits):
                raise ValueError('QA output tensor shape differs from input tokens')
            windows.append(dict(start_logits=logits[0][0].astype(float).tolist(),
                                end_logits=logits[1][0].astype(float).tolist(),
                                offsets=offsets,context_mask=mask,cls_index=0,
                                input_ids=ids,attention_mask=attention,sequence_ids=sequence_ids))
        before_decode=time.perf_counter()
        answer=decode(context,windows,LIMITS['max_answer_tokens'])
        answer.update(tokenization_seconds=tokenized-start,inference_seconds=inference_seconds,
                      decode_seconds=time.perf_counter()-before_decode,
                      total_pipeline_seconds=time.perf_counter()-start,
                      feature_count=len(windows),input_tokens=[len(w['input_ids']) for w in windows])
        if include_logits:answer['raw_windows']=windows
        return answer
