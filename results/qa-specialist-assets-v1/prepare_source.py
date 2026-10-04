"""Prepare local FP32/dynamic INT8 QA assets from one pinned public source model.

Weights and ONNX graphs stay in a caller-selected local directory. Evidence
contains identities, export parity on synthetic inputs, graph counts and logs.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes, git_identity
from lab.quantization_diagnostics import read, write, sha

PACKAGES = ('torch', 'transformers', 'onnx', 'onnxruntime', 'numpy', 'safetensors', 'tokenizers')
TOKENIZER_FILES = ('config.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json')


def prepare(args):
    evidence = reserve_directory(args.evidence_dir)
    state = dict(status='running', stage='preflight', python=platform.python_version(), **git_identity(Path.cwd()))
    try:
        reference = read(args.identity)
        actual = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                  for p in sorted(args.source.iterdir()) if p.is_file()}
        expected = {name: {k: info[k] for k in ('bytes', 'sha256')} for name, info in reference['files'].items()}
        if actual != expected or 'model.safetensors' not in actual:
            raise ValueError('Source files differ from the pinned upstream identity')
        assets = reserve_directory(args.output_dir)
        state.update(upstream=reference, source_files=actual, identity_sha256=sha(args.identity),
                     packages={p: importlib.metadata.version(p) for p in PACKAGES})
        shutil.copy2(__file__, evidence/'prepare_source.py')
        state['prepare_source_sha256'] = sha(evidence/'prepare_source.py')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        import numpy as np
        import torch
        import onnx
        import onnxruntime as ort
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        from onnxruntime.quantization import quantize_dynamic, QuantType
        torch.set_num_threads(4)
        model = AutoModelForQuestionAnswering.from_pretrained(args.source, local_files_only=True,
                    use_safetensors=True, trust_remote_code=False, attn_implementation='eager').eval().float()
        tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True, use_fast=True)
        class QA(torch.nn.Module):
            def __init__(self, model):
                super().__init__(); self.model = model
            def forward(self, input_ids, attention_mask):
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return output.start_logits, output.end_logits
        wrapped = QA(model).eval()
        inputs = tok('When did the library open?', 'The library opened in 1982.', return_tensors='pt')
        state['stage'] = 'export'
        fp32 = assets/'fp32.onnx'
        with torch.no_grad():
            torch.onnx.export(wrapped, (inputs['input_ids'], inputs['attention_mask']), fp32,
                input_names=['input_ids', 'attention_mask'], output_names=['start_logits', 'end_logits'],
                dynamic_axes={key: {0: 'batch', 1: 'sequence'} for key in
                              ('input_ids', 'attention_mask', 'start_logits', 'end_logits')},
                opset_version=17, dynamo=False, external_data=False)
        onnx.checker.check_model(str(fp32))
        graph = onnx.load(str(fp32))
        # Fixed classifier-head exclusion, chosen before quality evaluation.
        excluded = [n.name for n in graph.graph.node if 'qa_outputs' in n.name and n.op_type in ('MatMul', 'Gemm')]
        if len(excluded) != 1:
            raise ValueError('Expected exactly one floating QA projection node')
        state['stage'] = 'quantize'
        int8 = assets/'int8.onnx'
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True,
                         reduce_range=False, op_types_to_quantize=['MatMul', 'Gemm'],
                         nodes_to_exclude=excluded, extra_options={'MatMulConstBOnly': True})
        onnx.checker.check_model(str(int8))
        quant_graph = onnx.load(str(int8))
        counts = {name: dict(Counter(n.op_type for n in g.graph.node))
                  for name, g in [('fp32', graph), ('int8', quant_graph)]}
        if counts['int8'].get('MatMulInteger', 0) == 0:
            raise ValueError('No real integer MatMul nodes in INT8 graph')
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(fp32), sess_options=opts, providers=['CPUExecutionProvider'])
        parity = []
        synthetic = [
            ('When did the library open?', 'The Arbor Library opened in 1982. It contains 4,000 books.'),
            ('Who won the gold medal?', 'Mara won silver in 2010. Leon won gold.'),
            ('What was approved?', 'The committee reviewed a request. '*50 + 'The committee approved a new bridge.'),
        ]
        for question, context in synthetic:
            encoded = tok(question, context, return_tensors='pt')
            with torch.no_grad(): expected_logits = wrapped(encoded['input_ids'], encoded['attention_mask'])
            observed = session.run(None, {k: encoded[k].numpy() for k in ('input_ids', 'attention_mask')})
            errors = [float(np.max(np.abs(a.numpy()-b))) for a, b in zip(expected_logits, observed)]
            parity.append(dict(question=question, context=context, shape=list(encoded['input_ids'].shape),
                               max_abs_logit_errors=errors, tolerance=1e-3, passed=max(errors)<=1e-3))
        if not all(p['passed'] for p in parity):
            raise ValueError('FP32 ONNX export differs from original task model')
        for name in TOKENIZER_FILES: shutil.copy2(args.source/name, assets/name)
        shutil.copy2(args.source/'README.md', assets/'UPSTREAM-README.md')
        metadata = dict(model_id=reference['model'], revision=reference['revision'], source_files=actual,
                        python=platform.python_version(), packages=state['packages'],
                        tokenizer_files={name:sha(assets/name) for name in TOKENIZER_FILES},
                        models={name:dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size)
                                for name,path in [('fp32',fp32),('int8',int8)]},
                        quantization=dict(method='ONNX Runtime dynamic activation UINT8 / weight INT8',
                            weight_type='QInt8',per_channel=True,reduce_range=False,
                            MatMulConstBOnly=True,excluded_nodes=excluded,
                            exclusion_rule='fixed QA projection head remains FP32; no data-driven layer search'),
                        graph_operator_counts=counts, fp32_export_parity=parity,
                        scope='File bytes are not peak runtime memory; graph integer operators are not speedup evidence')
        write(assets/'manifest.json', metadata)
        shutil.copy2(assets/'manifest.json', evidence/'assets.json')
        state.update(status='complete', stage='complete', manifest_sha256=sha(assets/'manifest.json'))
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=repr(exc)); raise
    finally:
        write(evidence/'run.json', state); write(evidence/'checksums.json', file_hashes(evidence))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True)
    prepare(p.parse_args())


if __name__ == '__main__': main()
