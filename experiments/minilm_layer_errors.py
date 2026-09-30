"""Observe cumulative and same-input local MatMul errors; never time debug graphs."""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from experiments.minilm_ptq import feeds, save
from lab.evidence import reserve_directory, source_record, sha256


def error_sums(reference, candidate, mask):
    """[B,S,H] arrays; exclude padding, accumulate in float64 across all tokens."""
    a = reference[mask.astype(bool)].astype(np.float64)
    b = candidate[mask.astype(bool)].astype(np.float64)
    d = a - b
    return {'n': a.size, 'sse': float(np.sum(d*d)), 'reference_energy': float(np.sum(a*a)),
            'candidate_energy': float(np.sum(b*b)), 'dot': float(np.sum(a*b)),
            'max_abs': float(np.max(np.abs(d)))}


def metrics(total):
    return {**total, 'mse': total['sse']/total['n'],
            'nmse': total['sse']/max(total['reference_energy'], 1e-30),
            'cosine': total['dot']/max(np.sqrt(total['reference_energy']*total['candidate_energy']), 1e-30)}


def debug_session(path, optimization):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimization == 'all'
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def instrument(source, dest, names):
    model = onnx.load(source)
    known = {v.name: v for v in [*model.graph.value_info, *model.graph.output, *model.graph.input]}
    outputs = {o.name for o in model.graph.output}
    produced = {o for node in model.graph.node for o in node.output}
    for name in names:
        if name not in produced:
            raise ValueError(f'Missing tensor: {name}')
        if name not in outputs:
            model.graph.output.append(deepcopy(known[name]) if name in known else
                helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', 'sequence', None]))
            outputs.add(name)
    onnx.checker.check_model(model)
    onnx.save(model, dest)


def main(args):
    out, work = reserve_directory(args.output_dir), reserve_directory(args.work_dir)
    fp_path = Path('models/minilm/onnx/model.onnx')
    fp = onnx.load(fp_path)
    weights = {w.name: w for w in fp.graph.initializer}
    nodes = [n for n in fp.graph.node if n.op_type == 'MatMul' and n.input[1] in weights]
    blocks = [n for n in fp.graph.node if n.name.endswith('/output/LayerNorm/Add_1')
              and '/attention/' not in n.name]
    assert len(nodes) == 36 and len(blocks) == 6
    targets = [n.output[0] for n in nodes + blocks]
    inputs = list(dict.fromkeys(n.input[0] for n in nodes))
    paths = {'fp32': fp_path, **{v: args.baseline_work / (v+'.onnx')
             for v in ['int8_per_tensor', 'int8_per_channel']}}
    config = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'source': source_record(),
              'models_sha256': {v: sha256(p) for v,p in paths.items()},
              'dataset_sha256': sha256('data/stsb-validation.parquet'),
              'probe_rows': list(range(64)), 'probe_sentences': 128, 'batch': 16, 'max_length': 256,
              'text_order': 'sentence1 rows 0:64 followed by sentence2 rows 0:64',
              'mask': 'non-padding tokens only', 'optimization': args.optimization,
              'provider': 'CPUExecutionProvider', 'local': 'same FP32 input; isolated real dynamic U8S8 MatMul, per-channel weights',
              'cumulative': 'matching full-network MatMul and transformer block outputs',
              'selection_rule': 'exclude exactly one MatMul with largest aggregate local per-channel NMSE; tie by node name',
              'hypothesis': 'Keeping the largest local-NMSE MatMul in FP32 reduces embedding MSE versus full per-channel INT8; task Spearman and latency may improve or worsen',
              'limitations': 'exploratory validation-subset selection, no independent generalization test; local error is not causal task sensitivity'}
    save(out/'protocol.json', config)  # Persist before observing any errors.
    sessions = {}
    for variant, path in paths.items():
        debug_path = work/(variant+'-debug.onnx')
        instrument(path, debug_path, targets + (inputs if variant == 'fp32' else []))
        sessions[variant] = debug_session(debug_path, args.optimization)
    production = {}
    for variant, path in paths.items():
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        production[variant] = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    local_sessions = []
    for i, node in enumerate(nodes):
        weight = deepcopy(weights[node.input[1]])
        weight.name = 'weight'
        shape = list(weight.dims)
        graph = helper.make_graph([helper.make_node('MatMul', ['x','weight'], ['y'], name='matmul')],
            'local_matmul', [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['b','s',shape[0]])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['b','s',shape[1]])], [weight])
        model = helper.make_model(graph, opset_imports=fp.opset_import, ir_version=fp.ir_version)
        src, dst = work/f'local-{i}-fp.onnx', work/f'local-{i}-int8.onnx'
        onnx.save(model, src)
        quantize_dynamic(str(src), str(dst), op_types_to_quantize=['MatMul'],
                         per_channel=True, reduce_range=False, weight_type=QuantType.QInt8,
                         extra_options={'MatMulConstBOnly': True})
        local_sessions.append(debug_session(dst, args.optimization))
    rows = pq.read_table('data/stsb-validation.parquet').to_pylist()[:64]
    texts = [r['sentence1'] for r in rows] + [r['sentence2'] for r in rows]
    tok = Tokenizer.from_file('models/minilm/tokenizer.json')
    tok.enable_truncation(max_length=256)
    tok.enable_padding(pad_id=0, pad_token='[PAD]')
    raw, totals, checks = [], defaultdict(lambda: defaultdict(float)), []
    for batch_start in range(0,128,16):
        feed = feeds(tok, texts[batch_start:batch_start+16])
        values = {}
        for variant, sess in sessions.items():
            names = [o.name for o in sess.get_outputs()]
            values[variant] = dict(zip(names, sess.run(None, feed)))
        # Instrumentation/fusion changes must not silently invalidate interpretation.
        checks.append({'batch_start':batch_start, 'feed_sha256':hashlib.sha256(
            b''.join(feed[k].tobytes() for k in sorted(feed))).hexdigest(), 'shape':list(feed['input_ids'].shape),
            'debug_vs_optimized_final': {v: metrics(error_sums(
                production[v].run(None, feed)[0], values[v]['last_hidden_state'], feed['attention_mask']))
                for v in sessions}})
        for variant in ['int8_per_tensor','int8_per_channel']:
            for node in nodes + blocks:
                name = node.output[0]
                sums = error_sums(values['fp32'][name], values[variant][name], feed['attention_mask'])
                raw.append({'batch_start':batch_start,'kind':'cumulative','variant':variant,'node':node.name,**sums})
        for node, sess in zip(nodes, local_sessions):
            x = values['fp32'][node.input[0]]
            y = sess.run(None, {'x':x})[0]
            sums = error_sums(values['fp32'][node.output[0]], y, feed['attention_mask'])
            raw.append({'batch_start':batch_start,'kind':'local','variant':'int8_per_channel','node':node.name,**sums})
        print('probe batch',batch_start,'complete',flush=True)
    for row in raw:
        t = totals[(row['kind'],row['variant'],row['node'])]
        for key in ['n','sse','reference_energy','candidate_energy','dot']:
            t[key] += row[key]
        t['max_abs'] = max(t['max_abs'],row['max_abs'])
    summary = [{'kind':k,'variant':v,'node':n,**metrics(t)} for (k,v,n),t in totals.items()]
    ranking = sorted([s for s in summary if s['kind']=='local'], key=lambda s:(-s['nmse'],s['node']))
    save(out/'raw-errors.json',raw)
    save(out/'summary.json',summary)
    save(out/'probe-batches.json',checks)
    save(out/'local-ranking.json',ranking)
    save(out/'exclusion.json', {'nodes_to_exclude':[ranking[0]['node']],
          'selection_rule':config['selection_rule'],'hypothesis':config['hypothesis'],
          'selected_local_nmse':ranking[0]['nmse'], 'protocol_sha256':sha256(out/'protocol.json')})
    print(json.dumps(ranking[:5],indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--optimization', choices=['all', 'disabled'], default='all')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--baseline-work',type=Path,required=True)
    main(p.parse_args())
