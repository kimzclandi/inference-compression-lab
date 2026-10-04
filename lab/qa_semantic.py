"""Expose frozen encoder states locally; do not distribute model parameters."""
from pathlib import Path
import numpy as np
from lab.qa_specialist_runtime import validate_assets
from lab.quantization_diagnostics import sha

TAP='/model/roberta/encoder/layer.11/output/LayerNorm/LayerNormalization_output_0'

class SemanticRuntime:
    def __init__(self,asset_root,manifest_sha,local_model):
        import onnx
        import onnxruntime as ort
        validate_assets(asset_root,manifest_sha)
        source=Path(asset_root)/'int8.onnx';graph=onnx.load(source)
        if sum(TAP in n.output for n in graph.graph.node)!=1:raise ValueError('Encoder tap identity')
        graph.graph.output.append(onnx.helper.make_tensor_value_info(TAP,onnx.TensorProto.FLOAT,['batch','sequence',768]))
        local_model=Path(local_model)
        if 'runs' not in local_model.parts or local_model.exists():raise ValueError('Instrumented weights require a new local runs path')
        local_model.parent.mkdir(parents=True,exist_ok=True);onnx.checker.check_model(graph);onnx.save(graph,local_model)
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(local_model),sess_options=opts,providers=['CPUExecutionProvider'])
        self.model_sha=sha(local_model)

    def vector(self,prediction):
        w=prediction['raw_windows'][prediction['window_index']]
        values=self.session.run(['start_logits','end_logits',TAP],{k:np.asarray([w[k]],dtype=np.int64) for k in ('input_ids','attention_mask')})
        difference=max(float(np.max(np.abs(values[i][0]-np.asarray(w[name])))) for i,name in enumerate(('start_logits','end_logits')))
        if difference>1e-4:raise ValueError('Instrumentation altered logits beyond frozen tolerance')
        hidden=values[2][0]
        if hidden.shape!=(len(w['input_ids']),768) or not np.isfinite(hidden).all():raise ValueError('Invalid hidden shape/values')
        span=hidden[prediction['start_token']:prediction['end_token']+1].mean(axis=0)
        return np.concatenate([hidden[w['cls_index']],span]),difference
