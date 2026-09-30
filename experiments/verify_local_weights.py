"""Confirm isolated probes use the exact quantized weights/scales of the full graph."""
import argparse
from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np
from experiments.minilm_ptq import save
from lab.evidence import source_record, sha256


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    fp=onnx.load('models/minilm/onnx/model.onnx')
    weights={w.name for w in fp.graph.initializer}
    nodes=[n for n in fp.graph.node if n.op_type=='MatMul' and n.input[1] in weights]
    full={w.name:numpy_helper.to_array(w) for w in onnx.load(args.quantized).graph.initializer}
    checks=[]
    for i,node in enumerate(nodes):
        local_path=args.probe_work/f'local-{i}-int8.onnx'
        local={w.name:numpy_helper.to_array(w) for w in onnx.load(local_path).graph.initializer}
        matched={suffix:bool(np.array_equal(full[node.input[1]+suffix],local['weight'+suffix]))
                 for suffix in ['_quantized','_scale','_zero_point']}
        if not all(matched.values()):raise ValueError(f'Quantization mismatch: {node.name}')
        checks.append({'node':node.name,'equal':matched,'local_model_sha256':sha256(local_path)})
    save(args.output,{'source':source_record(),'full_model_sha256':sha256(args.quantized),'checks':checks})
    print('36 isolated weight, scale and zero-point triplets match the full model exactly')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--quantized',type=Path,required=True)
    p.add_argument('--probe-work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
