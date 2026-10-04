"""Independent fixed-ID local tap check: QA projection and semantic pooling."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from experiments.qa_semantic import ROOT,SPEC,base,preflight
from lab.qa_semantic import SemanticRuntime,TAP
from lab.quantization_diagnostics import write,read

def check(asset_root,model_path,output):
    import onnx
    s=preflight();rt=SemanticRuntime(asset_root,s['asset_manifest_sha256'],model_path)
    graph=onnx.load(Path(asset_root)/'int8.onnx');initial={x.name:x for x in graph.graph.initializer}
    w=onnx.numpy_helper.to_array(initial['onnx::MatMul_1511']);b=onnx.numpy_helper.to_array(initial['model.qa_outputs.bias'])
    result=[]
    for name,splits in [('training',['old_train','train_new']),('development',['calibration'])]:
        raw=[]
        for split in splits:raw+=base(split)[1]
        with np.load(ROOT/f'results/qa-semantic-v1/collection/{name}.npz',allow_pickle=False) as z:stored=z['features']
        selected=sorted(range(len(raw)),key=lambda i:hashlib.sha256(('semantic-audit-'+raw[i]['id']).encode()).hexdigest())[:4]
        for i in selected:
            p=raw[i];window=p['raw_windows'][p['window_index']]
            start,end,h=rt.session.run(['start_logits','end_logits',TAP],{k:np.asarray([window[k]],dtype=np.int64) for k in ('input_ids','attention_mask')})
            projection=h@w+b;projection_error=max(float(np.max(abs(projection[0,:,0]-start[0]))),float(np.max(abs(projection[0,:,1]-end[0]))))
            first,last=p['start_token'],p['end_token'];span=np.sum(h[0,first:last+1],axis=0,dtype=np.float64)/(last-first+1)
            expected=np.concatenate([h[0,window['cls_index']],span]);pool_error=float(np.max(abs(expected-stored[i])))
            if projection_error>1e-4 or pool_error>2e-6:raise ValueError('Independent semantic tap mismatch')
            result.append(dict(split=name,id=p['id'],projection_error=projection_error,pooling_error=pool_error))
    report=dict(all_pass=True,selection='Four smallest SHA256(semantic-audit- + id) per role; no outcome filtering',records=result,
        model_inference=True,encoder_trained=False,scope='Eight-row local implementation/projection check; not a new quality evaluation')
    write(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for name in ('asset-root','model-path','output'):p.add_argument('--'+name,type=Path,required=True)
    print(check(**vars(p.parse_args())))
