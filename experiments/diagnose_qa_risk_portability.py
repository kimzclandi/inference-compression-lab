"""Read-only frozen-optimizer diagnostics, never emits trained parameters."""
import inspect
import json
from pathlib import Path

from experiments.qa_risk import training_matrix
from lab.quantization_diagnostics import read,sha
import lab.qa_risk_calibration as calibration


def main():
    spec=read('configs/qa-risk/study.json')
    source=inspect.getsource(calibration.fit)
    # Instrument only the local diagnostic copy at the fixed failure branch.
    # Published training source, optimizer settings and inference heads stay unchanged.
    old="raise ValueError('Logistic head did not converge in the fixed 100 iterations.')"
    replacement="return dict(convergence=dict(converged=False,iterations=iteration,gradient_max_abs=maximum_gradient),trace=trace)"
    if source.count(old)!=1:raise ValueError('Frozen diagnostic insertion point changed')
    namespace=dict(vars(calibration));exec(source.replace(old,replacement),namespace)
    diagnostic_fit=namespace['fit'];report={'frozen_source_sha256':sha('lab/qa_risk_calibration.py'),'variants':{}}
    for variant in ('fp32','int8'):
        stored=read(Path('results/qa-risk-v2/training')/(variant+'-training-features.json'))
        _,recomputed=training_matrix(spec,variant,'train')
        item={'max_recomputed_feature_difference':max(abs(x-y) for a,b in zip(stored,recomputed) for x,y in zip(a['features'],b['features']))}
        for label,records in [('stored',stored),('recomputed',recomputed)]:
            try:
                result=diagnostic_fit([r['features'] for r in records],[r['target'] for r in records])
                item[label]={'convergence':result['convergence'],'last_trace':result['trace'][-4:]}
            except Exception as exc:item[label]={'error':repr(exc)}
        report['variants'][variant]=item
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
