"""One fixed EM-correctness ranking experiment; no learned parameters distributed."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from lab.evidence import reserve_directory
from lab.artifact_integrity import git_identity, file_hashes
from lab.quantization_diagnostics import read, write, rows, sha
from lab.qa_metrics import normalize
from lab.qa_gate import evaluate_quality_gate
from lab.selective_qa import evaluate_selective
from lab.qa_risk_calibration import extract_features, fit, predict_probability

SOURCE_FILES=('experiments/qa_risk.py','lab/qa_risk_calibration.py',
              'lab/qa_specialist_runtime.py','lab/extractive_qa.py',
              'lab/selective_qa.py','lab/qa_metrics.py','lab/qa_gate.py')
VARIANTS=('fp32','int8')


def model_digest(model):
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def preflight(spec_path):
    spec=read(spec_path)
    if spec.get('locked') is not True or set(spec['source_sha256'])!=set(SOURCE_FILES):
        raise ValueError('Incomplete frozen risk study')
    for name,digest in {**spec['source_sha256'],**spec['upstream_evidence_sha256']}.items():
        if sha(Path(name))!=digest:raise ValueError('Frozen source/evidence changed: '+name)
    for split,digest in spec['data_sha256'].items():
        data=Path('configs/qa-risk/dataset')/split/'data.jsonl'
        if sha(data)!=digest:raise ValueError('Frozen data changed: '+split)
    return spec


def snapshot(out,spec_path):
    shutil.copy2(spec_path,out/'protocol.json')
    for name in SOURCE_FILES:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)


def training_matrix(spec,variant,split):
    raw={}
    for old_split in ('calibration','evaluation'):
        for prediction in rows(Path(spec['development_root'])/(old_split+'-'+variant)/'predictions.jsonl'):
            if prediction['id'] in raw:raise ValueError('Duplicate upstream prediction')
            raw[prediction['id']]=prediction
    data=rows(Path('configs/qa-risk/dataset')/split/'data.jsonl')
    records=[]
    for row in data:
        pred=raw[row['id']]
        feature=extract_features(row['context'],pred)
        target=int(any(normalize(pred['prediction'])==normalize(a) for a in row['answers']))
        # Unanswerable rows have no gold spans and necessarily receive target zero.
        records.append(dict(id=row['id'],prediction=pred['prediction'],features=feature['features'],
                            feature_names=feature['feature_names'],diagnostics=feature['diagnostics'],target=target))
    return data,records


def select_variant(data,records,model,spec):
    predictions=[dict(id=r['id'],prediction=r['prediction'],confidence=predict_probability(model,r['features'])) for r in records]
    candidates=[]
    for threshold in spec['threshold_grid']:
        summary=evaluate_selective(data,predictions,threshold)
        gate=evaluate_quality_gate(summary['selective'],spec['quality_constraints'])
        candidates.append(dict(threshold=threshold,gate=gate,summary=summary))
    passing=[c for c in candidates if c['gate']['all_pass']]
    return dict(eligible=bool(passing),threshold=passing[0]['threshold'] if passing else None,
                candidates=candidates),predictions


def train(args):
    spec=preflight(args.spec)
    out=reserve_directory(args.output_dir)
    head_root=reserve_directory(args.head_dir)
    state=dict(status='running',**git_identity(Path.cwd()),python=platform.python_version(),
               numpy=importlib.metadata.version('numpy'),protocol_sha256=sha(args.spec))
    try:
        snapshot(out,args.spec)
        result=dict(protocol_sha256=sha(args.spec),variants={},scope=spec['scope'])
        for split in ('train','calibration'):
            shutil.copy2(Path('configs/qa-risk/dataset')/split/'data.jsonl',out/(split+'-data.jsonl'))
        for variant in VARIANTS:
            _,matrix=training_matrix(spec,variant,'train')
            cal,cal_matrix=training_matrix(spec,variant,'calibration')
            model=fit([r['features'] for r in matrix],[r['target'] for r in matrix])
            # Coefficients/scaler are deliberately confined to the ignored local runs/ directory.
            write(head_root/(variant+'.json'),model)
            write(out/(variant+'-training-features.json'),matrix)
            write(out/(variant+'-calibration-features.json'),cal_matrix)
            write(out/(variant+'-fit.json'),{k:v for k,v in model.items() if k not in ('weights','intercept','scaler')})
            selected,predictions=select_variant(cal,cal_matrix,model,spec)
            write(out/(variant+'-calibration-predictions.json'),predictions)
            result['variants'][variant]=dict(**selected,model_sha256=model_digest(model))
        result['any_eligible']=any(v['eligible'] for v in result['variants'].values())
        result['both_eligible']=all(v['eligible'] for v in result['variants'].values())
        write(out/'selection.json',result)
        state.update(status='complete',variants=list(VARIANTS))
        print(json.dumps(dict(status='complete',thresholds={v:r['threshold'] for v,r in result['variants'].items()})))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def rebuild_head(training_dir,variant,expected_digest):
    records=read(training_dir/(variant+'-training-features.json'))
    model=fit([r['features'] for r in records],[r['target'] for r in records])
    if model_digest(model)!=expected_digest:raise ValueError('Rebuilt head differs from frozen training identity')
    return model


def evaluate(args):
    spec=preflight(args.spec)
    selection=read(args.training_dir/'selection.json')
    if selection['protocol_sha256']!=sha(args.spec) or not selection['variants'][args.variant]['eligible']:
        raise ValueError('No calibration permission for this variant')
    if sha(args.training_dir/'selection.json')!=args.selection_sha256:
        raise ValueError('Explicit frozen selection hash required')
    model=rebuild_head(args.training_dir,args.variant,selection['variants'][args.variant]['model_sha256'])
    out=reserve_directory(args.output_dir)
    state=dict(status='running',variant=args.variant,**git_identity(Path.cwd()),
               python=platform.python_version(),protocol_sha256=sha(args.spec),
               selection_sha256=args.selection_sha256,head_sha256=model_digest(model),completed_predictions=0)
    try:
        snapshot(out,args.spec)
        shutil.copy2(args.training_dir/'selection.json',out/'selection.json')
        data_path=Path('configs/qa-risk/dataset/evaluation/data.jsonl')
        shutil.copy2(data_path,out/'data.jsonl');state['dataset_sha256']=sha(data_path)
        from lab.qa_specialist_runtime import ExtractiveRuntime,RUNTIME_PACKAGES
        state['packages']={n:importlib.metadata.version(n) for n in RUNTIME_PACKAGES}
        start=time.perf_counter();runtime=ExtractiveRuntime(args.asset_root,args.variant,spec['asset_manifest_sha256'])
        state['model_load_seconds']=time.perf_counter()-start
        shutil.copy2(args.asset_root/'manifest.json',out/'assets.json')
        predictions=[]
        data=rows(data_path)
        for row in data:
            start=time.perf_counter()
            prediction=runtime.predict(row['context'],row['question'])
            feature=extract_features(row['context'],prediction)
            # Preserve decoder score separately; only this pre-fitted head can replace the ranking score.
            prediction.update(id=row['id'],base_confidence=prediction['confidence'],risk_features=feature,
                              confidence=predict_probability(model,feature['features']))
            prediction['pipeline_seconds']=time.perf_counter()-start
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(prediction,allow_nan=False)+'\n')
            predictions.append(prediction);state['completed_predictions']+=1
        summary=evaluate_selective(data,predictions,selection['variants'][args.variant]['threshold'])
        gate=evaluate_quality_gate(summary['selective'],spec['quality_constraints'])
        write(out/'summary.json',dict(summary=summary,gate=gate))
        state['status']='complete'
        print(json.dumps(dict(variant=args.variant,gate=gate,selective=summary['selective'])))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        write(out/'run.json',state);write(out/'checksums.json',file_hashes(out))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    t=sub.add_parser('train');t.add_argument('--head-dir',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--training-dir',type=Path,required=True)
    e.add_argument('--selection-sha256',required=True);e.add_argument('--asset-root',type=Path,required=True)
    e.add_argument('--variant',choices=VARIANTS,required=True)
    for parser in (t,e):
        parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.action=='train':
        # Prevent accidental distribution of even tiny learned head parameters.
        if not args.head_dir.resolve().is_relative_to((Path.cwd()/'runs').resolve()):
            raise ValueError('Head parameters must remain inside ignored local runs/')
        train(args)
    else:evaluate(args)

if __name__=='__main__':main()
