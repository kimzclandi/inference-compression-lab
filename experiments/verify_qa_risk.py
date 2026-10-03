"""Rebuild fixed correctness-head evidence without distributing learned parameters.

Full verification requires the study's exact NumPy version (2.2.6). No learned
weights, intercept or scaler are returned by the public verification result.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

from lab.artifact_integrity import safe_path, verify_hashes
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal
from lab.qa_metrics import evaluate as evaluate_raw, normalize
from lab.qa_gate import evaluate_quality_gate
from lab.selective_qa import evaluate_selective
from lab.qa_risk_calibration import FEATURE_NAMES, FIT_CONFIG, extract_features, fit, predict_probability
from experiments.qa_risk import SOURCE_FILES, VARIANTS, model_digest
from experiments.verify_qa_specialist import (independently_decode, verify_assets,
    verify_run as verify_parent_run, require, indexed, number)
from experiments.verify_qa_remediation import paired_bootstrap

REPO = Path(__file__).resolve().parents[1]
PRIVATE_FIELDS = {'weights', 'intercept', 'scaler'}


def assert_no_parameters(value):
    if isinstance(value, dict):
        require(not (set(value) & PRIVATE_FIELDS), 'Learned parameters must not be published')
        for child in value.values(): assert_no_parameters(child)
    elif isinstance(value, list):
        for child in value: assert_no_parameters(child)


def source_snapshot(folder, spec, repo):
    require(set(spec['source_sha256']) == set(SOURCE_FILES), 'Frozen source coverage')
    verify_hashes(folder/'source', spec['source_sha256'], exclude=())
    for name, value in spec['source_sha256'].items():
        require(sha(safe_path(repo, name)) == value, 'Current implementation differs from frozen source: '+name)


def load_roles(spec, repo=REPO):
    require(set(spec['data_sha256']) == {'train','calibration','evaluation'}, 'Missing data role')
    roles = {}
    for role, expected in spec['data_sha256'].items():
        path = repo/'configs/qa-risk/dataset'/role/'data.jsonl'
        require(sha(path) == expected, 'Frozen role data changed: '+role)
        data = rows(path); indexed(data)
        require(len(data) == spec['dataset_sizes'][role] and all(r['split'] == role for r in data), 'Role size or label')
        require(all(type(r['is_impossible']) is bool and bool(r['answers']) != r['is_impossible'] for r in data),
                'Answerability label consistency')
        roles[role] = data
    def canonical(value): return ' '.join(value.casefold().split())
    for first, second in [('train','calibration'),('train','evaluation'),('calibration','evaluation')]:
        for field in ('id','source_title','family_id','context','question'):
            a={canonical(r[field]) if field in ('context','question') else r[field] for r in roles[first]}
            b={canonical(r[field]) if field in ('context','question') else r[field] for r in roles[second]}
            require(not a & b, 'Train/calibration/evaluation overlap: '+field)
    return roles


def verify_parents(spec, repo=REPO):
    expected_names = {f"{spec['development_root']}/{split}-{variant}/{name}"
        for split in ('calibration','evaluation') for variant in VARIANTS
        for name in ('data.jsonl','predictions.jsonl','run.json','protocol.json','assets.json','checksums.json')}
    require(set(spec['upstream_evidence_sha256']) == expected_names, 'Raw parent evidence coverage')
    for name, value in spec['upstream_evidence_sha256'].items():
        require(sha(safe_path(repo,name)) == value, 'Frozen raw parent changed: '+name)
    by_variant={}; parent_data=None
    for variant in VARIANTS:
        data=[]; predictions=[]
        for split in ('calibration','evaluation'):
            result=verify_parent_run(repo/spec['development_root']/(split+'-'+variant))
            data += result['data']; predictions += result['predictions']
            require(sha(repo/spec['development_root']/(split+'-'+variant)/'assets.json') == spec['asset_manifest_sha256'],
                    'Risk head uses another task backbone')
        indexed(data); by_variant[variant]=indexed(predictions)
        if parent_data is None: parent_data=indexed(data)
        else: require(parent_data == indexed(data), 'FP32/INT8 parent cohort differs')
    return parent_data, by_variant


def reconstruct_matrix(data, raw):
    matrix=[]
    for row in data:
        pred=raw[row['id']]
        feature=extract_features(row['context'],pred)
        target=int(any(normalize(pred['prediction']) == normalize(answer) for answer in row['answers']))
        matrix.append(dict(id=row['id'],prediction=pred['prediction'],features=feature['features'],
                           feature_names=feature['feature_names'],diagnostics=feature['diagnostics'],target=target))
    return matrix


def select(data, matrix, model, spec):
    predictions=[dict(id=r['id'],prediction=r['prediction'],confidence=predict_probability(model,r['features'])) for r in matrix]
    candidates=[]
    for threshold in spec['threshold_grid']:
        summary,_=evaluate_selective(data,predictions,threshold)
        gate=evaluate_quality_gate(summary['selective'],spec['quality_constraints'])
        candidates.append(dict(threshold=threshold,gate=gate,summary=summary))
    passing=[c['threshold'] for c in candidates if c['gate']['all_pass']]
    return dict(eligible=bool(passing),threshold=min(passing) if passing else None,candidates=candidates),predictions


def verify_training(folder, spec, protocol_sha256, *, repo=REPO):
    """Internal return includes in-memory heads; never serialize that return."""
    folder, repo=Path(folder),Path(repo)
    verify_hashes(folder,read(folder/'checksums.json'),exclude=('checksums.json',))
    expected={'protocol.json','run.json','train-data.jsonl','calibration-data.jsonl','selection.json'}
    expected |= {v+'-'+suffix for v in VARIANTS for suffix in
                 ('training-features.json','calibration-features.json','fit.json','calibration-predictions.json')}
    expected |= {'source/'+name for name in SOURCE_FILES}
    require(set(read(folder/'checksums.json')) == expected, 'Unexpected/missing training evidence files')
    run=read(folder/'run.json')
    require(run['status']=='complete' and run['variants']==list(VARIANTS), 'Training is incomplete')
    require(sha(folder/'protocol.json')==protocol_sha256==run['protocol_sha256'] and read(folder/'protocol.json')==spec,
            'Training protocol identity')
    require(spec['locked'] is True and spec['feature_names']==list(FEATURE_NAMES) and spec['fit_config']==FIT_CONFIG,
            'Frozen feature/optimizer contract')
    require(run['numpy']==spec['numpy_version']==importlib.metadata.version('numpy'), 'Exact NumPy version required for fit reconstruction')
    grid=spec['threshold_grid'];require(grid and len(set(grid))==len(grid) and grid==sorted(grid), 'Ordered unique threshold grid')
    for value in grid:number(value,'threshold',0,1)
    source_snapshot(folder,spec,repo)
    roles=load_roles(spec,repo);parent_data,parent_raw=verify_parents(spec,repo)
    old_ids={r['id'] for role in ('train','calibration') for r in roles[role]}
    require(old_ids==set(parent_data), 'All and only prior 384 examples must become development')
    for role in ('train','calibration'):
        require(sha(folder/(role+'-data.jsonl'))==spec['data_sha256'][role], 'Archived role dataset identity')
        for row in roles[role]:
            require({k:v for k,v in row.items() if k!='split'}==
                    {k:v for k,v in parent_data[row['id']].items() if k!='split'}, 'Development labels/content changed')
    heads={}; selection=dict(protocol_sha256=protocol_sha256,variants={},scope=spec['scope'])
    for variant in VARIANTS:
        matrix=reconstruct_matrix(roles['train'],parent_raw[variant])
        cal_matrix=reconstruct_matrix(roles['calibration'],parent_raw[variant])
        require(aggregates_equal(matrix,read(folder/(variant+'-training-features.json'))), 'Training features/targets do not reproduce')
        require(aggregates_equal(cal_matrix,read(folder/(variant+'-calibration-features.json'))), 'Calibration features/targets do not reproduce')
        model=fit([r['features'] for r in matrix],[r['target'] for r in matrix])
        public_fit=read(folder/(variant+'-fit.json'));assert_no_parameters(public_fit)
        expected_fit={k:v for k,v in model.items() if k not in PRIVATE_FIELDS}
        require(aggregates_equal(expected_fit,public_fit), 'Published fit trace/config differs from training-only reconstruction')
        chosen,predictions=select(roles['calibration'],cal_matrix,model,spec)
        require(aggregates_equal(predictions,read(folder/(variant+'-calibration-predictions.json'))), 'Calibration scores differ from reconstructed head')
        selection['variants'][variant]=dict(**chosen,model_sha256=model_digest(model));heads[variant]=model
    selection['any_eligible']=any(v['eligible'] for v in selection['variants'].values())
    selection['both_eligible']=all(v['eligible'] for v in selection['variants'].values())
    require(aggregates_equal(selection,read(folder/'selection.json')), 'Saved eligibility, threshold, curve or model hash does not reproduce')
    assert_no_parameters(selection)
    return dict(selection=selection,heads=heads,roles=roles)


def verify_evaluation(folder,variant,spec,protocol_sha256,training_dir,training):
    folder,training_dir=Path(folder),Path(training_dir)
    verify_hashes(folder,read(folder/'checksums.json'),exclude=('checksums.json',))
    expected={'protocol.json','run.json','selection.json','data.jsonl','assets.json','predictions.jsonl','summary.json'}
    expected |= {'source/'+name for name in SOURCE_FILES}
    require(set(read(folder/'checksums.json'))==expected,'Unexpected/missing evaluation evidence files')
    run=read(folder/'run.json');selected=training['selection']['variants'][variant]
    require(selected['eligible'] is True,'Ineligible variant must not consume evaluation')
    require(run['status']=='complete' and run['variant']==variant,'Incomplete evaluation or wrong variant')
    require(run['protocol_sha256']==protocol_sha256==sha(folder/'protocol.json') and read(folder/'protocol.json')==spec,
            'Evaluation protocol changed')
    require(sha(folder/'selection.json')==sha(training_dir/'selection.json')==run['selection_sha256'], 'Frozen evaluation selection binding')
    require(run['head_sha256']==selected['model_sha256']==model_digest(training['heads'][variant]), 'Evaluation used another head')
    require(sha(folder/'data.jsonl')==spec['data_sha256']['evaluation']==run['dataset_sha256'], 'Evaluation data identity')
    require(sha(folder/'assets.json')==spec['asset_manifest_sha256'], 'Evaluation backbone identity')
    source_snapshot(folder,spec,REPO)
    assets=read(folder/'assets.json');file_sizes=verify_assets(assets)
    from lab.qa_specialist_runtime import RUNTIME_PACKAGES
    require(run['packages']=={n:assets['packages'][n] for n in RUNTIME_PACKAGES}, 'Evaluation dependency identity')
    data=rows(folder/'data.jsonl');predictions=rows(folder/'predictions.jsonl');a,b=indexed(data),indexed(predictions)
    require(data==training['roles']['evaluation'] and set(a)==set(b) and
            type(run['completed_predictions']) is int and run['completed_predictions']==len(data), 'Evaluation exact coverage')
    for row in data:
        pred=b[row['id']];decoded=independently_decode(row['context'],pred['raw_windows'],spec['input_limits'])
        saved={k:pred[k] for k in decoded};saved['confidence']=pred['base_confidence']
        require(aggregates_equal(decoded,saved),'Evaluation raw span does not independently decode')
        feature=extract_features(row['context'],pred)
        require(aggregates_equal(feature,pred['risk_features']),'Evaluation feature extraction changed')
        confidence=predict_probability(training['heads'][variant],feature['features'])
        require(aggregates_equal(confidence,pred['confidence']),'Evaluation score differs from frozen head')
        require(pred['feature_count']==len(pred['raw_windows']) and pred['input_tokens']==[len(w['input_ids']) for w in pred['raw_windows']],
                'Evaluation window telemetry')
        for key in ('tokenization_seconds','inference_seconds','decode_seconds','total_pipeline_seconds','pipeline_seconds'):
            number(pred[key],key,0)
        require(pred['pipeline_seconds']+1e-6>=pred['total_pipeline_seconds'], 'Risk pipeline excludes backbone duration')
    summary,scored=evaluate_selective(data,predictions,selected['threshold'])
    gate=evaluate_quality_gate(summary['selective'],spec['quality_constraints'])
    require(aggregates_equal(dict(summary=summary,gate=gate),read(folder/'summary.json')), 'Evaluation metrics/gate do not reproduce')
    raw_metrics,raw_scored=evaluate_raw(data,predictions)
    return dict(data=data,predictions=predictions,summary=summary,scored=scored,gate=gate,
                raw_metrics=raw_metrics,raw_scored=raw_scored,file_size_comparison=file_sizes)


def verify(root,study=None):
    root=Path(root);training_dir=root/'training';protocol=training_dir/'protocol.json'
    spec=read(protocol);protocol_sha256=sha(protocol)
    if study is not None:require(sha(study)==protocol_sha256,'Expected study differs from archived protocol')
    training=verify_training(training_dir,spec,protocol_sha256)
    selection=training['selection'];variants={};evaluated={}
    for variant in VARIANTS:
        folder=root/('evaluation-'+variant);selected=selection['variants'][variant]
        if not selected['eligible']:
            require(not folder.exists(),'Calibration failure forbids evaluation for '+variant)
            variants[variant]=dict(calibration_eligible=False,threshold=None,evaluation_evaluated=False,
                                   task_pass=False,bounded_task_eligible=False,answer_entrypoint='unavailable_quality')
        elif not folder.exists():
            variants[variant]=dict(calibration_eligible=True,threshold=selected['threshold'],evaluation_evaluated=False,
                                   task_pass=False,bounded_task_eligible=False,answer_entrypoint='unavailable_quality')
        else:
            result=verify_evaluation(folder,variant,spec,protocol_sha256,training_dir,training);evaluated[variant]=result
            passed=result['gate']['all_pass']
            variants[variant]=dict(calibration_eligible=True,threshold=selected['threshold'],evaluation_evaluated=True,
                task_pass=passed,bounded_task_eligible=passed,summary=result['summary'],gate=result['gate'],
                answer_entrypoint='bounded_local_prototype' if passed else 'unavailable_quality')
    compression=dict(status='not_evaluated',evaluated=False,passed=False,reason='Both fixed pipeline variants must have complete evaluation')
    raw_pair=None
    if set(evaluated)==set(VARIANTS):
        config=spec['compression_gate'];require(config['replicates']==5000 and config['confidence']==.95,'Bootstrap contract')
        a,b=evaluated['fp32'],evaluated['int8'];require(a['data']==b['data'],'Paired evaluation cohort differs')
        paired=paired_bootstrap(a['data'],[r['system_score'] for r in a['scored']],[r['system_score'] for r in b['scored']],
                                 seed=config['seed'],replicates=config['replicates'])
        raw_pair=paired_bootstrap(a['data'],a['raw_scored'],b['raw_scored'],seed=config['seed'],replicates=config['replicates'])
        criteria=dict(em_noninferiority=paired['em_ci95'][0]>=config['minimum_ci_lower_bound'],
            f1_noninferiority=paired['f1_ci95'][0]>=config['minimum_ci_lower_bound'],
            file_size=a['file_size_comparison']['int8_to_fp32_ratio']<=config['maximum_model_file_bytes_ratio'],
            fp32_task=variants['fp32']['task_pass'],int8_task=variants['int8']['task_pass'])
        compression=dict(status='evaluated',evaluated=True,passed=all(criteria.values()),criteria=criteria,paired=paired,
                         file_size_comparison=a['file_size_comparison'],scope='Separate per-precision fitted heads: complete pipeline comparison, not pure quantization effect')
    result=dict(evidence_valid=True,protocol_sha256=protocol_sha256,selection_sha256=sha(training_dir/'selection.json'),
        any_calibration_eligible=selection['any_eligible'],both_calibration_eligible=selection['both_eligible'],
        status='calibration_failed' if not selection['any_eligible'] else
               'evaluation_complete' if all(not v['calibration_eligible'] or v['evaluation_evaluated'] for v in variants.values()) else 'ready_for_evaluation',
        variants=variants,compression_gate=compression,raw_candidate_paired=raw_pair,
        compression_default_recommendation=False,compression_recommendation_reason='A compression recommendation additionally requires paired quality and independent performance evidence. Per-variant bounded QA eligibility is decided separately by its task gate.',
        parameters_distributed=False,numpy_version=spec['numpy_version'],scope=spec['scope'])
    assert_no_parameters(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--study',type=Path,default=Path('configs/qa-risk/study.json'))
    args=parser.parse_args();print(json.dumps(verify(args.root,args.study),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
