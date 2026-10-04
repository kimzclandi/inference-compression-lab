"""Audit the fixed expanded-ranking failure, without model inference.

Raw spans/features/labels use the independent standard-library reference.
Optional fitting replays verified archived matrices, never regenerated floats.
This verifier deliberately requires the recorded failed calibration outcome.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import math
from pathlib import Path

from lab.artifact_integrity import safe_path, verify_hashes
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal, write
from lab.qa_expanded_io import load_collection
from experiments.verify_qa_coverage_gap import metrics, passes, require

ROOT = Path(__file__).resolve().parents[1]


def normalized_hash(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def reference():
    path = ROOT/'results/qa-risk-review-v1/independent-feature-audit.py'
    spec = importlib.util.spec_from_file_location('expanded_independent_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_features(data, raw, matrix, ref):
    require([r['id'] for r in data] == [r['id'] for r in matrix] == [r['id'] for r in raw], 'Raw/data/matrix ordering')
    maximum_error = 0.; windows = Counter(); correct = 0
    for row, record, stored in zip(data, raw, matrix):
        features, prediction, diagnostics = ref.independent(row['context'], record['raw_windows'])
        ref.compare(record, prediction, row['id']+'.decoder')
        ref.compare(stored['features'], features, row['id']+'.features')
        ref.compare(stored['diagnostics'], diagnostics, row['id']+'.diagnostics')
        require(stored['feature_names'] == ref.NAMES, 'Feature names/order')
        require(stored['prediction'] == record['prediction'], 'Matrix answer changed')
        target = ref.em_label(prediction['prediction'], row['answers'])
        require(type(stored['target']) is int and stored['target'] == target, 'Correctness label')
        require(not row['is_impossible'] or target == 0, 'Impossible correctness label')
        maximum_error = max(maximum_error, *(abs(a-b) for a,b in zip(features, stored['features'])))
        windows[len(record['raw_windows'])] += 1; correct += target
    return dict(rows=len(data), windows=dict(windows), raw_em_correct=correct,
                maximum_independent_feature_difference=maximum_error)


def dataset_audit(spec, source_root=None):
    manifest = read(ROOT/'configs/qa-expanded/dataset/manifest.json')
    historical = read(ROOT/'configs/qa-risk/exclusions.json')
    blocked = {k:set(historical[k]) for k in ('ids','context_hashes','question_hashes')}
    old_titles = set()
    for split in ('train','calibration','evaluation'):
        for r in rows(ROOT/f'configs/qa-risk/dataset/{split}/data.jsonl'):
            old_titles.add(r['source_title']); blocked['ids'].add(r['id'])
            blocked['context_hashes'].add(normalized_hash(r['context']))
            blocked['question_hashes'].add(normalized_hash(r['question']))
    seen_titles = set(old_titles); report = {}; source_index = {}
    if source_root is not None:
        for name, digest in manifest['sources'].items():
            require(sha(source_root/name) == digest, 'Official source identity')
            source_index[name] = {q['id']:(a['title'], p['context'], q)
                for a in read(source_root/name)['data'] for p in a['paragraphs'] for q in p['qas']}
    for split, size in (('train_new',2048),('calibration',192),('evaluation',192)):
        data = rows(ROOT/f'configs/qa-expanded/dataset/{split}.jsonl'); info = manifest['splits'][split]
        require(len(data) == size == info['n'], 'Fixed dataset size')
        titles = {r['source_title'] for r in data}
        require(titles == set(info['titles']) and not titles & seen_titles, 'Head article-role overlap')
        if split == 'train_new': require(not titles & set(historical['titles']), 'Historical training article overlap')
        role = {k:set() for k in blocked}; quotas = Counter(); families = Counter()
        for r in data:
            require(r['split'] == split and type(r['is_impossible']) is bool, 'Row role/type')
            require(bool(r['answers']) != r['is_impossible'], 'Gold answerability')
            require(all(a and a in r['context'] for a in r['answers']), 'Gold answer offset/text')
            require(r['family_id'] == normalized_hash(r['context']), 'Context family identity')
            values = dict(ids=r['id'], context_hashes=r['family_id'], question_hashes=normalized_hash(r['question']))
            for key, value in values.items():
                require(value not in blocked[key], 'Historical/cross-role reuse: '+key)
                if key != 'context_hashes': require(value not in role[key], 'Duplicate '+key)
                role[key].add(value)
            quotas[(r['source_title'],r['is_impossible'])] += 1
            families[(r['family_id'],r['is_impossible'])] += 1
            if source_root is not None:
                source = 'train-v2.0.json' if split == 'train_new' else 'dev-v2.0.json'
                title, context, q = source_index[source][r['id']]
                require((title, context, q['question'], q['is_impossible'], list(dict.fromkeys(a['text'] for a in q['answers']))) ==
                        (r['source_title'],r['context'],r['question'],r['is_impossible'],r['answers']), 'Official source row mismatch')
                require(all(context[a['answer_start']:a['answer_start']+len(a['text'])] == a['text'] for a in q['answers']), 'Official source offsets')
        require(len(quotas) == 2*len(titles) and set(quotas.values()) == {info['per_class_per_article']}, 'Article class quotas')
        require(max(families.values()) <= info['max_questions_per_context_per_class'], 'Context class cap')
        require(len(role['context_hashes']) == info['context_families'], 'Family count')
        for key in blocked: blocked[key].update(role[key])
        seen_titles.update(titles)
        report[split] = dict(rows=len(data), articles=len(titles), context_families=len(role['context_hashes']),
            older_qwen_title_overlap=sorted(titles & set(historical['titles'])))
    return dict(roles=report, official_source_rows_checked=2432 if source_root is not None else 0,
                upstream_benchmark_exposure_excluded=False, semantic_near_duplicates_excluded=False)


def independent_probability(model, features):
    score = sum((x-m)/s*w for x,m,s,w in zip(features,model['scaler']['mean'],model['scaler']['scale'],model['weights']))+model['intercept']
    return 1/(1+math.exp(-score)) if score >= 0 else math.exp(score)/(1+math.exp(score))


def check_diagnosis(review, data, predictions, methods, spec):
    """Check published row ledgers and article counts, not just their hashes."""
    verify_hashes(review,read(review/'checksums.json'),exclude=('checksums.json',))
    diagnosis=read(review/'diagnosis.json'); ledger=read(review/'calibration-decisions.json')
    run=read(review/'run.json'); ref=reference(); expected=[]
    require(run['source_sha256']==sha(ROOT/'experiments/diagnose_qa_expanded.py'), 'Diagnosis implementation changed')
    require(diagnosis['protocol_sha256']==sha(ROOT/'configs/qa-expanded/study.json'), 'Diagnosis protocol')
    require(diagnosis['calibration_passed'] is False and diagnosis['evaluation_outputs_read'] is False and diagnosis['threshold_selected'] is None, 'Diagnosis claim')
    indexed={name:{p['id']:p for p in records} for name,records in predictions.items()}
    labels={r['id']:ref.em_label(indexed['original'][r['id']]['prediction'],r['answers']) for r in data}
    limits=spec['quality_constraints']
    for name, method in methods.items():
        points=diagnosis['curves'][name]
        require(len(points)==len(method['curve']), 'Diagnosis grid length')
        for point, truth in zip(points,method['curve']):
            require(aggregates_equal({k:point[k] for k in truth},truth), 'Diagnosis curve')
            failed=[]
            for metric,key in (('accepted_precision','min_accepted_precision'),('answerable_answer_coverage','min_answerable_answer_coverage'),('correct_answerable_coverage','min_correct_answerable_coverage')):
                if truth[metric] is None or truth[metric]<limits[key]: failed.append(metric)
            for metric,key in (('unanswerable_false_accept_rate','max_unanswerable_false_accept_rate'),('invalid_rate','max_invalid_rate')):
                if truth[metric]>limits[key]: failed.append(metric)
            require(point['failed_constraints']==failed, 'Diagnosis failure reason')
            titles={r['source_title'] for r in data}; require(set(point['per_article'])==titles,'Diagnosis article coverage')
            for title in titles:
                group=[r for r in data if r['source_title']==title]
                require(aggregates_equal(metrics(group,[indexed[name][r['id']] for r in group],point['threshold']),point['per_article'][title]), 'Diagnosis article arithmetic')
            for r in data:
                p=indexed[name][r['id']]; accepted=p['confidence']>=point['threshold']
                expected.append(dict(method=name,threshold=point['threshold'],id=r['id'],source_title=r['source_title'],
                    confidence=p['confidence'],raw_em_correct=bool(labels[r['id']]),is_impossible=r['is_impossible'],
                    accepted=accepted,accepted_correct=accepted and bool(labels[r['id']])))
    require(ledger==expected,'Diagnosis row ledger')
    changes=[]
    for r in data:
        key=r['id']; a=indexed['original'][key]['confidence']>=.7; b=indexed['expanded'][key]['confidence']>=.7
        if a!=b: changes.append(dict(id=key,source_title=r['source_title'],is_impossible=r['is_impossible'],
            raw_em_correct=bool(labels[key]),original_accept=a,expanded_accept=b,
            original_score=indexed['original'][key]['confidence'],expanded_score=indexed['expanded'][key]['confidence']))
    categories=dict(Counter(('gained_' if r['expanded_accept'] else 'lost_')+('correct' if r['raw_em_correct'] else 'incorrect') for r in changes))
    require(diagnosis['fixed_0_7_changed_decisions']==changes and diagnosis['fixed_0_7_change_categories']==categories,'Diagnosis paired changes')
    require(diagnosis['raw_correct_answerable']==sum(labels.values()) and diagnosis['answerable']==sum(not r['is_impossible'] for r in data),'Diagnosis raw ceiling')
    return dict(decision_rows=len(ledger),paired_changes=len(changes),all_pass=True)


def audit(folder, full_features=True, replay_fit=False, source_root=None):
    spec_path = ROOT/'configs/qa-expanded/study.json'; protocol = sha(spec_path); spec = read(spec_path)
    for key in ('source_sha256','input_sha256'):
        for name,digest in spec[key].items(): require(sha(safe_path(ROOT,name)) == digest, 'Frozen identity '+name)
    result = dict(evidence_valid=True, protocol_sha256=protocol, datasets=dataset_audit(spec,source_root),
        feature_reenumeration=full_features, archived_matrix_fit_replay=replay_fit, model_inference=False,
        audit_sha256=sha(Path(__file__)), reference_sha256=sha(ROOT/'results/qa-risk-review-v1/independent-feature-audit.py'))
    matrices = {}; collections = {}; feature_checks = {}; ref = reference()
    for split, dirname in (('train_new','train-new'),('calibration','calibration')):
        raw, matrix = load_collection(folder/dirname,split,protocol); matrices[split] = matrix
        data = rows(ROOT/f'configs/qa-expanded/dataset/{split}.jsonl'); collections[split] = data
        require([r['id'] for r in raw] == [r['id'] for r in data], 'Collection data coverage')
        if full_features: feature_checks[split] = check_features(data,raw,matrix,ref)
    training = folder/'training'; verify_hashes(training,read(training/'checksums.json'),exclude=('checksums.json',))
    require(sha(training/'protocol.json') == protocol, 'Training protocol')
    for name,digest in spec['source_sha256'].items(): require(sha(safe_path(training/'source',name)) == digest, 'Training source '+name)
    selection = read(training/'selection.json'); state = read(training/'run.json')
    require(selection['protocol_sha256'] == state['protocol_sha256'] == protocol, 'Training selection identity')
    require(state['status'] == 'complete' and state['attempts_per_method'] == 1 and state['model_inference'] is False, 'Training completion/attempt count')
    for split, dirname in (('train_new','train-new'),('calibration','calibration')):
        require(selection['collection_run_sha256'][split] == sha(folder/dirname/'run.json'), 'Collection link')
    old = read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
    if full_features:
        old_data = rows(ROOT/'configs/qa-risk/dataset/train/data.jsonl')
        old_raw = {r['id']:r for split in ('calibration','evaluation') for r in rows(ROOT/f'results/qa-specialist-v1/{split}-int8/predictions.jsonl')}
        feature_checks['old_train'] = check_features(old_data,[old_raw[r['id']] for r in old_data],old,ref)
    methods = {}; scored = {}; cal = collections['calibration']; features = matrices['calibration']
    for name, expected in (('original',old),('expanded',old+matrices['train_new'])):
        matrix = read(training/(name+'-training-features.json'))
        require(matrix == expected, 'Training concatenation/order')
        fit_record = read(training/(name+'-fit.json')); chosen = selection['methods'][name]
        require(chosen['status'] == 'complete' and chosen['n_train'] == len(expected) == fit_record['n_train'], 'Fit completion/size')
        require(not {'weights','scaler','intercept'} & fit_record.keys(), 'Distributed learned parameters')
        require(fit_record['convergence']['converged'] and fit_record['trace'][-1]['gradient_max_abs'] <= spec['fit_config']['gradient_max_abs_tolerance'], 'Fit convergence')
        predictions = read(training/(name+'-calibration-predictions.json'))
        scored[name] = predictions
        require([r['id'] for r in predictions] == [r['id'] for r in features], 'Score order')
        require(all(p['prediction'] == f['prediction'] for p,f in zip(predictions,features)), 'Raw answer changed by ranking')
        require([c['threshold'] for c in chosen['candidates']] == spec['threshold_grid'], 'Frozen grid')
        eligible = []; curve = []
        for c in chosen['candidates']:
            actual = metrics(cal,predictions,c['threshold']); ok = passes(actual,spec['quality_constraints'])
            require(aggregates_equal(actual,{k:c['summary']['selective'][k] for k in actual}), 'Calibration arithmetic')
            require(c['gate']['all_pass'] == ok, 'Calibration quality gate')
            if ok: eligible.append(c['threshold'])
            curve.append(dict(threshold=c['threshold'],**actual,passes=ok))
        require(chosen['eligible'] == bool(eligible) and chosen['threshold'] == (eligible[0] if eligible else None), 'Threshold selection')
        if replay_fit:
            from lab.qa_risk_calibration import fit
            from experiments.qa_risk import model_digest
            model = fit([r['features'] for r in matrix],[r['target'] for r in matrix])
            require(model_digest(model) == chosen['model_sha256'], 'Archived matrix head identity')
            require(aggregates_equal({k:v for k,v in model.items() if k not in ('weights','scaler','intercept')},fit_record), 'Archived fit trace')
            require(all(math.isclose(independent_probability(model,f['features']),p['confidence'],rel_tol=1e-12,abs_tol=1e-12) for f,p in zip(features,predictions)), 'Reconstructed independent scores')
        methods[name] = dict(converged=True,n_train=len(matrix),eligible=bool(eligible),threshold=chosen['threshold'],curve=curve,
                            model_sha256=chosen['model_sha256'],convergence=fit_record['convergence'])
    require(methods['original']['model_sha256'] == read(ROOT/'results/qa-risk-v2/training/selection.json')['variants']['int8']['model_sha256'], 'Historical head identity')
    require(not any(m['eligible'] for m in methods.values()), 'This frozen study records failure for both methods')
    require(selection['evaluation_permitted'] is False and state['evaluation_permitted'] is False, 'Failed calibration must block evaluation')
    require(not (folder/'evaluation').exists(), 'Forbidden evaluation after failed calibration')
    review=ROOT/'results/qa-expanded-review-v1'
    if review.exists():
        require(read(review/'diagnosis.json')['selection_sha256']==sha(training/'selection.json'),'Diagnosis selection identity')
        result['diagnosis_audit']=check_diagnosis(review,cal,scored,methods,spec)
    result.update(methods=methods,feature_checks=feature_checks,calibration_passed=False,evaluation_run=False,
        default_policy_changed=False,scope='Valid evidence of converged training and failed calibration, not quality acceptance. Fit replay uses independently checked archived floats; arbitrary regenerated cross-platform floats are not proven stable.')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/qa-expanded-v1')
    p.add_argument('--skip-feature-reenumeration',action='store_true',help='Checks hashes/counts only; does not establish raw-to-feature correctness')
    p.add_argument('--replay-fit',action='store_true'); p.add_argument('--source-root',type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=audit(a.root,not a.skip_feature_reenumeration,a.replay_fit,a.source_root)
    write(a.output,result); print({k:result[k] for k in ('evidence_valid','feature_reenumeration','archived_matrix_fit_replay','calibration_passed','evaluation_run')})
