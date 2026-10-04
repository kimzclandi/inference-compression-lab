"""One fixed INT8 disjoint-competitor coverage experiment, with no default change.

Inputs are already observed development evidence. Published risk evaluation was
used for hypothesis diagnosis and can only be a retrospective replay. Learned
parameters stay in memory. One baseline reconstruction and one candidate fit;
no retries, hyperparameter search, model inference, or unseen-evaluation claim.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil

from lab.artifact_integrity import file_hashes, git_identity, safe_path
from lab.evidence import reserve_directory
from lab.qa_metrics import normalize
from lab.qa_nonoverlap_gap import extract_nonoverlap_features
from lab.qa_risk_calibration import extract_features, fit, FIT_CONFIG, predict_probability
from lab.quantization_diagnostics import aggregates_equal, read, rows, sha, write
from lab.selective_qa import evaluate_selective
from lab.qa_gate import evaluate_quality_gate
from experiments.qa_risk import model_digest, select_variant
from experiments.my_span_gap_ablation import public_failure_trace

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/qa-coverage-gap/study.json'


def checked_protocol(path):
    spec = read(path)
    if spec['schema'] != 'qa-coverage-gap-v1' or spec['fit_config'] != FIT_CONFIG:
        raise ValueError('Unknown protocol or changed optimizer')
    if importlib.metadata.version('numpy') != spec['numpy_version']:
        raise ValueError('Pinned NumPy required')
    parent = read(ROOT / 'configs/qa-risk/study.json')
    for key in ('threshold_grid', 'quality_constraints'):
        if spec[key] != parent[key]:
            raise ValueError('Parent thresholds/quality gates changed')
    for section in ('source_sha256', 'input_sha256'):
        if not spec[section]:
            raise ValueError('Empty identity manifest')
        for name, digest in spec[section].items():
            if sha(safe_path(ROOT, name)) != digest:
                raise ValueError('Frozen input/source changed: ' + name)
    return spec


def indexed(values):
    result = {row['id']: row for row in values}
    if len(result) != len(values):
        raise ValueError('Duplicate ID')
    return result


def exact_correct(row, prediction):
    return int(any(normalize(prediction) == normalize(a) for a in row['answers']))


def baseline_diagnosis():
    data = indexed(rows(ROOT / 'results/qa-risk-v2/evaluation-int8/data.jsonl'))
    preds = indexed(rows(ROOT / 'results/qa-risk-v2/evaluation-int8/predictions.jsonl'))
    if data.keys() != preds.keys():
        raise ValueError('Diagnosis ID mismatch')
    records = []
    for uid, row in data.items():
        pred = preds[uid]; feature = extract_features(row['context'], pred)
        if not aggregates_equal(feature, pred['risk_features']):
            raise ValueError('Diagnosis raw feature mismatch')
        alt = feature['diagnostics']['alternative']
        records.append(dict(id=uid, source_title=row['source_title'],
            answerable=not row['is_impossible'], raw_correct=exact_correct(row, pred['prediction']),
            accepted=pred['confidence'] >= .7, confidence=pred['confidence'],
            alternative_overlaps=max(pred['start'], alt['start']) < min(pred['end'], alt['end'])))
    refused = [r for r in records if r['answerable'] and not r['accepted']]
    correct = [r for r in refused if r['raw_correct']]
    return dict(scope='Post-hoc diagnosis of already published outcomes; used for hypothesis selection',
                n=len(records), answerable=sum(r['answerable'] for r in records),
                refused_answerable=len(refused), refused_raw_correct=len(correct),
                correct_refused_with_overlapping_competitor=sum(r['alternative_overlaps'] for r in correct),
                raw_correct_answerable=sum(r['answerable'] and r['raw_correct'] for r in records),
                records=records)


def prepare():
    raw = indexed([p for split in ('calibration', 'evaluation')
                   for p in rows(ROOT / f'results/qa-specialist-v1/{split}-int8/predictions.jsonl')])
    data = {}; matrices = {'baseline': {}, 'disjoint': {}}; audit = {}
    for split, suffix in (('train', 'training'), ('calibration', 'calibration')):
        data[split] = rows(ROOT / f'configs/qa-risk/dataset/{split}/data.jsonl')
        saved = read(ROOT / f'results/qa-risk-v2/training/int8-{suffix}-features.json')
        if [r['id'] for r in saved] != [r['id'] for r in data[split]]:
            raise ValueError('Archived order differs from data')
        before = []; after = []; changes = []; undefined = []
        for row, archived in zip(data[split], saved):
            pred = raw[row['id']]; original = extract_features(row['context'], pred)
            record = dict(id=row['id'], prediction=pred['prediction'], **original,
                          target=exact_correct(row, pred['prediction']))
            if not aggregates_equal(record, archived):
                raise ValueError('Raw/features/labels mismatch: ' + row['id'])
            before.append(deepcopy(archived))
            try:
                feature = extract_nonoverlap_features(row['context'], pred)
            except ValueError as error:
                if str(error) != 'No disjoint differently normalized legal span':
                    raise
                undefined.append(row['id']); continue
            candidate = deepcopy(archived)
            candidate.update(feature_names=feature['feature_names'], diagnostics=feature['diagnostics'])
            candidate['features'][2] = feature['features'][2]
            after.append(candidate)
            if candidate['features'][2] != archived['features'][2]:
                changes.append(row['id'])
            if any(candidate['features'][j] != archived['features'][j] for j in (0, 1, 3, 4)):
                raise ValueError('Non-target feature changed')
        matrices['baseline'][split] = before; matrices['disjoint'][split] = after
        audit[split] = dict(n=len(before), changed_ids=changes, undefined_ids=undefined,
                           changed_count=len(changes), titles=dict(Counter(r['source_title'] for r in data[split])))
    if set(raw) != set(indexed(data['train'] + data['calibration'])):
        raise ValueError('Development coverage mismatch')
    if len(data['train']) != 256 or len(data['calibration']) != 128:
        raise ValueError('Wrong development split sizes')
    return data, matrices, audit


def arm(folder, data, matrices, spec, method):
    train, cal = matrices['train'], matrices['calibration']
    write(folder / 'training-features.json', train)
    write(folder / 'calibration-features.json', cal)
    if len(train) != 256 or len(cal) != 128:
        result = dict(status='feature_failed', reason='Undefined competitor; no imputation or row dropping')
        write(folder / 'status.json', result); return None, result
    try:
        model = fit([r['features'] for r in train], [r['target'] for r in train])
    except ValueError as error:
        result = dict(status='fit_failed', error=str(error), trace=public_failure_trace(error), retried=False)
        write(folder / 'status.json', result); return None, result
    # Reuse only the frozen numeric solver/scorer. Its legacy column labels do
    # not authorize serving this challenger: semantic identity is explicit here
    # and in every feature record, protocol hash and method-specific directory.
    public = {k: v for k, v in model.items() if k not in ('weights', 'intercept', 'scaler')}
    public['experiment_feature_names'] = train[0]['feature_names']
    public['default_runtime_compatible'] = method == 'baseline'
    write(folder / 'fit.json', public)
    selection, predictions = select_variant(data, cal, model, spec)
    if method == 'baseline':
        prior = read(ROOT / 'results/qa-risk-v2/training/selection.json')['variants']['int8']
        if model_digest(model) != prior['model_sha256'] or not aggregates_equal(selection, {k: prior[k] for k in selection}):
            raise ValueError('Baseline reconstruction differs from archived result')
    write(folder / 'selection.json', selection)
    write(folder / 'calibration-predictions.json', predictions)
    result = dict(status='complete', eligible=selection['eligible'], threshold=selection['threshold'],
                  fixed_0_7=next(c for c in selection['candidates'] if c['threshold'] == .7),
                  selected=next((c for c in selection['candidates'] if c['threshold'] == selection['threshold']), None),
                  model_sha256=model_digest(model), convergence=model['convergence'],
                  parameters_distributed=False, experiment_feature_names=train[0]['feature_names'])
    write(folder / 'status.json', result)
    return model, result


def permits_replay(baseline, candidate, spec):
    if candidate.get('selected') is None or baseline.get('selected') is None:
        return False
    b, c = (v['selected']['summary']['selective'] for v in (baseline, candidate))
    return (candidate['selected']['gate']['all_pass']
            and c['correct_answerable_coverage'] - b['correct_answerable_coverage'] >= spec['min_coverage_gain']
            and c['answerable_answer_coverage'] - b['answerable_answer_coverage'] >= spec['min_coverage_gain'])


def replay(out, models, results, spec):
    data = rows(ROOT / 'results/qa-risk-v2/evaluation-int8/data.jsonl')
    raw = indexed(rows(ROOT / 'results/qa-risk-v2/evaluation-int8/predictions.jsonl'))
    summaries = {}; predictions = {}; features = {}
    for method in ('baseline', 'disjoint'):
        predictions[method] = []; features[method] = []
        for row in data:
            pred = raw[row['id']]
            feature = (pred['risk_features'] if method == 'baseline'
                       else extract_nonoverlap_features(row['context'], pred))
            score = predict_probability(models[method], feature['features'])
            predictions[method].append(dict(id=row['id'], prediction=pred['prediction'], confidence=score))
            features[method].append(dict(id=row['id'], **feature))
        threshold = results[method]['threshold']
        summary, detail = evaluate_selective(data, predictions[method], threshold)
        summaries[method] = dict(threshold=threshold, summary=summary,
                                gate=evaluate_quality_gate(summary['selective'], spec['quality_constraints']))
        write(out / (method + '-predictions.json'), predictions[method])
        write(out / (method + '-features.json'), features[method])
        write(out / (method + '-decisions.json'), detail)
    write(out / 'summary.json', dict(methods=summaries,
        scope='Retrospective replay of data used in diagnosis; not independent confirmation or deployment permission'))
    return summaries


def run(output, protocol=PROTOCOL):
    out = reserve_directory(output)
    state = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                 **git_identity(ROOT), python=platform.python_version(),
                 numpy=importlib.metadata.version('numpy'), model_inference=False,
                 fresh_evaluation=False, parameters_distributed=False,
                 authorship='AI-assisted development and experiment')
    try:
        spec = checked_protocol(protocol); state['protocol_sha256'] = sha(protocol)
        shutil.copy2(protocol, out / 'protocol.json')
        for name in spec['source_sha256']:
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(out / 'diagnosis.json', baseline_diagnosis())
        data, matrices, audit = prepare(); write(out / 'feature-audit.json', audit)
        models = {}; results = {}
        for method in ('baseline', 'disjoint'):
            models[method], results[method] = arm(reserve_directory(out / method), data['calibration'], matrices[method], spec, method)
        eligible = permits_replay(results['baseline'], results['disjoint'], spec)
        write(out / 'comparison.json', dict(methods=results, retrospective_replay_permitted=eligible,
            rule='All original point gates and >=5 percentage points improvement in both answerable and correct-answer coverage on calibration',
            scope=spec['scope']))
        if eligible:
            replay(reserve_directory(out / 'retrospective'), models, results, spec)
        state.update(status='complete', candidate_calibration_improved=eligible,
                     retrospective_replay_run=eligible, default_policy_changed=False)
        checked_protocol(protocol)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(out / 'run.json', state); write(out / 'checksums.json', file_hashes(out))
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    print(run(args.output_dir, args.protocol))
