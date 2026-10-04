"""Collect frozen INT8 logits, expand a fixed five-feature ranking head, evaluate once.

No head parameters enter result folders. Evaluation requires an explicit hash
of a saved calibration selection. All output directories must be new.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil

from lab.artifact_integrity import file_hashes, git_identity, safe_path, verify_hashes
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, rows, sha, write
from lab.qa_risk_calibration import fit, extract_features, predict_probability, FIT_CONFIG
from lab.qa_metrics import normalize
from lab.qa_gate import evaluate_quality_gate
from lab.selective_qa import evaluate_selective
from lab.qa_expanded_io import write_shard, load_collection, paired_coverage
from experiments.qa_risk import select_variant, model_digest
from experiments.my_span_gap_ablation import public_failure_trace

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/'configs/qa-expanded/study.json'


def preflight():
    spec = read(SPEC)
    if spec['schema'] != 'qa-expanded-ranking-v1' or spec['fit_config'] != FIT_CONFIG:
        raise ValueError('Invalid expanded-ranking protocol')
    if importlib.metadata.version('numpy') != spec['numpy_version']:
        raise ValueError('Pinned NumPy required')
    for mapping in ('source_sha256', 'input_sha256'):
        if not spec[mapping]: raise ValueError('Empty source/input manifest')
        for name, digest in spec[mapping].items():
            if sha(safe_path(ROOT, name)) != digest:
                raise ValueError('Frozen source/input changed: ' + name)
    return spec


def start(output, action, spec):
    out = reserve_directory(output)
    state = dict(status='running', action=action, started_at_utc=datetime.now(timezone.utc).isoformat(),
                 **git_identity(ROOT), protocol_sha256=sha(SPEC), python=platform.python_version(),
                 numpy=importlib.metadata.version('numpy'), parameters_distributed=False,
                 authorship='AI-assisted implementation and execution')
    shutil.copy2(SPEC, out/'protocol.json')
    for name in spec['source_sha256']:
        target = out/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    return out, state


def finish(out, state):
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(out/'run.json', state); write(out/'checksums.json', file_hashes(out))


def collect(split, asset_root, output):
    if split not in ('train_new', 'calibration'):
        raise ValueError('Evaluation inference must use the gated evaluate action')
    spec = preflight(); out, state = start(output, 'collect', spec)
    try:
        from lab.qa_specialist_runtime import ExtractiveRuntime
        runtime = ExtractiveRuntime(asset_root, 'int8', spec['asset_manifest_sha256'])
        collect_into(runtime, split, out, state, spec)
        state['status'] = 'complete'; preflight()
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally: finish(out, state)
    return state


def collect_into(runtime, split, out, state, spec):
    data_path = ROOT/f'configs/qa-expanded/dataset/{split}.jsonl'
    data = rows(data_path); shutil.copy2(data_path, out/'data.jsonl')
    state.update(split=split, completed_predictions=0, dataset_sha256=sha(data_path),
                 asset_manifest_sha256=spec['asset_manifest_sha256'], model_inference=True)
    features = []; buffer = []; shard = 0
    for row in data:
        p = runtime.predict(row['context'], row['question'])
        feature = extract_features(row['context'], p)
        p['id'] = row['id']; buffer.append(p)
        features.append(dict(id=row['id'], prediction=p['prediction'], **feature,
            target=int(any(normalize(p['prediction']) == normalize(a) for a in row['answers']))))
        state['completed_predictions'] += 1
        if len(buffer) == 64:
            write_shard(out/f'predictions-{shard:04d}.jsonl.gz', buffer); shard += 1; buffer = []
            print(f"{split}: {state['completed_predictions']}/{len(data)}", flush=True)
    if buffer: write_shard(out/f'predictions-{shard:04d}.jsonl.gz', buffer)
    write(out/'features.json', features)
    print(f'{split}: complete {len(data)}', flush=True)
    return data, features


def train(train_dir, cal_dir, head_dir, output):
    spec = preflight(); out, state = start(output, 'train', spec)
    try:
        heads = reserve_heads(head_dir)
        _, new = load_collection(train_dir, 'train_new', sha(SPEC))
        _, cal = load_collection(cal_dir, 'calibration', sha(SPEC))
        old = read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
        data = rows(ROOT/'configs/qa-expanded/dataset/calibration.jsonl')
        if len(new) != 2048 or len(old) != 256 or len(cal) != 192:
            raise ValueError('Fixed matrix sizes changed')
        if len({r['id'] for r in old+new+cal}) != len(old+new+cal):
            raise ValueError('Matrix roles overlap')
        selection = dict(protocol_sha256=sha(SPEC), methods={},
                         collection_run_sha256={'train_new':sha(train_dir/'run.json'), 'calibration':sha(cal_dir/'run.json')})
        for method, matrix in (('original', old), ('expanded', old+new)):
            write(out/(method+'-training-features.json'), matrix)
            try:
                model = fit([r['features'] for r in matrix], [r['target'] for r in matrix])
            except ValueError as error:
                failure = dict(eligible=False, status='fit_failed', threshold=None,
                               error=str(error), trace=public_failure_trace(error), retry=False)
                write(out/(method+'-failure.json'), failure); selection['methods'][method] = failure
                continue
            if method == 'original':
                prior = read(ROOT/'results/qa-risk-v2/training/selection.json')['variants']['int8']['model_sha256']
                if model_digest(model) != prior: raise ValueError('Original head identity mismatch')
            write(heads/(method+'.json'), model)
            write(out/(method+'-fit.json'), {k:v for k,v in model.items() if k not in ('weights','intercept','scaler')})
            chosen, predictions = select_variant(data, cal, model, spec)
            selection['methods'][method] = dict(chosen, status='complete', model_sha256=model_digest(model), n_train=len(matrix))
            write(out/(method+'-calibration-predictions.json'), predictions)
        selection['evaluation_permitted'] = selection['methods']['expanded']['eligible']
        write(out/'selection.json', selection)
        state.update(status='complete', evaluation_permitted=selection['evaluation_permitted'],
                     model_inference=False, attempts_per_method=1)
        preflight()
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally: finish(out, state)
    return state


def reserve_heads(path):
    if (ROOT/'runs').is_symlink() or not Path(path).resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Learned heads must remain under the local ignored runs directory')
    return reserve_directory(path)


def rebuild(training_dir, head_dir, output):
    """Reconstruct archived inputs for local use, not another training search."""
    spec = preflight(); out, state = start(output, 'rebuild', spec)
    try:
        verify_hashes(training_dir, read(training_dir/'checksums.json'), exclude=('checksums.json',))
        selection = read(training_dir/'selection.json')
        if selection['protocol_sha256'] != sha(SPEC): raise ValueError('Wrong reconstruction protocol')
        heads = reserve_heads(head_dir); identities = {}
        for name, selected in selection['methods'].items():
            if selected['status'] != 'complete': continue
            matrix = read(training_dir/(name+'-training-features.json'))
            model = fit([r['features'] for r in matrix], [r['target'] for r in matrix])
            if model_digest(model) != selected['model_sha256']:
                raise ValueError('Archived-input reconstruction differs: ' + name)
            write(heads/(name+'.json'), model); identities[name] = model_digest(model)
        state.update(status='complete', reconstructed=identities, model_inference=False)
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally: finish(out, state)
    return state


def evaluate(training_dir, head_dir, asset_root, output, selection_sha256):
    spec = preflight()
    verify_hashes(training_dir, read(training_dir/'checksums.json'), exclude=('checksums.json',))
    selection = read(training_dir/'selection.json')
    if sha(training_dir/'selection.json') != selection_sha256 or not selection['evaluation_permitted']:
        raise ValueError('Explicit eligible frozen selection required before evaluation')
    if selection['protocol_sha256'] != sha(SPEC): raise ValueError('Wrong selection protocol')
    models = {}
    for name in ('original', 'expanded'):
        models[name] = read(head_dir/(name+'.json'))
        if model_digest(models[name]) != selection['methods'][name]['model_sha256']:
            raise ValueError('Local head identity mismatch')
    out, state = start(output, 'evaluate', spec)
    state['selection_sha256'] = selection_sha256
    try:
        shutil.copy2(training_dir/'selection.json', out/'selection.json')
        from lab.qa_specialist_runtime import ExtractiveRuntime
        runtime = ExtractiveRuntime(asset_root, 'int8', spec['asset_manifest_sha256'])
        data, features = collect_into(runtime, 'evaluation', out, state, spec)
        policies = {'original_frozen': ('original', .7),
                    'expanded': ('expanded', selection['methods']['expanded']['threshold'])}
        if selection['methods']['original']['eligible']:
            policies['original_recalibrated'] = ('original', selection['methods']['original']['threshold'])
        results = {}; indicators = {}
        for name, (method, threshold) in policies.items():
            predictions = [dict(id=r['id'], prediction=r['prediction'], confidence=predict_probability(models[method],r['features'])) for r in features]
            summary, details = evaluate_selective(data, predictions, threshold)
            results[name] = dict(threshold=threshold, summary=summary,
                gate=evaluate_quality_gate(summary['selective'], spec['quality_constraints']))
            indicators[name] = [dict(id=r['id'], accepted_correct=bool(r['decision']['accepted'] and r['system_score']['em']==1.)) for r in details]
            write(out/(name+'-predictions.json'), predictions)
            write(out/(name+'-decisions.json'), details)
        comparisons = {name: paired_coverage(data, indicators[name], indicators['expanded'], **spec['bootstrap'])
                       for name in results if name != 'expanded'}
        primary = comparisons.get('original_recalibrated')
        success = bool(primary and results['expanded']['gate']['all_pass']
                       and primary['correct_coverage_difference'] >= spec['min_correct_coverage_gain']
                       and primary['ci95'][0] > 0)
        write(out/'summary.json', dict(policies=results, paired_correct_coverage=comparisons,
            training_improvement_confirmed_on_local_cohort=success, default_policy_changed=False,
            scope=spec['scope']))
        state.update(status='complete', improvement_gate_passed=success, default_policy_changed=False)
        preflight()
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally: finish(out, state)
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    c = sub.add_parser('collect'); c.add_argument('--split', choices=['train_new','calibration'], required=True)
    c.add_argument('--asset-root', type=Path, required=True)
    t = sub.add_parser('train'); t.add_argument('--train-dir', type=Path, required=True); t.add_argument('--cal-dir', type=Path, required=True)
    t.add_argument('--head-dir', type=Path, required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--training-dir', type=Path, required=True)
    e.add_argument('--head-dir', type=Path, required=True); e.add_argument('--asset-root', type=Path, required=True)
    e.add_argument('--selection-sha256', required=True)
    r = sub.add_parser('rebuild'); r.add_argument('--training-dir', type=Path, required=True)
    r.add_argument('--head-dir', type=Path, required=True)
    for x in (c,t,e,r):x.add_argument('--output-dir', type=Path, required=True)
    a = vars(p.parse_args()); action = a.pop('action'); a['output'] = a.pop('output_dir')
    print(globals()[action](**a))
