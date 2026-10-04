"""One fixed, development-only selected-window margin-gap ablation.

Only feature index 2 changes. Baseline and candidate use verified archived
values for all other inputs, the frozen optimizer, and the same calibration
threshold grid. No model inference, new evaluation, or deployment policy edit.
Learned coefficients/scalers stay in memory and are never written.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys

from lab.artifact_integrity import file_hashes, git_identity, safe_path
from lab.evidence import reserve_directory
from lab.extractive_qa import decode
from lab.qa_metrics import normalize
from lab.qa_risk_calibration import extract_features, fit, FIT_CONFIG
from lab.quantization_diagnostics import aggregates_equal, read, rows, sha, write
from experiments.qa_risk import model_digest, select_variant

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/qa-span-gap-ablation/study.json'
METHODS = ('all_windows', 'selected_window')
VARIANTS = ('fp32', 'int8')


def selected_window_features(context, prediction):
    """Validate all windows, but replace only the third feature's competitor set.

    Input: original decoder record plus raw_windows (no labels).
    Output: five features; dimensions 0,1,3,4 stay exactly unchanged, including
    log1p of the ORIGINAL number of windows. No local alternative fails closed.
    """
    original = extract_features(context, prediction)
    index = original['diagnostics']['selected_window_index']
    window = prediction['raw_windows'][index]
    local_prediction = decode(context, [window], max_answer_tokens=30)
    local_prediction['raw_windows'] = [window]
    local = extract_features(context, local_prediction)
    if any(original['features'][i] != local['features'][i] for i in (0, 1, 3)):
        raise ValueError('Single-window restriction changed the selected candidate')
    result = deepcopy(original)
    result['features'][2] = local['features'][2]
    alternative = deepcopy(local['diagnostics']['alternative'])
    alternative['window_index'] = index
    result['diagnostics']['alternative'] = alternative
    result['diagnostics']['competitor_scope'] = 'selected_window_only'
    if result['features'][2] < original['features'][2]:
        raise ValueError('Restricting the alternative set cannot reduce the margin gap')
    return result


def two_window_example():
    # Synthetic token records, not claimed to be actual tokenizer outputs.
    # False-mask separators prevent compound spans from hiding the example.
    context = 'Alpha Beta Alpha Gamma'
    def window(a, b, offsets):
        return dict(start_logits=[0., a / 2, 0., b / 2],
                    end_logits=[0., a / 2, 0., b / 2],
                    offsets=[[0, 0], offsets[0], [0, 0], offsets[1]],
                    context_mask=[False, True, False, True], cls_index=0)
    windows = [window(10., 2., [[0, 5], [6, 10]]),
               window(9.5, 9., [[11, 16], [17, 22]])]
    prediction = decode(context, windows)
    prediction['raw_windows'] = windows
    original = extract_features(context, prediction)
    local = selected_window_features(context, prediction)
    if original['features'][2] != 1. or local['features'][2] != 8.:
        raise ValueError('Hand-derived counterexample does not match implementation')
    return dict(scope='Synthetic decoder-contract example, not QA evaluation',
                context=context, prediction=prediction,
                all_windows=original, selected_window=local,
                explanation='Alpha margin=10 wins. Duplicate Alpha at 9.5 is excluded by normalized '
                            'answer identity. Global competitor Gamma=9 gives gap=1; selected-window '
                            'competitor Beta=2 gives gap=8. Only the third feature changes.')


def checked_protocol(path):
    protocol = read(path)
    if protocol['schema'] != 'qa-span-gap-ablation-v1' or protocol['methods'] != list(METHODS):
        raise ValueError('Unknown ablation protocol')
    if protocol['variants'] != list(VARIANTS) or protocol['fit_config'] != FIT_CONFIG:
        raise ValueError('Fixed variants/optimizer changed')
    if importlib.metadata.version('numpy') != protocol['numpy_version']:
        raise ValueError('Pinned NumPy version required')
    for section in ('source_sha256', 'input_sha256'):
        if not protocol[section]:
            raise ValueError('Empty ablation identity manifest')
        for name, expected in protocol[section].items():
            if name.startswith(('configs/qa-risk/dataset/evaluation/', 'results/qa-risk-v2/evaluation')):
                raise ValueError('New risk evaluation is forbidden in this development-only run')
            if sha(safe_path(ROOT, name)) != expected:
                raise ValueError('Pinned source/input changed: ' + name)
    parent = read(ROOT / 'configs/qa-risk/study.json')
    for key in ('threshold_grid', 'quality_constraints', 'numpy_version', 'fit_config'):
        if protocol[key] != parent[key]:
            raise ValueError('Parent settings changed: ' + key)
    return protocol, parent


def prepare_variant(variant, protocol, parent):
    raw = {}
    # These former specialist evaluation rows are now explicitly development.
    # This never reads the separate 128-row qa-risk-v2 evaluation.
    for role in ('calibration', 'evaluation'):
        for pred in rows(ROOT / parent['development_root'] / (role + '-' + variant) / 'predictions.jsonl'):
            if pred['id'] in raw:
                raise ValueError('Duplicate parent prediction ID')
            raw[pred['id']] = pred
    roles = {}; matrices = {method: {} for method in METHODS}; audit = {}
    for split in ('train', 'calibration'):
        data = rows(ROOT / 'configs/qa-risk/dataset' / split / 'data.jsonl')
        if len(data) != protocol['dataset_sizes'][split] or len({r['id'] for r in data}) != len(data):
            raise ValueError('Role size/ID mismatch')
        saved_name = variant + ('-training-features.json' if split == 'train' else '-calibration-features.json')
        stored = read(ROOT / 'results/qa-risk-v2/training' / saved_name)
        if [r['id'] for r in data] != [r['id'] for r in stored]:
            raise ValueError('Stored feature row order differs from dataset')
        roles[split] = data; baseline = []; local = []; changes = []; failures = []
        windows_count = {}; max_raw_delta = 0.
        for row, archived in zip(data, stored):
            pred = raw[row['id']]
            original = extract_features(row['context'], pred)
            target = int(any(normalize(pred['prediction']) == normalize(a) for a in row['answers']))
            recomputed = dict(id=row['id'], prediction=pred['prediction'], features=original['features'],
                              feature_names=original['feature_names'], diagnostics=original['diagnostics'], target=target)
            if not aggregates_equal(recomputed, archived):
                raise ValueError('Archived raw/features/labels mismatch: ' + row['id'])
            max_raw_delta = max(max_raw_delta, *(abs(a-b) for a,b in zip(original['features'], archived['features'])))
            window_count = len(pred['raw_windows'])
            windows_count[str(window_count)] = windows_count.get(str(window_count), 0) + 1
            base = deepcopy(archived); baseline.append(base)
            try:
                feature = selected_window_features(row['context'], pred)
            except ValueError as error:
                if str(error).startswith('No differently normalized legal span'):
                    failures.append(dict(id=row['id'], error=str(error)))
                    continue
                raise
            candidate = deepcopy(archived)
            # Freeze the other four archived binary64 inputs; isolate feature 3.
            candidate['features'][2] = feature['features'][2]
            candidate['diagnostics'] = feature['diagnostics']
            local.append(candidate)
            if candidate['features'][2] != base['features'][2]:
                changes.append(dict(id=row['id'], source_title=row['source_title'],
                    window_count=window_count, target=target,
                    old_gap=base['features'][2], new_gap=candidate['features'][2],
                    old_alternative=original['diagnostics']['alternative'],
                    new_alternative=feature['diagnostics']['alternative']))
            if any(candidate['features'][i] != base['features'][i] for i in (0,1,3,4)):
                raise ValueError('Non-target feature changed')
        matrices['all_windows'][split] = baseline
        matrices['selected_window'][split] = local
        audit[split] = dict(n=len(data), window_counts=windows_count, changed_gap_rows=changes,
                            local_feature_failures=failures, max_raw_archive_float_difference=max_raw_delta,
                            other_four_features_and_labels_unchanged=True)
    if set(r['id'] for r in roles['train']) & set(r['id'] for r in roles['calibration']):
        raise ValueError('Train/calibration overlap')
    if set(raw) != set(r['id'] for data in roles.values() for r in data):
        raise ValueError('Development rows must cover all and only the old 384 examples')
    return roles, matrices, audit


def public_failure_trace(error):
    # Read only the existing solver's nonparametric trace from its failure frame.
    # Do not change fit(), suppress its exception, retry, or export beta/scaler.
    tb = error.__traceback__
    while tb:
        frame = tb.tb_frame
        if frame.f_code.co_name == 'fit' and Path(frame.f_code.co_filename).resolve() == ROOT / 'lab/qa_risk_calibration.py':
            return deepcopy(frame.f_locals.get('trace', []))
        tb = tb.tb_next
    return []


def fit_arm(folder, data, matrices, protocol):
    train, cal = matrices['train'], matrices['calibration']
    write(folder / 'training-features.json', train)
    write(folder / 'calibration-features.json', cal)
    if len(train) != 256 or len(cal) != 128:
        result = dict(status='feature_failed', reason='Undefined local alternatives; no row dropping or imputation')
        write(folder / 'status.json', result)
        return result
    try:
        model = fit([r['features'] for r in train], [r['target'] for r in train])
    except ValueError as error:
        result = dict(status='fit_failed', error=str(error), trace=public_failure_trace(error), retry=False)
        write(folder / 'status.json', result)
        return result
    public = {k: v for k, v in model.items() if k not in ('weights', 'intercept', 'scaler')}
    write(folder / 'fit.json', public)
    selection, predictions = select_variant(data, cal, model, protocol)
    write(folder / 'selection.json', selection)
    write(folder / 'calibration-predictions.json', predictions)
    fixed = next(c for c in selection['candidates'] if c['threshold'] == .7)
    chosen = next((c for c in selection['candidates'] if c['threshold'] == selection['threshold']), None)
    result = dict(status='complete', eligible=selection['eligible'], threshold=selection['threshold'],
                  fixed_0_7=fixed, selected=chosen, model_sha256=model_digest(model),
                  convergence=model['convergence'], objective_start=model['trace'][0]['objective'],
                  objective_end=model['trace'][-1]['objective'], parameters_distributed=False)
    write(folder / 'status.json', result)
    return result


def run(output_dir, protocol_path=PROTOCOL):
    out = reserve_directory(output_dir)
    state = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                 **git_identity(ROOT), python=platform.python_version(), platform=platform.platform(),
                 argv=sys.argv, evaluation_consumed=False, model_inference_run=False,
                 authorship='AI-assisted implementation and experiment; user mastery unverified')
    try:
        protocol, parent = checked_protocol(protocol_path)
        state['protocol_sha256'] = sha(protocol_path)
        state['numpy'] = importlib.metadata.version('numpy')
        shutil.copy2(protocol_path, out / 'protocol.json')
        for name in protocol['source_sha256']:
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(out / 'counterexample.json', two_window_example())
        results = {}; audits = {}
        for variant in VARIANTS:
            data, matrices, audit = prepare_variant(variant, protocol, parent)
            audits[variant] = audit; results[variant] = {}
            write(out / (variant + '-feature-audit.json'), audit)
            for method in METHODS:
                folder = reserve_directory(out / variant / method)
                results[variant][method] = fit_arm(folder, data['calibration'], matrices[method], protocol)
        checked_protocol(protocol_path)
        write(out / 'comparison.json', dict(variants=results, feature_audits=audits,
            inference_or_evaluation_run=False, scope=protocol['scope']))
        state['status'] = ('complete' if all(r['status']=='complete' for v in results.values() for r in v.values())
                           else 'complete_with_arm_failures')
        return results, state['status']
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        write(out / 'run.json', state)
        write(out / 'checksums.json', file_hashes(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    results, status = run(args.output_dir)
    print(json.dumps(dict(status=status, variants={v:{m:{k:r.get(k) for k in ('status','eligible','threshold')}
          for m,r in methods.items()} for v,methods in results.items()}), ensure_ascii=False))
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
