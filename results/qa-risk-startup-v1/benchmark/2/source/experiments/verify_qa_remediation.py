"""Offline artifact, scoring and paired-statistics checks for selective QA evidence."""
import argparse
import json
import math
from pathlib import Path
import random
import re

from lab.artifact_integrity import verify_hashes
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal
from lab.qa_metrics import evaluate
from lab.selective_qa import evaluate_selective, parse_output
from lab.qa_gate import evaluate_quality_gate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _indexed(items):
    result = {item['id']: item for item in items}
    require(len(result) == len(items) and bool(items), 'Exact nonempty ID coverage required')
    return result


def paired_bootstrap(data, baseline, candidate, *, seed=2026100403, replicates=5000):
    """Candidate minus reference; resample context families within each fixed article.

    Inputs contain exact matching IDs; scored records contain EM/F1 in [0, 1].
    Articles are fixed strata and are NOT sampled. Each resampled family keeps
    all its questions; the pooled denominator can vary with family sizes.
    Intervals describe the selected article distribution, not unseen domains.
    """
    data, baseline, candidate = list(data), list(baseline), list(candidate)
    samples, a, b = _indexed(data), _indexed(baseline), _indexed(candidate)
    require(set(samples) == set(a) == set(b), 'Paired comparison requires exact ID coverage')
    require(type(replicates) is int and replicates >= 100, 'Insufficient bootstrap replicates')
    strata, family_titles = {}, {}
    for record in data:
        title, family, identifier = record['source_title'], record['family_id'], record['id']
        require(bool(title) and bool(family), 'Article and family identifiers are required')
        require(family not in family_titles or family_titles[family] == title,
                'Context family belongs to multiple articles')
        family_titles[family] = title
        for metric in ('em', 'f1'):
            for value in (a[identifier][metric], b[identifier][metric]):
                require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                        'Scores must be finite numbers in [0, 1]')
        strata.setdefault(title, {}).setdefault(family, []).append(tuple(
            b[identifier][metric] - a[identifier][metric] for metric in ('em', 'f1')))
    groups = [[(sum(v[0] for v in values), sum(v[1] for v in values), len(values))
               for _, values in sorted(families.items())] for _, families in sorted(strata.items())]
    rng = random.Random(seed); distributions = [[], []]
    for _ in range(replicates):
        total_em, total_f1, n = 0., 0., 0
        for families in groups:
            for _ in families:
                delta_em, delta_f1, count = rng.choice(families)
                total_em += delta_em; total_f1 += delta_f1; n += count
        distributions[0].append(total_em / n); distributions[1].append(total_f1 / n)
    result = dict(n=len(data), articles=len(strata), families=len(family_titles),
                  seed=seed, replicates=replicates,
                  method='paired article-stratified context-family percentile bootstrap',
                  quantiles='sorted draws at floor((replicates-1)*p), p=0.025 and 0.975',
                  scope='Conditional on these fixed articles; articles are not resampled.')
    for metric, values in zip(('em', 'f1'), distributions):
        values.sort(); last = len(values) - 1
        result[metric + '_difference'] = sum(b[i][metric] - a[i][metric] for i in sorted(a)) / len(a)
        result[metric + '_ci95'] = [values[int(last * .025)], values[int(last * .975)]]
    result['em_gained_ids'] = sorted(i for i in a if b[i]['em'] > a[i]['em'])
    result['em_lost_ids'] = sorted(i for i in a if b[i]['em'] < a[i]['em'])
    return result


def tensor_bytes_comparison(fp16_identity, q8_identity):
    """Serialized tensor payload bytes, explicitly not RSS, peak MLX memory or KV."""
    totals = {}
    for label, identity in [('fp16', fp16_identity), ('q8', q8_identity)]:
        tensors = identity.get('tensors')
        require(isinstance(tensors, dict) and bool(tensors), 'Missing tensor identity records')
        for tensor in tensors.values():
            require(type(tensor.get('bytes')) is int and tensor['bytes'] > 0,
                    'Tensor payload bytes must be positive integers')
        totals[label] = sum(tensor['bytes'] for tensor in tensors.values())
    return dict(fp16_tensor_bytes=totals['fp16'], q8_tensor_bytes=totals['q8'],
                q8_to_fp16_ratio=totals['q8'] / totals['fp16'],
                reduction_fraction=1 - totals['q8'] / totals['fp16'],
                scope='All recorded tensor payloads including quantization scales/biases; '
                      'not checkpoint container size, RSS, runtime peak memory or logical KV bytes.')


def _finite(value, name, *, minimum=0, maximum=None):
    require(type(value) in (float, int) and math.isfinite(value) and value >= minimum and
            (maximum is None or value <= maximum), 'Invalid finite numeric field: ' + name)
    return value


def _hash(value, name):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'Invalid SHA256 field: ' + name)


def verify_model_layout(layout, bits):
    distributions = layout.get('dtype_distribution')
    require(isinstance(distributions, dict) and distributions, 'Missing physical tensor distribution')
    for group in distributions.values():
        require(type(group['tensors']) is int and group['tensors'] > 0 and
                type(group['tensor_bytes']) is int and group['tensor_bytes'] > 0,
                'Invalid physical tensor counts')
    require(layout['parameter_tensor_bytes'] == sum(g['tensor_bytes'] for g in distributions.values()) and
            layout['parameter_tensor_count'] == sum(g['tensors'] for g in distributions.values()),
            'Physical tensor bytes/counts disagree with dtype distribution')
    modules = layout['module_counts']
    names = {'quantized_linear', 'floating_linear', 'quantized_embedding', 'floating_embedding'}
    require(set(modules) == names and all(type(v) is int and v >= 0 for v in modules.values()),
            'Invalid module layout')
    quantized = modules['quantized_linear'] + modules['quantized_embedding']
    if bits is None:
        require(quantized == 0 and not layout['quantizer_counts'], 'FP16 contains quantized modules')
        require(set(distributions) == {'mlx.core.float16'}, 'FP16 contains non-FP16 parameter storage')
    else:
        require(quantized > 0 and layout['quantizer_counts'] == {f'bits={bits},group_size=64': quantized},
                'Executed quantizer layout differs from fixed bits/group size')
    norms = layout['normalization_tensors']
    require(bool(norms) and layout['normalizations_all_fp16'] is True and
            len({n['name'] for n in norms}) == len(norms) and
            all(n['is_fp16'] is True and n['dtype'] == 'mlx.core.float16' and
                type(n['tensor_bytes']) is int and n['tensor_bytes'] > 0 for n in norms),
            'Normalization tensor FP16 contract')
    return layout['parameter_tensor_bytes']


def physical_tensor_comparison(fp16_layout, q8_layout):
    a, b = verify_model_layout(fp16_layout, None), verify_model_layout(q8_layout, 8)
    return dict(fp16_tensor_bytes=a, q8_tensor_bytes=b, q8_to_fp16_ratio=b/a,
                reduction_fraction=1-b/a,
                scope='Physical parameter tensor storage including quantization metadata; '
                      'not weight-file bytes, RSS, runtime peak memory, or logical KV bytes.')


def verify_run(folder, *, threshold=None, require_locked=True):
    """Verify one complete runner output and recompute raw/contracted scoring.

    This is offline: hashes bind archived source and registered model files;
    it does not claim to rerun model inference or authenticate a fabricated
    archive. Trust in the fixed protocol comes from its pre-inference commit.
    """
    from experiments.qa_remediation import SOURCE_FILES, validate_locked_run
    folder = Path(folder)
    verify_hashes(folder, read(folder / 'checksums.json'), exclude=('checksums.json',))
    run, spec = read(folder / 'run.json'), read(folder / 'protocol.json')
    require(run['status'] == 'complete' and run.get('stage') == 'complete', 'Run is incomplete')
    require(sha(folder / 'protocol.json') == run['spec_sha256'], 'Protocol identity changed')
    require(sha(folder / 'data.jsonl') == run['dataset_sha256'], 'Dataset identity changed')
    require(set(run['source_sha256']) == set(SOURCE_FILES), 'Source manifest coverage is incomplete')
    verify_hashes(folder / 'source', run['source_sha256'], exclude=())
    require(sha(folder / 'source/lab/qa_metrics.py') == sha(Path('lab/qa_metrics.py')),
            'Historical scoring source changed; inspect before recomputing')
    if 'prompt_source_sha256' in spec:
        require(run['source_sha256']['lab/grounded_qa.py'] == spec['prompt_source_sha256'],
                'Frozen prompt/inference source changed')
    if require_locked:
        require(spec.get('locked') is True, 'Release evidence requires a locked protocol')
    files = run['source_model_files']
    require(isinstance(files, dict) and files and 'config.json' in files, 'Missing source model manifest')
    for name, info in files.items():
        _hash(info['sha256'], 'source model ' + name)
        require(type(info['bytes']) is int and info['bytes'] > 0, 'Invalid source model file bytes')
    matched = validate_locked_run(spec, label=run['model_label'], mode=run['mode'], bits=run['bits'],
                                  model_files_sha256={name: info['sha256'] for name, info in files.items()},
                                  data_sha256=run['dataset_sha256'])
    require(matched == run['locked_spec_validation'], 'Locked run registration record changed')
    verify_model_layout(run['model_layout'], run['bits'])
    config = run['loaded_config']
    require(config['torch_dtype'] == 'float16', 'Executed floating dtype differs from fixed FP16')
    if run['bits'] is None:
        require(not any(key in config for key in ('quantization', 'quantization_config')),
                'FP16 loaded config unexpectedly declares quantization')
    else:
        require(all(config.get(key) == {'bits': run['bits'], 'group_size': 64}
                    for key in ('quantization', 'quantization_config')),
                'Loaded quantization config differs from the recorded layout')
    data, predictions = rows(folder / 'data.jsonl'), rows(folder / 'predictions.jsonl')
    _indexed(data)
    require(predictions == run['predictions'], 'Raw prediction log differs from completed run')
    by_id = _indexed(predictions)
    require(set(by_id) == set(_indexed(data)), 'Prediction ID coverage differs from dataset')
    for row in data:
        pred = by_id[row['id']]
        parsed = parse_output(row['context'], pred['prediction'])
        for field in ('prompt_sha256', 'input_token_ids_sha256'):
            _hash(pred[field], field)
        require(type(pred['input_tokens']) is int and 0 < pred['input_tokens'] <= spec['max_input_tokens'],
                'Input token limit/empty input')
        tokens = pred['token_ids']
        require(isinstance(tokens, list) and 0 < len(tokens) <= spec['max_new_tokens'] and
                all(type(t) is int and t >= 0 for t in tokens) and
                len(tokens) == pred['generated_tokens'], 'Generated token coverage/limit')
        require(pred['stop_reason'] in ('eos', 'max_new_tokens') and
                (pred['stop_reason'] != 'max_new_tokens' or len(tokens) == spec['max_new_tokens']),
                'Generation stop reason')
        for field in ('ttft_seconds', 'decode_seconds', 'total_seconds', 'total_pipeline_seconds'):
            _finite(pred[field], field)
        require(math.isclose(pred['ttft_seconds'] + pred['decode_seconds'], pred['total_seconds'],
                             rel_tol=0, abs_tol=1e-9), 'Generation timing scope arithmetic')
        require(pred['total_pipeline_seconds'] + 1e-9 >= pred['total_seconds'],
                'Pipeline excludes recorded generation duration')
        confidence = _finite(pred['confidence'], 'confidence', maximum=1)
        audit = pred['audit']
        if parsed['status'] == 'answer':
            require(isinstance(audit, dict), 'Valid span is missing its answer audit')
            y, n = audit['yes_logit'], audit['no_logit']
            require(type(y) in (float, int) and type(n) in (float, int) and
                    math.isfinite(y) and math.isfinite(n), 'Nonfinite answer audit logits')
            # Independently reconstruct the two-label softmax with a stable shift.
            offset = max(y, n); yes, no = math.exp(y-offset), math.exp(n-offset)
            expected = yes / (yes + no)
            require(math.isclose(confidence, expected, rel_tol=0, abs_tol=1e-12) and
                    math.isclose(audit['confidence'], expected, rel_tol=0, abs_tol=1e-12),
                    'Audit confidence does not match saved logits')
            _finite(audit['binary_mass'], 'binary audit mass', maximum=1)
            require(type(audit['yes_token']) is int and type(audit['no_token']) is int and
                    audit['yes_token'] >= 0 and audit['no_token'] >= 0 and
                    audit['yes_token'] != audit['no_token'], 'Audit labels are not distinct tokens')
            require(type(audit['input_tokens']) is int and
                    0 < audit['input_tokens'] <= spec['max_input_tokens'], 'Audit input token limit')
            _hash(audit['prompt_sha256'], 'audit prompt')
        else:
            require(audit is None and confidence == 0, 'Non-span output unexpectedly audited')
    metrics, scored = evaluate(data, predictions)
    require(aggregates_equal(metrics, run['metrics']) and scored == run['scored'],
            'Stored QA metrics/per-question scores do not reproduce')
    selective, examples = evaluate_selective(data, predictions, threshold)
    return dict(n=len(data), model_label=run['model_label'], mode=run['mode'], bits=run['bits'],
                data_sha256=run['dataset_sha256'], protocol_sha256=run['spec_sha256'],
                source_sha256=run['source_sha256'], source_model_files=files,
                model_layout=run['model_layout'], protocol=spec,
                data=data, predictions=predictions, metrics=metrics, scored=scored,
                selective=selective, per_example=examples)


def calibration_selection(root, spec, protocol_sha256):
    """Compute every fixed threshold; a failure returns null, never a fallback."""
    root = Path(root); variants = {}; verified = {}
    for precision in ('fp16', 'q8'):
        folder = root / ('calibration-' + precision)
        result = verify_run(folder)
        require(result['protocol_sha256'] == protocol_sha256 and result['protocol'] == spec,
                'Calibration protocols differ from fixed study')
        require(result['data_sha256'] == spec['data_hashes']['calibration'] and result['n'] == 128,
                'Calibration cohort identity/coverage')
        require(result['bits'] == (None if precision == 'fp16' else 8) and
                result['model_label'] == 'calibration-' + precision, 'Calibration precision/label')
        curve = []
        for threshold in spec['threshold_grid']:
            summary, _ = evaluate_selective(result['data'], result['predictions'], threshold)
            gate = evaluate_quality_gate(summary['selective'], spec['quality_constraints'])
            curve.append(dict(threshold=threshold, feasible=gate['all_pass'], metrics=summary, quality_gate=gate))
        feasible = [row for row in curve if row['feasible']]
        variants[precision] = dict(calibration_feasible=bool(feasible),
                                   threshold=min(row['threshold'] for row in feasible) if feasible else None,
                                   run_sha256=sha(folder/'run.json'),
                                   checksums_sha256=sha(folder/'checksums.json'), curve=curve)
        verified[precision] = result
    require(verified['fp16']['source_model_files'] == verified['q8']['source_model_files'],
            'FP16/Q8 were not derived from identical original model files')
    require(verified['fp16']['source_sha256'] == verified['q8']['source_sha256'],
            'Calibration precisions used different inference source')
    require([(p['id'], p['prompt_sha256'], p['input_token_ids_sha256'], p['input_tokens'])
             for p in verified['fp16']['predictions']] ==
            [(p['id'], p['prompt_sha256'], p['input_token_ids_sha256'], p['input_tokens'])
             for p in verified['q8']['predictions']], 'Precision comparison used different prompts')
    passed = all(row['calibration_feasible'] for row in variants.values())
    return dict(schema_version=1, protocol_sha256=protocol_sha256,
                calibration_feasible=passed,
                status='calibration_passed' if passed else 'calibration_failed',
                variants=variants, selection_rule='smallest_feasible_calibration_threshold_per_precision',
                quality_constraints=spec['quality_constraints'], threshold_grid=spec['threshold_grid'],
                confirmation_allowed=passed,
                failure_action=None if passed else 'unavailable_quality; do not consume confirmation',
                physical_tensor_comparison=physical_tensor_comparison(
                    verified['fp16']['model_layout'], verified['q8']['model_layout'])), verified


def _write_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_selection(root, study):
    root, study = Path(root), Path(study)
    require(not (root/'selection.json').exists(), 'Selection already exists; never overwrite it')
    spec, digest = read(study), sha(study)
    selection, _ = calibration_selection(root, spec, digest)
    protocol = root/'protocol.json'
    if protocol.exists():
        require(sha(protocol) == digest, 'Top-level protocol differs from registered study')
    else:
        with protocol.open('xb') as stream:
            stream.write(study.read_bytes())
    _write_exclusive(root/'selection.json', selection)
    if selection['calibration_feasible']:
        _write_exclusive(root/'confirmation-protocol.json', dict(
            protocol_sha256=digest, selection_sha256=sha(root/'selection.json'),
            data_sha256=spec['data_hashes']['confirmation'],
            thresholds={name: row['threshold'] for name, row in selection['variants'].items()},
            prerequisite='Commit this selection and its calibration evidence before any confirmation inference.'))
    return selection


def verify(root, study=None):
    root = Path(root)
    protocol_path = root/'protocol.json'
    spec, digest = read(protocol_path), sha(protocol_path)
    if study is not None:
        require(sha(study) == digest, 'Archived study does not match expected fixed study')
    selection, calibration = calibration_selection(root, spec, digest)
    require(aggregates_equal(selection, read(root/'selection.json')), 'Saved threshold selection/curves do not reproduce')
    common = dict(evidence_valid=True, calibration_feasible=selection['calibration_feasible'],
                  protocol_sha256=digest, selection_sha256=sha(root/'selection.json'),
                  physical_tensor_comparison=selection['physical_tensor_comparison'],
                  scope=spec['scope'])
    if not selection['calibration_feasible']:
        require(not any((root/name).exists() for name in
                        ('confirmation-fp16', 'confirmation-q8', 'confirmation-baseline',
                         'confirmation-protocol.json')), 'Failed calibration must not consume confirmation')
        return dict(**common, status='calibration_failed', overall_success=False,
                    confirmation_evaluated=False, answer_entrypoint='unavailable_quality',
                    variants={name: dict(threshold=value['threshold'],
                                         calibration_feasible=value['calibration_feasible'])
                              for name, value in selection['variants'].items()},
                    compression_confirmed=False)
    confirmation_protocol = read(root/'confirmation-protocol.json')
    thresholds = {name: row['threshold'] for name, row in selection['variants'].items()}
    require(confirmation_protocol['protocol_sha256'] == digest and
            confirmation_protocol['selection_sha256'] == sha(root/'selection.json') and
            confirmation_protocol['data_sha256'] == spec['data_hashes']['confirmation'] and
            confirmation_protocol['thresholds'] == thresholds, 'Confirmation selection binding changed')
    existing = [(root/('confirmation-'+name)).exists() for name in ('fp16', 'q8')]
    if not any(existing):
        return dict(**common, status='ready_for_confirmation', overall_success=False,
                    confirmation_evaluated=False, answer_entrypoint='unavailable_quality',
                    compression_confirmed=False)
    require(all(existing), 'Partial confirmation evidence cannot be reported as complete')
    variants, confirmed = {}, {}
    for precision in ('fp16', 'q8'):
        result = verify_run(root/('confirmation-'+precision), threshold=thresholds[precision])
        require(result['protocol_sha256'] == digest and result['n'] == 256 and
                result['data_sha256'] == spec['data_hashes']['confirmation'],
                'Confirmation protocol/cohort identity or coverage')
        require(result['bits'] == (None if precision == 'fp16' else 8) and
                result['model_label'] == 'confirmation-'+precision, 'Confirmation precision/label')
        require(result['source_model_files'] == calibration[precision]['source_model_files'] and
                result['source_sha256'] == calibration[precision]['source_sha256'] and
                result['model_layout'] == calibration[precision]['model_layout'],
                'Confirmation model/source differs from calibration')
        gate = evaluate_quality_gate(result['selective']['selective'], spec['quality_constraints'])
        variants[precision] = dict(threshold=thresholds[precision], metrics=result['selective'], quality_gate=gate)
        confirmed[precision] = result
    require(confirmed['fp16']['data'] == confirmed['q8']['data'], 'Confirmation datasets differ')
    gate_spec = spec['compression_gate']
    require(gate_spec['replicates'] == 5000 and gate_spec['confidence'] == .95,
            'Fixed bootstrap protocol changed')
    paired = paired_bootstrap(confirmed['fp16']['data'],
                              [r['system_score'] for r in confirmed['fp16']['per_example']],
                              [r['system_score'] for r in confirmed['q8']['per_example']],
                              seed=gate_spec['seed'], replicates=gate_spec['replicates'])
    physical = physical_tensor_comparison(confirmed['fp16']['model_layout'], confirmed['q8']['model_layout'])
    criteria = dict(em_noninferiority=paired['em_ci95'][0] >= gate_spec['minimum_ci_lower_bound'],
                    f1_noninferiority=paired['f1_ci95'][0] >= gate_spec['minimum_ci_lower_bound'],
                    tensor_bytes=physical['q8_to_fp16_ratio'] <= gate_spec['maximum_parameter_tensor_bytes_ratio'],
                    fp16_quality=variants['fp16']['quality_gate']['all_pass'],
                    q8_quality=variants['q8']['quality_gate']['all_pass'])
    passed = all(criteria.values())
    return dict(**common, status='confirmation_complete', overall_success=passed,
                confirmation_evaluated=True, compression_confirmed=passed,
                answer_entrypoint='eligible_research_prototype' if passed else 'unavailable_quality',
                variants=variants, paired=paired, compression_gate=dict(criteria=criteria, passed=passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=Path('configs/qa-remediation/study.json'))
    parser.add_argument('--write-selection', action='store_true')
    parser.add_argument('--single-run', action='store_true')
    args = parser.parse_args()
    if args.single_run:
        result = verify_run(args.root)
        output = {key: result[key] for key in ('n', 'model_label', 'mode', 'bits', 'metrics', 'selective')}
    elif args.write_selection:
        selection = write_selection(args.root, args.study)
        output = {key: selection[key] for key in ('status', 'calibration_feasible', 'confirmation_allowed')}
    else:
        output = verify(args.root, args.study)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
