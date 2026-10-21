"""Offline identity, independent span decoding and fixed selective-QA verification.

This verifies a recorded public-benchmark experiment, not unseen-model data,
production suitability, or the authenticity of arbitrarily fabricated logits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from lab.artifact_integrity import safe_path, verify_hashes
from lab.quantization_diagnostics import read, rows, sha, aggregates_equal
from lab.qa_gate import evaluate_quality_gate
from lab.selective_qa import evaluate_selective
from experiments.verify_qa_remediation import paired_bootstrap

VARIANTS = ('fp32', 'int8')
REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite numeric field: ' + name)
    require(minimum is None or value >= minimum, 'Numeric field below bound: ' + name)
    require(maximum is None or value <= maximum, 'Numeric field above bound: ' + name)
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid integer field: ' + name)
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256: ' + name)


def indexed(items):
    require(isinstance(items, list) and bool(items), 'Nonempty record list required')
    require(all(isinstance(x.get('id'), str) and x['id'] for x in items), 'Record ID missing')
    output = {x['id']: x for x in items}
    require(len(output) == len(items), 'Duplicate IDs')
    return output


def independently_decode(context, windows, limits):
    """Enumerate legal token pairs directly, independently of lab.extractive_qa."""
    require(isinstance(context, str) and context.strip() and len(context) <= limits['max_context_characters'],
            'Context input contract')
    require(isinstance(windows, list) and 1 <= len(windows) <= limits['max_windows'], 'Window count')
    candidates = []
    fields = ('start_logits', 'end_logits', 'offsets', 'context_mask', 'input_ids', 'attention_mask', 'sequence_ids')
    for wi, window in enumerate(windows):
        require(all(isinstance(window.get(key), list) for key in fields), 'Raw window token arrays missing')
        n = len(window['input_ids'])
        require(1 <= n <= limits['max_sequence_tokens'] and all(len(window[k]) == n for k in fields),
                'Raw window token coverage')
        require(window.get('cls_index') == 0 and type(window.get('cls_index')) is int and
                window['input_ids'][0] == 0 and window['sequence_ids'][0] is None,
                'RoBERTa CLS identity')
        previous = None
        for i in range(n):
            integer(window['input_ids'][i], 'token ID')
            require(type(window['attention_mask'][i]) is int and window['attention_mask'][i] == 1,
                    'Unpadded window attention mask')
            sid = window['sequence_ids'][i]
            require(sid is None or (type(sid) is int and sid in (0, 1)), 'Invalid sequence ID')
            mask = window['context_mask'][i]
            require(type(mask) is bool and mask == (sid == 1), 'Context mask differs from token provenance')
            for key in ('start_logits', 'end_logits'):
                number(window[key][i], key)
            offsets = window['offsets'][i]
            require(isinstance(offsets, list) and len(offsets) == 2, 'Invalid offsets')
            left, right = offsets
            integer(left, 'offset start'); integer(right, 'offset end')
            require(left <= right, 'Reversed offsets')
            if mask:
                require(left <= right <= len(context), 'Context offsets outside exact source')
                if previous is not None:
                    require(left >= previous[0] and right >= previous[1], 'Nonmonotonic context offsets')
                previous = (left, right)
        require(not window['context_mask'][0], 'CLS is marked as context')
        require(sum(sid == 0 for sid in window['sequence_ids']) <= limits['max_question_tokens'],
                'Question token limit')
        null = number(window['start_logits'][0] + window['end_logits'][0], 'null score')
        valid = []
        for first in range(n):
            for last in range(first, min(n, first + limits['max_answer_tokens'])):
                if not all(window['context_mask'][first:last + 1]):
                    continue
                if (window['offsets'][first][0] == window['offsets'][first][1] or
                        window['offsets'][last][0] == window['offsets'][last][1]):
                    continue
                left, right = window['offsets'][first][0], window['offsets'][last][1]
                raw = context[left:right]
                stripped = raw.strip()
                if not stripped:
                    continue
                left += len(raw) - len(raw.lstrip())
                right -= len(raw) - len(raw.rstrip())
                score = number(window['start_logits'][first] + window['end_logits'][last], 'span score')
                valid.append((score, -first, -last, left, right))
        require(bool(valid), 'Window has no legal span')
        score, neg_first, neg_last, left, right = max(valid)
        first, last = -neg_first, -neg_last
        margin = number(score - null, 'span/null margin')
        if margin >= 0:
            confidence = 1 / (1 + math.exp(-margin))
        else:
            exp = math.exp(margin); confidence = exp / (1 + exp)
        candidates.append(dict(prediction=context[left:right], start=left, end=right,
            start_token=first, end_token=last, span_score=score,
            start_logit=window['start_logits'][first], end_logit=window['end_logits'][last],
            window_index=wi, null_score=null, margin=margin,
            cls_start_logit=window['start_logits'][0], cls_end_logit=window['end_logits'][0],
            confidence=confidence))
    chosen = min(candidates, key=lambda x: (-x['margin'], x['window_index']))
    return dict(**chosen, context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                max_answer_tokens=limits['max_answer_tokens'], windows=candidates)


def verify_assets(manifest):
    require(manifest['model_id'] == 'deepset/roberta-base-squad2' and
            manifest['revision'] == 'adc3b06f79f797d1c575d5479d6f5efe54a9e3b4', 'Pinned upstream model identity')
    require(set(manifest['models']) == set(VARIANTS), 'Model precision coverage')
    names = []
    for label, info in manifest['models'].items():
        require(info['file'] == label + '.onnx', 'Unexpected ONNX asset path')
        digest(info['sha256'], label); integer(info['bytes'], label + ' file bytes', 1)
        names.append(info['file'])
    require(len(set(names)) == len(names), 'Precision aliases same asset')
    expected_tokenizer = {'config.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json'}
    require(set(manifest['tokenizer_files']) == expected_tokenizer, 'Tokenizer file identity coverage')
    require(set(manifest['source_files']) == expected_tokenizer | {'README.md', 'model.safetensors'},
            'Upstream source identity coverage')
    for name, info in manifest['source_files'].items():
        safe_path(Path('.'), name); digest(info['sha256'], name); integer(info['bytes'], name, 1)
    for name, value in manifest['tokenizer_files'].items():
        require(value == manifest['source_files'][name]['sha256'], 'Tokenizer differs from upstream source')
    q = manifest['quantization']
    require(q['weight_type'] == 'QInt8' and q['per_channel'] is True and q['reduce_range'] is False and
            q['MatMulConstBOnly'] is True and q['excluded_nodes'] == ['/model/qa_outputs/MatMul'],
            'Fixed dynamic INT8 / floating classifier contract changed')
    counts = manifest['graph_operator_counts']
    require(counts['fp32'].get('MatMulInteger', 0) == 0 and counts['int8'].get('MatMulInteger', 0) > 0,
            'Recorded graph has no integer transformation')
    parity = manifest['fp32_export_parity']
    require(isinstance(parity, list) and len(parity) == 3, 'Export parity coverage')
    for record in parity:
        require(record['passed'] is True and record['tolerance'] == .001 and
                len(record['max_abs_logit_errors']) == 2, 'Export parity contract')
        for error in record['max_abs_logit_errors']:
            number(error, 'export logit error', 0, .001)
    return dict(fp32_onnx_bytes=manifest['models']['fp32']['bytes'],
                int8_onnx_bytes=manifest['models']['int8']['bytes'],
                int8_to_fp32_ratio=manifest['models']['int8']['bytes'] / manifest['models']['fp32']['bytes'],
                scope='ONNX file bytes only, not RSS, peak runtime memory, activation storage, or speedup.')


def verify_run(folder, *, threshold=None):
    from experiments.qa_specialist import SOURCE_FILES
    from lab.qa_specialist_runtime import LIMITS, RUNTIME_PACKAGES
    folder = Path(folder)
    verify_hashes(folder, read(folder/'checksums.json'), exclude=('checksums.json',))
    run, spec = read(folder/'run.json'), read(folder/'protocol.json')
    require(spec.get('locked') is True and run['status'] == 'complete' and run['stage'] == 'complete',
            'Run is not complete and locked')
    require(sha(folder/'protocol.json') == run['protocol_sha256'], 'Protocol identity')
    require(sha(folder/'data.jsonl') == run['dataset_sha256'] == spec['data_sha256'][run['split']],
            'Frozen dataset identity')
    require(run['split'] in ('calibration', 'evaluation') and run['variant'] in VARIANTS, 'Run partition/precision')
    require(run['source_sha256'] == spec['source_sha256'] and set(run['source_sha256']) == set(SOURCE_FILES),
            'Source identity coverage')
    verify_hashes(folder/'source', run['source_sha256'], exclude=())
    for name, value in run['source_sha256'].items():
        require(sha(REPO/name) == value, 'Current inference/scoring differs from archived source: ' + name)
    require(spec['input_limits'] == LIMITS, 'Frozen runtime input contract')
    require(sha(folder/'assets.json') == spec['asset_manifest_sha256'], 'Asset manifest differs from frozen study')
    assets = read(folder/'assets.json')
    require(assets == run['assets'], 'Embedded asset manifest differs from original bytes')
    require(spec['model_id'] == assets['model_id'] and spec['revision'] == assets['revision'],
            'Study names another source model')
    file_size = verify_assets(assets)
    require(run['provider'] == 'CPUExecutionProvider' and run['intra_op_threads'] == 4 and
            run['inter_op_threads'] == 1, 'Runtime provider/thread contract')
    require(run['packages'] == {name: assets['packages'][name] for name in RUNTIME_PACKAGES},
            'Runtime packages differ from prepared assets')
    number(run['model_load_seconds'], 'model load duration', 0)
    data, predictions = rows(folder/'data.jsonl'), rows(folder/'predictions.jsonl')
    by_data, by_pred = indexed(data), indexed(predictions)
    require(type(spec['dataset_sizes'][run['split']]) is int and
            len(data) == spec['dataset_sizes'][run['split']], 'Fixed dataset size differs from evidence')
    require(set(by_data) == set(by_pred), 'Exact prediction ID coverage')
    require(type(run['completed_predictions']) is int and run['completed_predictions'] == len(predictions),
            'Completed prediction count differs from raw evidence')
    for row in data:
        require(row['split'] == run['split'] and type(row['is_impossible']) is bool,
                'Dataset partition or answerability type')
        require(isinstance(row['question'], str) and row['question'].strip() and len(row['question']) <= 1000,
                'Question character contract')
        require(isinstance(row['answers'], list) and all(isinstance(a, str) and a for a in row['answers']) and
                bool(row['answers']) != row['is_impossible'], 'Answerability/gold label consistency')
        require(row['family_id'] and row['source_title'], 'Missing article/family unit')
        pred = by_pred[row['id']]
        reconstructed = independently_decode(row['context'], pred['raw_windows'], LIMITS)
        require(all(key in pred for key in reconstructed), 'Missing decoded candidate field')
        require(aggregates_equal(reconstructed, {key: pred[key] for key in reconstructed}),
                'Saved span, null score, margin, confidence or window candidates do not re-decode')
        require(pred['feature_count'] == len(pred['raw_windows']) and type(pred['feature_count']) is int and
                pred['input_tokens'] == [len(w['input_ids']) for w in pred['raw_windows']], 'Window telemetry coverage')
        for key in ('tokenization_seconds', 'inference_seconds', 'decode_seconds', 'total_pipeline_seconds'):
            number(pred[key], key, 0)
        require(pred['total_pipeline_seconds'] + 1e-6 >= sum(pred[k] for k in
                ('tokenization_seconds', 'inference_seconds', 'decode_seconds')), 'Pipeline excludes component timing')
    if run['split'] == 'evaluation':
        require(sha(folder/'selection.json') == run['selection_sha256'], 'Evaluation selection identity')
    selective, per_example = evaluate_selective(data, predictions, threshold)
    return dict(n=len(data), variant=run['variant'], split=run['split'],
                protocol=spec, protocol_sha256=run['protocol_sha256'], dataset_sha256=run['dataset_sha256'],
                source_sha256=run['source_sha256'], assets=assets, file_size_comparison=file_size,
                data=data, predictions=predictions, selective=selective, per_example=per_example, run=run)


def _same_inputs(first, second):
    require(first['data'] == second['data'], 'Precision comparison datasets differ')
    a, b = indexed(first['predictions']), indexed(second['predictions'])
    for identifier in a:
        keys = ('input_ids', 'attention_mask', 'offsets', 'sequence_ids', 'context_mask', 'cls_index')
        aa = [{k: w[k] for k in keys} for w in a[identifier]['raw_windows']]
        bb = [{k: w[k] for k in keys} for w in b[identifier]['raw_windows']]
        require(aa == bb, 'Precision comparison tokenization differs')
    require(first['assets'] == second['assets'] and first['source_sha256'] == second['source_sha256'],
            'Precision comparison source/model identities differ')


def calibration_selection(root, spec, protocol_sha256):
    root = Path(root); variants = {}; verified = {}
    grid = spec['threshold_grid']
    require(isinstance(grid, list) and bool(grid) and len(set(grid)) == len(grid), 'Nonempty unique threshold grid')
    for t in grid: number(t, 'threshold', 0, 1)
    for variant in VARIANTS:
        folder = root/('calibration-' + variant); result = verify_run(folder)
        require(result['protocol'] == spec and result['protocol_sha256'] == protocol_sha256 and
                result['split'] == 'calibration' and result['variant'] == variant, 'Calibration study/variant identity')
        curve = []
        for threshold in grid:
            metrics, _ = evaluate_selective(result['data'], result['predictions'], threshold)
            gate = evaluate_quality_gate(metrics['selective'], spec['quality_constraints'])
            curve.append(dict(threshold=threshold, feasible=gate['all_pass'], metrics=metrics, quality_gate=gate))
        feasible = [item['threshold'] for item in curve if item['feasible']]
        variants[variant] = dict(calibration_feasible=bool(feasible), threshold=min(feasible) if feasible else None,
            run_sha256=sha(folder/'run.json'), checksums_sha256=sha(folder/'checksums.json'), curve=curve)
        verified[variant] = result
    _same_inputs(verified['fp32'], verified['int8'])
    passed = all(value['calibration_feasible'] for value in variants.values())
    return dict(schema_version=1, protocol_sha256=protocol_sha256, calibration_feasible=passed,
                status='calibration_passed' if passed else 'calibration_failed', variants=variants,
                selection_rule='smallest_feasible_calibration_threshold_per_precision',
                threshold_grid=grid, quality_constraints=spec['quality_constraints'], evaluation_allowed=passed,
                failure_action=None if passed else 'unavailable_quality; do not consume evaluation',
                file_size_comparison=verified['fp32']['file_size_comparison'])


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False); out.write('\n')


def write_selection(root, study):
    root, study = Path(root), Path(study)
    require(not (root/'selection.json').exists() and not (root/'evaluation-protocol.json').exists(),
            'Never overwrite selection or evaluation authorization')
    spec, protocol_sha = read(study), sha(study)
    selection = calibration_selection(root, spec, protocol_sha)
    if (root/'protocol.json').exists():
        require(sha(root/'protocol.json') == protocol_sha, 'Archived top-level protocol differs')
    else:
        with (root/'protocol.json').open('xb') as out: out.write(study.read_bytes())
    exclusive(root/'selection.json', selection)
    if selection['calibration_feasible']:
        exclusive(root/'evaluation-protocol.json', dict(protocol_sha256=protocol_sha,
            selection_sha256=sha(root/'selection.json'), data_sha256=spec['data_sha256']['evaluation'],
            thresholds={k: v['threshold'] for k, v in selection['variants'].items()},
            prerequisite='Commit this selection and calibration evidence before evaluation inference.',
            scope='Fixed public benchmark evaluation; not unseen-model confirmation.'))
    return selection


def verify(root, study=None):
    root = Path(root); spec = read(root/'protocol.json'); protocol_sha = sha(root/'protocol.json')
    if study is not None: require(sha(study) == protocol_sha, 'Archived study differs from expected protocol')
    selection = calibration_selection(root, spec, protocol_sha)
    require(aggregates_equal(selection, read(root/'selection.json')), 'Saved threshold selection does not reproduce')
    common = dict(evidence_valid=True, calibration_feasible=selection['calibration_feasible'],
                  protocol_sha256=protocol_sha, selection_sha256=sha(root/'selection.json'),
                  file_size_comparison=selection['file_size_comparison'],
                  scope=spec.get('scope', 'Fixed public benchmark evaluation; not unseen-model confirmation.'))
    if not selection['calibration_feasible']:
        require(not any((root/name).exists() for name in ('evaluation-fp32','evaluation-int8','evaluation-protocol.json')),
                'Failed calibration must not consume evaluation')
        return dict(**common, status='calibration_failed', evaluation_evaluated=False, overall_success=False,
                    compression_confirmed=False, answer_entrypoint='unavailable_quality')
    authorization = read(root/'evaluation-protocol.json')
    thresholds = {k: v['threshold'] for k, v in selection['variants'].items()}
    require(authorization['protocol_sha256'] == protocol_sha and
            authorization['selection_sha256'] == sha(root/'selection.json') and
            authorization['data_sha256'] == spec['data_sha256']['evaluation'] and
            authorization['thresholds'] == thresholds, 'Evaluation authorization binding changed')
    existing = [(root/('evaluation-' + name)).exists() for name in VARIANTS]
    if not any(existing):
        return dict(**common, status='ready_for_evaluation', evaluation_evaluated=False, overall_success=False,
                    compression_confirmed=False, answer_entrypoint='unavailable_quality')
    require(all(existing), 'Partial evaluation cannot pass')
    verified, variants = {}, {}
    for variant in VARIANTS:
        folder = root/('evaluation-' + variant); result = verify_run(folder, threshold=thresholds[variant])
        require(result['protocol'] == spec and result['protocol_sha256'] == protocol_sha and
                result['split'] == 'evaluation' and result['variant'] == variant, 'Evaluation protocol/precision')
        require(sha(folder/'selection.json') == sha(root/'selection.json'), 'Evaluation used another selection')
        gate = evaluate_quality_gate(result['selective']['selective'], spec['quality_constraints'])
        variants[variant] = dict(threshold=thresholds[variant], metrics=result['selective'], quality_gate=gate)
        verified[variant] = result
    _same_inputs(verified['fp32'], verified['int8'])
    config = spec['compression_gate']
    require(config['replicates'] == 5000 and config['confidence'] == .95, 'Frozen bootstrap configuration')
    paired = paired_bootstrap(verified['fp32']['data'],
        [x['system_score'] for x in verified['fp32']['per_example']],
        [x['system_score'] for x in verified['int8']['per_example']], seed=config['seed'], replicates=config['replicates'])
    criteria = dict(em_noninferiority=paired['em_ci95'][0] >= config['minimum_ci_lower_bound'],
        f1_noninferiority=paired['f1_ci95'][0] >= config['minimum_ci_lower_bound'],
        model_file_bytes=common['file_size_comparison']['int8_to_fp32_ratio'] <= config['maximum_model_file_bytes_ratio'],
        fp32_quality=variants['fp32']['quality_gate']['all_pass'], int8_quality=variants['int8']['quality_gate']['all_pass'])
    passed = all(criteria.values())
    return dict(**common, status='evaluation_complete', evaluation_evaluated=True, overall_success=passed,
                compression_confirmed=passed, answer_entrypoint='bounded_local_prototype' if passed else 'unavailable_quality',
                variants=variants, paired=paired, compression_gate=dict(criteria=criteria, passed=passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=Path('configs/qa-specialist/study.json'))
    parser.add_argument('--write-selection', action='store_true'); parser.add_argument('--single-run', action='store_true')
    args = parser.parse_args()
    if args.single_run:
        result = verify_run(args.root)
        output = {k: result[k] for k in ('n','variant','split','selective','file_size_comparison')}
    elif args.write_selection:
        output = write_selection(args.root, args.study)
    else: output = verify(args.root, args.study)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
