"""Persist one failed Attention case; replay archived arrays without a device."""
import hashlib
import json
from pathlib import Path

import numpy as np


class AttentionValidationFailure(RuntimeError):
    """The original exception is chained; public receipts omit private messages."""


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def comparison_metrics(actual, reference, atol, rtol):
    """Replay descriptive FP32 metrics; the runner retains its torch comparator."""
    result = dict(actual_shape=None if actual is None else list(actual.shape),
                  reference_shape=None if reference is None else list(reference.shape),
                  actual_finite=None if actual is None else bool(np.isfinite(actual).all()),
                  reference_finite=None if reference is None else bool(np.isfinite(reference).all()),
                  shapes_equal=None, max_abs_error=None, normalized_error=None,
                  array_tolerance_pass=None)
    if actual is None or reference is None:
        return result
    result['shapes_equal'] = actual.shape == reference.shape
    if not result['shapes_equal'] or not result['actual_finite'] or not result['reference_finite']:
        return result
    if not actual.size:
        return result
    # Match the existing runner's float32 descriptive arithmetic. Infinity from
    # a subtraction/ratio overflow is represented by null, never invalid JSON.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        difference = np.abs(actual - reference)
        bound = atol + rtol * np.abs(reference)
        absolute = float(np.max(difference))
        normalized = float(np.max(difference / bound))
    result.update(max_abs_error=absolute if np.isfinite(absolute) else None,
                  normalized_error=normalized if np.isfinite(normalized) else None,
                  array_tolerance_pass=bool(np.all(difference <= bound)))
    return result


def validate_case(output, checks, case, inputs, arms, evaluate, compare, atol, rtol):
    """Evaluate a reference and each arm, saving each receipt before proceeding.

    ``inputs`` contains host Q/K/V arrays. ``evaluate(arm)`` returns the complete
    float32 host output; it may include device transfer/synchronization. ``compare``
    is the unchanged backend assertion. Failure stops this shape before timing.
    Only the failed case's arrays are retained, including any available outputs.
    """
    output = Path(output)
    reference = actual = None
    arm, stage = 'reference', 'reference_execution'
    try:
        reference = np.asarray(evaluate('eager_fp32'))
        stage = 'reference_validation'
        expected_shape = (*inputs[0].shape[:-1], inputs[2].shape[-1])
        if reference.shape != expected_shape or reference.dtype != np.float32 or not np.isfinite(reference).all():
            raise ValueError('Invalid reference shape, dtype or finite values')
        for arm in arms:
            actual = None
            stage = 'backend_execution'
            actual = np.asarray(evaluate(arm))
            stage = 'comparison'
            metrics = comparison_metrics(actual, reference, atol, rtol)
            if actual.dtype != np.float32 or not metrics['shapes_equal'] or not metrics['actual_finite']:
                raise ValueError('Invalid backend shape, dtype or finite values')
            compare(actual, reference)
            checks.append(dict(**case, arm=arm, status='passed', stage=stage, **metrics))
            _save(output / 'correctness.json', checks)
    except Exception as error:
        metrics = comparison_metrics(actual, reference, atol, rtol)
        receipt = dict(schema=1, **case, arm=arm, status='failed', stage=stage,
                       error_type=type(error).__name__, atol=atol, rtol=rtol,
                       metrics=metrics, arrays_file='failure-case.npz',
                       scope='Failed-case array replay only; no CUDA rerun or usable performance conclusion.')
        arrays = dict(zip(('q', 'k', 'v'), inputs))
        if reference is not None:
            arrays['reference'] = reference
        if actual is not None:
            arrays['actual'] = actual
        temporary = output / 'failure-case.npz.tmp'
        with temporary.open('wb') as file:
            np.savez_compressed(file, **arrays)
        temporary.replace(output / receipt['arrays_file'])
        receipt['arrays_sha256'] = _sha(output / receipt['arrays_file'])
        receipt['arrays'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                             for key, value in arrays.items()}
        checks.append(dict(**case, arm=arm, status='failed', stage=stage, **metrics))
        _save(output / 'correctness.json', checks)
        _save(output / 'failure.json', receipt)
        raise AttentionValidationFailure('Attention validation failed; inspect failure.json') from error


def verify_failure(output):
    """Validate hashes and recompute available failed-case metrics, never accept it."""
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    if run['status'] != 'failed' or run['mode'] != 'cuda' or run['cuda_performance_measured']:
        raise ValueError('Expected a failed CUDA study without a complete performance result')
    spec_name = 'configs/attention-backend-v1.json'
    required_sources = {spec_name, 'lab/attention_reference.py', 'lab/attention_failure.py',
                        'experiments/attention_backend_study.py'}
    if set(run['source_sha256']) != required_sources:
        raise ValueError('Missing or unexpected archived source identity')
    for name, expected in run['source_sha256'].items():
        if _sha(output / 'source' / name) != expected:
            raise ValueError('Archived source hash mismatch')
    required = {'failure.json', 'failure-case.npz', 'correctness.json'}
    allowed = required | {'timings.json', 'memory.json'}
    artifacts = run['artifact_sha256']
    actual_files = {p.name for p in output.iterdir()
                    if p.is_file() and p.suffix in ('.json', '.npz') and p.name != 'run.json'}
    if not required <= set(artifacts) <= allowed or set(artifacts) != actual_files:
        raise ValueError('Missing or unexpected failure artifacts; no complete summary is allowed')
    for name in artifacts:
        if artifacts[name] != _sha(output / name):
            raise ValueError('Failure artifact hash mismatch')
    receipt = json.loads((output / 'failure.json').read_text())
    if receipt['schema'] != 1 or receipt['status'] != 'failed' or receipt['arrays_file'] != 'failure-case.npz':
        raise ValueError('Invalid failure receipt')
    spec = json.loads((output / 'source' / spec_name).read_text())
    if receipt['atol'] != spec['atol'] or receipt['rtol'] != spec['rtol']:
        raise ValueError('Failure tolerance differs from archived protocol')
    if receipt['batch'] not in spec['batches'] or [receipt['query_length'], receipt['key_length']] not in spec['shapes']:
        raise ValueError('Failure case differs from archived protocol')
    stage = receipt['stage']
    if stage not in ('reference_execution', 'reference_validation', 'backend_execution', 'comparison'):
        raise ValueError('Unknown failure stage')
    if (stage.startswith('reference') and receipt['arm'] != 'reference') or (
            not stage.startswith('reference') and receipt['arm'] not in spec['arms']):
        raise ValueError('Failure arm differs from archived protocol')
    if receipt['arrays_sha256'] != _sha(output / 'failure-case.npz'):
        raise ValueError('Failure array hash mismatch')
    with np.load(output / 'failure-case.npz', allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    if not {'q', 'k', 'v'} <= set(arrays) <= {'q', 'k', 'v', 'reference', 'actual'}:
        raise ValueError('Invalid failure array keys')
    expected_outputs = {'reference_execution': set(), 'reference_validation': {'reference'},
                        'backend_execution': {'reference'}, 'comparison': {'reference', 'actual'}}[stage]
    if set(arrays) != {'q', 'k', 'v'} | expected_outputs:
        raise ValueError('Available arrays disagree with failure stage')
    for key, length in (('q', receipt['query_length']), ('k', receipt['key_length']), ('v', receipt['key_length'])):
        value = arrays[key]
        if value.shape != (receipt['batch'], spec['heads'], length, spec['head_dim']) or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError('Failure inputs differ from archived protocol')
    if stage in ('backend_execution', 'comparison'):
        reference = arrays['reference']
        expected_shape = (receipt['batch'], spec['heads'], receipt['query_length'], spec['head_dim'])
        if reference.shape != expected_shape or reference.dtype != np.float32 or not np.isfinite(reference).all():
            raise ValueError('Invalid reference cannot precede backend/comparison stage')
    metadata = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()}
    if metadata != receipt['arrays']:
        raise ValueError('Failure array metadata mismatch')
    metrics = comparison_metrics(arrays.get('actual'), arrays.get('reference'), receipt['atol'], receipt['rtol'])
    if metrics != receipt['metrics']:
        raise ValueError('Failure numerical replay mismatch')
    checks = json.loads((output / 'correctness.json').read_text())
    last = checks[-1]
    if any(last[key] != receipt[key] for key in ('batch', 'query_length', 'key_length', 'arm', 'status', 'stage')):
        raise ValueError('Failure record mismatch')
    if any(last[key] != value for key, value in metrics.items()):
        raise ValueError('Failure record numerical mismatch')
    return dict(failure_evidence_valid=True, stage=receipt['stage'],
                available_arrays=sorted(arrays), metrics=metrics,
                cuda_reexecuted=False, study_accepted=False,
                performance_conclusion_available=False,
                scope='Hash and available-array metric replay; backend exceptions are recorded, not independently reproduced.')
