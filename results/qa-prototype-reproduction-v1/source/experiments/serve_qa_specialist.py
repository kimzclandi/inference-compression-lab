"""Evidence-gated local English passage QA prototype, not a general deployment.

The fixed public-benchmark point gates authorize only this bounded prototype.
Initialization verifies evidence, rebuilds the head only from its 256 training
records, and loads caller-supplied local assets. It downloads no weights.
Initialization time is separate from request pipeline time. The CLI accepts a
JSON object containing exactly context and question; gold labels are rejected.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from lab.qa_risk_calibration import extract_features, predict_probability
from lab.quantization_diagnostics import read, sha


REPO = Path(__file__).resolve().parents[1]
STUDY_SHA256 = '365a1ff260d9f887558bc5ce5421e4028702ad435717aefc8f17787cea08081b'
SELECTION_SHA256 = 'b2999477ee82318a98266ef0674c22a526cddea1eec541c591a7bec7e354b74a'
ASSET_MANIFEST_SHA256 = 'c7709c2098b46dc59223b451c076974e1035f591c1e03ca3c2af2747d1f5c7ed'
SCOPE = 'english_supplied_passage_bounded_local_prototype'
SCORE_KIND = 'fitted_ranking_score_not_guaranteed_correctness_probability'
EXPECTED_POLICY = dict(schema_version=1, enabled=True, variant='int8',
    study_sha256=STUDY_SHA256, selection_sha256=SELECTION_SHA256,
    asset_manifest_sha256=ASSET_MANIFEST_SHA256, threshold=0.7, scope=SCOPE,
    evidence_scope='public_benchmark_point_gates_not_general_deployment',
    input_limits=dict(max_question_characters=1000, max_question_tokens=64,
        max_context_characters=12000, max_sequence_tokens=384, stride=128,
        max_windows=8, max_answer_tokens=30),
    minimum_training_rows=256)


class QualityUnavailable(ValueError):
    pass


class ModelUnavailable(ValueError):
    pass


class InvalidInput(ValueError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _same_typed_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(actual) == set(expected) and
                all(_same_typed_value(actual[key], expected[key]) for key in expected))
    return actual == expected


def load_policy(path):
    try:
        policy = _json(Path(path).read_text(encoding='utf-8'))
        if not isinstance(policy, dict) or type(policy.get('enabled')) is not bool:
            raise ValueError('An explicit boolean policy.enabled is required.')
        expected = {**EXPECTED_POLICY, 'enabled': policy['enabled']}
        if not _same_typed_value(policy, expected):
            raise ValueError('Policy must exactly match the fixed scope, identities, threshold and limits.')
        if policy['enabled'] is not True:
            raise ValueError('This QA policy is disabled.')
        return policy
    except Exception as error:
        raise QualityUnavailable(str(error)) from error


def validate_verified_quality(policy, verification):
    """Bind a previously recomputed evidence result without fitting/loading.

    Release verification can call this after its own full ``verify_qa_risk``
    run. A claimed result alone is not evidence: the caller owns recomputation.
    """
    try:
        if not _same_typed_value(policy, EXPECTED_POLICY):
            raise ValueError('Only the complete fixed enabled policy is eligible.')
        variant = verification['variants']['int8']
        if (verification.get('evidence_valid') is not True or
            verification.get('protocol_sha256') != STUDY_SHA256 or
            verification.get('selection_sha256') != SELECTION_SHA256 or
            variant.get('bounded_task_eligible') is not True or
            variant.get('evaluation_evaluated') is not True or
            variant.get('task_pass') is not True or variant.get('threshold') != .7 or
            variant.get('gate', {}).get('all_pass') is not True):
            raise ValueError('Complete INT8 evidence does not authorize the bounded task.')
        return dict(variant='int8', threshold=.7, bounded_task_eligible=True, scope=SCOPE)
    except Exception as error:
        raise QualityUnavailable(str(error)) from error


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {'context', 'question'}:
        raise InvalidInput('Input JSON must contain exactly context and question; no labels or extra fields.')
    for name, limit in (('context', 12000), ('question', 1000)):
        value = payload[name]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise InvalidInput(f'{name} must be nonempty text of at most {limit} characters.')
    return payload


def read_request(path):
    try:
        # UTF-8 and escaped JSON for the bounded text fit comfortably below
        # this parser limit; reject oversized input before loading a model.
        if str(path) == '-':
            text = sys.stdin.read(131073)
        else:
            with Path(path).open(encoding='utf-8') as source:
                text = source.read(131073)
        if len(text) > 131072:
            raise InvalidInput('Input JSON exceeds the fixed parser limit.')
        return validate_request(_json(text))
    except InvalidInput:
        raise
    except Exception as error:
        raise InvalidInput(str(error)) from error


def _verify_evidence(root, study):
    from experiments.verify_qa_risk import verify
    return verify(root, study)


def _rebuild_head(training_dir, expected_digest):
    from experiments.qa_risk import rebuild_head
    return rebuild_head(training_dir, 'int8', expected_digest)


def _load_runtime(asset_root):
    from lab.qa_specialist_runtime import ExtractiveRuntime
    return ExtractiveRuntime(asset_root, 'int8', ASSET_MANIFEST_SHA256, threads=4)


class LocalQAService:
    def __init__(self, runtime, head, startup_timing):
        self.runtime = runtime
        self.head = head
        self.startup_timing = startup_timing

    def answer(self, payload):
        """Score one request; no training or threshold adaptation happens here."""
        started = time.perf_counter()
        validate_request(payload)
        context, question = payload['context'], payload['question']
        try:
            # The frozen runtime validates actual token and window counts.
            # Only these known input-limit errors are classified as bad input;
            # malformed model output remains an unavailable-model failure.
            prediction = self.runtime.predict(context, question)
        except ValueError as error:
            if str(error).startswith(('Question exceeds the frozen token limit',
                                      'Context exceeds the bounded window count')):
                raise InvalidInput(str(error)) from error
            raise ModelUnavailable(str(error)) from error
        except Exception as error:
            raise ModelUnavailable(str(error)) from error
        try:
            feature = extract_features(context, prediction)
            score = predict_probability(self.head, feature['features'])
        except ValueError as error:
            if str(error).startswith('No differently normalized legal span'):
                raise InvalidInput('The fixed feature contract requires a differently normalized alternative span.') from error
            raise ModelUnavailable(str(error)) from error
        except Exception as error:
            raise ModelUnavailable(str(error)) from error
        accepted = score >= EXPECTED_POLICY['threshold']
        result = dict(status='answer' if accepted else 'abstain',
            reason='fixed_threshold_met' if accepted else 'below_fixed_threshold',
            score=score, score_kind=SCORE_KIND, threshold=EXPECTED_POLICY['threshold'], scope=SCOPE)
        if accepted:
            result.update(answer=prediction['prediction'], start=prediction['start'],
                          end=prediction['end'], context_sha256=prediction['context_sha256'])
        result['timing'] = dict(request_pipeline_seconds=time.perf_counter() - started,
                               includes='input checks, tokenization, ORT CPU, decoding, risk features and score',
                               excludes='startup verification/head rebuild/model load and JSON serialization')
        return result


def load_service(policy_path, evidence_root, study_path, asset_root):
    """Verify once, reconstruct training-only parameters, then load local ORT."""
    started = time.perf_counter()
    policy = load_policy(policy_path)
    try:
        root, study = Path(evidence_root), Path(study_path)
        if sha(study) != STUDY_SHA256 or sha(root/'training'/'selection.json') != SELECTION_SHA256:
            raise ValueError('Study or calibration selection differs from the pinned policy identity.')
        spec = read(study)
        if spec['asset_manifest_sha256'] != ASSET_MANIFEST_SHA256 or spec['dataset_sizes']['train'] != 256:
            raise ValueError('Backbone identity or training-only row count changed.')
        verification = _verify_evidence(root, study)
        validate_verified_quality(policy, verification)
        selection = read(root/'training'/'selection.json')
        chosen = selection['variants']['int8']
        if chosen['eligible'] is not True or chosen['threshold'] != .7:
            raise ValueError('The fixed INT8 calibration policy is not eligible.')
        verified_at = time.perf_counter()
        head = _rebuild_head(root/'training', chosen['model_sha256'])
        if type(head.get('n_train')) is not int or head['n_train'] != 256:
            raise ValueError('The rebuilt head must use only the 256 frozen training records.')
        rebuilt_at = time.perf_counter()
    except Exception as error:
        raise QualityUnavailable(str(error)) from error
    try:
        assets = Path(asset_root)
        if sha(assets/'manifest.json') != ASSET_MANIFEST_SHA256:
            raise ValueError('Local asset manifest differs from the accepted backbone.')
        runtime = _load_runtime(assets)
    except Exception as error:
        raise ModelUnavailable(str(error)) from error
    ready = time.perf_counter()
    return LocalQAService(runtime, head, dict(
        evidence_verification_seconds=verified_at-started,
        training_head_rebuild_seconds=rebuilt_at-verified_at,
        local_model_load_seconds=ready-rebuilt_at, total_startup_seconds=ready-started,
        scope='One-time initialization; excluded from request-only benchmark timings.'))


def serve(args):
    try:
        # Disabled/malformed policies exit before reading input or loading any
        # evidence/model. Invalid JSON exits before expensive initialization.
        load_policy(args.policy)
        payload = read_request(args.input_json)
        service = load_service(args.policy, args.evidence_root, args.study, args.asset_root)
        result = service.answer(payload)
        result['startup_timing'] = service.startup_timing
        return result, 0
    except QualityUnavailable as error:
        return dict(status='unavailable_quality', reason=str(error), scope=SCOPE), 2
    except InvalidInput as error:
        return dict(status='invalid_input', reason=str(error), scope=SCOPE), 1
    except ModelUnavailable as error:
        return dict(status='unavailable_model', reason=str(error), scope=SCOPE), 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=REPO/'configs/qa-risk/policy.json')
    parser.add_argument('--evidence-root', type=Path, default=REPO/'results/qa-risk-v2')
    parser.add_argument('--study', type=Path, default=REPO/'configs/qa-risk/study.json')
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--input-json', required=True, help='UTF-8 JSON file, or - for stdin')
    result, code = serve(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
