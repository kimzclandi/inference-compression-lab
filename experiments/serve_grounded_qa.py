"""Evidence-gated, one-shot local QA research CLI; this is not a production service.

Disabled policies return unavailable_quality before reading inputs, evidence,
or importing a model runtime. Enabled policies require a passing, independently
held-out confirmation run and exact evidence/model/source bindings. A local
operator can edit this code and its evidence; this is not a security boundary.

Evidence envelope: quality.json beside an untouched complete study directory,
designated by policy.study_dir; policy.run_dir names its confirmation-fp16 or
confirmation-q8 child. quality.json contains split='confirmation', threshold,
constraints, and selective. policy.source_evidence_sha256 maps relative files
in the envelope to their SHA256; no evidence symlinks or escaping paths.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from lab.artifact_integrity import safe_path
from lab.qa_gate import DEFAULT_CONSTRAINTS, validate_policy
from lab.quantization_diagnostics import aggregates_equal, read, rows, sha
from lab.selective_qa import apply_threshold, evaluate_selective, parse_output


REPO_ROOT = Path(__file__).resolve().parents[1]
BOUND_CODE = ('lab/grounded_qa.py', 'experiments/qwen_quantization.py',
              'experiments/qa_remediation.py', 'lab/qa_metrics.py')
RUNTIME_PACKAGES = ('mlx', 'mlx-lm', 'numpy', 'transformers')


def _evidence_file(root, relative):
    if not isinstance(relative, str):
        raise ValueError('Evidence paths must be strings.')
    path = safe_path(root, relative)
    if not path.is_file():
        raise ValueError('Missing evidence file: ' + relative)
    return path


def _model_hashes(model_root):
    """HF snapshots may have read-only file symlinks into their local blob store."""
    if not model_root.is_dir():
        raise ValueError('An existing local model directory is required.')
    actual = {}
    for path in sorted(model_root.iterdir()):
        if path.is_dir() or not path.is_file():
            raise ValueError('Only top-level model files are covered by the frozen manifest.')
        actual[path.name] = sha(path)
    if not actual or 'config.json' not in actual or not any(name.endswith('.safetensors') for name in actual):
        raise ValueError('The source model requires config.json and safetensors weights.')
    config = read(model_root / 'config.json')
    if not isinstance(config, dict) or any(key in config for key in ('quantization', 'quantization_config')):
        raise ValueError('The local source must contain original floating weights.')
    return actual


def _verify_complete_study(study_root):
    from experiments.verify_qa_remediation import verify
    previous = Path.cwd()
    try:
        # The offline verifier checks current scoring source relative to the
        # repository. Keep the one-shot CLI usable from another working dir.
        os.chdir(REPO_ROOT)
        return verify(study_root, REPO_ROOT / 'configs/qa-remediation/study.json')
    finally:
        os.chdir(previous)


def _verify_runtime_versions(run):
    """Inspect installed metadata without importing any model runtime."""
    recorded = run.get('packages')
    if not isinstance(recorded, dict) or set(recorded) != set(RUNTIME_PACKAGES):
        raise ValueError('Confirmation must record the complete runtime package version set.')
    if any(not isinstance(version, str) or not version for version in recorded.values()):
        raise ValueError('Recorded runtime package versions must be nonempty strings.')
    if run.get('python') != platform.python_version():
        raise ValueError('Current Python version differs from confirmation; exact research reproduction is required.')
    actual = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if actual != recorded:
        raise ValueError('Installed MLX/tokenizer dependency versions differ from confirmation.')


def verify_enabled_evidence(policy, evidence_root, model_root):
    """Recompute confirmation quality before any MLX import or model loading."""
    if policy.get('enabled') is not True:
        raise ValueError('Enabled evidence verification requires enabled=true.')
    if evidence_root is None or model_root is None:
        raise ValueError('Enabled policies require --evidence-dir and --model.')
    evidence_root, model_root = Path(evidence_root).absolute(), Path(model_root).absolute()
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError('Evidence root must be an existing nonsymlink directory.')
    # Check the full ancestor chain, not just the final evidence directory.
    if any(part.is_symlink() for part in evidence_root.absolute().parents):
        raise ValueError('Evidence root cannot traverse symlink directories.')
    manifest = policy.get('source_evidence_sha256')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Enabled policies require an exact evidence-file hash mapping.')
    for name, expected in manifest.items():
        if sha(_evidence_file(evidence_root, name)) != expected:
            raise ValueError('Evidence SHA256 mismatch: ' + str(name))
    run_dir = policy.get('run_dir', 'run')
    run_root = safe_path(evidence_root, run_dir) if isinstance(run_dir, str) else None
    if run_root is None or not run_root.is_dir():
        raise ValueError('Policy run_dir must identify a safe existing run directory.')
    study_dir = policy.get('study_dir')
    study_root = safe_path(evidence_root, study_dir) if isinstance(study_dir, str) else None
    if study_root is None or not study_root.is_dir():
        raise ValueError('Enabled policy requires a safe complete study_dir.')
    if 'bits' not in policy or not (policy['bits'] is None or type(policy['bits']) is int and policy['bits'] == 8):
        raise ValueError('Only explicitly declared FP16 or Q8 is eligible.')
    precision = 'fp16' if policy['bits'] is None else 'q8'
    if run_root != study_root / ('confirmation-' + precision):
        raise ValueError('Run directory must be the selected study confirmation precision.')
    quality_name = policy.get('quality_file', 'quality.json')
    required = {f'{run_dir}/{name}' for name in ('run.json', 'data.jsonl', 'predictions.jsonl', 'protocol.json')}
    required.update(f'{run_dir}/source/{name}' for name in BOUND_CODE)
    required.update(f'{study_dir}/{name}' for name in ('selection.json', 'protocol.json', 'confirmation-protocol.json'))
    required.add(quality_name)
    if not required <= manifest.keys():
        raise ValueError('Evidence manifest must bind all inference, data, predictions and quality files.')
    # This checks both calibrations, the fixed threshold selection, both
    # confirmations, quality gates and compression noninferiority. A manually
    # chosen threshold after viewing confirmation cannot substitute for it.
    study = _verify_complete_study(study_root)
    if study.get('overall_success') is not True:
        raise ValueError('The complete frozen study did not pass calibration and confirmation.')
    selection = read(study_root / 'selection.json')
    selected_threshold = selection.get('variants', {}).get(precision, {}).get('threshold')
    verified_threshold = study.get('variants', {}).get(precision, {}).get('threshold')
    if policy.get('threshold') != selected_threshold or policy.get('threshold') != verified_threshold:
        raise ValueError('Policy threshold differs from the frozen calibration selection.')
    quality = read(_evidence_file(evidence_root, quality_name))
    if quality.get('split') != 'confirmation':
        raise ValueError('Calibration alone cannot enable serving; confirmation evidence is required.')
    if quality.get('threshold') != policy.get('threshold'):
        raise ValueError('Policy threshold differs from the confirmed threshold.')
    if quality.get('constraints') != DEFAULT_CONSTRAINTS:
        raise ValueError('Quality evidence must use the complete fixed deployment constraints.')
    run = read(run_root / 'run.json')
    protocol = read(run_root / 'protocol.json')
    data, predictions = rows(run_root / 'data.jsonl'), rows(run_root / 'predictions.jsonl')
    if not data or any(row.get('split') != 'confirmation' for row in data):
        raise ValueError('Every bound data row must belong to the confirmation split.')
    if run.get('status') != 'complete' or not protocol.get('locked'):
        raise ValueError('Only completed, locked confirmation runs can enable serving.')
    if run.get('mode') != policy.get('mode') or run.get('bits') != policy.get('bits'):
        raise ValueError('Policy prompt mode or precision differs from confirmation.')
    if protocol.get('max_input_tokens') != 2048 or protocol.get('max_new_tokens') != 48:
        raise ValueError('The research CLI requires frozen 2048-input/48-output token limits.')
    if run.get('dataset_sha256') != sha(run_root / 'data.jsonl') or run.get('spec_sha256') != sha(run_root / 'protocol.json'):
        raise ValueError('Run dataset/protocol identity differs from bound evidence.')
    if run.get('predictions') != predictions:
        raise ValueError('Run predictions differ from the bound raw prediction log.')
    if any(prediction.get('stop_reason') != 'eos' for prediction in predictions):
        raise ValueError('Confirmation must contain complete generations; truncated runs cannot enable this CLI.')
    recorded_files = run.get('source_model_files')
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise ValueError('Confirmation has no complete source model manifest.')
    recorded_hashes = {name: record.get('sha256') for name, record in recorded_files.items()}
    if recorded_hashes != policy.get('model_files_sha256'):
        raise ValueError('Policy model identity differs from confirmation.')
    actual_hashes = _model_hashes(model_root)
    if actual_hashes != recorded_hashes:
        raise ValueError('Local model files do not exactly match the confirmed model manifest.')
    # Recompute registration; never trust run.locked_spec_validation by itself.
    from experiments.qa_remediation import validate_locked_run
    registered = validate_locked_run(protocol, label=run.get('model_label'), mode=policy.get('mode'),
                                     bits=policy.get('bits'), model_files_sha256=actual_hashes,
                                     data_sha256=run['dataset_sha256'])
    if not registered['locked']:
        raise ValueError('Unregistered confirmation identity.')
    for name in BOUND_CODE:
        evidence_hash = sha(run_root / 'source' / name)
        if run.get('source_sha256', {}).get(name) != evidence_hash or sha(REPO_ROOT / name) != evidence_hash:
            raise ValueError('Current inference/scoring source differs from confirmation: ' + name)
    if policy.get('grounded_sha256') != sha(REPO_ROOT / 'lab/grounded_qa.py'):
        raise ValueError('Policy must bind the current grounded QA source hash.')
    _verify_runtime_versions(run)
    summary, _ = evaluate_selective(data, predictions, policy.get('threshold'))
    if not aggregates_equal(summary['selective'], quality.get('selective')):
        raise ValueError('Claimed selective quality differs from raw prediction recomputation.')
    verified = validate_policy(policy, summary['selective'], DEFAULT_CONSTRAINTS)
    return dict(quality_gate=verified['quality_gate'], seed=protocol['seed'],
                file_hashes_verified=True, confirmation_examples=len(data))


def _load_and_predict(model_path, policy, context, question, seed):
    # All runtime imports occur only after the evidence and deployment gates.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from experiments.qa_remediation import _load_runtime
    from lab.grounded_qa import predict
    mx, fetch_from_hub, quantize_model, tree_map, _, _ = _load_runtime()
    model, config, tok = fetch_from_hub(Path(model_path), lazy=False, trust_remote_code=False)
    if any(key in config for key in ('quantization', 'quantization_config')):
        raise ValueError('Loaded source is unexpectedly prequantized.')
    model.update(tree_map(lambda x: x.astype(mx.float16) if mx.issubdtype(x.dtype, mx.floating) else x,
                          model.parameters()))
    config['torch_dtype'] = 'float16'
    if policy['bits'] == 8:
        model, config = quantize_model(model, config, 64, 8)
    mx.eval(model.parameters())
    mx.random.seed(seed)
    return predict(model, tok, context, question, policy['mode'], 48, 2048)


def _input_text(value, file_path, name):
    if (value is None) == (file_path is None):
        raise ValueError('Specify exactly one of --' + name + ' or --' + name + '-file.')
    text = Path(file_path).read_text(encoding='utf-8') if file_path is not None else value
    if not isinstance(text, str) or not text.strip():
        raise ValueError(name + ' must be nonempty text.')
    return text


def serve(args):
    """Return one JSON-safe response and exit code; errors are never refusals."""
    try:
        policy = read(args.policy)
        if not isinstance(policy, dict) or type(policy.get('enabled')) is not bool:
            raise ValueError('Policy requires an explicit boolean enabled field.')
        if not policy['enabled']:
            return dict(status='unavailable_quality', reason='quality_gate_not_passed',
                        detail=policy.get('reason', 'This research QA configuration is disabled.'),
                        scope='local_research_cli_not_production'), 2
        verification = verify_enabled_evidence(policy, args.evidence_dir, args.model)
        context = _input_text(args.context, args.context_file, 'context')
        question = _input_text(args.question, args.question_file, 'question')
        prediction = _load_and_predict(args.model, policy, context, question, verification['seed'])
        decision = apply_threshold(parse_output(context, prediction['prediction']),
                                   prediction['confidence'], policy['threshold'])
        if prediction.get('stop_reason') != 'eos':
            return dict(status='invalid', action='abstain', reason='generation_not_complete',
                        context_sha256=decision['context_sha256'],
                        scope='local_research_cli_not_production'), 0
        result = dict(status=decision['status'], action=decision['action'], reason=decision['reason'],
                      confidence=decision['confidence'], context_sha256=decision['context_sha256'],
                      scope='local_research_cli_not_production')
        if decision['accepted']:
            result.update(answer=decision['answer'], start=decision['start'], end=decision['end'])
        return result, 0
    except Exception as error:
        return dict(status='error', error_type=type(error).__name__, message=str(error),
                    scope='local_research_cli_not_production'), 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--model', type=Path)
    for name in ('context', 'question'):
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--' + name)
        group.add_argument('--' + name + '-file', type=Path)
    result, code = serve(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
