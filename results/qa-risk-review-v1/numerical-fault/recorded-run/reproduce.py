"""Synthetic before/after runtime fault reproduction; no QA labels or outputs."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import shutil
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data(count):
    return ([[float(i % 17), float((i % 17)**2), float(i % 3),
              float(i % 4), float(i % 5)] for i in range(count)],
            [i % 2 for i in range(count)])


def outcome(module, count):
    matrix, labels = data(count)
    try:
        model = module.fit(matrix, labels)
        scores = [module.predict_probability(model, row) for row in matrix]
        weights, intercept = model['weights'], model['intercept']
        centers, scales = model['scaler']['mean'], model['scaler']['scale']
        logits = [intercept + math.fsum(weight * (value - center) / scale
                  for weight, value, center, scale in zip(weights, row, centers, scales))
                  for row in matrix]
        signed = [-logit if label else logit for logit, label in zip(logits, labels)]
        independent_objective = math.fsum(max(z, 0) + math.log1p(math.exp(-abs(z)))
                                         for z in signed)
        independent_objective += .5 * math.fsum(weight * weight for weight in weights)
        return dict(status='complete', n=count, convergence=model['convergence'],
                    objective=model['trace'][-1]['objective'],
                    independent_stdlib_objective=independent_objective,
                    objective_absolute_difference=abs(independent_objective - model['trace'][-1]['objective']),
                    synthetic_model_sha256=hashlib.sha256(json.dumps(model, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()).hexdigest()), scores
    except Exception as error:
        chain = []
        current = error
        while current is not None:
            chain.append(dict(type=type(current).__name__, message=str(current)))
            current = current.__cause__
        return dict(status='failed', n=count, error_chain=chain), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    for original, name in ((args.before, 'source-before.py'), (args.after, 'source-after.py'),
                           (Path(__file__), 'reproduce.py')):
        shutil.copy2(original, out / name)
    # Ensure project modules remain importable when executing this nested file.
    sys.path.insert(0, str(Path.cwd()))
    before = load('risk_before', out / 'source-before.py')
    after = load('risk_after', out / 'source-after.py')
    import numpy as np
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    (out / 'numpy-build.txt').write_text(config.getvalue())
    report = dict(scope='Synthetic 5-feature runtime regression; no QA calibration or evaluation results used.',
                  python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                  before_sha256=digest(out / 'source-before.py'), after_sha256=digest(out / 'source-after.py'),
                  unchanged_fit_config=before.FIT_CONFIG == after.FIT_CONFIG,
                  fit_config=after.FIT_CONFIG, cases=[])
    for count in (32, 128, 256, 512, 1024):
        old, old_scores = outcome(before, count)
        new, new_scores = outcome(after, count)
        repeated, _ = outcome(after, count)
        report['cases'].append(dict(n=count, before=old, after=new,
            after_repeat_exact_match=new == repeated,
            score_max_absolute_difference=(max(abs(a-b) for a,b in zip(old_scores,new_scores))
                                           if old_scores is not None and new_scores is not None else None)))
    report['corrected_all_pass'] = (report['unchanged_fit_config'] and all(
        case['after']['status'] == 'complete' and case['after_repeat_exact_match'] and
        case['after']['objective_absolute_difference'] < 1e-10 for case in report['cases']))
    report['interpretation'] = ('The recorded BLAS backend is identified in numpy-build.txt. '
        'A before-failure and after-success on finite synthetic inputs supports an implementation/runtime '
        'workaround, not a definitive upstream-library root-cause diagnosis or a quality improvement claim.')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    hashes = {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()}
    (out / 'checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(corrected_all_pass=report['corrected_all_pass'], output=str(out))))
    if not report['corrected_all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
