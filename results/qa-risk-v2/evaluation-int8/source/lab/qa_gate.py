"""Fail-closed empirical QA quality gates and frozen deployment policy checks.

Passing these point-estimate constraints is not a population-risk guarantee.
Wilson intervals are reported separately; they assume binomial observations
and do not account for correlations within contexts or articles.
"""
import math
import re


DEFAULT_CONSTRAINTS = {
    'min_accepted_precision': .9,
    'min_answerable_answer_coverage': .4,
    'min_correct_answerable_coverage': .35,
    'max_unanswerable_false_accept_rate': .1,
    'max_invalid_rate': .05,
}

_SPECS = (
    ('accepted_precision', 'min_accepted_precision', 'accepted_correct', 'accepted', '>='),
    ('answerable_answer_coverage', 'min_answerable_answer_coverage', 'accepted_answerable', 'answerable', '>='),
    ('correct_answerable_coverage', 'min_correct_answerable_coverage', 'accepted_correct', 'answerable', '>='),
    ('unanswerable_false_accept_rate', 'max_unanswerable_false_accept_rate', 'accepted_unanswerable', 'unanswerable', '<='),
    ('invalid_rate', 'max_invalid_rate', 'invalid', 'n', '<='),
)


def _unit(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def _count(value):
    return type(value) is int and value >= 0


def wilson_interval(successes, trials):
    """Two-sided 95% Wilson interval; reject invalid/empty binomial counts."""
    if not _count(successes) or not _count(trials) or trials == 0 or successes > trials:
        raise ValueError('Wilson interval requires integers 0 <= successes <= trials and trials > 0.')
    z = 1.959963984540054
    estimate = successes / trials
    scale = 1 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / scale
    radius = z * math.sqrt(estimate * (1 - estimate) / trials + z * z / (4 * trials * trials)) / scale
    return dict(lower=0.0 if successes == 0 else max(0.0, center - radius),
                upper=1.0 if successes == trials else min(1.0, center + radius),
                estimate=estimate, successes=successes, trials=trials,
                confidence_level=.95, method='two_sided_wilson',
                scope='descriptive_binomial_interval_without_context_or_article_dependence_adjustment')


def evaluate_quality_gate(selective, constraints=None):
    """Check selective_qa's ``summary['selective']`` against all frozen limits.

    Returns criteria and ``all_pass``. Missing metrics, undefined/nonfinite
    rates, empty denominators, inconsistent counts, or forged rates fail.
    With explicit constraints, all five named constraints must be present.
    Intervals are informational, not additional gates or confidence promises.
    """
    limits = dict(DEFAULT_CONSTRAINTS) if constraints is None else constraints
    constraints_valid = (isinstance(limits, dict)
                         and all(_unit(limits.get(key)) for key in DEFAULT_CONSTRAINTS))
    values = selective if isinstance(selective, dict) else {}
    counts = {key: values.get(key) for key in ('n', 'accepted', 'accepted_correct', 'answerable',
              'unanswerable', 'accepted_answerable', 'accepted_unanswerable')}
    status_counts = values.get('model_status_counts')
    statuses_valid = (isinstance(status_counts, dict)
                      and all(_count(status_counts.get(key)) for key in ('answer', 'abstain', 'invalid')))
    counts['invalid'] = status_counts.get('invalid') if isinstance(status_counts, dict) else None
    counts_valid = all(_count(value) for value in counts.values()) and statuses_valid
    if counts_valid:
        counts_valid = (
            counts['n'] > 0
            and counts['answerable'] + counts['unanswerable'] == counts['n']
            and counts['accepted_answerable'] + counts['accepted_unanswerable'] == counts['accepted']
            and counts['accepted_correct'] <= counts['accepted_answerable'] <= counts['answerable']
            and counts['accepted_unanswerable'] <= counts['unanswerable']
            and counts['accepted'] <= status_counts['answer']
            and sum(status_counts[key] for key in ('answer', 'abstain', 'invalid')) == counts['n'])
    criteria = {
        'constraints_valid': dict(passed=constraints_valid, reason=None if constraints_valid else 'missing_or_invalid_constraints'),
        'counts_consistent': dict(passed=counts_valid, reason=None if counts_valid else 'missing_empty_or_inconsistent_counts'),
    }
    for metric, constraint, numerator, denominator, comparator in _SPECS:
        value = values.get(metric)
        limit = limits.get(constraint) if isinstance(limits, dict) else None
        count_ok = counts_valid and counts[denominator] > 0
        expected = counts[numerator] / counts[denominator] if count_ok else None
        valid = (constraints_valid and count_ok and _unit(value)
                 and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12))
        passed = valid and (value >= limit if comparator == '>=' else value <= limit)
        reason = ('missing_nonfinite_undefined_or_inconsistent_metric' if not valid else
                  None if passed else 'point_estimate_outside_frozen_limit')
        criteria[metric] = dict(value=value if _unit(value) else None,
                                recomputed_value=expected, limit=limit if _unit(limit) else None,
                                comparator=comparator, passed=bool(passed), reason=reason)
    intervals = {}
    for metric, numerator, denominator in (
            ('accepted_precision', 'accepted_correct', 'accepted'),
            ('unanswerable_false_accept_rate', 'accepted_unanswerable', 'unanswerable'),
            ('correct_answerable_coverage', 'accepted_correct', 'answerable')):
        intervals[metric] = (wilson_interval(counts[numerator], counts[denominator])
                             if counts_valid and counts[denominator] > 0 else None)
    return dict(all_pass=all(item['passed'] for item in criteria.values()), criteria=criteria,
                constraints=limits if constraints_valid else None, intervals=intervals,
                scope='empirical_point_estimate_gate_not_population_risk_guarantee',
                interval_role='report_only_not_used_to_pass_the_gate')


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _hash_manifest(value):
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(path, str) and bool(path.strip()) and _sha256(digest)
                    for path, digest in value.items()))


def validate_policy(policy, selective, constraints=None):
    """Validate a policy against recalculated gates, never a self-reported pass.

    ``model_files_sha256`` is a nonempty filename-to-SHA256 mapping.
    ``source_evidence_sha256`` is a SHA256 string or nonempty such mapping.
    Hash syntax is checked here; the caller must check actual local file
    contents against the hashes before loading. A disabled policy remains
    valid with failed quality gates but can never authorize answer serving.
    Only unmodified FP16 (bits=None) or Q8 are eligible. Block restoration
    configurations, including block10, are not approved for this policy.
    """
    if not isinstance(policy, dict):
        raise ValueError('Policy must be an object.')
    errors = []
    if type(policy.get('enabled')) is not bool:
        errors.append('enabled must be an explicit boolean')
    if policy.get('mode') not in ('legacy', 'grounded'):
        errors.append('mode must be legacy or grounded')
    if 'bits' not in policy or not (policy['bits'] is None or type(policy['bits']) is int and policy['bits'] == 8):
        errors.append('bits must be explicitly null (FP16) or integer 8')
    if not _unit(policy.get('threshold')):
        errors.append('threshold must be a finite number in [0, 1]')
    if not _hash_manifest(policy.get('model_files_sha256')):
        errors.append('model_files_sha256 must be a nonempty SHA256 manifest')
    source = policy.get('source_evidence_sha256')
    if not (_sha256(source) or _hash_manifest(source)):
        errors.append('source_evidence_sha256 must identify nonempty evidence')
    for key in ('block', 'blocks', 'fallback_block', 'fallback_blocks', 'restored_blocks', 'fp16_blocks'):
        if key in policy and policy[key] not in (None, [], (), ''):
            errors.append('block restoration is not allowed in the deployment policy')
    if 'block10' in str(policy.get('variant', '')).lower().replace('_', '').replace('-', ''):
        errors.append('block10 is not an approved deployment default')
    gate = evaluate_quality_gate(selective, constraints)
    if policy.get('enabled') is True and not gate['all_pass']:
        errors.append('enabled policy requires passing recalculated quality gates')
    if errors:
        raise ValueError('Invalid QA policy: ' + '; '.join(errors))
    return dict(valid=True, enabled=policy['enabled'], may_serve_answers=policy['enabled'] and gate['all_pass'],
                quality_gate=gate, file_hashes_verified=False)
