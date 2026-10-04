"""Complete RC4 offline acceptance, requiring NumPy 2.2.6 but no model or Git.

Run this module from the unpacked source archive itself. Historical stdlib-only
acceptance remains available separately as experiments.verify_release.
"""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from lab.quantization_diagnostics import read, sha
from experiments.verify_release import verify as verify_history
from experiments.verify_qa_specialist import verify as verify_specialist, require
from experiments.verify_qa_risk import verify as verify_risk

REPO = Path(__file__).resolve().parents[1]
SPECIALIST_STUDY_SHA256 = '15df43bd137bec2f0ba56c5523db04725fb8864f411558f82b79b7041360f9e9'
RISK_STUDY_SHA256 = '365a1ff260d9f887558bc5ce5421e4028702ad435717aefc8f17787cea08081b'


def validate_results(historical, specialist, risk):
    """Release claims must retain failed predecessors and separate task quality."""
    require(historical['technical_acceptance']=='pass' and historical['confirmation_gate_passed'] is False and
            historical['qa_remediation_quality_passed'] is False and
            historical['qa_answer_entrypoint']=='unavailable_quality', 'Historical failure or technical status changed')
    require(specialist['evidence_valid'] is True and specialist['calibration_feasible'] is True and
            specialist['evaluation_evaluated'] is True and specialist['status']=='evaluation_complete' and
            specialist['overall_success'] is False and specialist['compression_confirmed'] is False and
            specialist['answer_entrypoint']=='unavailable_quality', 'Frozen specialist failure must be retained')
    require(all(specialist['variants'][v]['quality_gate']['all_pass'] is False for v in ('fp32','int8')),
            'Original specialist evaluation quality failure changed')
    require(risk['evidence_valid'] is True and risk['status']=='evaluation_complete' and
            risk['any_calibration_eligible'] is True and risk['both_calibration_eligible'] is False,
            'Supervised ranking study completion/eligibility changed')
    fp32,int8=risk['variants']['fp32'],risk['variants']['int8']
    require(fp32['calibration_eligible'] is False and fp32['threshold'] is None and
            fp32['evaluation_evaluated'] is False and fp32['task_pass'] is False and
            fp32['answer_entrypoint']=='unavailable_quality', 'Failed FP32 head cannot enable evaluation or answering')
    require(int8['calibration_eligible'] is True and int8['threshold']==.7 and int8['evaluation_evaluated'] is True and
            int8['task_pass'] is True and int8['bounded_task_eligible'] is True and
            int8['gate']['all_pass'] is True and int8['answer_entrypoint']=='bounded_local_prototype',
            'INT8 bounded task eligibility changed')
    counts=int8['summary']['selective']
    require((counts['n'],counts['accepted'],counts['accepted_correct'],counts['answerable'],
             counts['unanswerable'],counts['accepted_unanswerable']) == (128,27,27,64,64,0),
            'Fixed evaluation counts differ from accepted RC4 evidence')
    require(risk['compression_gate']['status']=='not_evaluated' and
            risk['compression_gate']['evaluated'] is False and risk['compression_gate']['passed'] is False and
            risk['raw_candidate_paired'] is None and risk['compression_default_recommendation'] is False,
            'No paired compression quality claim is authorized')
    require(risk['parameters_distributed'] is False, 'Release cannot distribute fitted head parameters')


def verify(root=REPO):
    root=Path(root).resolve()
    require(root==REPO.resolve(), 'Run the RC4 verifier from the target source archive, not another checkout')
    require(importlib.metadata.version('numpy')=='2.2.6', 'Full RC4 verification explicitly requires NumPy 2.2.6')
    historical=verify_history(root)
    specialist_study=root/'configs/qa-specialist/study.json'
    risk_study=root/'configs/qa-risk/study.json'
    require(sha(specialist_study)==SPECIALIST_STUDY_SHA256, 'Frozen specialist study changed')
    require(sha(risk_study)==RISK_STUDY_SHA256, 'Frozen supervised ranking study changed')
    specialist=verify_specialist(root/'results/qa-specialist-v1',specialist_study)
    risk=verify_risk(root/'results/qa-risk-v2',risk_study)
    validate_results(historical,specialist,risk)
    policy=verify_serving_policy(root,risk)
    # Standalone independent arithmetic modules: no model, training, network or Git.
    performance={}
    for label,relative,function in (
        ('base','results/qa-specialist-review-v1/performance/audit.py','audit_performance'),
        ('risk','results/qa-risk-review-v1/risk-performance-audit.py','audit')):
        module_spec=importlib.util.spec_from_file_location('rc4_'+label+'_performance',root/relative)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        performance[label]=getattr(module,function)(root)
    license_present=(root/'LICENSE').is_file()
    return dict(technical_acceptance='pass',research_reproducibility='pass',
        bounded_local_qa_prototype=True,general_qa_deployment_quality=False,
        original_quantization_fallback_confirmed=False,compression_quality_noninferiority_confirmed=False,
        root_code_license_present=license_present,
        root_code_license_status='present_owner_authorization_not_inferred' if license_present else 'pending_owner_decision',
        release_created_by_this_verification=False,
        historical=historical,
        specialist_evaluation_passed=False,
        risk=dict(protocol_sha256=risk['protocol_sha256'],selection_sha256=risk['selection_sha256'],
                  eligible_variant='int8',threshold=.7,accepted_correct=27,accepted=27,
                  answerable_coverage=27/64,unanswerable_false_accepts=0,unanswerable_n=64,
                  accepted_precision_wilson95=risk['variants']['int8']['gate']['intervals']['accepted_precision'],
                  compression_gate='not_evaluated',parameters_distributed=False),
        serving_policy=policy,performance_evidence_valid=True,numpy_version='2.2.6',
        scope='Fixed public SQuAD2 evidence on one local system; 27 accepted evaluation answers and four articles '
              'do not establish population risk or general application deployment. '
              'All prior failures remain part of the release.')


def verify_serving_policy(root,risk):
    from experiments.verify_qa_risk_pruning import verify as verify_pruning
    from experiments.serve_qa_specialist import load_policy, validate_verified_quality
    policy=load_policy(root/'configs/qa-risk/policy.json')
    pruning = verify_pruning(root)
    result = validate_verified_quality(policy,risk)
    result['pruning_acceptance'] = pruning
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO)
    parser.add_argument('--require-license',action='store_true')
    args=parser.parse_args();result=verify(args.root)
    if args.require_license:require(result['root_code_license_present'],'Root code license decision is pending')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
