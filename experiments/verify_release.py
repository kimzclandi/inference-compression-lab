"""Offline technical acceptance of this research release, independent of Git/MLX."""
import argparse
from pathlib import Path
from lab.artifact_integrity import verify_hashes, safe_path
from lab.model_identity import verify_inference_files
from lab.quantization_diagnostics import read, sha, aggregates_equal
from experiments.verify_qwen_quantization import verify as verify_exploration, require
from experiments.verify_qwen_confirmation import verify as verify_confirmation
from experiments.verify_qwen_prefix_fair import verify as verify_fair
from experiments.verify_qwen_cache_lifecycle import verify as verify_lifecycle

# The historical ledger is frozen, not a caller-supplied selection of files.
# This reviewed-code anchor detects deleted ledger entries without needing Git.
# It is an integrity boundary, not a signature against a malicious publisher.
PROTECTED_RESULTS_SHA256 = '321548d8022738466b8b772d91ea604ecfdb3c003b71a67d34d4fdff0d782abc'


def verify(root):
    root = Path(root)
    protected_path = safe_path(root, 'configs/release/protected-results.json')
    require(sha(protected_path) == PROTECTED_RESULTS_SHA256, 'Historical evidence ledger changed')
    protected = read(protected_path)
    for name, digest in protected['sha256'].items():
        require(sha(safe_path(root,name)) == digest, 'Historical evidence changed: '+name)
    for folder, fn in [('qwen-quantization-v1',verify_exploration), ('qwen-confirmation-v1',verify_confirmation)]:
        r = fn(root/'results'/folder)
        require(aggregates_equal(r,read(root/'results'/folder/'summary.json')), 'Summary differs: '+folder)
    confirm = r
    require(sha(root/'configs/qwen-confirmation/study.json')==sha(root/'results/qwen-confirmation-v1/protocol.json'),'Published protocol differs from run')
    require(sha(root/'configs/qwen-confirmation/dataset/data.jsonl')==sha(root/'results/qwen-confirmation-v1/data.jsonl'),'Published data differs from run')
    require(sha(root/'configs/qwen-quantization/tensor-identities.json')==sha(root/'results/qwen-confirmation-v1/tensor-identities.json'),'Confirmation identity reference drift')
    verify_fair(root/'results/qwen-prefix-fair-v1')
    verify_lifecycle(root/'results/qwen-cache-lifecycle-gpu-v1')
    rebuild = root/'results/qwen-model-rebuild-v1'
    verify_hashes(rebuild,read(rebuild/'checksums.json'),exclude=('checksums.json',))
    run = read(rebuild/'run.json'); require(run['status']=='complete','Model rebuild failed')
    require(run['upstream_identity']==read(root/'configs/qwen-quantization/upstream-identity.json'),'Upstream identity reference drift')
    expected = read(root/'configs/qwen-quantization/tensor-identities.json')
    require(run['expected_identities_sha256']==sha(root/'configs/qwen-quantization/tensor-identities.json'),'Rebuild reference drift')
    require(set(run['models'])==set(expected),'Rebuild model coverage')
    verify_hashes(rebuild/'source', run['source_sha256'], exclude=())
    tokenizer = read(root/'configs/qwen-quantization/model-identities.json')['fp16']
    for v, record in run['models'].items():
        require(record['tensor_identity_verified'],'Unverified rebuilt model')
        verify_inference_files(record['files'],tokenizer)
        quality = read(root/f'results/qwen-confirmation-v1/{v}-quality.json')
        require(quality['model_files']==record['files'],'Confirmation did not use rebuilt models')
    archive_run = read(root/'results/qwen-confirmation-v1/run.json')
    require(archive_run['git_head'] is None,'Archive run must not rely on Git')
    for name in ['THIRD_PARTY.md','DATA_LICENSE.md','NOTICE-Domain-QA-Lab.txt','docs/release-reproduction.md']:
        require((root/name).is_file(),'Missing release document: '+name)
    return {'technical_acceptance':'pass', 'historical_files_preserved':len(protected['sha256']),
            'rebuilt_variants':len(run['models']), 'confirmation_samples':confirm['n'],
            'confirmation_gate_passed':confirm['gate']['passed'],
            'source_archive_inference_verified':True,
            'root_code_license_present':(root/'LICENSE').is_file(),
            'scope':'Reproducible personal research artifact; not a deployable QA system or confirmed quantization algorithm.'}


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--require-license',action='store_true');a=p.parse_args()
    result=verify(a.root)
    if a.require_license: require(result['root_code_license_present'],'Root code licensing decision is pending')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
