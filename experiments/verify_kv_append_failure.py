"""Check the preserved pre-timing v1 failure, not a performance acceptance."""
import hashlib
import json
from pathlib import Path


def verify(root=Path('results/kv-append-mps-v1')):
    info=json.loads((root/'run.json').read_text())
    if (info['status']!='failed' or info['artifact_sha256']!={}
            or info['error']!='ValueError: K/V or Attention correctness failure'):
        raise ValueError('Expected pre-timing correctness failure receipt')
    expected={'configs/kv-append-mps-v1.json','lab/append_only_kv.py','experiments/kv_append_mps.py',
              'experiments/attention_backend_study.py','experiments/attention_mps_study.py','lab/attention_reference.py'}
    if set(info['source_sha256'])!=expected:raise ValueError('Incomplete archived source')
    for name,digest in info['source_sha256'].items():
        if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Failed-study source changed')
    if set(p.name for p in root.iterdir())!={'run.json','source'}:
        raise ValueError('Unexpected results attached to failed study')
    print(json.dumps(dict(evidence_valid=True,expected_correctness_failure=True,performance_measured=False)))


if __name__=='__main__':verify()
