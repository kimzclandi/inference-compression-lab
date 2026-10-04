"""Compatibility entrypoint for the strengthened current pruning verifier.

The original verifier is retained in qa-risk-pruning-readiness-v1/reference.
Original experiment records, manifests and source snapshots are unchanged.
"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.verify_qa_risk_pruning import verify as _verify


def verify():
    return _verify(ROOT)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
