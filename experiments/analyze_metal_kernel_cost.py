"""Derive a deterministic cost model from the frozen Metal study summary."""

import argparse
import hashlib
import json
from pathlib import Path

from lab.kernel_cost_model import analyze_fixed_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "metal-residual-rmsnorm-v1" / "summary.json"
DEFAULT_OUTPUT = ROOT / "results" / "metal-residual-rmsnorm-v1" / "cost-model.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = analyze_fixed_summary(summary)
    result["source_identity"] = {
        "summary_path": str(args.summary.relative_to(ROOT)),
        "summary_sha256": sha256(args.summary),
        "analysis_source_sha256": sha256(Path(__file__)),
        "cost_model_source_sha256": sha256(ROOT / "lab" / "kernel_cost_model.py"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
