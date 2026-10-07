"""Summarize process-attributed XML tables exported from xctrace."""

import argparse
import json
from pathlib import Path

from lab.metal_trace_summary import (
    summarize_application_export,
    summarize_gpu_export,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    result = {
        "schema": "metal-residual-rmsnorm-xctrace-summary-v1",
        "status": "diagnostic_not_benchmark",
        "application": summarize_application_export(args.application),
        "gpu": summarize_gpu_export(args.gpu),
        "changes_fixed_acceptance": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
