"""Recompute the historical diagnostic from public, process-filtered CSV rows.

No profiling, model execution or benchmark is performed. The original XML
hashes remain in the unchanged receipt; CSV files expose only the fields needed
to independently recompute its interval statistics.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/metal-trace-rows-v1"


def verify(root=EVIDENCE):
    manifest = json.loads((root / "manifest.json").read_text())
    receipt = json.loads((ROOT / "results/metal-residual-rmsnorm-v1/trace-diagnostic.json").read_text())
    if manifest["source_xml_sha256"] != receipt["machine_local_export_sha256"]:
        raise ValueError("source XML identity differs from historical receipt")
    results = {}
    for arm in ("native", "metal"):
        results[arm] = {}
        for kind, key in (("application", None), ("gpu", "gpu_compute_interval")):
            name = f"{arm}-{kind}.csv"
            path = root / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
                raise ValueError(f"row hash differs: {name}")
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row["process"] != manifest["processes"][arm]:
                    raise ValueError("unexpected process")
                if int(row["start_ns"]) < 0 or int(row["duration_ns"]) <= 0:
                    raise ValueError("invalid interval")
            categories = {"Compute": key} if key else {
                "Command Buffer 0": "application_command_buffer",
                "Compute Command 0": "application_compute_command",
            }
            if {r["category"] for r in rows} != set(categories):
                raise ValueError("unexpected interval category")
            for category, metric in categories.items():
                selected = [r for r in rows if r["category"] == category]
                durations = sorted(int(r["duration_ns"]) for r in selected)
                actual = {"count": len(selected), "median_ns": statistics.median(durations),
                          "p95_ns": durations[math.ceil(.95 * len(durations)) - 1]}
                if key:
                    actual.update(sum_ns=sum(durations), observed_span_ns=max(
                        int(r["start_ns"]) + int(r["duration_ns"]) for r in selected
                    ) - min(int(r["start_ns"]) for r in selected))
                if actual != receipt[arm][metric]:
                    raise ValueError(f"historical statistics differ: {arm}/{metric}")
                results[arm][metric] = actual
    if receipt["changes_fixed_acceptance"] or receipt["status"] != "post_hoc_diagnostic_only":
        raise ValueError("diagnostic boundary changed")
    return results


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
