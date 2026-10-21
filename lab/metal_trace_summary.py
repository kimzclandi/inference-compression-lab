"""Summarize process-attributed intervals exported by ``xctrace``.

The parser intentionally consumes exported XML rather than a ``.trace``
bundle.  It reports interval counts and descriptive timing only; Instruments
changes execution enough that these values are diagnostics, not benchmarks.
"""

from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def _resolved(cell, definitions):
    reference = cell.get("ref")
    return definitions[reference] if reference else cell


def read_export(path, expected_schema):
    root = ET.parse(path).getroot()
    schema = root.find(".//schema")
    if schema is None or schema.get("name") != expected_schema:
        actual = None if schema is None else schema.get("name")
        raise ValueError(f"expected {expected_schema!r}, found {actual!r}")
    columns = [col.findtext("mnemonic") for col in schema.findall("col")]
    definitions = {
        element.get("id"): element
        for element in root.iter()
        if element.get("id") is not None
    }
    rows = []
    for row in root.findall(".//row"):
        values = {}
        for column, cell in zip(columns, list(row), strict=True):
            if cell.tag == "sentinel":
                values[column] = None
                continue
            target = _resolved(cell, definitions)
            values[column] = {
                "tag": target.tag,
                "raw": target.text,
                "fmt": target.get("fmt"),
            }
        rows.append(values)
    return rows


def _raw_int(value):
    if value is None or value["raw"] is None:
        return None
    return int(value["raw"])


def _duration_stats(values):
    if not values:
        return {"count": 0, "sum_ns": 0, "median_ns": None, "p95_ns": None}
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "count": len(ordered),
        "sum_ns": sum(ordered),
        "median_ns": statistics.median(ordered),
        "p95_ns": p95,
    }


def summarize_application_export(path, process_prefix="python ("):
    rows = read_export(path, "metal-application-intervals")
    target = [
        row
        for row in rows
        if row["process"] is not None
        and (row["process"]["fmt"] or "").startswith(process_prefix)
    ]
    grouped = {}
    for row in target:
        label = "" if row["event-label"] is None else row["event-label"]["fmt"] or ""
        category = label.split("  (", 1)[0].strip() or "unlabelled"
        grouped.setdefault(category, []).append(_raw_int(row["duration"]))
    return {
        "schema": "metal-application-interval-summary-v1",
        "source": str(Path(path)),
        "process_prefix": process_prefix,
        "target_rows": len(target),
        "categories": {
            name: _duration_stats([value for value in values if value is not None])
            for name, values in sorted(grouped.items())
        },
    }


def summarize_gpu_export(path, process_prefix="python ("):
    rows = read_export(path, "metal-gpu-intervals")
    target = [
        row
        for row in rows
        if row["process"] is not None
        and (row["process"]["fmt"] or "").startswith(process_prefix)
    ]
    compute = [
        row
        for row in target
        if row["channel-name"] is not None
        and row["channel-name"]["fmt"] == "Compute"
    ]
    intervals = [
        (_raw_int(row["start"]), _raw_int(row["duration"])) for row in compute
    ]
    intervals = [
        (start, duration)
        for start, duration in intervals
        if start is not None and duration is not None
    ]
    durations = [duration for _, duration in intervals]
    span_ns = None
    if intervals:
        span_ns = max(start + duration for start, duration in intervals) - min(
            start for start, _ in intervals
        )
    return {
        "schema": "metal-gpu-interval-summary-v1",
        "source": str(Path(path)),
        "process_prefix": process_prefix,
        "target_rows": len(target),
        "compute": {
            **_duration_stats(durations),
            "observed_span_ns": span_ns,
        },
    }
