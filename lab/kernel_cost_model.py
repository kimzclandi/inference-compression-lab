"""Conservative logical-traffic model for residual-add plus RMSNorm.

This module does not measure hardware bandwidth, cache traffic, occupancy or
dispatch cost.  It counts tensor bytes crossing primitive boundaries and a
documented set of scalar operations so that a fused implementation can be
compared with a materialized native pair without turning a static estimate
into a performance claim.
"""

import math


DTYPE_BYTES = {"float16": 2, "float32": 4}


def residual_rmsnorm_cost(rows, width, dtype):
    """Return logical byte traffic and an arithmetic-intensity proxy.

    Native materialization counts Add as two reads plus one write, followed by
    RMSNorm as residual/weight reads plus one output write.  The fused kernel
    still returns both residual and normalized outputs, so it removes only the
    primitive-boundary reread of the residual.  Cache effects and internal
    reduction passes are intentionally excluded from both arms.
    """
    if type(rows) is not int or rows <= 0 or type(width) is not int or width <= 0:
        raise ValueError("rows and width must be positive integers")
    if dtype not in DTYPE_BYTES:
        raise ValueError("dtype must be float16 or float32")

    elements = rows * width
    item_bytes = DTYPE_BYTES[dtype]
    native_bytes = 6 * elements * item_bytes
    fused_bytes = 5 * elements * item_bytes
    # D residual adds, D squares, D-1 reduction adds, one division, epsilon
    # addition and rsqrt, then D normalization and D weight multiplies.
    counted_scalar_operations = rows * (5 * width + 2)
    return {
        "rows": rows,
        "width": width,
        "dtype": dtype,
        "elements": elements,
        "item_bytes": item_bytes,
        "counted_scalar_operations": counted_scalar_operations,
        "native_logical_bytes": native_bytes,
        "fused_logical_bytes": fused_bytes,
        "logical_bytes_saved": native_bytes - fused_bytes,
        "logical_byte_reduction_fraction": (native_bytes - fused_bytes) / native_bytes,
        "traffic_only_speedup_ceiling": native_bytes / fused_bytes,
        "native_intensity_proxy_ops_per_byte": counted_scalar_operations / native_bytes,
        "fused_intensity_proxy_ops_per_byte": counted_scalar_operations / fused_bytes,
    }


def analyze_fixed_summary(summary, rows=(1, 64, 512, 2048), width=896,
                          dtype="float16"):
    """Attach fixed observed timings to the static cost model.

    The returned ratios remain descriptive post-hoc diagnostics.  They do not
    alter the frozen study, its gates or its rejected candidate.
    """
    if summary.get("schema") != "metal-residual-rmsnorm-independent-verification-v1":
        raise ValueError("unexpected summary schema")
    if summary.get("acceptance", {}).get("accepted") is not False:
        raise ValueError("the fixed rejected-candidate result must be retained")
    measurements = summary.get("micro_measurements")
    if not isinstance(measurements, dict):
        raise ValueError("missing micro measurements")

    records = []
    for row in rows:
        cell = measurements.get(str(row))
        if not isinstance(cell, dict):
            raise ValueError(f"missing row {row}")
        medians = {}
        for arm in ("native", "compiled", "metal"):
            value = cell.get(arm, {}).get("median_of_round_medians")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {arm} timing for row {row}")
            medians[arm] = float(value)
        cost = residual_rmsnorm_cost(row, width, dtype)
        observed_native = medians["native"] / medians["metal"]
        observed_compiled = medians["compiled"] / medians["metal"]
        cost.update({
            "observed_seconds": medians,
            "observed_speedup_native_over_metal": observed_native,
            "observed_speedup_compiled_over_metal": observed_compiled,
            "native_speedup_fraction_of_traffic_only_ceiling": (
                observed_native / cost["traffic_only_speedup_ceiling"]
            ),
        })
        records.append(cost)

    return {
        "schema": "metal-residual-rmsnorm-cost-model-v1",
        "status": "post_hoc_diagnostic_only",
        "changes_fixed_acceptance": False,
        "fixed_candidate_accepted": False,
        "model": {
            "native": "materialized Add followed by RMSNorm at primitive boundaries",
            "fused": "single kernel returning both rounded residual and normalized output",
            "excluded": [
                "cache-line effects",
                "internal reduction passes",
                "register and threadgroup-memory traffic",
                "occupancy",
                "dispatch and host enqueue cost",
                "measured DRAM bandwidth",
            ],
            "interpretation": (
                "The fused contract removes one logical residual reread, only one sixth "
                "of the modeled native bytes. A 1.2x traffic-only ceiling is an idealized "
                "upper bound under these assumptions, not a predicted speedup."
            ),
        },
        "records": records,
    }
