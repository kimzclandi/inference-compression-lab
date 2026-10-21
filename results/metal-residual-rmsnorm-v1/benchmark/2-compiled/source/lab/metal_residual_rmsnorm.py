"""Opt-in residual-add/RMSNorm fusion for the pinned Metal experiment.

Inputs: x and residual [..., D], weight [D], all FP16 or all FP32. Outputs
are the rounded residual h and weighted RMS-normalized y with the same shape.
The caller must supply finite inputs whose rounded residual and FP32 squared
sum remain finite. Value scans would synchronize the GPU and are deliberately
not hidden in this operator. This is inference-only; no gradient is supplied.

The Metal reduction and rounding adapt MLX v0.29.3's MIT implementation;
see the kernel's attribution and third_party/MLX-MIT.txt. Imports stay lazy so
ordinary CPU/Linux evidence checks need neither MLX nor an Apple GPU.
"""

from functools import lru_cache
import math
from pathlib import Path
import struct


MODES = ("native", "compiled", "metal")
KERNEL_PATH = Path(__file__).parent / "kernels" / "residual_rmsnorm.metal"


def validate_metadata(x_shape, residual_shape, weight_shape, x_dtype,
                      residual_dtype, weight_dtype, eps, mode="native"):
    """Validate the narrow contract without importing MLX or reading values.

    Dtypes are the strings ``float16`` or ``float32``. Return launch metadata;
    threads follows a fixed formula, never a measured tuning choice.
    """
    if mode not in MODES:
        raise ValueError("Unknown residual RMSNorm mode")
    shapes = (x_shape, residual_shape, weight_shape)
    if any(not isinstance(s, (tuple, list)) or not s or
           any(type(d) is not int or d <= 0 for d in s) for s in shapes):
        raise ValueError("Shapes must contain positive integer dimensions")
    x_shape, residual_shape, weight_shape = map(tuple, shapes)
    if x_shape != residual_shape:
        raise ValueError("Residual shape must match input; broadcasting is unsupported")
    width = x_shape[-1]
    if not 1 <= width <= 4096:
        raise ValueError("Supported last dimension is 1..4096")
    if weight_shape != (width,):
        raise ValueError("Weight must be one-dimensional with the input width")
    if x_dtype not in ("float16", "float32") or not (
            x_dtype == residual_dtype == weight_dtype):
        raise ValueError("Inputs and weight must share float16 or float32 dtype")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("Epsilon must be a positive finite FP32-representable scalar")
    try:
        eps32 = struct.unpack("f", struct.pack("f", eps))[0]
    except (OverflowError, struct.error):
        raise ValueError("Epsilon is not finite FP32") from None
    if not math.isfinite(eps32) or eps32 <= 0:
        raise ValueError("Epsilon must remain positive and finite in FP32")
    rows = math.prod(x_shape[:-1])
    threads = 32 * ((width + 127) // 128)
    if rows * threads > 2**32 - 1:
        raise ValueError("Grid exceeds the supported 32-bit launch extent")
    return {"rows": rows, "width": width, "threads": threads, "eps32": eps32}


def _backend():
    import mlx.core as mx

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("Require an available Metal GPU and default device mx.gpu")
    return mx


@lru_cache(maxsize=1)
def _metal_kernel():
    mx = _backend()
    return mx.fast.metal_kernel(
        name="icl_residual_rmsnorm_v1",
        input_names=["x", "residual", "weight", "epsilon"],
        output_names=["h", "y"],
        source=KERNEL_PATH.read_text(encoding="utf-8"),
        header="#include <metal_simdgroup>\n",
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=32)
def _pair(mode, eps):
    """Cache both compiled wrappers; eps is a constant in each traced pair."""
    mx = _backend()
    if mode == "compiled":
        def native_pair(x, residual, weight):
            h = x + residual
            return h, mx.fast.rms_norm(h, weight, eps)

        return mx.compile(native_pair, shapeless=False)

    kernel = _metal_kernel()
    # A 1D constant has pointer semantics in generated Metal. Construct/evaluate
    # it once per epsilon, outside subsequent timed calls and shape traces.
    epsilon = mx.array([eps], dtype=mx.float32)
    mx.eval(epsilon)

    def custom_pair(x, residual, weight):
        width = x.shape[-1]
        rows = x.size // width
        threads = 32 * ((width + 127) // 128)
        outputs = kernel(
            inputs=[x, residual, weight, epsilon],
            template=[("T", x.dtype)],
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[x.shape, x.shape],
            output_dtypes=[x.dtype, x.dtype],
            stream=mx.gpu,
        )
        return outputs[0], outputs[1]

    return mx.compile(custom_pair, shapeless=False)


def residual_rmsnorm(x, residual, weight, eps, mode="native"):
    """Return (h, y), with an explicit native/compiled/metal opt-in mode.

    Noncontiguous tensors are accepted: MLX may copy them before the custom
    kernel. Those copies are part of this function's execution, not hidden
    preprocessing. Every arm has the same metadata checks and GPU requirement.
    """
    if mode not in MODES:
        raise ValueError("Unknown residual RMSNorm mode")
    mx = _backend()
    if not all(isinstance(a, mx.array) for a in (x, residual, weight)):
        raise ValueError("Inputs and weight must be MLX arrays")
    metadata = validate_metadata(
        x.shape, residual.shape, weight.shape,
        str(x.dtype).split(".")[-1], str(residual.dtype).split(".")[-1],
        str(weight.dtype).split(".")[-1], eps, mode,
    )
    eps = metadata["eps32"]
    if mode == "native":
        h = x + residual
        return h, mx.fast.rms_norm(h, weight, eps)
    return _pair(mode, eps)(x, residual, weight)


def qwen_forward(model, inputs, cache, mode="native", trace=None):
    """Return [1, 1, vocabulary] logits, updating ordinary KV caches once.

    All modes compute every decoder position, then project only the last one.
    Only the attention residual and post-attention norm are replaced. ``trace``
    records Python call/shape evidence; it is not a GPU execution profiler.
    """
    if mode not in MODES:
        raise ValueError("Unknown residual RMSNorm mode")
    if trace is not None and not isinstance(trace, dict):
        raise ValueError("Trace must be a dictionary or None")
    from . import qwen_demand_prefill as demand

    backend = demand._backend()
    _backend()  # Explicitly disallow silent CPU fallback for all three arms.
    offset = demand._validate(model, inputs, cache, "head_only", None, None, backend)
    _, _, _, create_mask, _ = backend
    decoder = model.model
    hidden = decoder.embed_tokens(inputs)
    mask = create_mask(hidden, cache)
    pairs = [] if trace is not None else None
    for index, (layer, layer_cache) in enumerate(zip(decoder.layers, cache)):
        attention = layer.self_attn(layer.input_layernorm(hidden), mask, layer_cache)
        norm = layer.post_attention_layernorm
        h, normalized = residual_rmsnorm(hidden, attention, norm.weight, norm.eps, mode)
        hidden = h + layer.mlp(normalized)
        if pairs is not None:
            pairs.append({"layer": index, "shape": list(h.shape),
                          "dtype": str(h.dtype), "mode": mode})
    # Preserve the upstream full final-norm shape, identically in every arm.
    hidden = decoder.norm(hidden)[:, -1:, :]
    logits = decoder.embed_tokens.as_linear(hidden)
    if trace is not None:
        trace.update(mode=mode, input_shape=list(inputs.shape),
                     prior_cache_tokens=offset, output_shape=list(logits.shape),
                     cache_offsets_after=[c.offset for c in cache],
                     residual_norm_pairs=pairs,
                     residual_norm_pair_calls=len(pairs), gpu_profiler=False)
    return logits
