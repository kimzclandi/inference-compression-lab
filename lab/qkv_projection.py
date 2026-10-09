"""Opt-in QKV projection packing for the pinned Qwen2 Q8 decode workload.

Three affine 8-bit/group-64 quantized matrices are concatenated along their
output dimension, without dequantization or requantization. One upstream MLX
quantized matmul then produces Q/K/V, followed by the original bias addition
and three slices. This changes dispatch and matrix shape, not the quantizer or
Attention algorithm. Floating-point equivalence and speed require measurement.

Preparation owns additional packed arrays and evaluates them before use; its
memory and construction cost must be reported separately. Original projection
objects/arrays are retained unchanged. Both compiled modes still need an
explicit shape-specific warmup before a benchmark: constructing mx.compile is
not evidence that compilation has already finished.

Supported projection input is finite FP16 [1, 1, 896]; outputs are FP16
[1, 1, 896], [1, 1, 128], [1, 1, 128]. The hot path validates metadata only,
without hidden value scans or synchronization. The caller verifies finite
activations in a separate correctness phase. This is inference-only.
"""

from contextlib import contextmanager


PROJECTION_MODES = ("native", "compiled_three", "packed")
ADAPTER_MODES = ("native_adapter", "compiled_three", "packed")
OUTPUT_DIMS = (896, 128, 128)


def _shape(value):
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(d) is not int or d <= 0 for d in value)):
        raise ValueError("tensor dimensions must be positive integers")
    return tuple(value)


def validate_projection_metadata(metadata, x_shape=None, x_dtype=None):
    """Validate three CPU-only metadata dictionaries; return packing dimensions.

    Each projection supplies bits/group_size/mode and weight/scales/biases/bias
    entries. A tensor entry has ``shape`` and ``dtype`` strings; no values or
    MLX imports are needed. Quantization ``biases`` and output ``bias`` are
    distinct arrays and both are required for this narrow affine contract.
    """
    if not isinstance(metadata, (tuple, list)) or len(metadata) != 3:
        raise ValueError("require metadata for exactly Q, K and V")
    for entry, rows in zip(metadata, OUTPUT_DIMS):
        if not isinstance(entry, dict):
            raise ValueError("projection metadata must be dictionaries")
        if (type(entry.get("bits")) is not int or entry["bits"] != 8
                or type(entry.get("group_size")) is not int
                or entry["group_size"] != 64 or entry.get("mode") != "affine"):
            raise ValueError("require affine 8-bit/group-64 projections")
        expected = {
            "weight": ((rows, 224), "uint32"),
            "scales": ((rows, 14), "float16"),
            "biases": ((rows, 14), "float16"),
            "bias": ((rows,), "float16"),
        }
        for name, (shape, dtype) in expected.items():
            tensor = entry.get(name)
            if not isinstance(tensor, dict):
                raise ValueError(f"missing tensor metadata: {name}")
            if _shape(tensor.get("shape")) != shape or tensor.get("dtype") != dtype:
                raise ValueError(f"unexpected {name} shape or dtype")
    if (x_shape is None) != (x_dtype is None):
        raise ValueError("input shape and dtype must be supplied together")
    if x_shape is not None and (_shape(x_shape) != (1, 1, 896) or x_dtype != "float16"):
        raise ValueError("require FP16 input [1,1,896]")
    return dict(input_dims=896, output_dims=OUTPUT_DIMS, packed_output_dims=1152,
                bits=8, group_size=64, mode="affine")


def _backend():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen2 import Attention, Model

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("require Metal GPU as default device; no CPU fallback")
    return mx, nn, Model, Attention, KVCache, scaled_dot_product_attention


def _dtype(array):
    return str(array.dtype).split(".")[-1]


def _metadata(projection):
    result = {name: getattr(projection, name, None)
              for name in ("bits", "group_size", "mode")}
    for name in ("weight", "scales", "biases", "bias"):
        tensor = getattr(projection, name, None)
        result[name] = (None if tensor is None else
                        dict(shape=tuple(tensor.shape), dtype=_dtype(tensor)))
    return result


class PackedQKV:
    """Prepared immutable-by-contract packing of three original projections.

    ``project(x, mode)`` returns unreshaped Q/K/V. ``native`` calls the original
    modules; ``compiled_three`` compiles those same three calls together;
    ``packed`` compiles one packed matmul/bias/split graph. Constructor scans
    floating parameter arrays for finiteness and materializes concatenations.
    Do not mutate the original or packed parameters after preparation.
    """

    def __init__(self, q_proj, k_proj, v_proj):
        mx, nn, *_ = _backend()
        self._mx = mx
        self.originals = (q_proj, k_proj, v_proj)
        if any(type(p) is not nn.QuantizedLinear for p in self.originals):
            raise ValueError("require original QuantizedLinear projections")
        self.metadata = tuple(_metadata(p) for p in self.originals)
        validate_projection_metadata(self.metadata)
        # This value validation is preparation work, never hidden in project().
        finite = [mx.all(mx.isfinite(getattr(p, name))) for p in self.originals
                  for name in ("scales", "biases", "bias")]
        mx.eval(*finite)
        if not all(bool(value.item()) for value in finite):
            raise ValueError("projection parameters must be finite")
        for name in ("weight", "scales", "biases", "bias"):
            setattr(self, name, mx.concatenate(
                [getattr(p, name) for p in self.originals], axis=0))
        mx.eval(self.weight, self.scales, self.biases, self.bias)
        self.packed_nbytes = sum(getattr(self, name).nbytes
                                for name in ("weight", "scales", "biases", "bias"))

        def three(x):
            return tuple(p(x) for p in self.originals)

        def packed(x):
            qkv = mx.quantized_matmul(
                x, self.weight, scales=self.scales, biases=self.biases,
                transpose=True, group_size=64, bits=8, mode="affine")
            qkv = qkv + self.bias
            return qkv[..., :896], qkv[..., 896:1024], qkv[..., 1024:1152]

        self._native = three
        self._compiled_three = mx.compile(three, shapeless=False)
        self._packed = mx.compile(packed, shapeless=False)

    def project(self, x, mode="native"):
        mx = self._mx
        if mode not in PROJECTION_MODES:
            raise ValueError("unknown QKV projection mode")
        if mx.default_device() != mx.gpu:
            raise RuntimeError("projection requires GPU as default device")
        if not isinstance(x, mx.array):
            raise ValueError("projection input must be an MLX array")
        # Immutable parameter metadata was validated during preparation. Avoid
        # repeating twelve tensor-metadata scans for every layer and token.
        if tuple(x.shape) != (1, 1, 896) or _dtype(x) != "float16":
            raise ValueError("require FP16 input [1,1,896]")
        return {"native": self._native, "compiled_three": self._compiled_three,
                "packed": self._packed}[mode](x)


class PreparedQKV:
    """Model-bound projection preparation; construct before timed requests."""

    def __init__(self, model):
        _, _, Model, Attention, *_ = _backend()
        args = getattr(model, "args", None)
        if type(model) is not Model or args is None or (
                args.hidden_size, args.num_attention_heads,
                args.num_key_value_heads, args.num_hidden_layers) != (896, 14, 2, 24):
            raise ValueError("require pinned 24-layer Qwen2 head configuration")
        self.model = model
        self.originals = tuple(layer.self_attn for layer in model.model.layers)
        if len(self.originals) != 24 or any(type(a) is not Attention for a in self.originals):
            raise ValueError("require original Attention objects; nested adapters unsupported")
        if any((a.n_heads, a.n_kv_heads, a.scale) != (14, 2, 0.125)
               for a in self.originals):
            raise ValueError("Attention head metadata differs from pinned model")
        self.projections = tuple(PackedQKV(a.q_proj, a.k_proj, a.v_proj)
                                 for a in self.originals)
        self.packed_nbytes = sum(p.packed_nbytes for p in self.projections)
        self._active = False


def prepare_qkv(model):
    """Return model-bound packing; no cache updates or model forward is run."""
    return PreparedQKV(model)


@contextmanager
def qkv_decode_adapter(model, prepared, mode="native_adapter", *, counts=None,
                       observer=None):
    """Temporarily replace instance Attention modules for B1/L1 decode only.

    Prefill delegates to each exact original object. Native_adapter preserves
    this routing while using the original three projections as a control.
    RoPE, KV update, native SDPA and output projection are unchanged. The
    optional observer(layer_index,x,q,k,v,cache) runs after projection, before
    reshape/RoPE/cache mutation; it is for separate correctness diagnostics.
    Counts measure Python routing, not GPU kernel launches. Nested adapters and
    mismatched preparation are rejected; original identities restore even if a
    callback, projection, cache update or model execution raises.
    """
    mx, nn, _, _, KVCache, sdpa = _backend()
    if mode not in ADAPTER_MODES:
        raise ValueError("unknown QKV adapter mode")
    if (type(prepared) is not PreparedQKV or prepared.model is not model
            or prepared._active):
        raise ValueError("mismatched preparation or nested adapter")
    if counts is None:
        counts = {}
    if not isinstance(counts, dict) or (observer is not None and not callable(observer)):
        raise ValueError("counts must be a dictionary; observer must be callable")
    layers = model.model.layers
    if len(layers) != len(prepared.originals) or any(
            layer.self_attn is not original
            for layer, original in zip(layers, prepared.originals)):
        raise ValueError("Attention objects changed after preparation")

    class Adapter(nn.Module):
        def __init__(self, upstream, projection, index):
            super().__init__()
            self.upstream = upstream
            # PackedQKV is deliberately not an nn.Module: it never replaces
            # or renames the model's original learned parameter objects.
            self.projection = projection
            self.index = index

        def __call__(self, x, mask=None, cache=None):
            if not isinstance(x, mx.array) or x.ndim != 3 or x.shape[1] < 1:
                raise ValueError("require nonempty rank-three MLX activations")
            if x.shape[1] != 1:
                counts["prefill"] = counts.get("prefill", 0) + 1
                return self.upstream(x, mask, cache)
            if tuple(x.shape) != (1, 1, 896) or _dtype(x) != "float16":
                raise ValueError("decode requires FP16 [1,1,896]")
            if mask is not None or type(cache) is not KVCache:
                raise ValueError("unsupported decode mask/cache; no fallback")
            if type(cache.offset) is not int or not 0 <= cache.offset < 8192:
                raise ValueError("decode cache offset must be integer in [0,8191]")
            keys, values = cache.keys, cache.values
            if keys is None or values is None:
                if keys is not None or values is not None or cache.offset:
                    raise ValueError("incomplete decode cache state")
            else:
                expected_prefix = (1, 2)
                if (keys.ndim != 4 or values.ndim != 4
                        or tuple(keys.shape) != tuple(values.shape)
                        or tuple(keys.shape[:2]) != expected_prefix
                        or keys.shape[3] != 64 or keys.shape[2] < cache.offset
                        or _dtype(keys) != "float16" or _dtype(values) != "float16"):
                    raise ValueError("unsupported decode cache tensor metadata")
            project_mode = "native" if mode == "native_adapter" else mode
            q, k, v = self.projection.project(x, project_mode)
            if observer is not None:
                observer(self.index, x, q, k, v, cache)
            q = q.reshape(1, 1, 14, 64).transpose(0, 2, 1, 3)
            k = k.reshape(1, 1, 2, 64).transpose(0, 2, 1, 3)
            v = v.reshape(1, 1, 2, 64).transpose(0, 2, 1, 3)
            a = self.upstream
            q = a.rope(q, offset=cache.offset)
            k = a.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
            out = sdpa(q, k, v, cache=cache, scale=a.scale, mask=None)
            counts["decode"] = counts.get("decode", 0) + 1
            return a.o_proj(out.transpose(0, 2, 1, 3).reshape(1, 1, 896))

    prepared._active = True
    try:
        for index, (layer, original, projection) in enumerate(
                zip(layers, prepared.originals, prepared.projections)):
            layer.self_attn = Adapter(original, projection, index)
        yield counts
    finally:
        for layer, original in zip(layers, prepared.originals):
            layer.self_attn = original
        prepared._active = False
