"""Opt-in, inference-only B1/Hq14/Hkv2/D64 FP16 decode Attention.

Only unmasked, single-token queries are supported. FP32 online-softmax state is
merged across fixed 128-key partitions. K/V are indexed by actual strides;
no forced-contiguous conversion or physical K/V-head expansion is performed.
"""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

MODES = ('native', 'native_compiled', 'shared_compiled')


def validate_metadata(q_shape, k_shape, v_shape, dtypes):
    if tuple(q_shape) != (1, 14, 1, 64):
        raise ValueError('require q [1,14,1,64]')
    if len(k_shape) != 4 or tuple(k_shape[:2]) != (1, 2) or k_shape[3] != 64:
        raise ValueError('require k [1,2,N,64]')
    if tuple(v_shape) != tuple(k_shape) or type(k_shape[2]) is not int or not 1 <= k_shape[2] <= 8192:
        raise ValueError('matching K/V and 1<=N<=8192 required')
    if tuple(dtypes) != ('float16',) * 3:
        raise ValueError('FP16 query and unquantized FP16 K/V required')
    return (k_shape[2] + 127) // 128


@lru_cache(maxsize=1)
def _kernels():
    import mlx.core as mx
    folder = Path(__file__).parent / 'kernels'
    first = mx.fast.metal_kernel(name='icl_gqa_shared_partial_v1',
        input_names=['q', 'k', 'v'], output_names=['partial'],
        source=(folder / 'gqa_shared_partial.metal').read_text(),
        header='#include <metal_simdgroup>\n', ensure_row_contiguous=False)
    second = mx.fast.metal_kernel(name='icl_gqa_shared_merge_v1',
        input_names=['partial'], output_names=['out'],
        source=(folder / 'gqa_shared_merge.metal').read_text(), ensure_row_contiguous=False)
    return first, second


def _shared(q, k, v):
    import mlx.core as mx
    first, second = _kernels()
    parts = (k.shape[2] + 127) // 128
    partial = first(inputs=[q, k, v], template=[('T', mx.float16)],
        grid=(parts * 256, 2, 1), threadgroup=(256, 1, 1),
        output_shapes=[(14, parts, 66)], output_dtypes=[mx.float32])[0]
    return second(inputs=[partial], template=[('T', mx.float16)],
        grid=(14 * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[q.shape], output_dtypes=[mx.float16])[0]


@lru_cache(maxsize=2)
def _compiled(mode):
    import mlx.core as mx
    if mode == 'native_compiled':
        return mx.compile(lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=.125))
    return mx.compile(_shared)


def attention(q, k, v, mode='native'):
    import mlx.core as mx
    if mode not in MODES:
        raise ValueError('unknown Attention mode')
    validate_metadata(q.shape, k.shape, v.shape, [str(a.dtype).split('.')[-1] for a in (q, k, v)])
    if mx.default_device() != mx.gpu:
        raise RuntimeError('GPU required; no CPU fallback')
    if mode == 'native':
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=.125)
    return _compiled(mode)(q, k, v)


@contextmanager
def qwen_decode_adapter(model, mode, counts):
    """Replace only decode SDPA; native prefill and quantized projections stay.

    Instance-local adapters restore even on failure. Each layer retains its
    original projection/rope objects; no global monkeypatch or weight copy.
    Counts record Python routing, not GPU profiling. Native mode is unmodified.
    """
    import mlx.nn as nn
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen2 import Model, Attention
    if type(model) is not Model or (model.args.num_attention_heads,
            model.args.num_key_value_heads, model.args.hidden_size) != (14, 2, 896):
        raise ValueError('only pinned Qwen2 head configuration supported')
    if mode not in ('native', 'shared_compiled'):
        raise ValueError('unsupported model mode')
    original = [layer.self_attn for layer in model.model.layers]
    if any(type(a) is not Attention for a in original):
        raise ValueError('native original Attention required; nested adapters unsupported')
    if mode == 'native':
        yield
        return

    class Adapter(nn.Module):
        def __init__(self, upstream):
            super().__init__()
            self.upstream = upstream

        def __call__(self, x, mask=None, cache=None):
            if x.shape[1] != 1:
                counts['prefill'] = counts.get('prefill', 0) + 1
                return self.upstream(x, mask, cache)
            if x.shape[0] != 1 or mask is not None or type(cache) is not KVCache:
                raise ValueError('unsupported decode mask/batch/cache; no silent fallback')
            if type(cache.offset) is not int or not 0 <= cache.offset < 8192:
                raise ValueError('decode cache offset must be integer in [0,8191] before mutation')
            a = self.upstream
            q = a.q_proj(x).reshape(1, 1, 14, 64).transpose(0, 2, 1, 3)
            k = a.k_proj(x).reshape(1, 1, 2, 64).transpose(0, 2, 1, 3)
            v = a.v_proj(x).reshape(1, 1, 2, 64).transpose(0, 2, 1, 3)
            q = a.rope(q, offset=cache.offset)
            k = a.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
            out = attention(q, k, v, mode='shared_compiled')
            counts['decode'] = counts.get('decode', 0) + 1
            return a.o_proj(out.transpose(0, 2, 1, 3).reshape(1, 1, 896))

    try:
        for layer, upstream in zip(model.model.layers, original):
            layer.self_attn = Adapter(upstream)
        yield
    finally:
        for layer, upstream in zip(model.model.layers, original):
            layer.self_attn = upstream
