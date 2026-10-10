"""Apply a StreamingPolicy to MLX-LM KV caches after prefill (Mac/MLX only).

Position-preserving eviction: kept keys retain the RoPE rotation applied when
they were written, and `offset` keeps counting ORIGINAL positions so new tokens
are rotated at their true positions. The physical buffer is shorter than
`offset`, so this class keeps its own concatenating update path instead of
the upstream pre-allocated buffer (whose write index is `offset`).

Only single-token decode is supported after eviction (mask is None for T == 1).
"""
from __future__ import annotations

from lab.kv_eviction import StreamingPolicy


class EvictedKVCache:
    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        if keys.shape[2] != 1:
            raise ValueError('EvictedKVCache supports single-token decode only')
        self.keys = mx.concatenate([self.keys, keys], axis=2)
        self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += 1
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @property
    def physical_tokens(self):
        return int(self.keys.shape[2])

    @property
    def nbytes(self):
        return self.keys.nbytes + self.values.nbytes


def evict_caches(caches, policy: StreamingPolicy):
    """Return new EvictedKVCache list; native caches are left unchanged."""
    import mlx.core as mx
    out = []
    for c in caches:
        n = c.offset
        idx = mx.array(policy.keep_indices(n).astype('int32'))
        k = mx.take(c.keys[..., :n, :], idx, axis=2)
        v = mx.take(c.values[..., :n, :], idx, axis=2)
        out.append(EvictedKVCache(k, v, n))
    mx.eval([x.keys for x in out] + [x.values for x in out])
    return out


def evict_caches_per_head(caches, indices):
    """indices[layer]: int array [n_kv_heads, budget] of kept positions per head."""
    import mlx.core as mx
    out = []
    for c, idx in zip(caches, indices):
        n = c.offset
        ix = mx.array(idx.astype('int32'))[None, :, :, None]  # [1, kvh, budget, 1]
        k = mx.take_along_axis(c.keys[..., :n, :], ix, axis=2)
        v = mx.take_along_axis(c.values[..., :n, :], ix, axis=2)
        out.append(EvictedKVCache(k, v, n))
    mx.eval([x.keys for x in out] + [x.values for x in out])
    return out
