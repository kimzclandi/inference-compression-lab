"""StreamingLLM-style KV eviction: keep attention-sink tokens + a recent window.

Draft. Pure NumPy so the policy can be unit-tested on CPU; the MLX runner that
applies it to Qwen KV caches is not written yet.

Shapes: K, V are [layers, kv_heads, seq, head_dim]; eviction acts on axis 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamingPolicy:
    sink: int      # number of leading tokens always kept
    window: int    # number of most recent tokens kept

    def __post_init__(self) -> None:
        if self.sink < 0 or self.window < 1:
            raise ValueError("sink must be >=0 and window >=1")

    @property
    def budget(self) -> int:
        return self.sink + self.window

    @classmethod
    def from_ratio(cls, seq_len: int, ratio: float, sink: int = 4) -> "StreamingPolicy":
        """Budget = round(ratio * seq_len) tokens, of which `sink` are sinks."""
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        budget = max(sink + 1, round(ratio * seq_len))
        return cls(sink=sink, window=budget - sink)

    def keep_indices(self, seq_len: int) -> np.ndarray:
        """Sorted token positions kept for a cache holding `seq_len` tokens."""
        if seq_len <= self.budget:
            return np.arange(seq_len)
        sinks = np.arange(self.sink)
        recent = np.arange(seq_len - self.window, seq_len)
        return np.concatenate([sinks, recent])


def evict(k: np.ndarray, v: np.ndarray, policy: StreamingPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_kept, v_kept, kept_positions). Original arrays are not modified.

    kept_positions are the ORIGINAL positions; RoPE was already applied when the
    keys were written, so this draft keeps original positions (no re-indexing).
    Whether to re-index positions inside the cache, as StreamingLLM does, is an
    open protocol question (see docs/kv-eviction-streaming.md).
    """
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("k and v must share shape [layers, heads, seq, dim]")
    idx = policy.keep_indices(k.shape[2])
    return k[:, :, idx, :], v[:, :, idx, :], idx


def kv_bytes(seq_len: int, layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """Logical K+V payload bytes (not measured device memory)."""
    return 2 * layers * kv_heads * seq_len * head_dim * dtype_bytes
