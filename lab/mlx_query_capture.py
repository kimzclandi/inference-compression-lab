"""Capture rotated queries of the last `window` prompt tokens during prefill.

Wraps mlx_lm.models.qwen2.Attention.__call__ (mlx-lm 0.26.3). The wrapper
recomputes q_proj + RoPE for the window only, exactly as the upstream module
does, and then calls the unmodified original, so model outputs are unchanged
(the study checks this before use).
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def capture_window_queries(window):
    from mlx_lm.models import qwen2
    original = qwen2.Attention.__call__
    store = []

    def patched(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        if L > 1:
            if L < window:
                raise ValueError('prompt shorter than observation window')
            q = self.q_proj(x[:, L - window:])
            q = q.reshape(B, window, self.n_heads, -1).transpose(0, 2, 1, 3)
            base = cache.offset if cache is not None else 0
            store.append(self.rope(q, offset=base + L - window))
        return original(self, x, mask=mask, cache=cache)

    qwen2.Attention.__call__ = patched
    try:
        yield store
    finally:
        qwen2.Attention.__call__ = original
