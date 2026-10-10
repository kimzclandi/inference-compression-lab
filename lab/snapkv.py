"""SnapKV-style KV selection (Li et al., 2024), scored in NumPy.

The last `window` prompt tokens (they contain the question) act as an
observation window. Their attention over earlier positions, summed over the
window and over the query heads of each KV group, is max-pooled and the top
positions are kept per KV head; the window itself is always kept.

Scoring runs on CPU/NumPy so the exact selection code is unit-tested without
MLX. Shapes:
  q: [n_heads, window, d]   rotated queries of the window
  k: [n_kv_heads, n, d]     rotated keys of the whole prompt
Returns kept positions [n_kv_heads, budget], sorted ascending per head.
"""
from __future__ import annotations

import numpy as np


def window_scores(q, k, scale):
    n_heads, w, d = q.shape
    kvh, n, _ = k.shape
    if n_heads % kvh or n < w:
        raise ValueError('bad head grouping or window longer than prompt')
    g = n_heads // kvh
    qg = q.reshape(kvh, g, w, d).astype(np.float64)
    logits = np.einsum('hgwd,hnd->hgwn', qg, k.astype(np.float64)) * scale
    # Window query j sits at absolute position n - w + j; it may not see later keys.
    pos = np.arange(n)[None, :]
    qpos = (n - w + np.arange(w))[:, None]
    logits = np.where(pos <= qpos, logits, -np.inf)
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs.sum(axis=(1, 2))  # [kvh, n]


def max_pool_1d(x, kernel):
    if kernel <= 1:
        return x
    r = kernel // 2
    padded = np.pad(x, ((0, 0), (r, r)), constant_values=-np.inf)
    return np.max(np.stack([padded[:, i:i + x.shape[1]] for i in range(kernel)]), axis=0)


def select(q, k, *, budget, window, pool, scale):
    kvh, n, _ = k.shape
    if budget >= n:
        return np.tile(np.arange(n), (kvh, 1))
    if budget <= window:
        raise ValueError('budget must exceed the observation window')
    scores = max_pool_1d(window_scores(q, k, scale)[:, :n - window], pool)
    keep = budget - window
    # Stable tie-break toward earlier positions keeps selection deterministic.
    top = np.argsort(-scores, axis=1, kind='stable')[:, :keep]
    prefix = np.sort(top, axis=1)
    tail = np.tile(np.arange(n - window, n), (kvh, 1))
    return np.concatenate([prefix, tail], axis=1)
