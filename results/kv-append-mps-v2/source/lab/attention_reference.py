"""Independent NumPy oracle for contiguous-prefix causal attention (B,H,L,D)."""
import numpy as np


def prefix_mask(query_length, key_length):
    """Queries occupy the final L positions of K; True means visible."""
    if not 0 < query_length <= key_length:
        raise ValueError('Require 0 < query length <= key length')
    return np.arange(key_length)[None, :] <= (
        key_length - query_length + np.arange(query_length)[:, None])


def attention(q, k, v):
    """Float64 oracle, same head count, no padding/dropout/GQA; return B,H,L,Dv."""
    q, k, v = [np.asarray(x, dtype=np.float64) for x in (q, k, v)]
    if any(x.ndim != 4 or not np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('Finite rank-four inputs required')
    if (q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1] or min(*q.shape, *k.shape, *v.shape) < 1):
        raise ValueError('Incompatible shapes')
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    scores = np.where(prefix_mask(q.shape[-2], k.shape[-2]), scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v
