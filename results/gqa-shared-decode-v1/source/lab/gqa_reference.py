"""Independent float64 GQA math, explicitly avoiding NumPy BLAS dispatch."""


def dense_reference(q, k, v):
    import numpy as np
    q = np.asarray(q, dtype=np.float64)[0, :, 0, :]
    keys = np.asarray(k, dtype=np.float64)[0, np.arange(14)//7]
    values = np.asarray(v, dtype=np.float64)[0, np.arange(14)//7]
    scores = np.einsum('hd,hnd->hn', q, keys, optimize=False) * .125
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum('hn,hnd->hd', weights, values, optimize=False)[None, :, None, :]


def partitioned_reference(q, k, v, partition=128):
    import numpy as np
    q = np.asarray(q, dtype=np.float64)[0, :, 0, :]
    keys = np.asarray(k, dtype=np.float64)[0, np.arange(14)//7]
    values = np.asarray(v, dtype=np.float64)[0, np.arange(14)//7]
    maximum = np.full((14, 1), -np.inf)
    denominator = np.zeros((14, 1))
    numerator = np.zeros((14, 64))
    for start in range(0, keys.shape[1], partition):
        scores = np.einsum('hd,hnd->hn', q, keys[:, start:start+partition], optimize=False) * .125
        local_max = scores.max(axis=-1, keepdims=True)
        merged_max = np.maximum(maximum, local_max)
        weights = np.exp(scores-merged_max)
        factor = np.exp(maximum-merged_max)
        denominator = denominator*factor + weights.sum(axis=-1, keepdims=True)
        numerator = numerator*factor + np.einsum('hn,hnd->hd', weights,
            values[:, start:start+partition], optimize=False)
        maximum = merged_max
    return (numerator/denominator)[None, :, None, :]
