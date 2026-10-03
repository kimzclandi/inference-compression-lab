"""Bounded offline length bucketing; indices always refer to original requests."""


def batch_indices(lengths, batch_size, window_size, bucket):
    if batch_size < 1 or window_size < batch_size:
        raise ValueError('Require window_size >= batch_size >= 1')
    for start in range(0, len(lengths), window_size):
        indices = list(range(start, min(start + window_size, len(lengths))))
        if bucket:
            indices.sort(key=lambda i: (lengths[i], i))
        for offset in range(0, len(indices), batch_size):
            yield indices[offset:offset + batch_size]
