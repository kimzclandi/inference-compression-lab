"""Request-sized allocation granularity for fresh native MLX KVCache objects.

This is a framework configuration adapter, not a new cache/kernel implementation.
It does not enforce a hard capacity or support reuse of an already populated cache.
"""


def reserve_fresh_caches(caches, capacity, cache_type):
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('capacity must be a positive integer')
    if not caches:
        raise ValueError('at least one native cache is required')
    for cache in caches:
        if (type(cache) is not cache_type or cache.offset != 0
                or cache.keys is not None or cache.values is not None
                or cache.step != 256):
            raise ValueError('expected fresh native KVCache with step=256')
    for cache in caches:
        cache.step = capacity
    return caches
