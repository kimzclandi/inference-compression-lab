"""Backend-independent cleanup metadata; preserve an already active error."""


def collect_after_sync(cache, synchronize, original_error=None):
    """Synchronize then inspect possibly unallocated caches.

    A secondary synchronization/inspection failure must not replace a request's
    original exception. Without an original error, cleanup failure is raised.
    This does not recover a failed device or promise async cancellation.
    """
    result = dict(allocated_kv_bytes=0, layer_offsets=[], unallocated_layers=0,
                  cleanup_error_types=[])
    try:
        synchronize()
        for layer in cache:
            keys, values = layer.keys, layer.values
            if keys is None or values is None:
                result['unallocated_layers'] += 1
            result['allocated_kv_bytes'] += sum(x.nbytes for x in (keys, values) if x is not None)
            result['layer_offsets'].append(int(layer.offset))
    except BaseException as error:
        if original_error is None:
            raise
        result['cleanup_error_types'].append(type(error).__name__)
        result['allocated_kv_bytes'] = None
    return result
