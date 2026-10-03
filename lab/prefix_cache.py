"""Exact-token, model-bound LRU prefix cache; single caller, no backend dependency."""
from collections import OrderedDict
import hashlib
import json
import time


def token_key(model_id, tokens):
    if not isinstance(model_id, str) or not model_id or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Require a model identity and nonempty nonnegative integer token IDs')
    return hashlib.sha256(json.dumps([model_id, list(tokens)], separators=(',', ':')).encode()).hexdigest()


class PrefixCache:
    """Builder returns an owned immutable snapshot and logical tensor bytes.

    Clone must not mutate its input (including before raising). Callbacks must not
    reenter this store. Single caller only. Builder/clone failure preserves resident
    entries, LRU and successful-operation counters; failures increments instead.
    The byte limit bounds resident payload, NOT temporary clones/allocator/RSS.
    Model identity is immutable; weights/tokenizer/config must also remain fixed.
    """
    def __init__(self, model_id, clone, *, max_entries=2, max_bytes=64*1024*1024):
        if (not isinstance(model_id, str) or not model_id or
                type(max_entries) is not int or max_entries < 1 or
                type(max_bytes) is not int or max_bytes < 1):
            raise ValueError('A string model identity and positive integer cache limits are required')
        self._model_id = model_id
        self.clone = clone
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = self.failures = 0

    @property
    def model_id(self):
        return self._model_id

    def acquire(self, tokens, builder, *, clone=None, profile=None):
        try:
            return self._acquire(tokens, builder, clone=clone, profile=profile)
        except Exception:
            self.failures += 1
            raise

    def _acquire(self, tokens, builder, *, clone, profile):
        clone = self.clone if clone is None else clone
        started = time.perf_counter() if profile is not None else None
        tokens = tuple(tokens)
        key = token_key(self.model_id, tokens)
        found = key in self.entries
        if profile is not None:
            profile['lookup_seconds'] = time.perf_counter() - started
        if found:
            stored_tokens, snapshot, size = self.entries[key]
            if stored_tokens != tokens:
                raise RuntimeError('Cache-key collision')
            request = clone(snapshot)
            self.entries.move_to_end(key)
            self.hits += 1
            return request, 'hit'
        snapshot, size = builder(tokens)
        if type(size) is not int or size < 0:
            raise ValueError('Invalid snapshot byte count')
        request = clone(snapshot)  # Finish fallible callback before any cache commit.
        if size > self.max_bytes:
            self.misses += 1
            self.bypasses += 1
            return request, 'bypass'
        # Stage metadata using references, not copies of the tensor snapshots.
        entries = self.entries.copy()
        stored_bytes, evictions = self.bytes, 0
        while entries and (len(entries) >= self.max_entries or stored_bytes + size > self.max_bytes):
            _, (_, _, removed) = entries.popitem(last=False)
            stored_bytes -= removed
            evictions += 1
        entries[key] = (tokens, snapshot, size)
        self.entries, self.bytes = entries, stored_bytes + size
        self.evictions += evictions
        self.misses += 1
        return request, 'miss'

    def clear(self):
        """Drop resident entries; preserve cumulative success/failure counters."""
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return {'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'bypasses': self.bypasses, 'failures': self.failures,
                'entries': len(self.entries), 'stored_tensor_bytes': self.bytes}
