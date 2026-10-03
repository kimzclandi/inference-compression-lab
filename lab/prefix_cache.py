"""Exact-token, model-bound LRU prefix cache; single caller, no backend dependency."""
from collections import OrderedDict
import hashlib
import json
import time


def token_key(model_id, tokens):
    if not model_id or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Require a model identity and nonempty nonnegative integer token IDs')
    return hashlib.sha256(json.dumps([model_id, list(tokens)], separators=(',', ':')).encode()).hexdigest()


class PrefixCache:
    """Builder returns (immutable snapshot, logical tensor bytes); clone isolates users.

    The byte limit bounds stored tensor payload, not allocator/RSS/GPU process memory.
    The caller must bind model_id to weights, tokenizer and cache configuration.
    """
    def __init__(self, model_id, clone, *, max_entries=2, max_bytes=64*1024*1024):
        if not model_id or max_entries < 1 or max_bytes < 1:
            raise ValueError('A model identity and positive cache limits are required')
        self.model_id = model_id
        self.clone = clone
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0

    def acquire(self, tokens, builder, *, clone=None, profile=None):
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
            self.entries.move_to_end(key)
            self.hits += 1
            return clone(snapshot), 'hit'
        self.misses += 1
        snapshot, size = builder(tokens)
        if type(size) is not int or size < 0:
            raise ValueError('Invalid snapshot byte count')
        if size > self.max_bytes:
            self.bypasses += 1
            return clone(snapshot), 'bypass'
        while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (tokens, snapshot, size)
        self.bytes += size
        return clone(snapshot), 'miss'

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return {'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'bypasses': self.bypasses, 'entries': len(self.entries),
                'stored_tensor_bytes': self.bytes}
