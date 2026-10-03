"""Synchronized serial Qwen2 MLX decoding with optional exact prefix reuse.

This is an offline experimental runtime, not an HTTP or concurrent serving engine.
"""
import copy
import time
from lab.prefix_cache import PrefixCache


class QwenPrefixRuntime:
    def __init__(self, model, model_id, *, max_entries=2, max_bytes=64*1024*1024):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache, KVCache
        self.mx = mx
        self.model = model
        self.make_cache = lambda: make_prompt_cache(model)
        self.cache_class = KVCache
        if not all(type(c) is KVCache for c in self.make_cache()):
            raise ValueError('Only ordinary floating KVCache is supported')
        self.store = PrefixCache(model_id, self.clone, max_entries=max_entries, max_bytes=max_bytes)

    def clone(self, snapshot):
        cache = []
        for state in snapshot:
            layer = self.cache_class()
            layer.state = tuple(copy.deepcopy(a) for a in state)
            cache.append(layer)
        self.mx.eval([c.state for c in cache])
        return cache

    def build(self, tokens):
        cache = self.make_cache()
        self.model(self.mx.array([tokens]), cache=cache)
        snapshot = tuple(tuple(copy.deepcopy(a) for a in c.state) for c in cache)
        self.mx.eval(snapshot)
        return snapshot, sum(int(a.nbytes) for state in snapshot for a in state)

    def prepare(self, prefix):
        self.mx.synchronize()
        start = time.perf_counter()
        cache, status = self.store.acquire(prefix, self.build)
        self.mx.synchronize()
        return {'seconds': time.perf_counter()-start, 'status': status,
                'prefix_tokens': len(prefix), 'cache': self.store.stats()}

    def generate(self, tokens, *, prefix=(), max_new_tokens=32, eos_ids=(), stop_at_eos=True):
        if not tokens or max_new_tokens < 1:
            raise ValueError('Require nonempty prompt and positive output limit')
        if len(prefix) >= len(tokens) or list(tokens[:len(prefix)]) != list(prefix):
            raise ValueError('Prefix must be an exact token prefix and leave at least one suffix token')
        mx = self.mx
        mx.synchronize()
        start = time.perf_counter()
        if prefix:
            cache, status = self.store.acquire(prefix, self.build)
        else:
            cache, status = self.make_cache(), 'disabled'
        x = mx.array([tokens[len(prefix):]])
        output, times = [], []
        for _ in range(max_new_tokens):
            logits = self.model(x, cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(nxt)
            mx.synchronize()
            token = int(nxt.item())
            times.append(time.perf_counter())
            output.append(token)
            if stop_at_eos and token in eos_ids:
                break
            x = nxt.reshape(1, 1)
        total = times[-1]-start
        decode = times[-1]-times[0]
        return {'token_ids': output, 'input_tokens': len(tokens), 'prefix_tokens': len(prefix),
                'suffix_tokens': len(tokens)-len(prefix), 'status': status,
                'ttft_seconds': times[0]-start, 'total_seconds': total,
                'decode_seconds': decode,
                'decode_tokens_per_second': (len(output)-1)/decode if len(output)>1 else None,
                'stop_reason': 'eos' if stop_at_eos and output[-1] in eos_ids else 'token_limit',
                'cache': self.store.stats()}
