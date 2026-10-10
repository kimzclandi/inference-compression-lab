"""Dependency/position contracts without importing Metal or loading weights.

The small NumPy Qwen-shaped model is a mathematical fixture, not a claim that
different real MLX kernels produce bitwise-identical floating-point results.
"""
import copy
import types
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from lab import qwen_demand_prefill as demand


class TinyCache:
    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self.keys = keys if self.keys is None else np.concatenate((self.keys, keys), axis=2)
        self.values = values if self.values is None else np.concatenate((self.values, values), axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values


def tiny_attention(queries, keys, values, *, cache, scale, mask):
    repeats = queries.shape[1] // keys.shape[1]
    keys = np.repeat(keys, repeats, axis=1)
    values = np.repeat(values, repeats, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", queries, keys) * scale
    if mask is not None:
        qlen, klen = scores.shape[-2:]
        visible = np.arange(klen)[None, :] <= np.arange(klen - qlen, klen)[:, None]
        scores = np.where(visible, scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", probabilities, values)


def tiny_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)


class TinyProjection:
    def __init__(self, rng, width_in, width_out, events, name):
        self.weight = rng.normal(0, 0.1, (width_in, width_out))
        self.bias = rng.normal(0, 0.01, width_out)
        self.events, self.name = events, name

    def __call__(self, x):
        self.events.append((self.name, tuple(x.shape)))
        return np.einsum("...i,ij->...j", x, self.weight) + self.bias


class TinyAttention:
    def __init__(self, rng, events, name):
        self.n_heads, self.n_kv_heads = 2, 1
        self.scale = 4 ** -0.5
        self.q_proj = TinyProjection(rng, 8, 8, events, name + ".q")
        self.k_proj = TinyProjection(rng, 8, 4, events, name + ".k")
        self.v_proj = TinyProjection(rng, 8, 4, events, name + ".v")
        self.o_proj = TinyProjection(rng, 8, 8, events, name + ".o")

    @staticmethod
    def rope(x, offset):
        position = np.arange(offset, offset + x.shape[2])[:, None]
        phase = position * np.array([0.1, 0.03])[None, :]
        result = np.empty_like(x)
        result[..., :2] = x[..., :2] * np.cos(phase) - x[..., 2:] * np.sin(phase)
        result[..., 2:] = x[..., :2] * np.sin(phase) + x[..., 2:] * np.cos(phase)
        return result

    def __call__(self, x, mask, cache):
        length = x.shape[1]
        q = self.q_proj(x).reshape(1, length, 2, 4).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(1, length, 1, 4).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(1, length, 1, 4).transpose(0, 2, 1, 3)
        q, k = self.rope(q, cache.offset), self.rope(k, cache.offset)
        k, v = cache.update_and_fetch(k, v)
        result = tiny_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(result.transpose(0, 2, 1, 3).reshape(1, length, 8))


class TinyBlock:
    input_layernorm = staticmethod(tiny_norm)
    post_attention_layernorm = staticmethod(tiny_norm)

    def __init__(self, rng, events, index):
        self.self_attn = TinyAttention(rng, events, str(index))
        self.feedforward = TinyProjection(rng, 8, 8, events, str(index) + ".mlp")

    def mlp(self, x):
        return np.tanh(self.feedforward(x))

    def __call__(self, x, mask, cache):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyEmbedding:
    def __init__(self, rng, events):
        self.weight = rng.normal(0, 1, (19, 8))
        self.events = events

    def __call__(self, x):
        return self.weight[x]

    def as_linear(self, x):
        self.events.append(("head", tuple(x.shape)))
        return np.einsum("...d,vd->...v", x, self.weight)


class TinyDecoder:
    norm = staticmethod(tiny_norm)

    def __init__(self, rng, events):
        self.embed_tokens = TinyEmbedding(rng, events)
        self.layers = [TinyBlock(rng, events, i) for i in range(3)]

    def __call__(self, x, cache):
        x = self.embed_tokens(x)
        for layer, c in zip(self.layers, cache):
            x = layer(x, "causal", c)
        return self.norm(x)


class TinyModel:
    model_type = "qwen2"

    def __init__(self):
        self.args = types.SimpleNamespace(tie_word_embeddings=True, num_hidden_layers=3,
                                         hidden_size=8, num_attention_heads=2,
                                         num_key_value_heads=1)
        self.events = []
        self.model = TinyDecoder(np.random.default_rng(932), self.events)
        self.calls = []

    def __call__(self, inputs, cache):
        self.calls.append(tuple(inputs.shape))
        return self.model.embed_tokens.as_linear(self.model(inputs, cache))


@unittest.skipIf(np is None, "NumPy optional numerical fixture unavailable")
class DemandPrefillTests(unittest.TestCase):
    def setUp(self):
        self.evaluated = []
        backend = types.SimpleNamespace(array=np.ndarray, int32=np.dtype("int32"),
                                        int64=np.dtype("int64"), uint32=np.dtype("uint32"),
                                        issubdtype=np.issubdtype, floating=np.floating,
                                        eval=lambda state: self.evaluated.append(state))
        self.patch = patch.object(demand, "_backend", return_value=(
            backend, TinyModel, TinyCache, lambda h, cache: "causal", tiny_attention))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    @staticmethod
    def tokens(length):
        return (np.arange(length, dtype=np.int32) % 19)[None, :]

    @staticmethod
    def caches():
        return [TinyCache() for _ in range(3)]

    def test_all_arms_last_logits_and_every_kv_with_prior_cache_and_continuation(self):
        for prior in (0, 5):
            for length in (1, 2, 7, 32):
                with self.subTest(prior=prior, length=length):
                    seed_model = TinyModel()
                    seed_cache = self.caches()
                    if prior:
                        seed_model(self.tokens(prior), cache=seed_cache)
                    baseline_cache = copy.deepcopy(seed_cache)
                    reference = demand.prefill(seed_model, self.tokens(length), baseline_cache, "full")
                    for arm in demand.MODES:
                        model, cache = TinyModel(), copy.deepcopy(seed_cache)
                        actual = demand.prefill(model, self.tokens(length), cache, arm)
                        self.assertTrue(np.isfinite(actual).all())
                        self.assertTrue(np.isfinite(reference).all())
                        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
                        self.assertEqual(actual.shape, (1, 1, 19))
                        for left, right in zip(cache, baseline_cache):
                            self.assertEqual(left.offset, prior + length)
                            np.testing.assert_allclose(left.keys, right.keys, rtol=1e-12, atol=1e-12)
                            np.testing.assert_allclose(left.values, right.values, rtol=1e-12, atol=1e-12)
                        continued = demand.prefill(model, self.tokens(1), cache, arm)
                        expected = demand.prefill(seed_model, self.tokens(1), copy.deepcopy(baseline_cache), "full")
                        np.testing.assert_allclose(continued, expected, rtol=1e-12, atol=1e-12)

    def test_final_query_shapes_and_absolute_rope_offsets(self):
        model, cache, trace = TinyModel(), self.caches(), {}
        model(self.tokens(5), cache=cache)
        model.events.clear()
        demand.prefill(model, self.tokens(7), cache, "final_query", trace=trace)
        for prefix in ("0", "1"):
            self.assertIn((prefix + ".q", (1, 7, 8)), model.events)
        for name in ("2.q", "2.o", "2.mlp", "head"):
            self.assertIn((name, (1, 1, 8)), model.events)
        for name in ("2.k", "2.v"):
            self.assertIn((name, (1, 7, 8)), model.events)
        self.assertEqual(trace["shapes"]["query_rope_offset"], 11)
        self.assertEqual(trace["shapes"]["key_rope_offset"], 5)
        self.assertEqual(trace["cache_offsets_after"], [12, 12, 12])

    def test_head_only_keeps_complete_final_block(self):
        model = TinyModel()
        demand.prefill(model, self.tokens(7), self.caches(), "head_only")
        self.assertIn(("2.mlp", (1, 7, 8)), model.events)
        self.assertIn(("head", (1, 1, 8)), model.events)

    def test_split_last_uses_two_calls_and_evaluates_only_kv(self):
        model = TinyModel()
        cache = self.caches()
        demand.prefill(model, self.tokens(7), cache, "split_last")
        self.assertEqual(model.calls, [(1, 6), (1, 1)])
        self.assertEqual(len(self.evaluated), 1)
        self.assertEqual(len(self.evaluated[0]), 3)
        self.assertEqual(self.evaluated[0][0][0].shape, (1, 1, 6, 4))

    def test_single_token_always_uses_standard_model(self):
        for mode in demand.MODES:
            model = TinyModel()
            demand.prefill(model, self.tokens(1), self.caches(), mode)
            self.assertEqual(model.calls, [(1, 1)])
        self.assertEqual(self.evaluated, [])

    def test_invalid_inputs_fail_before_mutation(self):
        malformed = [np.ones((2, 3), dtype=np.int32), np.ones((1, 0), dtype=np.int32),
                     np.ones(3, dtype=np.int32), np.ones((1, 3), dtype=np.float32), [[1, 2]]]
        for inputs in malformed:
            model, cache = TinyModel(), self.caches()
            with self.assertRaises(ValueError):
                demand.prefill(model, inputs, cache, "final_query")
            self.assertEqual(model.events, [])
            self.assertEqual([c.offset for c in cache], [0, 0, 0])

    def test_unknown_mode_external_mask_and_embeddings_rejected(self):
        for kwargs in ({"mode": "typo"}, {"mask": "causal"}, {"input_embeddings": object()}, {"trace": []}):
            with self.assertRaises(ValueError):
                demand.prefill(TinyModel(), self.tokens(2), self.caches(), **kwargs)

    def test_unsupported_model_and_cache_rejected(self):
        class DerivedModel(TinyModel):
            pass
        class DerivedCache(TinyCache):
            pass
        invalid = [(DerivedModel(), self.caches()), (TinyModel(), self.caches()[:-1]),
                   (TinyModel(), tuple(self.caches())), (TinyModel(), [DerivedCache() for _ in range(3)]),
                   (TinyModel(), [TinyCache()] * 3)]
        for model, cache in invalid:
            with self.assertRaises(ValueError):
                demand.prefill(model, self.tokens(2), cache)
        model = TinyModel()
        model.args.tie_word_embeddings = False
        with self.assertRaises(ValueError):
            demand.prefill(model, self.tokens(2), self.caches())
        model = TinyModel()
        model.args.sliding_window = 64
        with self.assertRaises(ValueError):
            demand.prefill(model, self.tokens(2), self.caches())

    def test_inconsistent_cache_state_rejected_before_mutation(self):
        for damage in ("offset", "missing", "shape", "dtype", "integer"):
            model, cache = TinyModel(), self.caches()
            model(self.tokens(3), cache=cache)
            model.events.clear()
            if damage == "offset":
                cache[0].offset = 4
            elif damage == "missing":
                cache[0].keys = None
            elif damage == "shape":
                cache[0].values = cache[0].values[:, :, :, :2]
            elif damage == "dtype":
                cache[0].values = cache[0].values.astype(np.float32)
            else:
                cache[0].keys = cache[0].keys.astype(np.int32)
                cache[0].values = cache[0].values.astype(np.int32)
            with self.assertRaises(ValueError):
                demand.prefill(model, self.tokens(2), cache, "final_query")
            self.assertEqual(model.events, [])


if __name__ == "__main__":
    unittest.main()
