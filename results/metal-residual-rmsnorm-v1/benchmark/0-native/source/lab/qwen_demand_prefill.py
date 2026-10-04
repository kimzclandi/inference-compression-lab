"""Opt-in last-position prefill for the pinned MLX Qwen2 experiment.

This is inference graph pruning, not a new attention or quantization kernel.
All prompt positions remain in every layer's floating KV cache. Only the final
block's last query can affect the requested last-position logits. Shapes may
select different upstream kernels, so algebraic equivalence does not promise
bitwise floating-point or greedy-token equivalence.

Inputs are integer token IDs [1, N], N >= 1; output is logits [1, 1, V]. This
module deliberately supports only ordinary, unpadded causal Qwen2 inference,
with tied embeddings and an explicitly supplied list of ordinary KVCache.
The experiment runner additionally binds the model files and library versions.
"""

MODES = ("full", "head_only", "split_last", "final_query")


def _backend():
    # Keep imports lazy: offline evidence checks and ordinary CI need no Metal.
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen2 import Model

    return mx, Model, KVCache, create_attention_mask, scaled_dot_product_attention


def _validate(model, inputs, cache, mode, mask, input_embeddings, backend):
    mx, model_class, cache_class, _, _ = backend
    if mode not in MODES:
        raise ValueError("Unknown prefill mode")
    if mask is not None or input_embeddings is not None:
        raise ValueError("External mask and input embeddings are unsupported")
    if type(model) is not model_class or model.model_type != "qwen2":
        raise ValueError("Only the standard MLX Qwen2 Model is supported")
    args = model.args
    if not args.tie_word_embeddings or getattr(args, "sliding_window", None) is not None:
        raise ValueError("Require tied embeddings and full causal attention")
    if not isinstance(inputs, mx.array) or inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1:
        raise ValueError("Require nonempty unpadded token IDs with shape [1, N]")
    if inputs.dtype not in (mx.int32, mx.int64, mx.uint32):
        raise ValueError("Token IDs must have an integer dtype")
    layers = model.model.layers
    if not layers or len(layers) != args.num_hidden_layers:
        raise ValueError("Unexpected Qwen2 decoder layer count")
    if not isinstance(cache, list) or len(cache) != len(layers):
        raise ValueError("Require one explicit floating KVCache per decoder layer")
    if len({id(c) for c in cache}) != len(cache):
        raise ValueError("Each decoder layer must own a distinct KV cache")
    if any(type(c) is not cache_class for c in cache):
        raise ValueError("Quantized, rotating and custom caches are unsupported")
    offsets = [c.offset for c in cache]
    if any(type(n) is not int or n < 0 for n in offsets) or len(set(offsets)) != 1:
        raise ValueError("All layer cache offsets must be equal nonnegative integers")
    dimensions = (args.hidden_size, args.num_attention_heads, args.num_key_value_heads)
    if (any(type(n) is not int or n < 1 for n in dimensions)
            or args.hidden_size % args.num_attention_heads
            or args.num_attention_heads % args.num_key_value_heads):
        raise ValueError("Unexpected attention head dimensions")
    head_dim = args.hidden_size // args.num_attention_heads
    for c in cache:
        if c.keys is None or c.values is None:
            if c.keys is not None or c.values is not None or c.offset:
                raise ValueError("Incomplete floating KV cache state")
            continue
        for tensor in (c.keys, c.values):
            if (tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1, args.num_key_value_heads)
                    or tensor.shape[2] < c.offset or tensor.shape[3] != head_dim):
                raise ValueError("Unexpected floating KV cache tensor shape")
            if not mx.issubdtype(tensor.dtype, mx.floating):
                raise ValueError("Only floating cache tensors are supported")
        if c.keys.dtype != c.values.dtype:
            raise ValueError("Key/value cache dtypes differ")
    return offsets[0]


def prefill(model, inputs, cache, mode="full", *, mask=None, input_embeddings=None, trace=None):
    """Compute last logits, mutating request-owned KV exactly once per token.

    ``trace`` is optional shape evidence, not an execution profiler. No trace or
    extra device synchronization is added to ordinary timed calls, except the
    intentional KV-only evaluation defining the ``split_last`` control.
    """
    backend = _backend()
    mx, _, _, create_mask, attention = backend
    offset = _validate(model, inputs, cache, mode, mask, input_embeddings, backend)
    if trace is not None and not isinstance(trace, dict):
        raise ValueError("trace must be a dictionary or None")
    length = inputs.shape[1]
    shapes = {}

    # Decode has no unused positions; keep the upstream one-token path.
    if length == 1 or mode == "full":
        logits = model(inputs, cache=cache)[:, -1:, :]
    elif mode == "head_only":
        hidden = model.model(inputs, cache=cache)[:, -1:, :]
        shapes["head_input_shape"] = list(hidden.shape)
        logits = model.model.embed_tokens.as_linear(hidden)
    elif mode == "split_last":
        # The ignored output remains lazy. Evaluate only KV, avoiding an
        # artificial full-output evaluation absent from upstream generation.
        model(inputs[:, :-1], cache=cache)
        mx.eval([c.state for c in cache])
        logits = model(inputs[:, -1:], cache=cache)[:, -1:, :]
        shapes["model_call_input_shapes"] = [[1, length - 1], [1, 1]]
    else:
        decoder = model.model
        hidden = decoder.embed_tokens(inputs)
        causal_mask = create_mask(hidden, cache)
        for layer, layer_cache in zip(decoder.layers[:-1], cache[:-1]):
            hidden = layer(hidden, causal_mask, layer_cache)

        block = decoder.layers[-1]
        attn = block.self_attn
        last_cache = cache[-1]
        normalized = block.input_layernorm(hidden)
        query_input = normalized[:, -1:, :]
        queries = attn.q_proj(query_input)
        keys = attn.k_proj(normalized)
        values = attn.v_proj(normalized)
        queries = queries.reshape(1, 1, attn.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        # Queries retain their original absolute position, while K covers all
        # new positions beginning at the unchanged pre-call cache offset.
        queries = attn.rope(queries, offset=offset + length - 1)
        keys = attn.rope(keys, offset=offset)
        keys, values = last_cache.update_and_fetch(keys, values)
        # The last query may see every stored key in this unpadded causal scope.
        result = attention(queries, keys, values, cache=last_cache,
                           scale=attn.scale, mask=None)
        result = result.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        hidden = hidden[:, -1:, :] + attn.o_proj(result)
        hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
        hidden = decoder.norm(hidden)
        logits = decoder.embed_tokens.as_linear(hidden)
        shapes.update(query_input_shape=list(query_input.shape),
                      key_value_input_shape=list(normalized.shape),
                      attention_query_shape=list(queries.shape),
                      attention_key_shape=list(keys.shape),
                      head_input_shape=list(hidden.shape),
                      query_rope_offset=offset + length - 1,
                      key_rope_offset=offset)

    if trace is not None:
        trace.update(mode=mode, input_shape=list(inputs.shape),
                     prior_cache_tokens=offset, output_shape=list(logits.shape),
                     cache_offsets_after=[c.offset for c in cache],
                     shapes=shapes)
    return logits
