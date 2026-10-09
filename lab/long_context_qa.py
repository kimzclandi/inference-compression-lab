"""Deterministic long-context QA prompts for KV-eviction studies (CPU-testable).

Every dev question is shown ALL distinct passages of the dev set: its own
passage at a seeded slot, the other passages in a seeded order. Using all
passages (instead of a token threshold) keeps prompt construction independent
of the tokenizer, so it can be frozen and tested without the model.
"""
from __future__ import annotations

import random


def passage_pool(rows):
    """One passage per context_id, sorted by context_id for determinism."""
    pool = {}
    for row in rows:
        pool.setdefault(row['context_id'], row['context'])
    return sorted(pool.items())


def build_user_message(passages, question):
    body = '\n\n'.join(f'Passage {i + 1}:\n{text}' for i, text in enumerate(passages))
    return f'{body}\n\nQuestion: {question}'


def arrange(row, pool, *, seed):
    """Return (passages, target_slot) for one row."""
    if row['context_id'] not in {cid for cid, _ in pool}:
        raise ValueError(f"target context {row['context_id']} missing from pool")
    rng = random.Random(f"{seed}:{row['id']}")
    others = [text for cid, text in pool if cid != row['context_id']]
    rng.shuffle(others)
    slot = rng.randint(0, len(others))
    return others[:slot] + [row['context']] + others[slot:], slot


def build_prompt(row, pool, encode, *, seed):
    """encode(user_message) -> token ids with the frozen chat template applied."""
    passages, slot = arrange(row, pool, seed=seed)
    tokens = encode(build_user_message(passages, row['question']))
    return dict(tokens=tokens, passages=passages, target_slot=slot, n_passages=len(passages))
