"""Frozen StreamingLLM-style KV eviction study on Qwen2.5-0.5B Q8 (Mac/MLX only).

Order: provenance -> correctness gate (no-eviction parity) -> quality (74 rows x 3
conditions) -> timing -> summary with the frozen adoption gate. Any failure stops
the run and is recorded in run.json; nothing is retried or retuned.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback

from lab.kv_eviction import StreamingPolicy
from lab.long_context_qa import build_prompt, passage_pool
from lab.qa_metrics import score

SPEC = Path('configs/kv-eviction-streaming-v1.json')
SOURCES = [str(SPEC), 'lab/kv_eviction.py', 'lab/mlx_kv_eviction.py', 'lab/long_context_qa.py',
           'lab/qa_metrics.py', 'experiments/kv_eviction_streaming.py']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def policy_for(cond, n):
    if cond['name'] == 'full_cache':
        return None
    return StreamingPolicy.from_ratio(n, cond['budget_ratio'], sink=cond['sink'])


def summarize(spec, quality, timing):
    by = {}
    for r in quality:
        by.setdefault(r['condition'], []).append(r)
    def em(rows, pred=lambda r: True):
        sel = [r['em'] for r in rows if pred(r)]
        return 100.0 * sum(sel) / len(sel)
    full = by['full_cache']
    base = dict(overall=em(full), answerable=em(full, lambda r: not r['is_impossible']),
                impossible=em(full, lambda r: r['is_impossible']))
    gate, out = spec['adoption_gate'], {'full_cache': base}
    for cond in spec['conditions'][1:]:
        rows = by[cond['name']]
        res = dict(overall=em(rows), answerable=em(rows, lambda r: not r['is_impossible']),
                   impossible=em(rows, lambda r: r['is_impossible']),
                   answerable_em_target_last=em(rows, lambda r: not r['is_impossible'] and r['target_is_last_passage'])
                   if any(not r['is_impossible'] and r['target_is_last_passage'] for r in rows) else None)
        res['kv_reduction'] = 1 - sum(r['kv_bytes'] for r in rows) / sum(r['kv_bytes'] for r in full)
        ratios = []
        for rnd in range(spec['timing']['rounds']):
            f = statistics.median(t['tok_s'] for t in timing if t['round'] == rnd and t['condition'] == 'full_cache')
            c = statistics.median(t['tok_s'] for t in timing if t['round'] == rnd and t['condition'] == cond['name'])
            ratios.append(c / f)
        res['decode_ratio_by_round'] = ratios
        res['decode_ratio_median'] = statistics.median(ratios)
        checks = dict(
            answerable=base['answerable'] - res['answerable'] <= gate['answerable_em_drop_max_points'],
            overall=base['overall'] - res['overall'] <= gate['overall_em_drop_max_points'],
            kv=res['kv_reduction'] >= gate['logical_kv_reduction_min'],
            speed=res['decode_ratio_median'] >= gate['decode_tokens_per_s_ratio_min']
            and sum(x >= gate['decode_tokens_per_s_ratio_min'] for x in ratios) >= gate['decode_faster_rounds_min'])
        res['gate_checks'] = checks
        res['adopt'] = all(checks.values())
        out[cond['name']] = res
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', protocol_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                source_sha256={}, artifacts={})
    for name in SOURCES:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        info['source_sha256'][name] = digest(target)
    save(args.output / 'run.json', info)
    try:
        assert not info['dirty'], 'working tree must be clean so protocol_commit identifies the code'
        import mlx.core as mx
        import numpy as np
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from lab.mlx_kv_eviction import evict_caches
        for pkg, key in [('mlx', 'mlx'), ('mlx-lm', 'mlx_lm')]:
            assert importlib.metadata.version(pkg) == spec[key], (pkg, 'version changed')
        assert all(digest(args.model / n) == h for n, h in spec['model_files_sha256'].items()), 'model changed'
        assert mx.default_device() == mx.gpu, 'GPU required'
        info['device'] = str(mx.default_device())
        info['system'] = subprocess.check_output(['sw_vers'], text=True)
        model, tok = load(str(args.model))
        mx.eval(model.parameters())
        system = json.loads(Path(spec['data']['system_prompt_source']).read_text())['system_prompt']
        rows = [json.loads(l) for l in Path(spec['data']['source']).read_text().splitlines()]
        assert len(rows) == spec['data']['rows']
        pool = passage_pool(rows)
        encode = lambda user: tok.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            tokenize=True, add_generation_prompt=True)
        prompts = {r['id']: build_prompt(r, pool, encode, seed=spec['seed'])
                   for r in rows}
        save(args.output / 'prompts.json', {k: dict(n_tokens=len(v['tokens']), target_slot=v['target_slot'],
                                                    n_passages=v['n_passages']) for k, v in prompts.items()})
        eos = set(tok.eos_token_ids)

        def run(tokens, policy, max_new, stop_eos, keep_logits=False):
            caches = make_prompt_cache(model)
            y = model(mx.array([tokens]), cache=caches)[:, -1, :]
            mx.eval(y)
            t0 = time.perf_counter()
            if policy is not None:
                caches = evict_caches(caches, policy)
            evict_s = time.perf_counter() - t0
            kv = sum(c.keys[..., :c.offset, :].nbytes + c.values[..., :c.offset, :].nbytes for c in caches) \
                if policy is None else sum(c.nbytes for c in caches)
            out, logits = [], []
            t1 = time.perf_counter()
            for _ in range(max_new):
                t = mx.argmax(y, axis=-1)
                mx.eval(t)
                tid = int(t.item())
                if keep_logits:
                    logits.append(np.array(y).copy())
                out.append(tid)
                if stop_eos and tid in eos:
                    break
                y = model(t[:, None], cache=caches)[:, -1, :]
            mx.eval(y)
            mx.synchronize()
            return dict(tokens=out, kv_bytes=int(kv), evict_s=evict_s, decode_s=time.perf_counter() - t1), logits

        # 1. Correctness gate: no-eviction EvictedKVCache path must equal native decode.
        checks = []
        for r in rows[:spec['correctness_gate']['rows']]:
            toks = prompts[r['id']]['tokens']
            nat, nl = run(toks, None, spec['decode']['max_new_tokens'], True, True)
            noop, el = run(toks, StreamingPolicy(sink=0, window=len(toks)), spec['decode']['max_new_tokens'], True, True)
            ok_tokens = nat['tokens'] == noop['tokens']
            ok_logits = len(nl) == len(el) and all(np.allclose(a, b, atol=spec['correctness_gate']['atol'],
                                                               rtol=spec['correctness_gate']['rtol']) for a, b in zip(nl, el))
            checks.append(dict(id=r['id'], tokens_equal=ok_tokens, logits_allclose=bool(ok_logits)))
            save(args.output / 'correctness.json', checks)
            assert ok_tokens and ok_logits, 'no-eviction parity failed; stopping before quality/timing'

        # 2. Quality.
        quality = []
        for r in rows:
            p = prompts[r['id']]
            n = len(p['tokens'])
            for cond in spec['conditions']:
                pol = policy_for(cond, n)
                res, _ = run(p['tokens'], pol, spec['decode']['max_new_tokens'], spec['decode']['stop_at_eos'])
                text = tok.decode([t for t in res['tokens'] if t not in eos], skip_special_tokens=True)
                s = score(r, {'prediction': text})
                quality.append(dict(id=r['id'], condition=cond['name'], is_impossible=r['is_impossible'],
                                    prediction=text, em=s['em'], category=s['category'],
                                    kv_bytes=res['kv_bytes'], n_prompt=n, target_slot=p['target_slot'],
                                    n_passages=p['n_passages'],
                                    target_is_last_passage=p['target_slot'] == p['n_passages'] - 1))
            save(args.output / 'quality.json', quality)
        print('quality complete', flush=True)

        # 3. Timing (decode only, forced length, shuffled arms).
        rng = random.Random(spec['seed'])
        tspec = spec['timing']
        timing = []
        for r in rows[:tspec['rows']]:
            toks = prompts[r['id']]['tokens']
            for cond in spec['conditions']:
                for _ in range(tspec['warmups']):
                    run(toks, policy_for(cond, len(toks)), tspec['forced_decode_tokens'], False)
            for rnd in range(tspec['rounds']):
                for rep in range(tspec['repeats']):
                    conds = list(spec['conditions'])
                    rng.shuffle(conds)
                    for order, cond in enumerate(conds):
                        res, _ = run(toks, policy_for(cond, len(toks)), tspec['forced_decode_tokens'], False)
                        timing.append(dict(id=r['id'], condition=cond['name'], round=rnd, repeat=rep, order=order,
                                           decode_s=res['decode_s'], evict_s=res['evict_s'],
                                           tok_s=tspec['forced_decode_tokens'] / res['decode_s']))
                        save(args.output / 'timing.json', timing)
        save(args.output / 'summary.json', summarize(spec, quality, timing))
        info['status'] = 'complete'
    except BaseException as exc:
        info['status'] = 'failed'
        info['error'] = type(exc).__name__ + ': ' + str(exc)
        traceback.print_exc()
        raise
    finally:
        for name in ['prompts.json', 'correctness.json', 'quality.json', 'timing.json', 'summary.json']:
            if (args.output / name).exists():
                info['artifacts'][name] = digest(args.output / name)
        save(args.output / 'run.json', info)


if __name__ == '__main__':
    main()
