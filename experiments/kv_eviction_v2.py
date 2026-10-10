"""Frozen v2 KV eviction study: StreamingLLM vs SnapKV on 3-passage QA (Mac/MLX only).

Order: provenance -> correctness gates -> full_cache quality -> baseline gate ->
eviction quality -> timing -> summary. A failed gate stops the run and is recorded.
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

from lab import snapkv
from lab.kv_eviction import StreamingPolicy
from lab.long_context_qa import build_prompt, passage_pool
from lab.qa_metrics import score

SPEC = Path('configs/kv-eviction-v2.json')
SOURCES = [str(SPEC), 'lab/kv_eviction.py', 'lab/snapkv.py', 'lab/mlx_kv_eviction.py', 'lab/mlx_query_capture.py',
           'lab/long_context_qa.py', 'lab/qa_metrics.py', 'experiments/kv_eviction_v2.py']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def budget_for(cond, n):
    return max(cond.get('sink', 0) + 1, round(cond['budget_ratio'] * n))


def _em(rows, pred=lambda r: True):
    sel = [r['em'] for r in rows if pred(r)]
    return 100.0 * sum(sel) / len(sel)


def baseline(spec, quality):
    full = [r for r in quality if r['condition'] == 'full_cache']
    base = dict(overall=_em(full), answerable=_em(full, lambda r: not r['is_impossible']),
                impossible=_em(full, lambda r: r['is_impossible']))
    return {'full_cache': base,
            'baseline_invalid': base['answerable'] < spec['baseline_gate']['answerable_em_min']}


def summarize(spec, quality, timing):
    by = {}
    for r in quality:
        by.setdefault(r['condition'], []).append(r)

    em = _em
    full = by['full_cache']
    out = baseline(spec, quality)
    base = out['full_cache']
    if out['baseline_invalid']:
        return out
    gate = spec['adoption_gate']
    full_em = {r['id']: r['em'] for r in full}
    for cond in spec['conditions'][1:]:
        rows = by[cond['name']]
        res = dict(overall=em(rows), answerable=em(rows, lambda r: not r['is_impossible']),
                   impossible=em(rows, lambda r: r['is_impossible']),
                   lost=sum(1 for r in rows if full_em[r['id']] == 1 and r['em'] == 0),
                   gained=sum(1 for r in rows if full_em[r['id']] == 0 and r['em'] == 1))
        res['kv_reduction'] = 1 - sum(r['kv_bytes'] for r in rows) / sum(r['kv_bytes'] for r in full)
        ratios = []
        for rnd in range(spec['timing']['rounds']):
            f = statistics.median(t['tok_s'] for t in timing if t['round'] == rnd and t['condition'] == 'full_cache')
            c = statistics.median(t['tok_s'] for t in timing if t['round'] == rnd and t['condition'] == cond['name'])
            ratios.append(c / f)
        res['decode_ratio_by_round'] = ratios
        res['decode_ratio_median'] = statistics.median(ratios)
        res['evict_ms_median'] = 1000 * statistics.median(t['evict_s'] for t in timing if t['condition'] == cond['name'])
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
        from mlx_lm.models import qwen2
        from mlx_lm.models.cache import make_prompt_cache
        from lab.mlx_kv_eviction import evict_caches, evict_caches_per_head
        from lab.mlx_query_capture import capture_window_queries
        for pkg, key in [('mlx', 'mlx'), ('mlx-lm', 'mlx_lm')]:
            assert importlib.metadata.version(pkg) == spec[key], (pkg, 'version changed')
        assert all(digest(args.model / n) == h for n, h in spec['model_files_sha256'].items()), 'model changed'
        assert mx.default_device() == mx.gpu, 'GPU required'
        info['device'] = str(mx.default_device())
        info['system'] = subprocess.check_output(['sw_vers'], text=True)
        info['chip'] = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        model, tok = load(str(args.model))
        mx.eval(model.parameters())
        assert isinstance(model.model.layers[0].self_attn, qwen2.Attention), 'unexpected attention class'
        scale = model.model.layers[0].self_attn.scale
        W, pool = spec['snapkv']['window'], spec['snapkv']['pool']
        system = json.loads(Path(spec['data']['system_prompt_source']).read_text())['system_prompt']
        rows = [json.loads(l) for l in Path(spec['data']['source']).read_text().splitlines()]
        assert len(rows) == spec['data']['rows']
        pool_passages = passage_pool(rows)
        encode = lambda user: tok.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            tokenize=True, add_generation_prompt=True)
        prompts = {r['id']: build_prompt(r, pool_passages, encode, seed=spec['seed'],
                                         n_distractors=spec['data']['n_distractors']) for r in rows}
        save(args.output / 'prompts.json', {k: dict(n_tokens=len(v['tokens']), target_slot=v['target_slot'],
                                                    n_passages=v['n_passages']) for k, v in prompts.items()})
        eos = set(tok.eos_token_ids)

        def run(tokens, cond, max_new, stop_eos, keep_logits=False, capture=None):
            caches = make_prompt_cache(model)
            method = cond.get('method') if cond else None
            want_q = method == 'snapkv' or capture
            if want_q:
                with capture_window_queries(W) as qs:
                    y = model(mx.array([tokens]), cache=caches)[:, -1, :]
                    mx.eval(y)
            else:
                y = model(mx.array([tokens]), cache=caches)[:, -1, :]
                mx.eval(y)
            n = len(tokens)
            t0 = time.perf_counter()
            if method == 'stream':
                caches = evict_caches(caches, StreamingPolicy(sink=cond['sink'], window=budget_for(cond, n) - cond['sink']))
            elif method == 'snapkv':
                b = budget_for(cond, n)
                idx = [snapkv.select(np.array(q[0].astype(mx.float32)), np.array(c.keys[0, :, :n, :].astype(mx.float32)),
                                     budget=b, window=W, pool=pool, scale=scale) for q, c in zip(qs, caches)]
                caches = evict_caches_per_head(caches, idx)
            elif method == 'noop':
                caches = evict_caches(caches, StreamingPolicy(sink=0, window=n))
            evict_s = time.perf_counter() - t0
            kv = sum(c.keys[..., :c.offset, :].nbytes + c.values[..., :c.offset, :].nbytes for c in caches) \
                if method is None else sum(c.nbytes for c in caches)
            out, logits = [], []
            t1 = time.perf_counter()
            for _ in range(max_new):
                t = mx.argmax(y, axis=-1)
                mx.eval(t)
                tid = int(t.item())
                if keep_logits:
                    logits.append(np.array(y.astype(mx.float32)).copy())
                out.append(tid)
                if stop_eos and tid in eos:
                    break
                y = model(t[:, None], cache=caches)[:, -1, :]
            mx.eval(y)
            mx.synchronize()
            return dict(tokens=out, kv_bytes=int(kv), evict_s=evict_s, decode_s=time.perf_counter() - t1), logits

        g = spec['correctness_gate']
        same = lambda a, b: len(a) == len(b) and all(np.allclose(x, y, atol=g['atol'], rtol=g['rtol']) for x, y in zip(a, b))
        checks = []
        for r in rows[:g['rows']]:
            toks = prompts[r['id']]['tokens']
            nat, nl = run(toks, None, spec['decode']['max_new_tokens'], True, True)
            noop, el = run(toks, {'method': 'noop'}, spec['decode']['max_new_tokens'], True, True)
            cap, cl = run(toks, None, spec['decode']['max_new_tokens'], True, True, capture=True)
            row = dict(id=r['id'], noop_tokens_equal=nat['tokens'] == noop['tokens'], noop_logits_allclose=bool(same(nl, el)),
                       capture_tokens_equal=nat['tokens'] == cap['tokens'], capture_logits_allclose=bool(same(nl, cl)))
            checks.append(row)
            save(args.output / 'correctness.json', checks)
            assert all(row[k] for k in row if k != 'id'), 'correctness gate failed; stopping'

        quality = []

        def quality_pass(conds):
            for r in rows:
                p = prompts[r['id']]
                for cond in conds:
                    res, _ = run(p['tokens'], None if cond['name'] == 'full_cache' else cond,
                                 spec['decode']['max_new_tokens'], spec['decode']['stop_at_eos'])
                    text = tok.decode([t for t in res['tokens'] if t not in eos], skip_special_tokens=True)
                    s = score(r, {'prediction': text})
                    quality.append(dict(id=r['id'], condition=cond['name'], is_impossible=r['is_impossible'],
                                        prediction=text, em=s['em'], category=s['category'], kv_bytes=res['kv_bytes'],
                                        n_prompt=len(p['tokens']), target_slot=p['target_slot']))
                save(args.output / 'quality.json', quality)

        quality_pass(spec['conditions'][:1])
        base = baseline(spec, quality)
        if base['baseline_invalid']:
            save(args.output / 'summary.json', base)
            info['status'] = 'baseline_invalid'
            print('baseline invalid: full_cache answerable EM', base['full_cache']['answerable'], flush=True)
            return
        quality_pass(spec['conditions'][1:])
        print('quality complete', flush=True)

        rng = random.Random(spec['seed'])
        tspec = spec['timing']
        timing = []
        for r in rows[:tspec['rows']]:
            toks = prompts[r['id']]['tokens']
            for cond in spec['conditions']:
                for _ in range(tspec['warmups']):
                    run(toks, None if cond['name'] == 'full_cache' else cond, tspec['forced_decode_tokens'], False)
            for rnd in range(tspec['rounds']):
                for rep in range(tspec['repeats']):
                    conds = list(spec['conditions'])
                    rng.shuffle(conds)
                    for order, cond in enumerate(conds):
                        res, _ = run(toks, None if cond['name'] == 'full_cache' else cond, tspec['forced_decode_tokens'], False)
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
