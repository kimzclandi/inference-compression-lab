"""Real Qwen generation study: prefix reuse, cold costs, correctness and LRU limits."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from lab.evidence import reserve_directory, sha256, source_record
from lab.qwen_prefix import QwenPrefixRuntime


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def common_prefix(sequences):
    prefix = list(sequences[0])
    for seq in sequences[1:]:
        length = 0
        for a, b in zip(prefix, seq):
            if a != b: break
            length += 1
        prefix = prefix[:length]
    return prefix[:min(len(s) for s in sequences)-1]


def identity(path):
    files = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, fingerprint


def benchmark(rt, tok, spec, out):
    # Synthetic token streams exercise exact shapes, not real task quality.
    body = ('The archive records robot inspections, room numbers, maintenance dates, '
            'battery measurements and component replacements. ')*500
    base = tok.apply_chat_template([{'role': 'user', 'content': body}],
                                   tokenize=True, add_generation_prompt=True)
    workloads = []
    for length in spec['prefix_lengths']:
        prefix = base[:length]
        requests = []
        for index in range(spec['questions_per_prefix']):
            suffix = tok.encode(f'\nQuestion {index}: Describe the inspection process briefly.\nAnswer:',
                                add_special_tokens=False)
            requests.append(prefix+suffix)
        workloads.append({'prefix_tokens': length, 'prefix': prefix, 'requests': requests})
    save(out/'workloads.json', workloads)
    records, builds = [], []
    rng = random.Random(spec['seed'])
    for work in workloads:
        prefix = work['prefix']
        for mode in ['cold', 'cached_workload', 'warm_hit']:
            for _ in range(spec['warmup_per_mode']):
                rt.store.clear()
                if mode == 'warm_hit': rt.prepare(prefix)
                rt.generate(work['requests'][0], prefix=prefix if mode != 'cold' else (),
                            max_new_tokens=spec['generated_tokens'], stop_at_eos=False)
        for round_id in range(spec['rounds']):
            order = ['cold', 'cached_workload', 'warm_hit']
            rng.shuffle(order)
            for mode in order:
                rt.store.clear()
                if mode == 'warm_hit':
                    builds.append({'prefix_tokens': len(prefix), 'round': round_id, **rt.prepare(prefix)})
                outputs = []
                rt.mx.synchronize()
                start = time.perf_counter()
                for index, tokens in enumerate(work['requests']):
                    result = rt.generate(tokens, prefix=prefix if mode != 'cold' else (),
                                         max_new_tokens=spec['generated_tokens'], stop_at_eos=False)
                    outputs.append({'request': index, **result})
                rt.mx.synchronize()
                records.append({'prefix_tokens': len(prefix), 'round': round_id, 'mode': mode,
                                'workload_seconds': time.perf_counter()-start, 'outputs': outputs})
            save(out/'timings.json', records)
            save(out/'cache-builds.json', builds)
            print('benchmark', len(prefix), 'round', round_id+1, flush=True)
    return records, builds


def strict_correct(row, answer):
    answer = answer.strip()
    if row['is_impossible']:
        return answer == 'NO_ANSWER'
    return answer in row['answers']


def quality(rt, tok, model, spec, out):
    from mlx_lm.generate import generate_step
    rows = [json.loads(line) for line in Path('configs/qwen-prefix/qa-dev.jsonl').read_text().splitlines()]
    system = json.loads(Path('configs/qwen-prefix/source-qa-protocol.json').read_text())['system_prompt']
    groups = defaultdict(list)
    for row in rows:
        tokens = tok.apply_chat_template([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"}],
            tokenize=True, add_generation_prompt=True)
        if len(tokens) > spec['qa_max_input_tokens']:
            raise ValueError('Quality prompt exceeds frozen length limit; truncation is forbidden')
        groups[row['context_id']].append((row, tokens))
    rt.store.clear()
    predictions, native_parity = [], []
    for context_id, requests in sorted(groups.items()):
        prefix = common_prefix([tokens for _, tokens in requests])
        for row, tokens in requests:
            cold = rt.generate(tokens, max_new_tokens=spec['qa_max_new_tokens'], eos_ids=tok.eos_token_ids)
            cached = rt.generate(tokens, prefix=prefix, max_new_tokens=spec['qa_max_new_tokens'],
                                 eos_ids=tok.eos_token_ids)
            for result in (cold, cached):
                result['text'] = tok.decode(result['token_ids'], skip_special_tokens=True)
                result['strict_correct'] = strict_correct(row, result['text'])
            if len(native_parity) < 2:
                expected = []
                for token, _ in generate_step(rt.mx.array(tokens), model,
                                              max_tokens=spec['qa_max_new_tokens'], kv_bits=None):
                    expected.append(int(token))
                    if int(token) in tok.eos_token_ids: break
                native_parity.append({'id': row['id'], 'reference_token_ids': expected,
                                      'matches': expected == cold['token_ids']})
            predictions.append({'id': row['id'], 'context_id': context_id,
                                'is_impossible': row['is_impossible'], 'references': row['answers'],
                                'prefix_tokens': len(prefix), 'cold': cold, 'cached': cached,
                                'token_parity': cold['token_ids'] == cached['token_ids']})
        save(out/'quality-predictions.json', predictions)
        print('quality', len(predictions), '/', len(rows), flush=True)
    save(out/'native-decoder-parity.json', native_parity)
    return predictions, native_parity


def real_cache_contracts(rt, tok, spec, out):
    # Interleave contexts to exercise request isolation, eviction and miss fallback.
    rt.store.clear()
    base = tok.encode('A robot reads a report. '*50, add_special_tokens=False)
    prefixes = [base[:63]+[n] for n in (20, 21, 22)]
    suffix = tok.encode('\nSummarize briefly:', add_special_tokens=False)
    checks = []
    reference = rt.generate(prefixes[0]+suffix, max_new_tokens=8, stop_at_eos=False)
    for index in [0, 1, 0, 2, 1, 0]:
        result = rt.generate(prefixes[index]+suffix, prefix=prefixes[index],
                             max_new_tokens=8, stop_at_eos=False)
        checks.append({'prefix_index': index, 'status': result['status'], 'cache': result['cache'],
                       'matches_cold_a': result['token_ids'] == reference['token_ids'] if index == 0 else None})
    try:
        rt.generate(prefixes[0]+suffix, prefix=prefixes[1], max_new_tokens=1)
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    tiny = QwenPrefixRuntime(rt.model, rt.store.model_id, max_bytes=1)
    bypass = tiny.generate(prefixes[0]+suffix, prefix=prefixes[0], max_new_tokens=8, stop_at_eos=False)
    result = {'interleaved': checks, 'mismatched_prefix_rejected': mismatch_rejected,
              'oversized_bypassed': bypass['status'] == 'bypass' and tiny.store.bytes == 0,
              'oversized_output_matches_cold': bypass['token_ids'] == reference['token_ids']}
    save(out/'real-cache-contracts.json', result)
    return result


def summarize(records, builds, predictions, native, contracts):
    performance = []
    for length in sorted({r['prefix_tokens'] for r in records}):
        cells = {mode: [r for r in records if r['prefix_tokens'] == length and r['mode'] == mode]
                 for mode in ['cold', 'cached_workload', 'warm_hit']}
        med = lambda mode, key: statistics.median(o[key] for r in cells[mode] for o in r['outputs'])
        cold_ttft, hit_ttft = med('cold', 'ttft_seconds'), med('warm_hit', 'ttft_seconds')
        cold_group = statistics.median(r['workload_seconds'] for r in cells['cold'])
        cached_group = statistics.median(r['workload_seconds'] for r in cells['cached_workload'])
        miss_ttft = statistics.median(r['outputs'][0]['ttft_seconds'] for r in cells['cached_workload'])
        pair_ratios = []
        parity = []
        for baseline, candidate in zip(cells['cold'], cells['warm_hit']):
            pair_ratios.append(1-statistics.median(o['ttft_seconds'] for o in candidate['outputs'])/
                               statistics.median(o['ttft_seconds'] for o in baseline['outputs']))
            parity += [a['token_ids'] == b['token_ids'] for a, b in zip(baseline['outputs'], candidate['outputs'])]
        performance.append({'prefix_tokens': length, 'cold_ttft_seconds': cold_ttft,
                            'hit_ttft_seconds': hit_ttft, 'miss_ttft_seconds': miss_ttft,
                            'hit_ttft_reduction': 1-hit_ttft/cold_ttft,
                            'cold_decode_tps': med('cold', 'decode_tokens_per_second'),
                            'hit_decode_tps': med('warm_hit', 'decode_tokens_per_second'),
                            'cold_four_request_seconds': cold_group,
                            'cached_four_request_seconds': cached_group,
                            'four_request_reduction_including_miss': 1-cached_group/cold_group,
                            'round_hit_ttft_reductions': pair_ratios,
                            'cache_build_median_seconds': statistics.median(b['seconds'] for b in builds if b['prefix_tokens']==length),
                            'stored_tensor_bytes': cells['warm_hit'][0]['outputs'][0]['cache']['stored_tensor_bytes'],
                            'fixed_token_parity_count': sum(parity), 'fixed_token_comparisons': len(parity)})
    quality_result = {'questions': len(predictions), 'contexts': len({p['context_id'] for p in predictions}),
                      'token_parity_count': sum(p['token_parity'] for p in predictions),
                      'cold_strict_correct': sum(p['cold']['strict_correct'] for p in predictions),
                      'cached_strict_correct': sum(p['cached']['strict_correct'] for p in predictions),
                      'always_abstain_correct': sum(p['is_impossible'] for p in predictions),
                      'divergent_ids': [p['id'] for p in predictions if not p['token_parity']],
                      'cache_hits': sum(p['cached']['status']=='hit' for p in predictions),
                      'native_greedy_parity': all(p['matches'] for p in native)}
    return {'performance': performance, 'quality': quality_result, 'contracts': contracts,
            'acceptance': {'qa_token_parity': quality_result['token_parity_count']==len(predictions),
                           'native_parity': quality_result['native_greedy_parity'],
                           'prefix_mismatch_rejected': contracts['mismatched_prefix_rejected'],
                           'bounded_cache': contracts['oversized_bypassed'],
                           'eviction_replay_parity': all(r['matches_cold_a'] for r in contracts['interleaved'] if r['prefix_index']==0)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir(): raise ValueError('Local model required; implicit download forbidden')
    out = reserve_directory(args.output_dir)
    spec = json.loads(Path('configs/qwen-prefix/study.json').read_text())
    files, fingerprint = identity(args.model)
    import mlx.core as mx
    from mlx_lm import load
    if not mx.metal.is_available(): raise RuntimeError('Apple GPU unavailable; no silent backend fallback')
    model, tok = load(str(args.model))
    mx.eval(model.parameters()); mx.synchronize()
    config = json.loads((args.model/'config.json').read_text())
    if config.get('model_type') != 'qwen2' or config.get('quantization', {}).get('bits') != 8:
        raise ValueError('This frozen experiment requires the prepared Qwen2 Q8 model')
    manifest = {'environment': {'platform': platform.platform(), 'python': sys.version,
                               'device': mx.metal.device_info(),
                               'versions': {p: importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','tokenizers']}},
                'source': source_record(), 'spec': spec, 'model_files_sha256': files,
                'model_fingerprint': fingerprint, 'model_config': config,
                'input_sha256': {str(p): sha256(p) for p in sorted(Path('configs/qwen-prefix').iterdir())},
                'loaded_model_active_bytes': mx.get_active_memory()}
    save(out/'manifest.json', manifest)
    rt = QwenPrefixRuntime(model, fingerprint, max_entries=spec['max_cache_entries'],
                           max_bytes=spec['max_cache_bytes'])
    records, builds = benchmark(rt, tok, spec, out)
    predictions, native = quality(rt, tok, model, spec, out)
    contracts = real_cache_contracts(rt, tok, spec, out)
    save(out/'summary.json', summarize(records, builds, predictions, native, contracts))
    save(out/'complete.json', {'sha256': {p.name: sha256(p) for p in sorted(out.glob('*.json'))}})


if __name__ == '__main__':
    main()
