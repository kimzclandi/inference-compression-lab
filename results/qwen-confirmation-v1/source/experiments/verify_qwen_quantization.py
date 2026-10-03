"""Recompute Qwen diagnostic selection, QA scores and matched performance without MLX."""
import argparse
import math
from pathlib import Path
import statistics
from lab.artifact_integrity import verify_hashes
from lab.qa_metrics import evaluate
from lab.quantization_diagnostics import (read, rows, sha, write, index_by_id,
                                          paired, rank_blocks, performance, gate, aggregates_equal)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(folder):
    folder = Path(folder)
    spec = read(folder / 'protocol.json'); run = read(folder / 'run.json')
    require(run['status'] == 'complete', 'Incomplete/failed experiment')
    checksums = read(folder / 'checksums.json')
    verify_hashes(folder, checksums)
    require(sha(folder / 'protocol.json') == run['protocol_sha256'], 'Protocol hash mismatch')
    for name, expected in run['source_sha256'].items():
        require(sha(folder / 'source' / name) == expected, 'Source snapshot mismatch')
    data = rows(folder / 'data.jsonl')
    require(len(data) == spec['quality']['expected_n'], 'Wrong quality sample count')
    selection = read(folder / 'selection.json'); screen = read(folder / 'screen.json')
    require(rank_blocks(screen, spec) == selection['ranking'], 'Selection arithmetic drift')
    require(selection['selected'] == selection['ranking'][0]['block'], 'Selection rule changed')
    require(selection['control'] == (selection['selected'] + 12) % 24, 'Control changed')
    require(len(selection['costs']) == 24, 'Block cost coverage')
    require(len({c['fp16_tensor_bytes'] - c['q4_tensor_bytes'] for c in selection['costs']}) == 1,
            'Unequal intervention tensor cost')
    identities = read(folder / 'model-identities.json')
    quality, scored, logits, result = {}, {}, {}, {}
    for variant in spec['variants']:
        quality[variant] = read(folder / f'{variant}-quality.json')
        predictions = quality[variant]['predictions']
        metrics, scored[variant] = evaluate(data, predictions)
        require(aggregates_equal(metrics, quality[variant]['metrics']), 'Quality arithmetic drift: ' + variant)
        require(scored[variant] == quality[variant]['scored'], 'Per-item score drift')
        logits[variant] = index_by_id(read(folder / f'{variant}-logits.json'))
        require(set(logits[variant]) == set(index_by_id(data)), 'Diagnostic sample coverage')
        for p in predictions:
            d = logits[variant][p['id']]; ref = logits['fp16'][p['id']]
            require(d['input_token_ids_sha256'] == p['input_token_ids_sha256'], 'Logit prompt mismatch')
            require(d['top1'] == p['token_ids'][0] == d['top10'][0]['token'], 'Greedy/diagnostic mismatch')
            require(d['top2'] == d['top10'][1]['token'], 'Top2 mismatch')
            require(d['reference_top1'] == ref['top1'] and d['reference_top2'] == ref['top2'], 'Wrong reference pair')
            require(d['fp16_margin'] == ref['margin'], 'Reference margin drift')
            require(d['reference_pair_margin'] == d['reference_pair_logits'][0] - d['reference_pair_logits'][1], 'Pair margin arithmetic')
            require(d['margin'] == d['top10'][0]['logit'] - d['top10'][1]['logit'], 'Top2 margin arithmetic')
            top_ids = [r['token'] for r in d['top10']]
            require(len(top_ids) == len(set(top_ids)) == 10, 'Top10 duplicates')
            require(d['top10'] == sorted(d['top10'], key=lambda x: (-x['logit'], x['token'])), 'Top10 order')
            if ref['top1'] in top_ids:
                require(d['reference_top1_rank'] == top_ids.index(ref['top1']) + 1, 'Reference rank mismatch')
            else:
                require(d['reference_top1_rank'] > 10, 'Reference rank out of bounds')
        sig = [(p['id'], p['input_tokens'], p['input_token_ids_sha256'], p['prompt_sha256']) for p in predictions]
        ref_sig = [(p['id'], p['input_tokens'], p['input_token_ids_sha256'], p['prompt_sha256']) for p in quality['fp16']['predictions']]
        require(sig == ref_sig, 'Quality inputs differ across precision')
        model_files = quality[variant]['model_files']
        if variant in identities:
            for name, meta in identities[variant].items():
                require(model_files[name] == meta, 'Historical model identity mismatch')
        else:
            export = read(folder / f'{variant}-export.json')
            require(export['model_files'] == model_files and export['source_tensors_verified'], 'Export identity mismatch')
            prefix = f"model.layers.{selection[variant]}."
            for name, tensor in export['tensors'].items():
                require(tensor['source'] == ('fp16' if name.startswith(prefix) else 'q4'), 'Wrong source tensor')
            # In-memory intervention and reloaded export must agree on diagnostics.
            for r in screen:
                if r['block'] == selection[variant]:
                    d = logits[variant][r['id']]
                    require(d['top10'] == r['top10'] and d['reference_pair_logits'] == r['reference_pair_logits'],
                            'Reloaded model differs from screened intervention')
        for index, layout in quality[variant]['layout'].items():
            fp = variant == 'fp16' or (variant in ['selected', 'control'] and int(index) == selection[variant])
            require(layout['fp16'] == (7 if fp else 0) and layout['quantized'] == (0 if fp else 7), 'Wrong executed layout')
        result[variant] = {'quality': metrics, 'em_count': int(sum(r['em'] for r in scored[variant])),
                           'format_valid_count': sum(r['format_valid'] for r in scored[variant]),
                           'weight_bytes': model_files['model.safetensors']['bytes']}
    require(paired(scored['fp16'], scored['q4'])['lost'] == spec['diagnosis_ids'], 'Diagnostic losses drift')
    # Equal tensor cost was preregistered. Safetensors headers can differ in byte
    # count because tensor offsets/order differ; report actual file sizes separately.
    export_costs = {v: sum(t['bytes'] for t in read(folder / f'{v}-export.json')['tensors'].values())
                    for v in ['selected', 'control']}
    require(export_costs['selected'] == export_costs['control'], 'Unequal exported tensor cost')
    for v in export_costs:
        result[v]['export_tensor_bytes'] = export_costs[v]
        result[v]['serialization_overhead_bytes'] = result[v]['weight_bytes'] - export_costs[v]
        require(result[v]['serialization_overhead_bytes'] > 0, 'Invalid export file size')
    replay = read(folder / 'historical-replay.json')
    require(all(v['n'] == 74 and not v['token_mismatches'] and not v['prompt_mismatches'] for v in replay.values()), 'Historical replay mismatch')
    commands = rows(folder / 'commands.jsonl'); variants = spec['variants']
    expected_order = [(r, v) for r in range(spec['benchmark']['rounds'])
                      for v in variants[r:] + variants[:r]]
    require([(c['round'], c['variant']) for c in commands] == expected_order, 'Unbalanced/missing benchmark cells')
    reference_sig = None; token_reference = {}; perfs = {v: [] for v in variants}
    for c in commands:
        variant = c['variant']; cell = read(folder / f"bench-{c['round']}-{variant}/run.json")
        require(c['returncode'] == 0 and cell['status'] == 'complete', 'Failed benchmark cell')
        require(cell['model_weight_sha256'] == quality[variant]['model_files']['model.safetensors']['sha256'], 'Benchmark used different model')
        timing = cell['timings']; require(len(timing) == 8, 'Benchmark request coverage')
        sig = [(p['id'], p['input_tokens'], p['input_token_ids_sha256'], p['prompt_sha256']) for p in timing]
        if reference_sig is None:
            reference_sig = sig
        require(sig == reference_sig, 'Benchmark input mismatch')
        for p in timing:
            require(math.isclose(p['total_seconds'], p['ttft_seconds'] + p['decode_seconds'], abs_tol=1e-12), 'Timing reconciliation')
            require(p['stop_reason'] == 'fixed_length', 'Early-stop benchmark is invalid')
        tokens = [p['token_ids'] for p in timing]
        if variant in token_reference:
            require(tokens == token_reference[variant], 'Within-variant greedy drift')
        token_reference[variant] = tokens
        perf = performance(timing); require(aggregates_equal(perf, cell['performance']), 'Performance arithmetic drift')
        require(all(value > 0 for value in cell['memory'].values()), 'Invalid memory counter')
        perfs[variant].append(dict(perf, **cell['memory']))
    comparisons = {}
    fp_predictions = index_by_id(quality['fp16']['predictions'])
    for variant in variants:
        result[variant]['performance'] = {key: statistics.median(p[key] for p in perfs[variant])
                                          for key in perfs[variant][0]}
        result[variant]['performance_runs'] = perfs[variant]
        mismatches = [r['id'] for r in quality[variant]['predictions']
                      if r['token_ids'] != fp_predictions[r['id']]['token_ids']]
        flips = [i for i, d in logits[variant].items() if d['top1'] != logits['fp16'][i]['top1']]
        margins = [logits['fp16'][i]['margin'] for i in flips]
        comparisons[variant] = dict(paired(scored['fp16'], scored[variant]),
            different_sequences=len(mismatches), first_token_flips=len(flips),
            flip_ids=flips, median_fp16_margin_on_flips=statistics.median(margins) if margins else None,
            reference_top2_overtakes=sum(d['reference_pair_margin'] < 0 for d in logits[variant].values()))
    return {'scope': spec['dev_role'], 'selected_block': selection['selected'], 'control_block': selection['control'],
            'variants': result, 'vs_fp16': comparisons, 'selected_vs_q4': paired(scored['q4'], scored['selected']),
            'selected_vs_control': paired(scored['control'], scored['selected']),
            'candidate_gate': gate(result, spec['exploratory_candidate_gate']),
            'acceptance': {'historical_replay': replay, 'balanced_benchmarks': len(commands),
                           'screen_interventions': len(screen), 'same_cost_control': True}}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('folder', type=Path)
    p.add_argument('--write-summary', action='store_true'); a = p.parse_args()
    result = verify(a.folder)
    if a.write_summary:
        write(a.folder / 'summary.json', result)
    elif (a.folder / 'summary.json').exists():
        require(aggregates_equal(read(a.folder / 'summary.json'), result), 'Frozen summary drift')
    print('Verified:', result['acceptance']['screen_interventions'], 'interventions,',
          result['acceptance']['balanced_benchmarks'], 'balanced fresh-process benchmarks.')
    for v, r in result['variants'].items():
        print(v, 'EM', r['em_count'], '/74; first-token flips', result['vs_fp16'][v]['first_token_flips'])
    print('Exploratory candidate gate:', result['candidate_gate'])


if __name__ == '__main__':
    main()
