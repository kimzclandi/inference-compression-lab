"""Offline recomputation; a negative confirmation is valid research evidence."""
import argparse
from pathlib import Path
from lab.artifact_integrity import verify_hashes, safe_path
from lab.confirmation import summarize
from lab.qa_metrics import evaluate
from lab.quantization_diagnostics import read, write, sha, rows, aggregates_equal, index_by_id
from experiments.prepare_qwen_confirmation import normalized_hash
from experiments.verify_qwen_quantization import require


def verify(folder):
    folder = Path(folder)
    verify_hashes(folder, read(folder / 'checksums.json'))
    run = read(folder / 'run.json'); spec = read(folder / 'protocol.json')
    require(run['status'] == 'complete', 'Incomplete confirmation')
    require(run['completed_variants'] == spec['variants'], 'Variant coverage/order')
    require(run['model_identity_verified'] == {v: True for v in spec['variants']}, 'Model identity not checked')
    require(sha(folder / 'protocol.json') == run['protocol_sha256'], 'Protocol identity')
    require(sha(folder / 'data.jsonl') == spec['data_sha256'], 'Frozen data identity')
    require(sha(folder / 'prompt.json') == spec['prompt_sha256'], 'Frozen prompt identity')
    require(sha(folder / 'source/lab/qa_metrics.py') == spec['scorer_sha256'], 'Unchanged scorer')
    for name, expected in run['source_sha256'].items():
        require(sha(safe_path(folder / 'source', name)) == expected, 'Source snapshot changed')
    require(spec['blocks'] == {'selected': 10, 'control': 22}, 'Fixed blocks changed')
    data = rows(folder / 'data.jsonl'); denied = read(folder / 'exclusions.json')
    manifest = read(folder / 'dataset-manifest.json'); selection = read(folder / 'selection.json')
    require(manifest['selection'] == selection and manifest['exclusions_sha256'] == sha(folder / 'exclusions.json'), 'Selection provenance')
    require(manifest['data_sha256'] == spec['data_sha256'], 'Dataset manifest identity')
    require(len(data) == spec['quality']['expected_n'] == manifest['n'], 'Confirmation coverage')
    for row in data:
        require(row['id'] not in denied['ids'] and row['source_title'] not in denied['titles']
                and normalized_hash(row['context']) not in denied['context_hashes']
                and normalized_hash(row['question']) not in denied['question_hashes'], 'Overlap with known prior data')
        require(row['family_id'] == normalized_hash(row['context']), 'Wrong bootstrap family')
    require(set(r['source_title'] for r in data) == set(manifest['titles']), 'Article coverage')
    for title in manifest['titles']:
        for impossible in [False, True]:
            group = [r for r in data if r['source_title'] == title and r['is_impossible'] == impossible]
            require(len(group) == selection['per_class_per_article'] == len({r['family_id'] for r in group}), 'Article/class balance')
    metrics, scored = {}, {}; reference = None
    for v in spec['variants']:
        result = read(folder / f'{v}-quality.json'); predictions = result['predictions']
        metrics[v], scored[v] = evaluate(data, predictions)
        require(aggregates_equal(metrics[v], result['metrics']) and scored[v] == result['scored'], 'Score arithmetic drift')
        sig = [(p['id'],p['prompt_sha256'],p['input_token_ids_sha256'],p['input_tokens']) for p in predictions]
        if reference is None: reference = sig
        require(sig == reference, 'Different precision received different prompts')
        logits = index_by_id(read(folder / f'{v}-logits.json'))
        require(set(logits) == set(index_by_id(data)), 'Logit coverage')
        for pred in predictions:
            d = logits[pred['id']]
            require(0 < len(pred['token_ids']) <= spec['quality']['max_new_tokens'], 'Generation length')
            require(d['top1'] == pred['token_ids'][0] == d['top10'][0]['token'], 'Greedy first token')
            require(d['input_token_ids_sha256'] == pred['input_token_ids_sha256'], 'Diagnostic prompt')
        require(set(result['layout']) == {str(i) for i in range(24)}, 'Layer coverage')
        for i, layout in result['layout'].items():
            fp = v == 'fp16' or int(i) == spec['blocks'].get(v)
            require(layout['fp16'] == (7 if fp else 0) and layout['quantized'] == (0 if fp else 7), 'Executed precision layout')
    return summarize(data, metrics, scored, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('folder', type=Path)
    p.add_argument('--write-summary', action='store_true'); a = p.parse_args()
    result = verify(a.folder)
    if a.write_summary: write(a.folder / 'summary.json', result)
    elif (a.folder / 'summary.json').exists():
        require(aggregates_equal(result, read(a.folder / 'summary.json')), 'Frozen summary drift')
    for v, r in result['variants'].items(): print(v, r['em_count'], '/', result['n'])
    print('Confirmation gate:', result['gate'])


if __name__ == '__main__':
    main()
