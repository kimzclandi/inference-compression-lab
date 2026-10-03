"""Show one frozen first-token failure, or replay it on the five local models."""
import argparse
import gc
from pathlib import Path
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, rows, write, index_by_id

DEFAULT_CASE = '56e1c0f6cd28a01900c67b2c'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path('results/qwen-quantization-v1'))
    p.add_argument('--case-id', default=DEFAULT_CASE)
    p.add_argument('--live', action='store_true')
    p.add_argument('--model-root', type=Path)
    p.add_argument('--mixed-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args(); spec = read(a.evidence / 'protocol.json')
    row = index_by_id(rows(a.evidence / 'data.jsonl'))[a.case_id]
    print('Case:', a.case_id); print('Question:', row['question']); print('Gold:', row['answers'])
    print('Source:', 'NEW local inference' if a.live else 'FROZEN evidence; no new inference')
    records = []
    if a.live:
        if None in (a.model_root, a.mixed_root, a.output_dir):
            p.error('--live requires --model-root, --mixed-root and a NEW --output-dir')
        out = reserve_directory(a.output_dir)
        import os
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        import mlx.core as mx
        from mlx_lm import load
        from experiments.qwen_quantization import (prompt_ids, first_logits, diagnose,
                                                    generate, assert_model_layout, model_files)
        cfg = read(a.evidence / 'prompt.json'); selection = read(a.evidence / 'selection.json')
        reference = None
    for variant in spec['variants']:
        frozen_quality = read(a.evidence / f'{variant}-quality.json')
        frozen_pred = index_by_id(frozen_quality['predictions'])[a.case_id]
        frozen_logit = index_by_id(read(a.evidence / f'{variant}-logits.json'))[a.case_id]
        if a.live:
            path = a.model_root / f'student-{variant}' if variant in ['fp16', 'q4', 'q8'] else a.mixed_root / variant
            actual = model_files(path)
            for name, meta in frozen_quality['model_files'].items():
                if actual[name] != meta:
                    raise ValueError('Model identity differs: ' + variant + '/' + name)
            model, tok = load(str(path), tokenizer_config={'local_files_only': True})
            assert_model_layout(model, selection.get(variant), None if variant == 'fp16' else 8 if variant == 'q8' else 4)
            ids, _ = prompt_ids(row, tok, cfg)
            d = diagnose(first_logits(model, ids), tok, reference)
            if variant == 'fp16':
                reference = d
            pred = generate(model, tok, ids)
            if pred['token_ids'] != frozen_pred['token_ids'] or d['top10'] != frozen_logit['top10']:
                raise ValueError('Live replay differs; do not silently reuse old evidence')
            del model, tok; gc.collect(); mx.clear_cache()
        else:
            d, pred = frozen_logit, frozen_pred
        records.append({'variant': variant, 'id': a.case_id, 'diagnostic': d, 'prediction': pred})
        print(f"{variant:8} top1={d['top10'][0]['text']!r:14} "
              f"margin={d['margin']:.6f} reference_pair_margin={d['reference_pair_margin']:.6f} "
              f"FP16_top1_rank={d['reference_top1_rank']:2} answer={pred['prediction']!r}")
    if a.live:
        write(out / 'demo.json', records)
        print('Live tokens and top10 match frozen evidence; diagnostics are not a performance benchmark.')


if __name__ == '__main__':
    main()
