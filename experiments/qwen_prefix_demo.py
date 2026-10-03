"""Local repeated-document QA demo with cold/cached output comparison."""
import argparse
from pathlib import Path
import json
from lab.evidence import reserve_directory
from lab.qwen_prefix import QwenPrefixRuntime
from experiments.qwen_prefix_study import common_prefix, identity, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--document', type=Path, required=True)
    p.add_argument('--question', action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.model.is_dir(): raise ValueError('Local model required')
    out = reserve_directory(args.output_dir)
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(str(args.model))
    mx.eval(model.parameters()); mx.synchronize()
    _, fingerprint = identity(args.model)
    rt = QwenPrefixRuntime(model, fingerprint)
    system = json.loads(Path('configs/qwen-prefix/source-qa-protocol.json').read_text())['system_prompt']
    text = args.document.read_text()
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': system},
               {'role': 'user', 'content': f'Passage:\n{text}\n\nQuestion: {q}'}],
               tokenize=True, add_generation_prompt=True) for q in args.question]
    if any(len(tokens)>2048 for tokens in prompts): raise ValueError('Demo prompt exceeds2048 tokens')
    prefix = common_prefix(prompts)
    results = []
    for question, tokens in zip(args.question, prompts):
        cold = rt.generate(tokens, max_new_tokens=48, eos_ids=tok.eos_token_ids)
        cached = rt.generate(tokens, prefix=prefix, max_new_tokens=48, eos_ids=tok.eos_token_ids)
        row = {'question': question, 'answer': tok.decode(cached['token_ids'], skip_special_tokens=True),
               'cache_status': cached['status'], 'same_tokens': cold['token_ids']==cached['token_ids'],
               'cold': cold, 'cached': cached}
        results.append(row)
        print(json.dumps({k: row[k] for k in ['question','answer','cache_status','same_tokens']}, ensure_ascii=False))
    save(out/'demo.json', {'model_fingerprint': fingerprint, 'results': results,
         'note': 'Functional demonstration. Not warmed performance evidence or general answer-quality validation.'})


if __name__ == '__main__': main()
