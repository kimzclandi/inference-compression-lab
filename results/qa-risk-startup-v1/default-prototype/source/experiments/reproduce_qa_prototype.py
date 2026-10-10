"""Three disclosed QA examples and real input failures; reproduction, not evaluation."""
import argparse
from pathlib import Path
import shutil

from lab.evidence import reserve_directory
from lab.artifact_integrity import git_identity,file_hashes
from lab.quantization_diagnostics import read,write,rows,sha
from experiments.serve_qa_specialist import (load_service,InvalidInput,ModelUnavailable)


def run(args):
    out=reserve_directory(args.output_dir);record=dict(status='running',**git_identity(Path.cwd()),cases=[],
        scope='Selected published examples, after evaluation. A functional demonstration, not new confirmation or representative sample.')
    try:
        config=read('configs/qa-risk/demo.json')
        data={r['id']:r for r in rows('configs/qa-risk/dataset/evaluation/data.jsonl')}
        predictions={p['id']:p for p in rows('results/qa-risk-v2/evaluation-int8/predictions.jsonl')}
        policy=Path('configs/qa-risk/policy.json');root=Path('results/qa-risk-v2');study=Path('configs/qa-risk/study.json')
        service=load_service(policy,root,study,args.asset_root)
        record['startup_timing']=service.startup_timing
        for case in config['cases']:
            row=data[case['id']];payload={k:row[k] for k in ('context','question')}
            result=service.answer(payload)
            if result['status']!=case['expected_status']:raise ValueError('Published decision failed to reproduce')
            if result['status']=='answer' and (result['answer']!=predictions[row['id']]['prediction'] or
                    payload['context'][result['start']:result['end']]!=result['answer']):
                raise ValueError('Published answer/offset failed to reproduce')
            if abs(result['score']-predictions[row['id']]['confidence'])>1e-9:
                raise ValueError('Published ranking score failed to reproduce')
            record['cases'].append(dict(id=row['id'],case=case['label'],input=payload,output=result,
                gold_for_post_run_review_only=row['answers'],is_impossible=row['is_impossible']))
        for label,payload in [('unexpected_gold_field',dict(context='alpha beta',question='what?',answers=['alpha'])),
                              ('over_question_token_limit',dict(context='alpha beta',question='word '*100)),
                              ('empty_context',dict(context='',question='what?'))]:
            try:service.answer(payload)
            except InvalidInput as exc:record['cases'].append(dict(case=label,status='invalid_input',reason=str(exc)))
            else:raise ValueError('Invalid request unexpectedly accepted: '+label)
        try:load_service(policy,root,study,args.asset_root/'missing-assets')
        except ModelUnavailable as exc:record['cases'].append(dict(case='missing_assets',status='unavailable_model',reason=str(exc)))
        else:raise ValueError('Missing assets unexpectedly accepted')
        record.update(status='complete',source_sha256={'experiments/reproduce_qa_prototype.py':sha(__file__),
            'experiments/serve_qa_specialist.py':sha('experiments/serve_qa_specialist.py')},
            data_sha256=sha('configs/qa-risk/dataset/evaluation/data.jsonl'))
        for name in record['source_sha256']:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    except BaseException as exc:record.update(status='failed',error=repr(exc));raise
    finally:write(out/'run.json',record);write(out/'checksums.json',file_hashes(out))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args())

if __name__=='__main__':main()
