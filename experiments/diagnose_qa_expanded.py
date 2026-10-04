"""Describe fixed calibration failures; no fitting, threshold selection or test read."""
import argparse
from collections import Counter
from pathlib import Path

from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes
from lab.quantization_diagnostics import read, rows, sha, write
from experiments.verify_qa_expanded import audit
from experiments.verify_qa_coverage_gap import metrics

ROOT = Path(__file__).resolve().parents[1]


def diagnose(output):
    # This light pass checks identities/counts. Full raw-feature verification is
    # a separate required command, explicitly recorded in its own artifact.
    folder=ROOT/'results/qa-expanded-v1'; verified=audit(folder,full_features=False)
    out=reserve_directory(output); data=rows(folder/'calibration/data.jsonl')
    features=read(folder/'calibration/features.json'); labels={r['id']:r['target'] for r in features}
    predictions={name:read(folder/f'training/{name}-calibration-predictions.json') for name in ('original','expanded')}
    title_groups={title:[r for r in data if r['source_title']==title] for title in sorted({r['source_title'] for r in data})}
    curves={}; row_decisions=[]
    for name, records in predictions.items():
        index={r['id']:r for r in records}; curves[name]=[]
        for point in verified['methods'][name]['curve']:
            threshold=point['threshold']; failed=[]
            constraints=read(ROOT/'configs/qa-expanded/study.json')['quality_constraints']
            for metric, minimum in (('accepted_precision','min_accepted_precision'),('answerable_answer_coverage','min_answerable_answer_coverage'),('correct_answerable_coverage','min_correct_answerable_coverage')):
                if point[metric] is None or point[metric] < constraints[minimum]: failed.append(metric)
            if point['unanswerable_false_accept_rate'] > constraints['max_unanswerable_false_accept_rate']: failed.append('unanswerable_false_accept_rate')
            if point['invalid_rate'] > constraints['max_invalid_rate']: failed.append('invalid_rate')
            per_article={title:metrics(group,[index[r['id']] for r in group],threshold) for title,group in title_groups.items()}
            curves[name].append(dict(**point,failed_constraints=failed,per_article=per_article))
            for row in data:
                p=index[row['id']]; accepted=p['confidence'] >= threshold
                row_decisions.append(dict(method=name,threshold=threshold,id=row['id'],source_title=row['source_title'],
                    confidence=p['confidence'],raw_em_correct=bool(labels[row['id']]),is_impossible=row['is_impossible'],
                    accepted=accepted,accepted_correct=accepted and bool(labels[row['id']])))
    base={p['id']:p for p in predictions['original']}; candidate={p['id']:p for p in predictions['expanded']}
    changed=[]
    for row in data:
        key=row['id']; a=base[key]['confidence']>=.7; b=candidate[key]['confidence']>=.7
        if a != b: changed.append(dict(id=key,source_title=row['source_title'],is_impossible=row['is_impossible'],
            raw_em_correct=bool(labels[key]),original_accept=a,expanded_accept=b,
            original_score=base[key]['confidence'],expanded_score=candidate[key]['confidence']))
    diagnosis=dict(protocol_sha256=verified['protocol_sha256'],selection_sha256=sha(folder/'training/selection.json'),
        training_converged=True,calibration_passed=False,evaluation_outputs_read=False,threshold_selected=None,
        explanation='A fixed 0.7 comparison and the full preregistered grid describe failure; neither creates a deployable threshold or a new selection.',
        raw_correct_answerable=sum(labels.values()),answerable=sum(not r['is_impossible'] for r in data),
        curves=curves,fixed_0_7_changed_decisions=changed,
        fixed_0_7_change_categories=dict(Counter(('gained_' if r['expanded_accept'] else 'lost_')+('correct' if r['raw_em_correct'] else 'incorrect') for r in changed)),
        scope='Descriptive calibration diagnosis after outcomes; not independent evaluation, model-unseen evidence, or proof that more data cannot help.')
    write(out/'diagnosis.json',diagnosis); write(out/'calibration-decisions.json',row_decisions)
    write(out/'run.json',dict(status='complete',model_inference=False,fit=False,evaluation_outputs_read=False,
        source_sha256=sha(Path(__file__)),authorship='AI-assisted implementation and execution'))
    write(out/'checksums.json',file_hashes(out)); return diagnosis


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); result=diagnose(a.output_dir)
    print({k:result[k] for k in ('training_converged','calibration_passed','raw_correct_answerable','answerable','fixed_0_7_change_categories')})
