"""Derive one fixed richer matrix from already published raw development data."""
from pathlib import Path
from lab.qa_rich_features import enrich
from lab.qa_expanded_io import load_collection
from lab.quantization_diagnostics import read,rows,sha,write
from lab.evidence import reserve_directory
from lab.artifact_integrity import file_hashes
ROOT=Path(__file__).resolve().parents[1]

def prepare(output):
    out=reserve_directory(output);spec=sha(ROOT/'configs/qa-expanded/study.json')
    old=read(ROOT/'results/qa-risk-v2/training/int8-training-features.json')
    old_data={r['id']:r for r in rows(ROOT/'configs/qa-risk/dataset/train/data.jsonl')}
    old_raw={r['id']:r for s in ('calibration','evaluation') for r in rows(ROOT/f'results/qa-specialist-v1/{s}-int8/predictions.jsonl')}
    enriched=[enrich(old_data[r['id']]['question'],old_data[r['id']]['context'],old_raw[r['id']],r) for r in old]
    for split,dirname in [('train_new','train-new'),('calibration','calibration')]:
        raw,matrix=load_collection(ROOT/'results/qa-expanded-v1'/dirname,split,spec)
        data=rows(ROOT/f'configs/qa-expanded/dataset/{split}.jsonl')
        values=[enrich(d['question'],d['context'],p,f) for d,p,f in zip(data,raw,matrix)]
        if split=='train_new':enriched+=values;write(out/'training-features.json',enriched)
        else:write(out/'development-features.json',values)
    write(out/'run.json',dict(status='complete',training_rows=2304,development_rows=192,features=16,model_inference=False,labels_used_for_features=False))
    write(out/'checksums.json',file_hashes(out))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
