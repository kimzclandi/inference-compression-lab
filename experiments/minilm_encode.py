"""Run a verified local CPU deployment; outputs dimensions/norms and similarity."""
import argparse
import json
from pathlib import Path
import numpy as np
from lab.evidence import sha256
from lab.minilm_runtime import MiniLMRuntime


def main(args):
    config=json.loads(args.deployment.read_text())
    if config['status']!='accepted_for_local_demo':raise ValueError('Candidate did not pass local engineering acceptance')
    if sha256(config['tokenizer_path'])!=config['tokenizer_sha256']:raise ValueError('Tokenizer hash mismatch')
    rt=MiniLMRuntime(config['model_path'],config['tokenizer_path'],threads=config['threads'],
                    max_length=config['eval_max_length'],expected_sha256=config['model_sha256'])
    e=rt.encode(args.text)
    result={'shape':list(e.shape),'norms':np.linalg.norm(e,axis=1).tolist(),
            'cosine_matrix':(e@e.T).tolist(),'providers':rt.session.get_providers(),
            'model_sha256':config['model_sha256'],'threads':config['threads'],
            'note':'Local functional demo; not a benchmark or task-quality score'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--text',action='append',required=True)
    p.add_argument('--output',type=Path)
    main(p.parse_args())
