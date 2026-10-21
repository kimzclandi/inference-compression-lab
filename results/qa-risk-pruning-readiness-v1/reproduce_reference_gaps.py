"""Run from repository root; mutate disposable copies only. No model required.
This intentionally reproduces the four historical false acceptances.
The active checker rejects these cases in tests/test_qa_risk_pruning_acceptance.py.
"""
from pathlib import Path
import importlib.util,json,shutil,tempfile,hashlib,sys
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('old_verify',ROOT/'results/qa-risk-pruning-readiness-v1/reference/verify_measurements.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def resign(folder):
 dump(folder/'checksums.json',{p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'checksums.json'})
results=[]
for mode in ('empty_source_bindings','forged_quality_summary','unknown_replay_id','forged_timing_counts'):
 with tempfile.TemporaryDirectory() as td:
  target=Path(td)/'repo';shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('.git','__pycache__'))
  evidence=target/'results/qa-risk-pruning-v1';audit=evidence/'audit-final';bench=evidence/'benchmark'
  # The reference checker must run against its historical runner identity.
  shutil.copy2(bench/'source/experiments/qa_risk_pruning.py',target/'experiments/qa_risk_pruning.py')
  if mode=='empty_source_bindings':
   for p in [audit/'run.json',bench/'run.json',*[bench/str(i)/'run.json' for i in range(6)]]:
    value=json.loads(p.read_text());value['source_sha256']={};dump(p,value)
   for i in range(6):resign(bench/str(i))
  elif mode=='forged_quality_summary':
   p=audit/'summary.json';value=json.loads(p.read_text());value['historical_quality_replayed']['selective']['accepted_correct']=128;dump(p,value)
  elif mode=='unknown_replay_id':
   p=audit/'records.json';value=json.loads(p.read_text());value[0]['id']='not-a-real-id';dump(p,value)
  else:
   p=bench/'summary.json';value=json.loads(p.read_text());value['measured_requests']=999999;dump(p,value)
  resign(audit);resign(bench);v.ROOT=target
  try:v.verify();result={'case':mode,'incorrectly_accepted':True}
  except Exception as e:result={'case':mode,'incorrectly_accepted':False,'error':str(e)}
  results.append(result);print(result,flush=True)
if not all(r['incorrectly_accepted'] for r in results):
 raise SystemExit('A reference-verifier failure did not reproduce')
print(json.dumps(results,indent=2))
