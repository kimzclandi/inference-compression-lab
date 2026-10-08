"""Publish only target-process interval projections from the fixed sync traces."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from experiments.metal_mechanism_diagnostic import save,sha
from lab.interval_coverage import coverage
from lab.metal_trace_summary import read_export

SPEC=Path('configs/metal-sync-granularity-v1.json')
FIELDS=['source_row_index','process','category','start_ns','duration_ns']


def select(rows,process,kind):
    selected=[]
    for index,row in enumerate(rows):
        if row['process'] is None or row['process']['fmt']!=process:continue
        cell=row['channel-name' if kind=='gpu' else 'event-label']
        if cell is None:continue
        category=(cell['fmt'] or '').split('  (',1)[0].strip()
        if kind=='gpu' and category!='Compute':continue
        if row['start'] is None or row['duration'] is None:raise ValueError('Missing attributed timing')
        selected.append(dict(source_row_index=index,process=process,category=category,
            start_ns=int(row['start']['raw']),duration_ns=int(row['duration']['raw'])))
    if not selected:raise ValueError('No exact-process intervals')
    return selected


def summarize(rows,process,kind):
    seen=set()
    for row in rows:
        if row['process']!=process or row['source_row_index'] in seen:raise ValueError('Process mismatch or duplicate source index')
        seen.add(row['source_row_index'])
        if kind=='gpu' and row['category']!='Compute':raise ValueError('Non-compute GPU row')
    result=coverage([(r['start_ns'],r['duration_ns']) for r in rows])
    result['categories']=dict(sorted(Counter(r['category'] for r in rows).items()))
    return result


def project(inputs,output):
    spec=json.loads(SPEC.read_text());output.mkdir(parents=True,exist_ok=False)
    receipt={'schema':'metal-sync-trace-projection-v1','protocol_sha256':sha(SPEC),
        'profile_source_sha256':sha('experiments/metal_sync_granularity.py'),
        'source_xml_sha256':{},'arms':{},'summary':{},'files':{},
        'scope':spec['profile']['scope'],
        'privacy':'Only exact target Python process intervals are public. Full trace/XML remain local.',
        'authenticity_limit':'Hashes bind local source exports; public CSV independently supports arithmetic, not recording authenticity.'}
    for arm in spec['profile']['arms']:
        progress=inputs/f'{arm}-progress.jsonl'
        events=[json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
        if [r['stage'] for r in events]!=['started','warm','complete'] or any(r['arm']!=arm or r['pid']!=events[0]['pid'] for r in events):raise ValueError('Incomplete or inconsistent workload receipt')
        if events[-1]['profiled_calls']!=spec['profile']['batches']*spec['chain_length']:raise ValueError('Profiled call count differs')
        process=f"python ({events[0]['pid']})"
        toc=ET.parse(inputs/f'{arm}-toc.xml')
        target=toc.find('.//info/target/process')
        if target is None or target.get('pid')!=str(events[0]['pid']) or target.get('return-exit-status')!='0':raise ValueError('Trace target did not exit successfully')
        if target.get('arguments')!=f'-m experiments.metal_sync_granularity profile --arm {arm}':raise ValueError('Wrong profiler command')
        info={key:toc.findtext('.//summary/'+key) for key in ('start-date','end-date','duration','instruments-version','template-name')}
        receipt['arms'][arm]=dict(process=process,progress=events,recording=info,
            toc_sha256=sha(inputs/f'{arm}-toc.xml'),progress_sha256=sha(progress))
        receipt['summary'][arm]={}
        for kind in ('gpu','application'):
            source=inputs/f'{arm}-{kind}.xml'
            rows=select(read_export(source,f'metal-{kind}-intervals'),process,kind)
            name=f'{arm}-{kind}.csv'
            with (output/name).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=FIELDS,lineterminator="\n");writer.writeheader();writer.writerows(rows)
            receipt['source_xml_sha256'][name]=sha(source);receipt['files'][name]=sha(output/name)
            receipt['summary'][arm][kind]=summarize(rows,process,kind)
    # Exported counter metadata contains no unrelated processes.
    counter=inputs/'compiled-counter-info.xml'
    (output/'compiled-counter-info.xml').write_bytes(counter.read_bytes())
    names=[r['name']['fmt'] for r in read_export(counter,'gpu-counter-info')]
    receipt['available_counter_names_compiled_trace']=names
    receipt['files']['compiled-counter-info.xml']=sha(output/'compiled-counter-info.xml')
    save(output/'receipt.json',receipt)
    return receipt


def verify(output):
    receipt=json.loads((output/'receipt.json').read_text())
    if receipt['protocol_sha256']!=sha(SPEC):raise ValueError('Protocol drift')
    source=json.loads(Path('results/metal-sync-granularity-v1/run.json').read_text())['source_sha256']['experiments/metal_sync_granularity.py']
    if source!=receipt['profile_source_sha256']:raise ValueError('Profiler source differs from fixed experiment')
    expected={f'{a}-{k}.csv' for a in ('compiled','original','masked') for k in ('gpu','application')}|{'compiled-counter-info.xml'}
    if set(receipt['files'])!=expected:raise ValueError('File coverage differs')
    for name,digest in receipt['files'].items():
        if sha(output/name)!=digest:raise ValueError('Projection changed')
    names=[r['name']['fmt'] for r in read_export(output/'compiled-counter-info.xml','gpu-counter-info')]
    if names!=receipt['available_counter_names_compiled_trace']:raise ValueError('Counter metadata differs')
    spec=json.loads(SPEC.read_text())
    for arm in spec['profile']['arms']:
        metadata=receipt['arms'][arm];events=metadata['progress']
        if [r['stage'] for r in events]!=['started','warm','complete'] or any(r['arm']!=arm or f"python ({r['pid']})"!=metadata['process'] for r in events):raise ValueError('Progress coverage differs')
        if events[-1]['profiled_calls']!=spec['profile']['batches']*spec['chain_length']:raise ValueError('Wrong workload size')
        for kind in ('gpu','application'):
            with (output/f'{arm}-{kind}.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            for row in rows:
                for key in ('source_row_index','start_ns','duration_ns'):row[key]=int(row[key])
            if summarize(rows,metadata['process'],kind)!=receipt['summary'][arm][kind]:raise ValueError('Interval summary differs')
    return receipt['summary']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['project','verify'])
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='project':
        if a.input is None:p.error('--input required')
        result=project(a.input,a.output)['summary']
    else:result=verify(a.output)
    print(json.dumps(result,indent=2))
