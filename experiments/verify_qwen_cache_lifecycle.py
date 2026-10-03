"""Offline verification of lifecycle faults, LRU trace and measured medians."""
import argparse
from collections import OrderedDict, Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from experiments.verify_qwen_prefix_fair import read, require


def verify(folder):
    folder=Path(folder)
    if (folder/'complete.json').exists():
        for name,digest in read(folder/'complete.json')['sha256'].items():
            require(hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,'Checksum '+name)
    m=read(folder/'manifest.json');spec=m['spec'];records=read(folder/'timings.json')
    require(hashlib.sha256(json.dumps(m['model_files_sha256'],sort_keys=True).encode()).hexdigest()==m['model_fingerprint'],'Identity')
    keyed={(r['length'],r['round'],r['mode']):r for r in records}
    require(len(keyed)==len(records) and set(keyed)==set(itertools.product(spec['prefix_lengths'],range(spec['rounds']),spec['modes'])),'Cell coverage')
    works={w['length']:w for w in read(folder/'workloads.json')}
    for r in records:
        ref=keyed[r['length'],r['round'],'direct']['outputs']
        require(len(r['outputs'])==len(spec['trace']),'Trace length')
        lru=OrderedDict();counts=Counter()
        for i,(index,o) in enumerate(zip(spec['trace'],r['outputs'])):
            if r['mode']=='direct': status='direct'
            elif index in lru: status='hit';lru.move_to_end(index);counts['hits']+=1
            else:
                status='miss';counts['misses']+=1
                while lru and (len(lru)>=r['max_entries'] or (len(lru)+1)*r['length']*12288>r['max_bytes']):
                    lru.popitem(last=False);counts['evictions']+=1
                lru[index]=True
            require(o['status']==status,'LRU status')
            require(o['cache']['entries']==len(lru) and o['cache']['stored_tensor_bytes']==len(lru)*r['length']*12288,'Resident accounting')
            require(o['token_ids']==ref[i]['token_ids'] and len(o['token_ids'])==spec['generated_tokens'],'Token parity/length')
            require(o['input_tokens']==r['length']+len(works[r['length']]['suffix']),'Input length')
            require(math.isclose(o['total_seconds'],o['ttft_seconds']+o['decode_seconds']),'Timing accounting')
        require(all(r['cache_delta'][k]==counts[k] for k in r['cache_delta']),'Counters')
        require(r['cache_delta']==spec['predictions_before_execution'][r['mode']],'Preregistered prediction')
        require(r['seconds']>=sum(o['total_seconds'] for o in r['outputs']),'Wall timer')
    c=read(folder/'contracts.json')
    require({r['case'] for r in c['faults']}=={'hit_clone','miss_clone','bypass_clone','builder_after_prefill','invalid_size'},'Fault coverage')
    for row in c['faults']:
        before,after=row['before'],row['after']
        require(after['resident_tokens']==before['resident_tokens'],'Fault changed LRU')
        require(after['stats']=={**before['stats'],'failures':before['stats']['failures']+1},'Fault counters')
        require([r['token_ids'] for r in row['replay']]==row['reference_token_ids'],'Fault replay parity')
        require(all(r['status']=='hit' for r in row['replay']),'Fault evicted resident')
    require(c['mismatch_state_preserved'] and c['cleared']['stats']['entries']==c['cleared']['stats']['stored_tensor_bytes']==0,'Clear/mismatch')
    require(c['after_clear']['status']=='miss','Clear did not invalidate')
    for o in c['bypass']:require(o['status']=='bypass' and o['cache']['entries']==0 and o['token_ids']==c['after_clear']['token_ids'],'Bypass')
    summary=[]
    for length in spec['prefix_lengths']:
        med={mode:statistics.median(r['seconds'] for r in records if r['length']==length and r['mode']==mode) for mode in spec['modes']}
        summary.append({'length':length,'median_trace_seconds':med,
          'reduction_vs_direct':{mode:1-value/med['direct'] for mode,value in med.items() if mode!='direct'},
          'counters_per_round':{mode:spec['predictions_before_execution'][mode] for mode in spec['modes']}})
    return {'all_token_parity':True,'faults_preserve_residents':True,'predictions_confirmed':True,
            'requests':len(records)*len(spec['trace']),'performance':summary}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args()
    result=verify(args.folder)
    if (args.folder/'summary.json').exists():require(result==read(args.folder/'summary.json'),'Summary mismatch')
    print(json.dumps(result,indent=2))
