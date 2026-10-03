"""Real-model proof that direct segmentation avoids snapshots and restores positions."""
import argparse
from pathlib import Path
from unittest.mock import patch
from lab.evidence import reserve_directory, source_record, sha256
from lab.qwen_prefix import QwenPrefixRuntime
from experiments.qwen_prefix_study import identity, save
from experiments.verify_qwen_prefix_fair import require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = reserve_directory(args.output_dir)
    import mlx.core as mx
    from mlx_lm import load
    require(mx.metal.is_available(), 'Apple GPU required')
    files, fingerprint = identity(args.model)
    model, tok = load(str(args.model)); mx.eval(model.parameters()); mx.synchronize()
    rt = QwenPrefixRuntime(model, fingerprint)
    prefix = tok.encode('A robot checks its sensors and battery. '*20, add_special_tokens=False)[:64]
    suffix = tok.encode('\nDescribe the robot:', add_special_tokens=False)
    reference = rt.generate(prefix+suffix, max_new_tokens=8, stop_at_eos=False)
    events, retained_caches = [], []
    def observe(x, cache):
        events.append({'input_shape': list(x.shape), 'offsets_before': [c.offset for c in cache]})
        if len(events)==1: retained_caches.append(cache)
        return model(x, cache=cache)
    rt.model = observe
    with patch.object(rt, 'build', side_effect=AssertionError('Unexpected snapshot build')), \
         patch.object(rt, 'clone', side_effect=AssertionError('Unexpected snapshot clone')), \
         patch.object(rt.store, 'acquire', side_effect=AssertionError('Unexpected cache lookup')):
        direct = rt.generate(prefix+suffix, prefix=prefix, max_new_tokens=8,
                             stop_at_eos=False, reuse_prefix=False, segmented_snapshot=False)
    rt.model = model
    require(direct['token_ids']==reference['token_ids'], 'Direct/full parity')
    require(events[0]['input_shape']==[1,64] and set(events[0]['offsets_before'])=={0}, 'Prefix position')
    require(events[1]['input_shape']==[1,len(suffix)] and set(events[1]['offsets_before'])=={64}, 'Suffix position')
    for i, event in enumerate(events[2:]):
        require(event['input_shape']==[1,1] and set(event['offsets_before'])=={64+len(suffix)+i}, 'Decode position')
    other = prefix[:-1]+[prefix[-1]+1]
    rt.generate(other+suffix, prefix=other, max_new_tokens=8, stop_at_eos=False,
                reuse_prefix=False, segmented_snapshot=False)
    replay = rt.generate(prefix+suffix, prefix=prefix, max_new_tokens=8, stop_at_eos=False,
                         reuse_prefix=False, segmented_snapshot=False)
    require(replay['token_ids']==reference['token_ids'], 'Interleaved replay')
    require(rt.store.stats()['entries']==0 and rt.store.stats()['misses']==0, 'Direct touched shared cache')
    save(out/'manifest.json', {'source':source_record(), 'model_files_sha256':files, 'model_fingerprint':fingerprint,
                              'device':mx.metal.device_info()})
    save(out/'contracts.json', {'prefix':prefix,'suffix':suffix,'events':events, 'full':reference, 'direct':direct,
                               'replay':replay, 'acceptance':{'no_snapshot_build_clone_or_lookup':True,
                               'position_offsets':True,'interleaved_replay_parity':True}})
    save(out/'complete.json', {'sha256':{p.name:sha256(p) for p in sorted(out.glob('*.json'))}})
    print('Direct path: no copy/lookup, correct offsets, full/replay tokens match.')


if __name__=='__main__': main()
