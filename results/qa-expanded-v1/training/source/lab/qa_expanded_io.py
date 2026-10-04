"""Fixed expanded-ranking evidence IO and paired article/family bootstrap."""
import gzip
import json
from pathlib import Path
import random

from lab.artifact_integrity import verify_hashes, safe_path
from lab.quantization_diagnostics import read, sha


def write_shard(path, records):
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n' for r in records).encode()
    with Path(path).open('xb') as f:
        f.write(gzip.compress(payload, compresslevel=6, mtime=0))


def load_collection(folder, expected_split, protocol_sha):
    folder = Path(folder)
    verify_hashes(folder, read(folder/'checksums.json'), exclude=('checksums.json',))
    state = read(folder/'run.json')
    if state['status'] != 'complete' or state['split'] != expected_split or state['protocol_sha256'] != protocol_sha:
        raise ValueError('Incomplete or wrong collection identity')
    if sha(folder/'protocol.json') != protocol_sha:
        raise ValueError('Collection protocol changed')
    protocol = read(folder/'protocol.json')
    expected_data = protocol['input_sha256']['configs/qa-expanded/dataset/'+expected_split+'.jsonl']
    if sha(folder/'data.jsonl') != expected_data or state['dataset_sha256'] != expected_data:
        raise ValueError('Collection dataset changed')
    for name, digest in protocol['source_sha256'].items():
        if sha(safe_path(folder/'source', name)) != digest:
            raise ValueError('Collection source snapshot changed')
    records = []
    for p in sorted(folder.glob('predictions-*.jsonl.gz')):
        records.extend(json.loads(line) for line in gzip.decompress(p.read_bytes()).decode().splitlines())
    features = read(folder/'features.json')
    if (len(records) != state['completed_predictions'] or len(records) != len(features)
            or len({r['id'] for r in records}) != len(records)
            or [r['id'] for r in records] != [r['id'] for r in features]):
        raise ValueError('Collection ID/count/order mismatch')
    return records, features


def paired_coverage(data, base, candidate, seed=2026100415, replicates=5000):
    """Paired difference in correct-answer coverage among answerable rows.

    Resample articles, then context families within each drawn article. The
    small number of articles and fixed class-balanced cohort limit inference.
    Per-row records require id/accepted_correct booleans, not model scores.
    """
    a, b = ({r['id']: r['accepted_correct'] for r in records} for records in (base, candidate))
    ids = {r['id'] for r in data}
    if len(a) != len(base) or len(b) != len(candidate) or a.keys() != ids or b.keys() != ids:
        raise ValueError('Paired comparison needs exact IDs')
    if any(type(v) is not bool for v in [*a.values(), *b.values()]):
        raise ValueError('Correct-accept indicators must be boolean')
    groups = {}
    changes = []; gained = []; lost = []
    for r in data:
        if r['is_impossible']:
            continue
        delta = int(b[r['id']]) - int(a[r['id']])
        changes.append(delta)
        groups.setdefault(r['source_title'], {}).setdefault(r['family_id'], []).append(delta)
        if delta > 0: gained.append(r['id'])
        if delta < 0: lost.append(r['id'])
    if not changes or len(groups) < 2 or replicates < 100:
        raise ValueError('Insufficient paired bootstrap input')
    articles = [[(sum(v), len(v)) for _, v in sorted(f.items())] for _, f in sorted(groups.items())]
    rng = random.Random(seed); values = []
    for _ in range(replicates):
        total = count = 0
        for _ in articles:
            families = rng.choice(articles)
            for _ in families:
                delta, n = rng.choice(families); total += delta; count += n
        values.append(total/count)
    values.sort(); last = len(values)-1
    return dict(correct_coverage_difference=sum(changes)/len(changes),
                ci95=[values[int(last*.025)], values[int(last*.975)]],
                gained=gained, lost=lost, articles=len(articles),
                context_families=sum(len(x) for x in articles), seed=seed, replicates=replicates,
                scope='Hierarchical paired article/context bootstrap on this fixed balanced cohort; not a population guarantee')
