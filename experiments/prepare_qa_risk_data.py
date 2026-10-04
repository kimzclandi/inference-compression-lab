"""Freeze one supervised risk-head data allocation; old evaluation is development."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.prepare_qa_remediation import (
    KEYS, extend_exclusions, read_json, read_rows, select_split, validate_cohort, write_json,
)
from lab.evidence import reserve_directory, sha256


def rank(seed, text):
    return hashlib.sha256((str(seed) + text).encode('utf-8')).hexdigest()


def allocate_development(records, spec):
    """Relabel all previously observed specialist data by fixed article rank."""
    names = ('train', 'calibration')
    sizes = {name: next(s['articles'] for s in spec['splits'] if s['name'] == name) for name in names}
    titles = sorted({row['source_title'] for row in records}, key=lambda title: rank(spec['seed'], title))
    if len(titles) != sum(sizes.values()):
        raise ValueError('Old development pool does not have exactly the declared article count')
    if len(records) != len({row['id'] for row in records}):
        raise ValueError('Duplicate ID in old development pool')
    result = {}; first = 0
    for name in names:
        chosen = titles[first:first + sizes[name]]; first += sizes[name]
        selected = sorted([dict(row, split=name) for row in records if row['source_title'] in chosen],
                          key=lambda row: row['id'])
        validate_cohort(selected, chosen, spec, name, {key: [] for key in KEYS})
        result[name] = dict(records=selected, titles=chosen, source='previously_observed_specialist_development')
    # Reuse the four-axis rule as a check, not as a reason to silently drop rows.
    blocked = extend_exclusions({key: [] for key in KEYS}, result['train']['records'])
    validate_cohort(result['calibration']['records'], result['calibration']['titles'],
                    spec, 'calibration', blocked)
    return result


def prepare(raw, spec):
    if ([s['name'] for s in spec['splits']] != ['train', 'calibration', 'evaluation'] or
            any(type(s['articles']) is not int or s['articles'] <= 0 for s in spec['splits']) or
            type(spec['seed']) is not int or type(spec['per_class_per_article']) is not int or
            spec['per_class_per_article'] <= 0 or type(spec['max_context_chars']) is not int or
            spec['max_context_chars'] <= 0):
        raise ValueError('Invalid frozen risk-head allocation protocol')
    exclusions = read_json(spec['exclusions'])
    if sha256(spec['exclusions']) != spec['exclusions_sha256']:
        raise ValueError('Exclusion inventory identity changed')
    for path, expected in exclusions['local_sources_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Historical source identity changed: ' + path)
    merged = read_json(exclusions['prior_inventory'])
    historical = []
    for path in exclusions['historical_datasets']:
        rows = read_rows(path); historical.extend(rows); merged = extend_exclusions(merged, rows)
    if any(merged[key] != exclusions[key] for key in KEYS):
        raise ValueError('Incomplete cumulative exclusion inventory')
    if spec['development_sources'] != exclusions['historical_datasets']:
        raise ValueError('Development source coverage differs from frozen specialist pool')
    result = allocate_development(historical, spec)
    records, titles, eligible = select_split(raw, spec, exclusions, 'evaluation')
    result['evaluation'] = dict(records=records, titles=titles, eligible_articles=eligible,
                                source='remaining_public_squad_dev_articles')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--spec', type=Path, default=Path('configs/qa-risk/selection.json'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(); spec = read_json(args.spec)
    if sha256(args.raw) != spec['source_sha256']:
        raise ValueError('Raw SQuAD source identity changed')
    result = prepare(read_json(args.raw), spec)
    out = reserve_directory(args.output_dir); manifests = {}
    for split, cohort in result.items():
        folder = reserve_directory(out / split)
        (folder/'data.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'
                                              for row in cohort['records']), encoding='utf-8')
        manifest = dict(split=split,n=len(cohort['records']),titles=cohort['titles'],
                        families=len({row['family_id'] for row in cohort['records']}),
                        unanswerable=sum(row['is_impossible'] for row in cohort['records']),
                        data_sha256=sha256(folder/'data.jsonl'),selection_sha256=sha256(args.spec),
                        exclusions_sha256=spec['exclusions_sha256'],source=cohort['source'],
                        eligible_articles=cohort.get('eligible_articles'),license=spec['license'],scope=spec['scope'])
        write_json(folder/'manifest.json',manifest); manifests[split]=manifest
    write_json(out/'manifest.json',dict(selection=spec,selection_sha256=sha256(args.spec),splits=manifests,
                                      model_outputs_read_by_selector=False,
                                      cross_split_disjoint=['source_title','id','normalized context','normalized question']))
    sources=('experiments/prepare_qa_risk_data.py','experiments/prepare_qa_remediation.py','lab/evidence.py')
    write_json(out/'command.json',dict(argv=[sys.executable,*sys.argv],source_sha256=sha256(args.raw),
               selector_sources_sha256={p:sha256(p) for p in sources},selection_sha256=sha256(args.spec),
               exclusions_sha256=spec['exclusions_sha256']))
    print(json.dumps({name:{key:row[key] for key in ('n','titles','families','data_sha256','eligible_articles')}
                      for name,row in manifests.items()},ensure_ascii=False))


if __name__=='__main__':main()
