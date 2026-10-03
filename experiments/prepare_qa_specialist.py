"""Fix local public-benchmark QA specialist cohorts without loading a model."""
import argparse
import json
from pathlib import Path
import sys

from experiments.prepare_qa_remediation import (
    KEYS, extend_exclusions, read_json, read_rows, select_split, write_json,
)
from lab.evidence import reserve_directory, sha256


SPLITS = ('calibration', 'evaluation')


def select_cohorts(raw, spec, exclusions):
    """Preserve the old deterministic selector but name exposure accurately."""
    if ([s['name'] for s in spec['splits']] != list(SPLITS) or
            any(type(s['articles']) is not int or s['articles'] <= 0 for s in spec['splits']) or
            type(spec['per_class_per_article']) is not int or spec['per_class_per_article'] <= 0 or
            type(spec['seed']) is not int or type(spec['max_context_chars']) is not int or
            spec['max_context_chars'] <= 0):
        raise ValueError('Invalid fixed specialist selection protocol')
    result = {}; current = exclusions
    for split in spec['splits']:
        name = split['name']
        records, titles, eligible_count = select_split(raw, spec, current, name)
        result[name] = dict(records=records, titles=titles, eligible_articles=eligible_count,
                            effective_exclusion_counts={key: len(current[key]) for key in KEYS})
        current = extend_exclusions(current, records)
    return result


def checked_exclusions(spec):
    path = Path(spec['exclusions'])
    if sha256(path) != spec['exclusions_sha256']:
        raise ValueError('Exclusion inventory identity changed')
    exclusions = read_json(path)
    for local, expected in exclusions['local_sources_sha256'].items():
        if sha256(local) != expected:
            raise ValueError(f'Historical exclusion source identity changed: {local}')
    merged = read_json(exclusions['prior_inventory'])
    for local in exclusions['historical_datasets']:
        merged = extend_exclusions(merged, read_rows(local))
    if any(sorted(exclusions[key]) != merged[key] for key in KEYS):
        raise ValueError('Merged exclusion inventory is incomplete')
    return exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('configs/qa-specialist/selection.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    if sha256(args.raw) != spec['source_sha256']:
        raise ValueError('Upstream source identity changed')
    exclusions = checked_exclusions(spec)
    selected = select_cohorts(read_json(args.raw), spec, exclusions)
    out = reserve_directory(args.output_dir)
    manifests = {}
    for name, cohort in selected.items():
        folder = reserve_directory(out / name)
        (folder / 'data.jsonl').write_text(''.join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
            for row in cohort['records']), encoding='utf-8')
        manifest = dict(split=name, selection_sha256=sha256(args.spec),
                        exclusions_sha256=spec['exclusions_sha256'],
                        data_sha256=sha256(folder / 'data.jsonl'), titles=cohort['titles'],
                        n=len(cohort['records']),
                        families=len({row['family_id'] for row in cohort['records']}),
                        unanswerable=sum(row['is_impossible'] for row in cohort['records']),
                        eligible_articles=cohort['eligible_articles'],
                        effective_exclusion_counts=cohort['effective_exclusion_counts'],
                        license=spec['license'], scope=spec['scope'])
        write_json(folder / 'manifest.json', manifest)
        manifests[name] = manifest
    write_json(out / 'manifest.json', dict(
        selection=spec, selection_sha256=sha256(args.spec),
        exclusions_sha256=spec['exclusions_sha256'], splits=manifests,
        cross_split_disjoint=['source_title', 'id', 'normalized context', 'normalized question'],
        model_outputs_read=False))
    sources = ('experiments/prepare_qa_specialist.py',
               'experiments/prepare_qa_remediation.py', 'lab/evidence.py')
    write_json(out / 'command.json', dict(
        argv=[sys.executable, *sys.argv], source_sha256=sha256(args.raw),
        selector_sources_sha256={path: sha256(path) for path in sources},
        selection_sha256=sha256(args.spec), exclusions_sha256=spec['exclusions_sha256']))
    print(json.dumps({name: {'n': value['n'], 'families': value['families'], 'titles': value['titles'],
                              'data_sha256': value['data_sha256']}
                     for name, value in manifests.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
