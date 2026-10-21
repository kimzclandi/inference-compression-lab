"""Stdlib-only evidence fixtures: no model execution or performance claims."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments import verify_qwen_demand_prefill as verifier


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def seal(folder):
    save(folder / 'checksums.json', {p.relative_to(folder).as_posix(): verifier.sha(p)
         for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'checksums.json'})


def fixture(root):
    source_root = verifier.ROOT
    for name in list(verifier.SOURCES) + [verifier.PROTOCOL,
            'configs/qwen-prefix/qa-dev.jsonl', 'configs/qwen-prefix/source-qa-protocol.json']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    spec = verifier.read(root / verifier.PROTOCOL)
    config = dict(model_type='qwen2', num_hidden_layers=24, hidden_size=896,
                  num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
                  tie_word_embeddings=True, use_sliding_window=False, eos_token_id=151645)
    state = dict(status='complete', spec_sha256=verifier.PROTOCOL_SHA256,
                 source_sha256={name: verifier.sha(root / name) for name in verifier.SOURCES},
                 model_files_sha256=spec['model_files_sha256'], versions=spec['required_versions'],
                 upstream_source_sha256=spec['upstream_source_sha256'], model_config=config,
                 git_head='1' * 40, git_status='', platform='synthetic-test-fixture', python='synthetic',
                 device={'device_name': 'synthetic-not-real-benchmark'}, default_device='Device(gpu, 0)', pid=101)
    def initialize(folder, run):
        folder.mkdir(parents=True)
        shutil.copyfile(root / verifier.PROTOCOL, folder / 'protocol.json')
        for name in verifier.SOURCES:
            target = folder / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        save(folder / 'run.json', run)
    audit, benchmark = root / 'audit', root / 'benchmark'
    initialize(audit, dict(state, scope='Consumed public functional/numerical reproduction; no timing or new quality claim'))
    work = [dict(case_id=f'{n}-{i}', length=n, prompt_index=i, token_ids=[i+1] * n,
                 input_sha256=verifier.digest([i+1] * n))
            for n in spec['benchmark']['lengths'] for i in range(2)]
    save(audit / 'workloads.json', work)
    mechanism = []
    for n in spec['audit']['shape_lengths']:
        for prior in spec['audit']['prior_cache_tokens']:
            for step in range(spec['audit']['continuation_steps'] + 1):
                offset = n + prior + step
                cache = [dict(offset=offset, tensors=[dict(shape=[1, 2, offset, 64],
                             dtype='mlx.core.float16', sha256='2' * 64) for _ in range(2)]) for _ in range(24)]
                logits = dict(top1=7, reference_top1=7, reference_top2=8, reference_margin=0.25,
                              max_abs_error=0.0, rms_error=0.0, relative_l2_error=0.0,
                              sha256_float64='3' * 64)
                arms = {}
                for mode in verifier.ARMS:
                    trace = {}
                    if step == 0:
                        shapes = {}
                        if n > 1 and mode == 'head_only':
                            shapes = dict(head_input_shape=[1, 1, 896])
                        elif n > 1 and mode == 'split_last':
                            shapes = dict(model_call_input_shapes=[[1, n-1], [1, 1]])
                        elif n > 1 and mode == 'final_query':
                            shapes = dict(query_input_shape=[1, 1, 896], key_value_input_shape=[1, n, 896],
                                attention_query_shape=[1, 14, 1, 64], attention_key_shape=[1, 2, offset, 64],
                                head_input_shape=[1, 1, 896], query_rope_offset=offset-1, key_rope_offset=prior)
                        trace = dict(mode=mode, input_shape=[1, n], prior_cache_tokens=prior,
                                     output_shape=[1, 1, 151936], cache_offsets_after=[offset] * 24, shapes=shapes)
                    arms[mode] = dict(logits=copy.deepcopy(logits), cache=copy.deepcopy(cache),
                                      cache_equal_full=True, trace=trace)
                mechanism.append(dict(length=n, prior_tokens=prior, step=step,
                    input_sha256=verifier.digest([1] * n if step == 0 else [7]), arms=arms))
    save(audit / 'mechanism.json', mechanism)
    quality = []
    for line in (root / spec['quality']['path']).read_text().splitlines():
        row = json.loads(line)
        em = f1 = float(row['is_impossible'])
        prediction = dict(token_ids=[9, 151645], stop_reason='eos', prediction='NO_ANSWER', em=em, f1=f1)
        quality.append(dict(id=row['id'], input_token_ids=[1, 2], input_sha256=verifier.digest([1, 2]),
                            arms={mode: copy.deepcopy(prediction) for mode in verifier.ARMS}))
    save(audit / 'quality.json', quality)
    seal(audit)
    benchmark.mkdir()
    workers = []
    for round_id, order in enumerate(spec['benchmark']['order']):
        for mode in order:
            folder = f'{round_id}-{mode}'
            records = []
            for row in work:
                ttft = {'full': .12, 'head_only': .105, 'split_last': .095, 'final_query': .08}[mode]
                ttft += round_id * .001 + row['prompt_index'] * .002
                elapsed = [ttft + i * .003 for i in range(32)]
                record = {k: v for k, v in row.items() if k != 'token_ids'}
                record.update(token_ids=[7] * 32, stop_reason='token_limit', ttft_seconds=elapsed[0],
                    decode_seconds=elapsed[-1] - elapsed[0], total_seconds=elapsed[-1],
                    token_elapsed_seconds=elapsed, active_before_bytes=100, peak_active_bytes=120)
                records.append(record)
            worker = dict(state, round=round_id, mode=mode, records=records, pid=200 + len(workers))
            initialize(benchmark / folder, worker)
            (benchmark / (folder + '.log')).write_text('synthetic receipt fixture only\n')
            seal(benchmark / folder)
            workers.append(dict(round=round_id, mode=mode, folder=folder, status='complete', exit_code=0))
    save(benchmark / 'run.json', dict(status='complete', protocol=spec, spec_sha256=verifier.PROTOCOL_SHA256,
         audit_checksums_sha256=verifier.sha(audit / 'checksums.json'), source_sha256=state['source_sha256'],
         workers=workers, wall_seconds=50.0))
    seal(benchmark)
    return audit, benchmark


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = tempfile.TemporaryDirectory()
        fixture(Path(cls.original.name))

    @classmethod
    def tearDownClass(cls):
        cls.original.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'copy'
        shutil.copytree(self.original.name, self.root)
        self.audit, self.benchmark = self.root / 'audit', self.root / 'benchmark'

    def verify(self):
        return verifier.verify(self.audit, self.benchmark, root=self.root)

    def edit(self, path, function):
        value = verifier.read(path)
        function(value)
        save(path, value)

    def reseal(self):
        seal(self.audit)
        for folder in self.benchmark.iterdir():
            if folder.is_dir():
                seal(folder)
        self.edit(self.benchmark / 'run.json', lambda r: r.update(audit_checksums_sha256=verifier.sha(self.audit / 'checksums.json')))
        seal(self.benchmark)

    def test_complete_fixture_and_independent_round_arithmetic(self):
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertTrue(result['acceptance']['accepted'])
        self.assertEqual(result['process_count'], 16)
        self.assertAlmostEqual(result['measurements']['2048']['full']['ttft_seconds']['median_of_round_medians'], .1225)
        self.assertAlmostEqual(result['acceptance']['primary_comparisons']['full']['ttft_speedup'], .1225 / .0825)
        self.assertEqual(result['mechanism']['final_query']['rows'], 80)
        self.assertEqual(result['quality']['split_last']['count'], 74)

    def test_unsealed_tamper(self):
        with (self.audit / 'quality.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.verify()

    def test_resealed_false_score_rejected(self):
        self.edit(self.audit / 'quality.json', lambda r: r[0]['arms']['final_query'].update(em=.123))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'EM/F1'):
            self.verify()

    def test_real_quality_negative_is_valid_and_preserved(self):
        self.edit(self.audit / 'quality.json', lambda r: r[0]['arms']['final_query'].update(
            token_ids=[8, 151645], prediction='problem instance', em=1.0, f1=1.0))
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['accepted'])
        self.assertEqual(result['quality']['final_query']['sequence_equal_full'], 73)
        self.assertGreater(result['quality']['final_query']['em'], result['quality']['full']['em'])
        self.assertFalse(result['acceptance']['gates']['all_quality_sequences_equal_full'])

    def test_resealed_cache_equality_lie_rejected(self):
        self.edit(self.audit / 'mechanism.json', lambda r:
            r[0]['arms']['final_query']['cache'][0]['tensors'][0].update(sha256='4' * 64))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'KV equality'):
            self.verify()

    def test_honest_cache_and_top1_negative_retained(self):
        def mutate(rows):
            arm = rows[0]['arms']['final_query']
            arm['cache'][0]['tensors'][0]['sha256'] = '4' * 64
            arm['cache_equal_full'] = False
            arm['logits'].update(top1=8, max_abs_error=.3, rms_error=.1,
                                 relative_l2_error=.1, sha256_float64='5' * 64)
        self.edit(self.audit / 'mechanism.json', mutate)
        self.reseal()
        result = self.verify()
        self.assertFalse(result['acceptance']['gates']['audit_kv_equal_full'])
        self.assertFalse(result['acceptance']['gates']['audit_all_top1_equal_full'])

    def test_trace_offset_and_coverage_rejected(self):
        self.edit(self.audit / 'mechanism.json', lambda r: r[8]['arms']['final_query']['trace'].update(prior_cache_tokens=19))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'trace'):
            self.verify()

    def test_resealed_timestamp_components_rejected(self):
        self.edit(self.benchmark / '0-final_query/run.json', lambda r: r['records'][0].update(ttft_seconds=.001))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Timing components'):
            self.verify()

    def test_strong_control_failure_even_when_full_speedup_passes(self):
        for round_id in range(4):
            def mutate(run):
                for record in run['records']:
                    shift = .006
                    record['ttft_seconds'] += shift
                    record['total_seconds'] += shift
                    record['token_elapsed_seconds'] = [t + shift for t in record['token_elapsed_seconds']]
            self.edit(self.benchmark / f'{round_id}-final_query/run.json', mutate)
        # Candidate stays faster than full/head but fails to beat split by 1.05x.
        for round_id in range(4):
            def change_control(run):
                for record in run['records']:
                    shift = -.006
                    record['ttft_seconds'] += shift
                    record['total_seconds'] += shift
                    record['token_elapsed_seconds'] = [t + shift for t in record['token_elapsed_seconds']]
            self.edit(self.benchmark / f'{round_id}-split_last/run.json', change_control)
        self.reseal()
        result = self.verify()
        self.assertTrue(result['acceptance']['gates']['primary_ttft_vs_full'])
        self.assertFalse(result['acceptance']['gates']['primary_ttft_vs_split_last'])
        self.assertFalse(result['acceptance']['accepted'])

    def test_duplicate_process_rejected(self):
        self.edit(self.benchmark / '0-head_only/run.json', lambda r: r.update(pid=200))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Fresh-process'):
            self.verify()

    def test_worker_order_rejected(self):
        self.edit(self.benchmark / 'run.json', lambda r: r['workers'].reverse())
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Process order'):
            self.verify()

    def test_honest_benchmark_sequence_negative_retained(self):
        self.edit(self.benchmark / '0-final_query/run.json', lambda r: r['records'][0]['token_ids'].__setitem__(1, 8))
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['gates']['all_benchmark_sequences_equal_full'])
        self.assertEqual(result['benchmark_sequence_parity']['final_query']['sequence_equal_full'], 23)

    def test_model_identity_drift_rejected(self):
        self.edit(self.benchmark / '0-full/run.json', lambda r: r['model_files_sha256'].update(
            {'model.safetensors': '4' * 64}))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'model_files_sha256'):
            self.verify()

    def test_frozen_protocol_drift_rejected(self):
        self.edit(self.root / verifier.PROTOCOL, lambda r: r['acceptance'].update(
            min_ttft_speedup_vs_each_strong_control=1.01))
        with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
            self.verify()

    def test_same_tokens_different_decoding_rejected(self):
        self.edit(self.audit / 'quality.json', lambda r: r[0]['arms']['final_query'].update(
            prediction='another wrong answer'))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Same tokens'):
            self.verify()

    def test_upstream_drift_rejected(self):
        self.edit(self.benchmark / '0-full/run.json', lambda r: r['upstream_source_sha256'].update(
            {'mlx_lm.models.qwen2': '4' * 64}))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'upstream_source_sha256'):
            self.verify()

    def test_current_source_drift_rejected(self):
        with (self.root / verifier.SOURCES[0]).open('a') as stream:
            stream.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Current source'):
            self.verify()

    def test_extra_file_rejected_even_when_resealed(self):
        (self.audit / 'unexpected.json').write_text('{}')
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.verify()

    def test_nonfinite_and_duplicate_keys_rejected(self):
        for text in ['{"x": NaN}', '{"x": 1e999}', '{"x": 1, "x": 2}']:
            with self.subTest(text=text):
                path = self.root / 'bad.json'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    verifier.read(path)

    def test_scoring_matches_normalization_and_abstention_boundary(self):
        self.assertEqual(verifier.pair_score('The input, string!', 'input string'), (1.0, 1.0))
        self.assertEqual(verifier.pair_score('input', 'input string'), (0.0, 2 / 3))
        self.assertEqual(verifier.quality_score({'is_impossible': True}, ''), (0.0, 0.0))
        self.assertEqual(verifier.quality_score({'is_impossible': True}, 'NO_ANSWER'), (1.0, 1.0))


if __name__ == '__main__':
    unittest.main()
