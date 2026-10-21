"""Synthetic receipts test offline verification; these are not GPU evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments import verify_metal_residual_rmsnorm as verifier


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def seal(folder):
    save(folder / 'checksums.json', {p.relative_to(folder).as_posix(): verifier.sha(p)
         for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'checksums.json'})


def fixture(root):
    for name in list(verifier.SOURCES) + [verifier.PROTOCOL,
            'configs/qwen-prefix/qa-dev.jsonl', 'configs/qwen-prefix/source-qa-protocol.json']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verifier.ROOT / name, target)
    spec = verifier.read(root / verifier.PROTOCOL)
    config = dict(model_type='qwen2', num_hidden_layers=24, hidden_size=896,
                  num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
                  tie_word_embeddings=True, use_sliding_window=False, eos_token_id=151645)
    state = dict(status='complete', spec_sha256=verifier.PROTOCOL_SHA256,
                 source_sha256={name: verifier.sha(root / name) for name in verifier.SOURCES},
                 model_files_sha256=spec['model_files_sha256'], versions=spec['required_versions'],
                 upstream_source_sha256=spec['upstream_source_sha256'], model_config=config,
                 runtime_binary_sha256={'core.so': 'a' * 64},
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
    initialize(audit, dict(state, scope='Synthetic fixture, not GPU execution'))
    work = [dict(case_id=f'{n}-{i}', length=n, prompt_index=i, token_ids=[i+1] * n,
                 input_sha256=verifier.digest([i+1] * n))
            for n in spec['benchmark']['lengths'] for i in range(2)]
    save(audit / 'workloads.json', work)
    numerical = []
    for dtype in spec['audit']['dtypes']:
        for n in spec['audit']['rows']:
            for width in spec['audit']['widths']:
                for pattern in spec['audit']['patterns']:
                    arm = dict(finite=True, residual_exact=True, within_tolerance=True,
                               max_abs_error=0.0, max_ulp_error=0, max_tolerance_ratio=0.0,
                               residual_sha256='8' * 64, output_sha256='9' * 64)
                    numerical.append(dict(dtype=dtype, rows=n, width=width, pattern=pattern,
                        input_sha256=['6' * 64] * 3, arms={mode: copy.deepcopy(arm) for mode in verifier.ARMS}))
    save(audit / 'numerical.json', numerical)
    mechanism = []
    for n in spec['audit']['model_lengths']:
        for prior in spec['audit']['prior_cache_tokens']:
            for step in range(spec['audit']['continuation_steps'] + 1):
                offset = n + prior + step
                cache = [dict(offset=offset, tensors=[dict(shape=[1, 2, offset, 64],
                             dtype='mlx.core.float16', sha256='2' * 64) for _ in range(2)]) for _ in range(24)]
                logits = dict(top1=7, reference_top1=7, reference_top2=8, reference_margin=0.25,
                              max_abs_error=0.0, rms_error=0.0, relative_l2_error=0.0,
                              sha256_float64='3' * 64)
                upstream = dict(logits=copy.deepcopy(logits), cache=copy.deepcopy(cache))
                arms = {}
                for mode in verifier.ARMS:
                    length = n if step == 0 else 1
                    trace = dict(mode=mode, input_shape=[1, length],
                        prior_cache_tokens=prior if step == 0 else prior+n+step-1,
                        output_shape=[1, 1, 151936], cache_offsets_after=[offset] * 24,
                        residual_norm_pair_calls=24, gpu_profiler=False,
                        residual_norm_pairs=[dict(layer=i, shape=[1, length, 896],
                                                 dtype='mlx.core.float16', mode=mode) for i in range(24)])
                    arms[mode] = dict(copy.deepcopy(upstream), cache_equal_native=True, trace=trace)
                ids = [1] * n if step == 0 else [7]
                mechanism.append(dict(length=n, prior_tokens=prior, step=step,
                    input_token_ids=ids, input_sha256=verifier.digest(ids), arms=arms,
                    upstream=upstream, upstream_native_equal=True))
    save(audit / 'mechanism.json', mechanism)
    quality = []
    for line in (root / spec['quality']['path']).read_text().splitlines():
        row = json.loads(line)
        em = f1 = float(row['is_impossible'])
        prediction = dict(token_ids=[9, 151645], stop_reason='eos', prediction='NO_ANSWER', em=em, f1=f1)
        quality.append(dict(id=row['id'], input_token_ids=[1, 2], input_sha256=verifier.digest([1, 2]),
                            arms={mode: copy.deepcopy(prediction) for mode in verifier.ARMS}))
    save(audit / 'quality.json', quality)
    for name in verifier.AUDIT_EXTRA_FILES:
        target = audit / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('synthetic graph fixture, not GPU trace\n')
    seal(audit)
    benchmark.mkdir()
    workers = []
    for round_id, order in enumerate(spec['benchmark']['order']):
        for mode in order:
            folder = f'{round_id}-{mode}'
            records = []
            for row in work:
                ttft = {'native': .12, 'compiled': .11, 'metal': .09}[mode]
                ttft += round_id * .001 + row['prompt_index'] * .002
                elapsed = [ttft + i * .003 for i in range(32)]
                record = {k: v for k, v in row.items() if k != 'token_ids'}
                record.update(token_ids=[7] * 32, stop_reason='token_limit', ttft_seconds=elapsed[0],
                    decode_seconds=elapsed[-1] - elapsed[0], total_seconds=elapsed[-1],
                    token_elapsed_seconds=elapsed, active_before_bytes=100, peak_active_bytes=120)
                records.append(record)
            micro = []
            for n in spec['benchmark']['micro_rows']:
                seconds = {'native': .00003, 'compiled': .000025, 'metal': .00002}[mode]
                seconds += round_id * .000001
                samples = [dict(total_seconds=50 * seconds, per_call_seconds=seconds) for _ in range(20)]
                micro.append(dict(rows=n, width=896, input_sha256=['7' * 64] * 3,
                                  samples=samples, batch_calls=50))
            worker = dict(state, round=round_id, mode=mode, records=records, micro=micro, pid=200 + len(workers))
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

    def test_complete_fixture_and_independent_arithmetic(self):
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertTrue(result['acceptance']['accepted'])
        self.assertEqual(result['process_count'], 9)
        self.assertEqual(result['numerical']['metal']['cases'], 300)
        self.assertEqual(result['mechanism']['arms']['metal']['rows'], 32)
        self.assertAlmostEqual(result['model_measurements']['2048']['native']['ttft_seconds']['median_of_round_medians'], .122)
        self.assertAlmostEqual(result['micro_measurements']['2048']['metal']['median_of_round_medians'], .000021)
        self.assertAlmostEqual(result['acceptance']['micro_primary_comparisons']['compiled']['micro_speedup'], 26/21)

    def test_unsealed_tamper(self):
        with (self.audit / 'quality.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.verify()

    def test_source_drift(self):
        with (self.root / verifier.SOURCES[0]).open('a') as stream:
            stream.write('\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'Current source differs'):
            self.verify()

    def test_protocol_drift(self):
        self.edit(self.root / verifier.PROTOCOL, lambda r: r['acceptance'].update(min_micro_speedup_vs_each_control=1.0))
        with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
            self.verify()

    def test_resealed_false_quality_score(self):
        self.edit(self.audit / 'quality.json', lambda r: r[0]['arms']['metal'].update(em=.123))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'EM/F1'):
            self.verify()

    def test_quality_negative_retained(self):
        self.edit(self.audit / 'quality.json', lambda r: r[0]['arms']['metal'].update(
            token_ids=[8, 151645], prediction='problem instance', em=1.0, f1=1.0))
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['accepted'])
        self.assertFalse(result['acceptance']['gates']['all_quality_sequences_equal_native'])

    def test_numerical_tolerance_boolean_lie(self):
        self.edit(self.audit / 'numerical.json', lambda r: r[0]['arms']['metal'].update(
            max_abs_error=.01, max_tolerance_ratio=2.0, output_sha256='a' * 64))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'tolerance boolean'):
            self.verify()

    def test_numerical_negative_retained(self):
        self.edit(self.audit / 'numerical.json', lambda r: r[0]['arms']['metal'].update(
            max_abs_error=.01, max_tolerance_ratio=2.0, within_tolerance=False, output_sha256='a' * 64))
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['gates']['all_micro_numerical_within_tolerance'])

    def test_residual_rounding_negative_retained(self):
        self.edit(self.audit / 'numerical.json', lambda r: r[0]['arms']['metal'].update(
            residual_exact=False, residual_sha256='b' * 64))
        self.reseal()
        self.assertFalse(self.verify()['acceptance']['gates']['all_residual_outputs_exact'])

    def test_numerical_coverage_and_bool_type(self):
        self.edit(self.audit / 'numerical.json', lambda r: r[0]['arms']['metal'].update(finite=1))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.verify()

    def test_resealed_cache_equality_lie(self):
        self.edit(self.audit / 'mechanism.json', lambda r:
            r[0]['arms']['metal']['cache'][0]['tensors'][0].update(sha256='4' * 64))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'KV equality'):
            self.verify()

    def test_model_logit_bound_negative_retained(self):
        self.edit(self.audit / 'mechanism.json', lambda r: r[0]['arms']['metal']['logits'].update(
            max_abs_error=.125, rms_error=.01, relative_l2_error=.01, sha256_float64='4' * 64))
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['gates']['model_logits_within_bound'])
        self.assertTrue(result['acceptance']['gates']['all_mechanism_top1_equal_native'])

    def test_upstream_equality_lie(self):
        self.edit(self.audit / 'mechanism.json', lambda r:
            r[0]['upstream']['cache'][0]['tensors'][0].update(sha256='4' * 64))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Upstream equality'):
            self.verify()

    def test_teacher_forcing_change(self):
        self.edit(self.audit / 'mechanism.json', lambda r: r[1].update(
            input_token_ids=[8], input_sha256=verifier.digest([8])))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'teacher-forced'):
            self.verify()

    def test_micro_per_call_arithmetic(self):
        self.edit(self.benchmark / '0-metal/run.json', lambda r:
            r['micro'][0]['samples'][0].update(per_call_seconds=.001))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'per-call arithmetic'):
            self.verify()

    def test_micro_input_drift(self):
        self.edit(self.benchmark / '0-metal/run.json', lambda r:
            r['micro'][0].update(input_sha256=['a' * 64] * 3))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'input identity'):
            self.verify()

    def test_micro_sample_missing(self):
        self.edit(self.benchmark / '0-metal/run.json', lambda r: r['micro'][0]['samples'].pop())
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'sample coverage'):
            self.verify()

    def test_resealed_token_timestamps(self):
        self.edit(self.benchmark / '0-metal/run.json', lambda r: r['records'][0].update(ttft_seconds=.001))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Timing components'):
            self.verify()

    def test_model_same_metric_must_beat_both_controls(self):
        # metal beats native only on TTFT and compiled only on decode.
        for round_id in range(3):
            for mode in verifier.ARMS:
                def mutate(run):
                    for record in run['records']:
                        ttft = {'native': .104, 'compiled': .100, 'metal': .102}[run['mode']]
                        decode = {'native': .100, 'compiled': .104, 'metal': .102}[run['mode']]
                        elapsed = [ttft + i * decode / 31 for i in range(32)]
                        record.update(ttft_seconds=ttft, decode_seconds=decode,
                                      total_seconds=elapsed[-1], token_elapsed_seconds=elapsed)
                self.edit(self.benchmark / f'{round_id}-{mode}/run.json', mutate)
        self.reseal()
        self.assertFalse(self.verify()['acceptance']['gates']['same_model_metric_beats_both_controls'])

    def test_valid_micro_performance_negative_retained(self):
        for round_id in range(3):
            def mutate(run):
                for row in run['micro']:
                    for sample in row['samples']:
                        sample['total_seconds'] *= 2
                        sample['per_call_seconds'] *= 2
            self.edit(self.benchmark / f'{round_id}-metal/run.json', mutate)
        self.reseal()
        result = self.verify()
        self.assertTrue(result['evidence_valid'])
        self.assertFalse(result['acceptance']['accepted'])
        self.assertFalse(result['acceptance']['gates']['micro_speedup_vs_compiled'])

    def test_fresh_process_pid_reuse(self):
        self.edit(self.benchmark / '0-metal/run.json', lambda r: r.update(pid=200))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Fresh-process'):
            self.verify()

    def test_worker_order(self):
        self.edit(self.benchmark / 'run.json', lambda r: r['workers'].reverse())
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Process order'):
            self.verify()

    def test_budget_exceeded(self):
        self.edit(self.benchmark / 'run.json', lambda r: r.update(wall_seconds=901))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            self.verify()

    def test_measured_time_includes_micro(self):
        self.edit(self.benchmark / 'run.json', lambda r: r.update(wall_seconds=.1))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'micro and model durations'):
            self.verify()

    def test_nonfinite_and_duplicate_json_rejected(self):
        path = self.root / 'invalid.json'
        for text in ('{"a":NaN}', '{"a":1e999}', '{"a":1,"a":2}'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                verifier.read(path)

    def test_unknown_evidence_and_symlink_rejected(self):
        (self.audit / 'extra.json').write_text('{}')
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'file coverage'):
            self.verify()
        (self.audit / 'extra.json').unlink()
        (self.audit / 'link').symlink_to(self.audit / 'run.json')
        with self.assertRaisesRegex(ValueError, 'Symlinks'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
