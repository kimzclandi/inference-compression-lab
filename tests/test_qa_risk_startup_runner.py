"""Synthetic startup study control-flow tests; never launch a measured process."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments import qa_risk_startup as runner


SPEC = json.loads(runner.PROTOCOL.read_text())


def measurements(reference=(3., 3., 2.), pruned=(2., 2., 2.)):
    values = {'reference': iter(reference), 'pruned': iter(pruned)}
    return [dict(status='pass', arm=arm,
                 timing=dict(total_startup_seconds=next(values[arm]),
                             evidence_verification_seconds=1.),
                 verification_sha256='a' * 64, responses_sha256='b' * 64,
                 packages={'numpy': 'synthetic'}) for arm in SPEC['arms']]


class StartupSummaryTests(unittest.TestCase):
    def test_exact_speed_threshold_and_two_faster_pairs_pass(self):
        result = runner.summarize(measurements(), SPEC)
        total = result['metrics']['total_startup_seconds']
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(total['medians'], {'reference': 3., 'pruned': 2.})
        self.assertEqual(total['speedup'], 1.5)
        # The middle pair is executed pruned/reference; direction stays reference/pruned.
        self.assertEqual(total['paired_speedups'], [1.5, 1.5, 1.])
        self.assertEqual(total['pairs_faster'], 2)
        self.assertEqual(result['scope'], SPEC['performance']['timing'])
        self.assertFalse(result['new_quality_confirmation'])

    def test_all_pairs_faster_do_not_override_median_speed_gate(self):
        result = runner.summarize(measurements((1.4, 1.4, 1.4), (1., 1., 1.)), SPEC)
        self.assertEqual(result['status'], 'gate_failed')
        self.assertEqual(result['metrics']['total_startup_seconds']['pairs_faster'], 3)

    def test_median_speed_pass_does_not_override_pair_gate(self):
        result = runner.summarize(measurements((1., 6., 100.), (2., 7., 3.)), SPEC)
        self.assertEqual(result['status'], 'gate_failed')
        total = result['metrics']['total_startup_seconds']
        self.assertEqual(total['speedup'], 2.)
        self.assertEqual(total['pairs_faster'], 1)

    def test_incomplete_reordered_or_failed_records_are_rejected(self):
        short = measurements()[:-1]
        reordered = measurements()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        failed = measurements()
        failed[3]['status'] = 'failed'
        for records in (short, reordered, failed):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, 'sequence'):
                runner.summarize(records, SPEC)

    def test_verification_or_functional_disagreement_is_rejected(self):
        for field in ('verification_sha256', 'responses_sha256'):
            records = measurements()
            records[4][field] = 'c' * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'outputs differ'):
                runner.summarize(records, SPEC)

    def test_dependency_drift_is_rejected(self):
        records = measurements()
        records[-1]['packages']['numpy'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Dependency versions'):
            runner.summarize(records, SPEC)

    def test_nonpositive_nonfinite_or_out_of_budget_times_are_rejected(self):
        for field in ('total_startup_seconds', 'evidence_verification_seconds'):
            for invalid in (0., -1., float('nan'), float('inf'), -float('inf'), 900.):
                records = measurements()
                records[2]['timing'][field] = invalid
                with self.subTest(field=field, invalid=invalid), \
                     self.assertRaisesRegex(ValueError, 'Invalid startup measurement'):
                    runner.summarize(records, SPEC)


class StartupBenchmarkControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / 'benchmark'
        self.out.mkdir()
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.assets = self.root / 'unused-assets'
        self.sources = {'synthetic_source.py': 'd' * 64}
        self.records = measurements()
        self.audit_state = dict(status='pass', source_sha256=self.sources,
                                protocol_sha256=runner.PROTOCOL_SHA)
        self.write_audit()

    def write_audit(self):
        # These temporary synthetic fixtures are deliberately corrupted and rehashed.
        (self.audit / 'run.json').write_text(json.dumps(self.audit_state))
        (self.audit / 'summary.json').write_text(json.dumps({'verification_sha256': 'a' * 64}))
        (self.audit / 'checksums.json').write_text(json.dumps(
            runner.file_hashes(self.audit, exclude=('checksums.json',))))

    def synthetic_process(self, command, **kwargs):
        """Write a tiny receipt as the subprocess mock; no worker is invoked."""
        folder = Path(command[command.index('--output-dir') + 1])
        index = int(folder.name)
        folder.mkdir()
        runner.write(folder / 'run.json', self.audit_state)
        runner.write(folder / 'measurement.json', self.records[index])
        return subprocess.CompletedProcess(command, 0)

    def run_with(self, process=None, ticks=None):
        return (patch.object(runner, 'source_hashes', return_value=self.sources),
                patch.object(runner.subprocess, 'run', side_effect=process or self.synthetic_process),
                patch.object(runner.time, 'monotonic', side_effect=ticks or [1000.] * 7))

    def test_fixed_order_fresh_commands_and_shared_decreasing_budget(self):
        source, process, clock = self.run_with(ticks=[1000., 1000., 1010., 1020., 1030., 1040., 1050.])
        with source, process as launch, clock, patch('builtins.print'):
            result = runner.benchmark(SPEC, self.out, self.assets, self.audit)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(launch.call_count, 6)
        for index, (call, arm) in enumerate(zip(launch.call_args_list, SPEC['arms'])):
            command = call.args[0]
            self.assertEqual(command, [runner.sys.executable, '-B', '-m',
                'experiments.qa_risk_startup', 'worker', '--arm', arm,
                '--asset-root', str(self.assets), '--output-dir', str(self.out / str(index))])
            self.assertEqual(call.kwargs['cwd'], runner.ROOT)
            self.assertEqual(call.kwargs['timeout'], 900. - 10. * index)
            self.assertEqual(call.kwargs['stderr'], subprocess.STDOUT)
            for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                               'OMP_NUM_THREADS': '1', 'PYTHONPATH': ''}.items():
                self.assertEqual(call.kwargs['env'][key], value)
        self.assertEqual(runner.read(self.out / 'summary.json'), result)

    def test_old_source_failed_audit_or_changed_protocol_launches_nothing(self):
        for field, invalid in (('source_sha256', {'old.py': 'e' * 64}),
                               ('status', 'failed'), ('protocol_sha256', 'f' * 64)):
            saved = deepcopy(self.audit_state)
            self.audit_state[field] = invalid
            self.write_audit()
            source, process, clock = self.run_with()
            with self.subTest(field=field), source, process as launch, clock, \
                 self.assertRaisesRegex(ValueError, 'completed full equality audit'):
                runner.benchmark(SPEC, self.out, self.assets, self.audit)
            launch.assert_not_called()
            self.audit_state = saved

    def test_budget_exhaustion_launches_nothing(self):
        source, process, clock = self.run_with(ticks=[1000., 1900.])
        with source, process as launch, clock, self.assertRaisesRegex(TimeoutError, 'budget exhausted'):
            runner.benchmark(SPEC, self.out, self.assets, self.audit)
        launch.assert_not_called()
        self.assertFalse((self.out / 'summary.json').exists())

    def test_failed_process_stops_without_retry_and_keeps_logs(self):
        def fail_second(command, **kwargs):
            if command[-1] == str(self.out / '1'):
                kwargs['stdout'].write('synthetic process failure\n')
                return subprocess.CompletedProcess(command, 7)
            return self.synthetic_process(command, **kwargs)
        source, process, clock = self.run_with(process=fail_second)
        with source, process as launch, clock, patch('builtins.print'), \
             self.assertRaisesRegex(ValueError, 'preserve worker log: 1'):
            runner.benchmark(SPEC, self.out, self.assets, self.audit)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual((self.out / '1.log').read_text(), 'synthetic process failure\n')
        self.assertTrue((self.out / '0' / 'measurement.json').is_file())
        self.assertFalse((self.out / '2.log').exists())
        self.assertFalse((self.out / 'summary.json').exists())

    def test_subprocess_timeout_propagates_without_retry(self):
        source, process, clock = self.run_with(process=subprocess.TimeoutExpired('synthetic', 900.))
        with source, process as launch, clock, self.assertRaises(subprocess.TimeoutExpired):
            runner.benchmark(SPEC, self.out, self.assets, self.audit)
        launch.assert_called_once()
        self.assertTrue((self.out / '0.log').is_file())
        self.assertFalse((self.out / 'summary.json').exists())

    def test_failed_performance_gate_is_saved_without_more_processes(self):
        self.records = measurements((1.4, 1.4, 1.4), (1., 1., 1.))
        source, process, clock = self.run_with()
        with source, process as launch, clock, patch('builtins.print'):
            result = runner.benchmark(SPEC, self.out, self.assets, self.audit)
        self.assertEqual(launch.call_count, 6)
        self.assertEqual(result['status'], 'gate_failed')
        self.assertEqual(runner.read(self.out / 'summary.json'), result)

    def test_main_preserves_failure_receipt(self):
        folder = self.root / 'failed-run'
        argv = ['qa_risk_startup', 'benchmark', '--output-dir', str(folder),
                '--asset-root', str(self.assets), '--audit-root', str(self.audit)]
        with patch.object(runner.sys, 'argv', argv), \
             patch.object(runner, 'benchmark', side_effect=ValueError('synthetic failure')) as launch, \
             self.assertRaisesRegex(ValueError, 'synthetic failure'):
            runner.main()
        launch.assert_called_once()
        state = runner.read(folder / 'run.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['error_type'], 'ValueError')
        self.assertEqual(state['error'], 'synthetic failure')
        runner.verify_hashes(folder, runner.read(folder / 'checksums.json'), exclude=('checksums.json',))

    def test_existing_output_is_never_reused(self):
        marker = self.out / 'prior-evidence.txt'
        marker.write_text('preserve this evidence')
        argv = ['qa_risk_startup', 'benchmark', '--output-dir', str(self.out)]
        with patch.object(runner.sys, 'argv', argv), patch.object(runner, 'benchmark') as launch, \
             self.assertRaises(FileExistsError):
            runner.main()
        launch.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserve this evidence')
        self.assertEqual(list(self.out.iterdir()), [marker])

    def test_changed_worker_source_protocol_or_verification_stops_immediately(self):
        for field in ('source_sha256', 'protocol_sha256', 'verification_sha256'):
            with self.subTest(field=field):
                folder = self.out / field
                folder.mkdir()
                def changed_process(command, **kwargs):
                    receipt = Path(command[-1])
                    receipt.mkdir()
                    state = deepcopy(self.audit_state)
                    record = deepcopy(self.records[0])
                    if field == 'verification_sha256':
                        record[field] = 'c' * 64
                    else:
                        state[field] = 'changed'
                    runner.write(receipt / 'run.json', state)
                    runner.write(receipt / 'measurement.json', record)
                    return subprocess.CompletedProcess(command, 0)
                source, process, clock = self.run_with(process=changed_process)
                with source, process as launch, clock, self.assertRaisesRegex(ValueError, 'Worker source'):
                    runner.benchmark(SPEC, folder, self.assets, self.audit)
                launch.assert_called_once()
                self.assertFalse((folder / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
