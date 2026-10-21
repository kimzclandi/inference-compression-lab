"""Missing provenance, resigned claims and runtime drift must fail acceptance."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import qa_risk_pruning as runner
from experiments import verify_qa_risk_pruning as acceptance
from lab.artifact_integrity import file_hashes

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def resign(folder):
    write(folder / 'checksums.json', file_hashes(folder, exclude=('checksums.json',)))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = acceptance.read(ROOT / 'configs/qa-risk-pruning/study.json')
        names = set(spec['inputs_sha256']) | set(acceptance.FROZEN_SOURCES)
        names.add('configs/qa-risk-pruning/study.json')
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.folder = self.root / 'results/qa-risk-pruning-v1'
        for name in ('audit-final', 'benchmark'):
            shutil.copytree(ROOT / 'results/qa-risk-pruning-v1' / name, self.folder / name)

    def test_complete_fixture_passes(self):
        self.assertEqual(acceptance.verify(self.root)['status'], 'pass')

    def test_empty_resigned_source_bindings_rejected(self):
        audit, bench = self.folder / 'audit-final', self.folder / 'benchmark'
        for folder in (audit, bench, *(bench / str(i) for i in range(6))):
            state = acceptance.read(folder / 'run.json')
            state['source_sha256'] = {}
            write(folder / 'run.json', state)
        for i in range(6):
            resign(bench / str(i))
        resign(audit)
        resign(bench)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            acceptance.verify(self.root)

    def test_resigned_quality_claim_rejected(self):
        folder = self.folder / 'audit-final'
        value = acceptance.read(folder / 'summary.json')
        value['historical_quality_replayed']['selective']['accepted_correct'] = 128
        write(folder / 'summary.json', value)
        resign(folder)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            acceptance.verify(self.root)

    def test_resigned_missing_or_unknown_replay_identity_rejected(self):
        folder = self.folder / 'audit-final'
        value = acceptance.read(folder / 'records.json')
        value[0]['id'] = 'unknown-id'
        write(folder / 'records.json', value)
        resign(folder)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            acceptance.verify(self.root)

    def test_resigned_timing_count_and_omitted_measurement_rejected(self):
        bench = self.folder / 'benchmark'
        run = acceptance.read(bench / '0' / 'measurements.json')
        run['records'].pop()
        write(bench / '0' / 'measurements.json', run)
        summary = acceptance.read(bench / 'summary.json')
        summary['measured_requests'] = 999999
        write(bench / 'summary.json', summary)
        resign(bench / '0')
        resign(bench)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            acceptance.verify(self.root)

    def test_changed_current_extractor_rejected(self):
        path = self.root / 'lab/qa_risk_pruning.py'
        path.write_text(path.read_text() + '\n# drift\n')
        with self.assertRaisesRegex(ValueError, 'Current implementation differs'):
            acceptance.verify(self.root)

    def test_source_symlink_rejected_even_with_identical_bytes(self):
        path = self.root / 'lab/qa_risk_pruning.py'
        path.unlink()
        path.symlink_to(ROOT / 'lab/qa_risk_pruning.py')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            acceptance.verify(self.root)

    def test_edited_gate_rejected_before_running_experiment(self):
        path = self.root / 'configs/qa-risk-pruning/study.json'
        value = acceptance.read(path)
        value['performance']['full_request_speedup_min'] = .5
        write(path, value)
        with patch.object(runner, 'PROTOCOL', path):
            with self.assertRaisesRegex(ValueError, 'Fixed pruning protocol changed'):
                runner.protocol()

    def test_current_runner_changes_only_protocol_guard(self):
        old = ast.parse((ROOT / 'results/qa-risk-pruning-v1/benchmark/source/experiments/qa_risk_pruning.py').read_text())
        new = ast.parse((ROOT / 'experiments/qa_risk_pruning.py').read_text())
        def executable_without_guard(tree):
            result = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'PROTOCOL_SHA256' for x in node.targets):
                    continue
                node = deepcopy(node)
                if isinstance(node, ast.FunctionDef) and node.name == 'protocol' and isinstance(node.body[0], ast.If):
                    node.body.pop(0)
                result.append(ast.dump(node, include_attributes=False))
            return result
        self.assertEqual(executable_without_guard(new), executable_without_guard(old))


if __name__ == '__main__':
    unittest.main()
