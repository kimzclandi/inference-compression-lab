import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from experiments.verify_qwen_request_scheduling import verify
from lab.request_scheduling import AdmissionQueue, Request

class SchedulingEvidenceTests(unittest.TestCase):
    def test_archived_evidence_and_preserved_failed_gate(self):
        result = verify()
        self.assertEqual(result['timed_requests'], 240)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['traces']['burst']['accepted'])
        self.assertEqual([k for k, v in result['traces']['staggered']['gates'].items() if not v], ['mean_ttft'])
        self.assertEqual(result, json.loads(Path('results/qwen-request-scheduling-v1/summary.json').read_text()))

    def test_rejects_edited_timing_and_rehashed_receipt(self):
        root = Path('results/qwen-request-scheduling-v1')
        for altered in ('timing', 'receipt', 'source'):
            with self.subTest(altered=altered), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'evidence'
                shutil.copytree(root, target)
                path = target / {'timing': 'trials.json', 'receipt': 'run.json',
                                 'source': 'source/lab/request_scheduling.py'}[altered]
                if altered == 'source':
                    path.write_text(path.read_text() + '\n# changed\n')
                else:
                    data = json.loads(path.read_text())
                    if altered == 'timing': data[0]['rows'][0]['ttft_s'] = 0
                    else: data['artifacts'] = {}
                    path.write_text(json.dumps(data))
                with self.assertRaises(AssertionError): verify(target)

    def test_backdated_arrival_cannot_hide_saturated_request(self):
        q = AdmissionQueue([Request('old', 1, 1000, 32)] +
                           [Request(str(i), 2, 1, 1) for i in range(4)], 'short_budget')
        for _ in range(3): q.pop_ready(2)
        q.submit(Request('backdated', 0, 5000, 32))
        self.assertEqual(q.pop_ready(2).request_id, 'old')
        self.assertLessEqual(max(q.bypasses.values()), 3)

    def test_optimized_python_fails_closed(self):
        result = subprocess.run([sys.executable, '-O', '-m', 'experiments.verify_qwen_request_scheduling'],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('optimized Python is unsupported', result.stderr)

    def test_live_submissions_respect_arrival_and_duplicate_identity(self):
        q = AdmissionQueue([], 'short_budget')
        q.submit(Request('a', 1, 100, 32))
        q.submit(Request('b', 2, 1, 1))
        self.assertEqual(q.pop_ready(1.5).request_id, 'a')
        self.assertIsNone(q.pop_ready(1.5))
        self.assertEqual(q.pop_ready(2).request_id, 'b')
        with self.assertRaises(ValueError): q.submit(Request('a', 3, 1, 1))
