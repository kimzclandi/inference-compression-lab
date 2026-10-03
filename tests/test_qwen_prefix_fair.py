"""Offline evidence checks, including rejection of incomplete or corrupted records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.verify_qwen_prefix_fair import verify
from lab.prefix_cache import PrefixCache

ROOT = Path(__file__).resolve().parents[1]


class FairEvidenceTests(unittest.TestCase):
    def test_saved_run_recomputes(self):
        folder = ROOT/'results/qwen-prefix-fair-v1'
        self.assertEqual(verify(folder), json.loads((folder/'summary.json').read_text()))

    def invalid(self, mutate):
        source = ROOT/'results/qwen-prefix-fair-v1'
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for name in ('manifest.json', 'timings.json', 'workloads.json', 'hot-preparations.json'):
                value = json.loads((source/name).read_text())
                if name=='timings.json': mutate(value)
                (folder/name).write_text(json.dumps(value))
            with self.assertRaises(ValueError): verify(folder)

    def test_rejects_missing_cells(self):
        self.invalid(lambda rows: rows.pop())

    def test_rejects_token_mismatch(self):
        def mutate(rows):
            row = next(r for r in rows if r['mode']=='direct_segmented')
            row['outputs'][0]['token_ids'][0] += 1
        self.invalid(mutate)

    def test_rejects_hidden_copy_in_direct_path(self):
        def mutate(rows):
            row = next(r for r in rows if r['mode']=='direct_segmented' and r['profile'])
            phases = row['outputs'][0]['phases']
            phases['snapshot_copy_seconds'] = phases['prefix_prefill_seconds'] / 2
            phases['prefix_prefill_seconds'] /= 2
        self.invalid(mutate)

    def test_profile_clone_override_preserves_isolation(self):
        store = PrefixCache('fixed-model', copy.deepcopy)
        profile = {}
        clones = []
        def clone(snapshot):
            clones.append(copy.deepcopy(snapshot))
            return clones[-1]
        first, _ = store.acquire([1, 2], lambda _: ([[1, 2]], 8), clone=clone, profile=profile)
        first[0].append(99)
        second, status = store.acquire([1, 2], lambda _: self.fail('Hit rebuilt'), clone=clone, profile=profile)
        self.assertEqual(second, [[1, 2]])
        self.assertEqual(status, 'hit')
        self.assertEqual(len(clones), 2)
        self.assertGreaterEqual(profile['lookup_seconds'], 0)
