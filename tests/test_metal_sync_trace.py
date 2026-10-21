import copy
import unittest
from experiments.metal_sync_trace import select,summarize


class ProcessProjection(unittest.TestCase):
    def test_exact_pid_filter(self):
        def row(process):
            return dict(process={'fmt':process},**{'channel-name':{'fmt':'Compute'}},
                        start={'raw':'10'},duration={'raw':'3'})
        rows=select([row('python (12)'),row('python (123)')],'python (12)','gpu')
        self.assertEqual(len(rows),1)
        self.assertEqual(summarize(rows,'python (12)','gpu')['union_duration_ns'],3)

    def test_reject_duplicate_other_process_and_noncompute(self):
        row=dict(source_row_index=0,process='python (12)',category='Compute',start_ns=10,duration_ns=3)
        for rows in ([row,row],[dict(row,process='python (123)')],[dict(row,category='Blit')]):
            with self.assertRaises(ValueError):summarize(rows,'python (12)','gpu')
