import unittest
from lab.interval_coverage import coverage


class IntervalCoverage(unittest.TestCase):
    def test_overlapping_nested_and_touching_intervals(self):
        result=coverage([(10,10),(12,2),(20,5),(30,2)])
        self.assertEqual(result['summed_duration_ns'],19)
        self.assertEqual(result['union_duration_ns'],17)
        self.assertEqual(result['observed_span_ns'],22)
        self.assertEqual(result['total_gap_ns'],5)
        self.assertEqual(result['gap_count'],1)
        self.assertAlmostEqual(result['interval_coverage_fraction'],17/22)

    def test_nested_last_start_is_not_last_end(self):
        self.assertEqual(coverage([(0,100),(50,1)])['observed_span_ns'],100)

    def test_zero_span_and_invalid_records(self):
        self.assertIsNone(coverage([(0,0)])['interval_coverage_fraction'])
        for rows in ([],[(1,-1)],[(-1,1)],[(True,2)],[(1,2.)]):
            with self.assertRaises(ValueError):coverage(rows)
