import unittest
from lab.benchmark import measure, percentile
from lab.quantization import calibrate, fake_quantize, errors


class QuantizationTests(unittest.TestCase):
    def test_zero_and_saturation(self):
        config = calibrate([-1, 0, 1])
        self.assertEqual(fake_quantize([-2, 0, 2], config), [-1, 0, 1])
        self.assertEqual(fake_quantize([0], calibrate([0, 0])), [0])

    def test_in_range_error_bound(self):
        values = [i / 100 for i in range(-100, 101)]
        config = calibrate(values)
        self.assertLessEqual(errors(values, fake_quantize(values, config))['max_abs_error'],
                             config['scale'] / 2 + 1e-12)

    def test_invalid_data(self):
        for data in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                calibrate(data)
        with self.assertRaises(ValueError):
            errors([1], [])


class TimingTests(unittest.TestCase):
    def test_gpu_requires_sync(self):
        with self.assertRaises(ValueError):
            measure(lambda: None, device='cuda')

    def test_sample_count_and_sync(self):
        calls, syncs = [], []
        result = measure(lambda: calls.append(1), warmup=2, samples=3,
                         repeats=2, device='gpu', synchronize=lambda: syncs.append(1))
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(syncs), 14)
        self.assertEqual(len(result['repeats']), 2)
        self.assertTrue(all(x >= 0 for r in result['repeats'] for x in r['samples_ms']))

    def test_percentile(self):
        self.assertEqual(percentile([0, 100], 95), 95)


if __name__ == '__main__':
    unittest.main()
