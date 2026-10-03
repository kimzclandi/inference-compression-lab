"""Educational narrow-range symmetric fake quantization, not integer inference."""
import math


def calibrate(values, bits=8):
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError('bits must be an integer from 2 to 16')
    values = list(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('finite nonempty calibration data required')
    qmax = 2 ** (bits - 1) - 1
    bound = max(abs(x) for x in values)
    return {'scale': bound / qmax if bound else 1.0,
            'qmin': -qmax, 'qmax': qmax, 'zero_point': 0,
            'rounding': 'ties-to-even', 'bits': bits}


def fake_quantize(values, config):
    scale = config['scale']
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite scale required')
    restored = []
    for x in values:
        if not math.isfinite(x):
            raise ValueError('finite evaluation values required')
        q = min(config['qmax'], max(config['qmin'], round(x / scale)))
        restored.append(q * scale)
    return restored


def errors(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError('equal nonempty inputs required')
    differences = [abs(a - b) for a, b in zip(reference, candidate)]
    if any(not math.isfinite(x) for x in differences):
        raise ValueError('finite inputs required')
    return {'mse': sum(x*x for x in differences) / len(differences),
            'max_abs_error': max(differences)}
