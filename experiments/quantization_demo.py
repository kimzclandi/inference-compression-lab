import json
from pathlib import Path
from lab.quantization import calibrate, fake_quantize, errors


def main():
    calibration = [-2.0, -0.7, 0.0, 0.8, 2.0]
    evaluation = [-3.0, -0.5, 0.0, 0.5, 3.0]
    config = calibrate(calibration)
    output = fake_quantize(evaluation, config)
    result = {'synthetic': True, 'execution': 'Python float fake quantization',
              'model': None, 'config': config, 'reference': evaluation,
              'restored': output, 'errors': errors(evaluation, output),
              'claim': 'Numerical illustration only; no INT8 inference or speedup.'}
    Path('runs').mkdir(exist_ok=True)
    Path('runs/synthetic-quantization.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
