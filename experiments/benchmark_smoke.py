import json
from pathlib import Path
from lab.benchmark import measure


def main():
    result = measure(lambda: sum(range(1000)), scope='synthetic CPU sum')
    result.update({'synthetic': True, 'model': None,
                   'claim': 'Timer smoke test only; not a model benchmark.'})
    Path('runs').mkdir(exist_ok=True)
    Path('runs/synthetic-timer.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print('Synthetic timer smoke test saved; no model performance measured.')


if __name__ == '__main__':
    main()
