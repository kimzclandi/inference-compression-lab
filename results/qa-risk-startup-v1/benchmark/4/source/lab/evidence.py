"""Small, dependency-free provenance and no-clobber helpers."""
import hashlib
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve_directory(path):
    """Exclusive creation: existing directories (even empty) are never reused."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_record():
    files = sorted([*Path('experiments').glob('*.py'), *Path('lab').glob('*.py'),
                    *Path('.').glob('requirements*.txt')])
    return {'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
            'argv': [sys.executable, *sys.argv],
            'source_sha256': {str(p): sha256(p) for p in files}}
