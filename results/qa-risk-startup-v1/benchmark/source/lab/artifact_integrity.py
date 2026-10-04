"""Portable, exact file coverage for evidence and source archives."""
from pathlib import Path, PurePosixPath
import subprocess
from lab.quantization_diagnostics import sha


def safe_path(root, name):
    p = PurePosixPath(name)
    if not name or name == '.' or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name:
        raise ValueError('Noncanonical relative artifact path: ' + name)
    result = Path(root) / name
    part = Path(root)
    for component in p.parts:
        part = part / component
        if part.is_symlink():
            raise ValueError('Symlink artifact is not supported: ' + name)
    return result


def file_hashes(root, exclude=()):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink artifact: ' + str(path))
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in exclude:
                result[name] = sha(path)
    return result


def verify_hashes(root, expected, exclude=('checksums.json', 'summary.json')):
    if not expected:
        raise ValueError('Empty checksum manifest')
    for name in expected:
        safe_path(root, name)
    actual = file_hashes(root, exclude)
    if set(actual) != set(expected):
        raise ValueError('Checksum manifest does not cover exact artifact files')
    for name, digest in expected.items():
        if actual[name] != digest:
            raise ValueError('Artifact hash mismatch: ' + name)


def git_identity(root):
    """Do not accidentally bind an unpacked archive to an unrelated parent repo."""
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        if Path(git('rev-parse', '--show-toplevel')).resolve() != root:
            raise ValueError('Different repository root')
        return {'git_head': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain')}
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {'git_head': None, 'git_status': None, 'source_form': 'archive; per-file source hashes identify code'}
