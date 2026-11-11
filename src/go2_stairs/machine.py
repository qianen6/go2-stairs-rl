"""Machine-specific locations, overridable with environment variables.

GO2_RUNTIME_CWD    ASCII working directory for Isaac Sim child processes (default C:\\rl-runtimes if it exists,
                   otherwise the repository root). Kit on Windows misbehaves when started from non-ASCII paths.
GO2_ISAACLAB_ROOT  Isaac Lab source checkout used to record the runtime revision (default C:/rl-runtimes/IsaacLab).
"""
import importlib.metadata
import os
from pathlib import Path
import subprocess

REFERENCE_ISAACLAB_REVISION = '37ddf626871758333d6ed89cf64ad702aef127d0'  # Isaac Lab v2.3.2, as in runtime-lock.json
_DEFAULT_RUNTIME = Path(r'C:\rl-runtimes')


def repo_root():
    return Path(__file__).resolve().parents[2]


def runtime_cwd():
    value = os.environ.get('GO2_RUNTIME_CWD')
    if value:
        return Path(value)
    return _DEFAULT_RUNTIME if _DEFAULT_RUNTIME.is_dir() else repo_root()


def isaaclab_root():
    value = os.environ.get('GO2_ISAACLAB_ROOT')
    path = Path(value) if value else _DEFAULT_RUNTIME/'IsaacLab'
    return path if path.is_dir() else None


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


def runtime_identity(strict=False):
    """Record the software stack. Mismatch with the reference stack is a warning unless strict."""
    packages = {}
    for name in ('isaaclab', 'isaacsim', 'torch', 'rsl-rl-lib', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = isaaclab_root(); revision = dirty = None
    if root is not None:
        try:
            revision = _git(root, 'rev-parse', 'HEAD'); dirty = bool(_git(root, 'status', '--porcelain'))
        except (OSError, subprocess.CalledProcessError):
            pass
    matches = revision == REFERENCE_ISAACLAB_REVISION and dirty is False
    identity = dict(isaaclab_root=str(root) if root else None, isaaclab_revision=revision, isaaclab_dirty=dirty,
                    packages=packages, matches_reference_runtime=matches)
    if strict and not matches:
        raise ValueError(f'runtime differs from the reference Isaac Lab v2.3.2 clean checkout: {identity}')
    return identity
