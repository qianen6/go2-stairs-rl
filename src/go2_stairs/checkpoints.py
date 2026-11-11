"""Complete, non-overwriting training checkpoints; load only project-produced files."""

import hashlib
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch


def capture_rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    cuda = state['torch_cuda']
    if cuda and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG state requires the same number of CUDA devices')
    n = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _validate(state):
    required = {'model_state_dict', 'optimizer_state_dict', 'normalization', 'curriculum',
                'rng', 'iter', 'contract_hash', 'config'}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError('checkpoint lacks complete resumable state')
    for name in required - {'iter', 'contract_hash'}:
        if not isinstance(state[name], dict):
            raise ValueError(f'{name} must be a mapping')
    if type(state['iter']) is not int or state['iter'] < 0:
        raise ValueError('checkpoint iteration must be nonnegative')
    if not isinstance(state['contract_hash'], str) or not re.fullmatch('[0-9a-f]{64}', state['contract_hash']):
        raise ValueError('contract hash must be a lowercase SHA256 digest')
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'}.issubset(state['rng']):
        raise ValueError('checkpoint RNG state is incomplete')


def create_run_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_checkpoint(path, state):
    _validate(state)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes a complete file atomically, failing if it already exists.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, expected_contract_hash):
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    _validate(state)
    if state['contract_hash'] != expected_contract_hash:
        raise ValueError('checkpoint control contract mismatch')
    return state
