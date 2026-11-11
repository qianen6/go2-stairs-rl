"""Non-learning probes and verified completion records for short fine-tuning trials."""
from pathlib import Path
import hashlib
import json
import torch
import os
import tempfile


def gaussian_kl(mean0, std0, mean1, std1):
    if not torch.isfinite(std0).all() or not torch.isfinite(std1).all() or (std0 <= 0).any() or (std1 <= 0).any():
        raise ValueError('Gaussian standard deviations must be positive and finite')
    return (torch.log(std1/std0) + (std0.square()+(mean0-mean1).square())/(2*std1.square())-.5).sum(-1)


def write_final_record(root, payload):
    root = Path(root).resolve()
    checkpoint = Path(payload['checkpoint']).resolve()
    if root not in checkpoint.parents or payload.get('train_exit_code') != 0:
        raise ValueError('final record requires a successful checkpoint inside this run')
    result = dict(payload, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix='.final-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8'))
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, root/'FINAL.json')
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return result


def read_final_record(root):
    root = Path(root).resolve()
    result = json.loads((root/'FINAL.json').read_text(encoding='utf-8'))
    checkpoint = Path(result['checkpoint']).resolve()
    if root not in checkpoint.parents or result.get('train_exit_code') != 0 or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result['checkpoint_sha256']:
        raise ValueError('completed checkpoint integrity mismatch')
    return result


def restore_public_adam(runner, state, learning_rate):
    policy = runner.alg.policy
    if [name for name, _ in policy.named_parameters()] != list(state['model_state_dict']):
        raise ValueError('optimizer parameter order does not match the public model')
    policy.load_state_dict(state['model_state_dict'], strict=True)
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    for parameter, moments in runner.alg.optimizer.state.items():
        if any(moments[k].shape != parameter.shape or not torch.isfinite(moments[k]).all()
               for k in ['exp_avg', 'exp_avg_sq']):
            raise ValueError('invalid public optimizer moments')
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = learning_rate
    runner.alg.learning_rate = learning_rate
    runner.current_learning_iteration = 0


def continuation_gate(baseline, candidate):
    if set(candidate) != set(baseline) or any(v.get('invalid',0) for v in candidate.values()):
        return 'INVALID_EVALUATION'
    if any(candidate[name].get('n') != baseline[name].get('n') for name in baseline):
        return 'INVALID_EVALUATION'
    for name in ['flat','up10','down10','down20']:
        if any(candidate[name][key] < baseline[name][key] for key in ['completed','clean']):
            return 'BASIC_REGRESSION'
    if candidate['up20']['completed'] < baseline['up20']['completed'] or candidate['up20']['clean'] <= baseline['up20']['clean']:
        return 'NO_TARGET_GAIN'
    return 'CANDIDATE_FOR_INDEPENDENT_CONFIRMATION'
