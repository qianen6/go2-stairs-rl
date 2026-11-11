"""Versioned proprioceptive observation and joint-position contracts."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


def _check(value, width, name, batch=None, device=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f'{name} must be a rank-2 tensor')
    if value.dtype != torch.float32 or value.shape[1] != width or value.shape[0] < 1:
        raise ValueError(f'{name} must be nonempty float32 [N,{width}]')
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f'{name}: batch mismatch')
    if device is not None and value.device != device:
        raise ValueError(f'{name}: device mismatch')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name}: nonfinite input')


class FiniteActionMixin:
    """Place before an action-clipping wrapper in the method-resolution order."""
    def step(self, actions):
        _check(actions, 12, 'policy action', self.num_envs)
        return super().step(actions)


def actor_frame(ang_vel, gravity, command, q_rel, dq, previous_action):
    parts = (ang_vel, gravity, command, q_rel, dq, previous_action)
    widths = (3, 3, 3, 12, 12, 12)
    _check(ang_vel, 3, 'ang_vel')
    for part, width in zip(parts, widths):
        _check(part, width, 'actor component', ang_vel.shape[0], ang_vel.device)
    return torch.cat(parts, dim=-1)


def critic_frame(history225, linear_velocity3, heights121):
    _check(history225, 225, 'actor history')
    for part, width in ((linear_velocity3, 3), (heights121, 121)):
        _check(part, width, 'critic component', history225.shape[0], history225.device)
    return torch.cat((history225, linear_velocity3, heights121), dim=-1)


class ObservationHistory:
    """Five frames, oldest first; explicit reset for every environment."""

    def __init__(self, num_envs, device='cpu'):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError('num_envs must be a positive integer')
        self._buffer = torch.zeros(num_envs, 5, 45, dtype=torch.float32, device=device)
        self._ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids, frame45):
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError('env_ids must be a one-dimensional int64 tensor')
        ids = env_ids.to(self._buffer.device)
        if ids.numel() < 1 or ids.unique().numel() != ids.numel():
            raise ValueError('env_ids must be nonempty and unique')
        if bool(((ids < 0) | (ids >= len(self._buffer))).any()):
            raise ValueError('env_ids out of range')
        _check(frame45, 45, 'reset frame', len(ids), self._buffer.device)
        self._buffer[ids] = frame45.detach().unsqueeze(1).repeat(1, 5, 1)
        self._ready[ids] = True

    def push(self, frame45):
        if not bool(self._ready.all()):
            raise RuntimeError('reset all environments before push')
        _check(frame45, 45, 'history frame', len(self._buffer), self._buffer.device)
        self._buffer[:, :-1] = self._buffer[:, 1:].clone()
        self._buffer[:, -1] = frame45.detach()

    def tensor(self):
        if not bool(self._ready.all()):
            raise RuntimeError('reset all environments before reading history')
        return self._buffer.reshape(len(self._buffer), 225).clone()


class ObservationCache:
    """Deduplicate actor/critic observation reads while honoring asynchronous resets."""

    def __init__(self, num_envs, device='cpu'):
        self.history = ObservationHistory(num_envs, device)
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._step = None

    def mark_reset(self, env_ids=None):
        if env_ids is None:
            self._pending[:] = True
            return
        ids = torch.as_tensor(env_ids, device=self._pending.device)
        if ids.dtype != torch.long or ids.ndim != 1 or bool(((ids < 0) | (ids >= len(self._pending))).any()):
            raise ValueError('invalid reset indices')
        self._pending[ids] = True

    def update(self, step, frame45):
        if type(step) is not int or step < 0 or (self._step is not None and step < self._step):
            raise ValueError('observation step must be a monotonic nonnegative integer')
        _check(frame45, 45, 'cached frame', len(self._pending), self._pending.device)
        if self._step is not None and step > self._step:
            self.history.push(frame45)
        ids = self._pending.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.history.reset(ids, frame45[ids])
            self._pending[ids] = False
        self._step = step
        return self.history.tensor()

    def tensor(self):
        return self.history.tensor()


@dataclass(frozen=True)
class ControlManifest:
    joint_names: tuple
    runtime_joint_names: tuple
    q_nominal: tuple
    action_scale: tuple
    lower: tuple
    upper: tuple
    kp: tuple
    kd: tuple
    torque_limits: tuple
    schema_version: int = 1
    physics_dt: float = .002
    policy_dt: float = .020

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError('unsupported contract schema')
        for field in ('joint_names', 'runtime_joint_names'):
            names = tuple(getattr(self, field))
            if len(names) != 12 or len(set(names)) != 12 or not all(isinstance(n, str) and n for n in names):
                raise ValueError(f'{field} requires twelve unique names')
            object.__setattr__(self, field, names)
        if set(self.joint_names) != set(self.runtime_joint_names):
            raise ValueError('runtime and policy joint sets differ')
        for field in ('q_nominal', 'action_scale', 'lower', 'upper', 'kp', 'kd', 'torque_limits'):
            values = tuple(float(v) for v in getattr(self, field))
            if len(values) != 12 or not all(math.isfinite(v) and abs(v) <= torch.finfo(torch.float32).max for v in values):
                raise ValueError(f'{field} requires twelve float32-representable finite values')
            object.__setattr__(self, field, values)
        for i in range(12):
            if not self.lower[i] <= self.q_nominal[i] <= self.upper[i] or self.lower[i] >= self.upper[i]:
                raise ValueError('invalid joint bounds or nominal pose')
            if self.action_scale[i] <= 0 or self.kp[i] < 0 or self.kd[i] < 0 or self.torque_limits[i] <= 0:
                raise ValueError('invalid scale, gain or torque limit')
        if not all(math.isfinite(v) and v > 0 for v in (self.physics_dt, self.policy_dt)):
            raise ValueError('invalid control periods')
        ratio = self.policy_dt / self.physics_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError('policy period must be an integer number of physics steps')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def digest(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def joint_target(action12, manifest):
    _check(action12, 12, 'action')
    values = lambda field: torch.tensor(getattr(manifest, field), device=action12.device, dtype=torch.float32)
    target = values('q_nominal') + values('action_scale') * action12.clamp(-1., 1.)
    if not bool(torch.isfinite(target).all()):
        raise ValueError('nonfinite scaled joint targets')
    target = torch.maximum(values('lower'), torch.minimum(values('upper'), target))
    order = [manifest.joint_names.index(n) for n in manifest.runtime_joint_names]
    return target[:, order]


def projected_gravity(quaternion_wxyz):
    _check(quaternion_wxyz, 4, 'quaternion')
    scale = quaternion_wxyz.abs().amax(dim=-1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError('zero quaternion')
    scaled = quaternion_wxyz / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    w, x, y, z = (scaled / norm).unbind(-1)
    return torch.stack((2 * (w*y - x*z), -2 * (y*z + w*x), 2*(x*x + y*y) - 1), dim=-1)
