"""Isaac Lab adapter. Import only after starting AppLauncher.

Flat pretraining is implemented first. The independent stair geometry and
episode judge are not yet connected here; no staircase acceptance is implied.
"""
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg

from .contracts import ControlManifest, FiniteActionMixin, ObservationCache, actor_frame, critic_frame
from .assets import locked_go2_assets
from .training_health import relative_tracking_reward

JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                    for joint in ('hip', 'thigh', 'calf'))
DEFAULT_USD = 'C:/rl-runtimes/unitree_model/Go2/usd/go2.usd'


class CheckedRslRlVecEnvWrapper(FiniteActionMixin, RslRlVecEnvWrapper):
    """Reject Inf/NaN before the upstream wrapper can saturate them to +/-1."""

    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions)
        self._sums = torch.zeros(6, device=self.device)
        self._count = 0

    def step(self, actions):
        command = self.unwrapped.command_manager.get_command('base_velocity').clone()
        result = super().step(actions)
        data = self.unwrapped.scene['robot'].data
        self._sums += torch.stack((
            (actions.abs()>=1).float().mean(), (actions.abs()>.95).float().mean(),
            torch.linalg.vector_norm(command[:, :2], dim=1).mean(),
            torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=1).mean(),
            torch.linalg.vector_norm(command[:, :2]-data.root_lin_vel_b[:, :2], dim=1).mean(),
            (command[:, 2]-data.root_ang_vel_b[:, 2]).abs().mean()))
        self._count += 1
        return result

    def consume_training_metrics(self):
        if not self._count:
            raise RuntimeError('training metrics require a rollout')
        values = (self._sums/self._count).tolist()
        self._sums.zero_()
        self._count = 0
        return dict(zip(('hard_clip_fraction', 'sample_near_bound_fraction', 'commanded_speed',
                         'actual_speed', 'velocity_error', 'yaw_error'), values))


def relative_linear_tracking(env, std, command_name):
    return relative_tracking_reward(env.command_manager.get_command(command_name)[:, :2],
                                    env.scene['robot'].data.root_lin_vel_b[:, :2], std)


def relative_yaw_tracking(env, std, command_name):
    return relative_tracking_reward(env.command_manager.get_command(command_name)[:, 2:3],
                                    env.scene['robot'].data.root_ang_vel_b[:, 2:3], std)


class BoundedJointPositionAction(JointPositionAction):
    """Clamp the dimensionless action BEFORE the target affine transformation."""

    def process_actions(self, actions):
        if not bool(torch.isfinite(actions).all()):
            raise ValueError('nonfinite action')
        super().process_actions(actions.clamp(-1., 1.))
        bounds = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions = torch.maximum(bounds[..., 0],
                                                 torch.minimum(bounds[..., 1], self._processed_actions))


class ProprioceptiveHistory(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cache = ObservationCache(env.num_envs, env.device)
        self.ids, names = env.scene['robot'].find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError('Go2 joint contract mismatch')
        env._go2_actor_term = self
        self.noise = cfg.params['noise']

    def reset(self, env_ids=None):
        self.cache.mark_reset(env_ids)

    def __call__(self, env, noise=False):
        data = env.scene['robot'].data
        def noisy(value, amplitude):
            return value + (2 * torch.rand_like(value) - 1) * amplitude if self.noise else value
        frame = actor_frame(
            noisy(data.root_ang_vel_b, .2) * .2,
            noisy(data.projected_gravity_b, .05),
            env.command_manager.get_command('base_velocity'),
            noisy((data.joint_pos - data.default_joint_pos)[:, self.ids], .01),
            noisy(data.joint_vel[:, self.ids], 1.5) * .05,
            env.action_manager.get_term('joint_position').raw_actions,
        )
        return self.cache.update(env.common_step_counter, frame)


def privileged_observation(env):
    data = env.scene['robot'].data
    # The flat plane is z=0. Ground height relative to base, x-major 11x11 grid.
    heights = -data.root_pos_w[:, 2:3].expand(-1, 121)
    return critic_frame(env._go2_actor_term.cache.tensor(), data.root_lin_vel_b, heights)


@configclass
class PolicyObservations(ObservationGroupCfg):
    history = ObservationTermCfg(func=ProprioceptiveHistory, params={'noise': False})
    enable_corruption = False
    concatenate_terms = True


@configclass
class CriticObservations(ObservationGroupCfg):
    privileged = ObservationTermCfg(func=privileged_observation)
    enable_corruption = False
    concatenate_terms = True


@configclass
class Observations:
    policy = PolicyObservations()
    critic = CriticObservations()


@configclass
class Actions:
    joint_position = JointPositionActionCfg(
        class_type=BoundedJointPositionAction, asset_name='robot',
        joint_names=list(JOINT_NAMES), preserve_order=True, use_default_offset=True,
        scale={'.*_hip_joint': .125, '.*_thigh_joint': .25, '.*_calf_joint': .25}, clip=None)


def make_env(stage, cfg):
    if stage != 'flat':
        raise ValueError('Only the flat pretraining stage is integrated at present')
    tracking_std = float(cfg.get('tracking_std', .5))
    if not .05 <= tracking_std <= 1.:
        raise ValueError('tracking_std must be finite and between .05 and 1.0')
    usd = Path(cfg.get('usd_path', DEFAULT_USD))
    if not usd.is_file():
        raise FileNotFoundError(usd)
    asset_bundle = locked_go2_assets(usd)
    conf = RobotEnvCfg()
    conf.seed = int(cfg.get('seed', 0))
    conf.scene.num_envs = int(cfg.get('num_envs', 512))
    conf.sim.device = cfg.get('device', 'cuda:0')
    conf.scene.robot.spawn.usd_path = str(usd)
    conf.scene.terrain.terrain_type = 'plane'
    conf.scene.terrain.terrain_generator = None
    conf.scene.terrain.visual_material = None
    conf.scene.sky_light.spawn.texture_file = None
    conf.scene.height_scanner = None
    conf.curriculum.terrain_levels = None
    conf.curriculum.lin_vel_cmd_levels = None
    conf.observations = Observations()
    conf.observations.policy.history.params['noise'] = bool(cfg.get('noise', True))
    conf.actions = Actions()
    conf.sim.dt = .002
    conf.decimation = 10
    conf.sim.render_interval = 10
    conf.scene.contact_forces.update_period = .002
    conf.commands.base_velocity.debug_vis = False
    conf.commands.base_velocity.ranges.lin_vel_x = (0., .3)
    conf.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    conf.commands.base_velocity.ranges.ang_vel_z = (-.3, .3)
    conf.commands.base_velocity.limit_ranges = conf.commands.base_velocity.ranges.copy()
    conf.commands.base_velocity.resampling_time_range = (4., 6.)
    conf.commands.base_velocity.rel_standing_envs = .2
    # Separate basic locomotion learning from later robustness training.
    conf.events.physics_material = None
    conf.events.add_base_mass = None
    conf.events.push_robot = None
    # This threshold only affects zero-command episodes (nonzero commands bypass it).
    # The pilot uses the same .05 m/s speed boundary as its stop measurement.
    conf.rewards.joint_pos.params['velocity_threshold'] = .05
    conf.rewards.track_lin_vel_xy.params['std'] = tracking_std
    conf.rewards.track_ang_vel_z.params['std'] = tracking_std
    profile = cfg.get('reward_profile', 'legacy')
    if profile == 'bounded-v3':
        conf.rewards.track_lin_vel_xy.func = relative_linear_tracking
        conf.rewards.track_ang_vel_z.func = relative_yaw_tracking
    elif profile != 'legacy':
        raise ValueError('unknown reward profile')
    conf.viewer.resolution = (640, 480)
    env = ManagerBasedRLEnv(cfg=conf, render_mode=cfg.get('render_mode'))
    env.asset_bundle = asset_bundle
    env.task_parameters = {'stage': stage, 'tracking_std': tracking_std}
    if profile != 'legacy':
        env.task_parameters['reward_profile'] = profile
    return env


def control_manifest(env):
    robot = env.scene['robot']
    ids, _ = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    actuator = robot.actuators['GO2HV']
    def row(tensor):
        return tuple(tensor[0, ids].detach().cpu().tolist())
    return ControlManifest(
        joint_names=JOINT_NAMES, runtime_joint_names=JOINT_NAMES,
        q_nominal=row(robot.data.default_joint_pos), action_scale=(.125, .25, .25) * 4,
        lower=row(robot.data.soft_joint_pos_limits[..., 0]),
        upper=row(robot.data.soft_joint_pos_limits[..., 1]),
        kp=row(actuator.stiffness), kd=row(actuator.damping),
        torque_limits=row(actuator.effort_limit), physics_dt=env.physics_dt, policy_dt=env.step_dt)
