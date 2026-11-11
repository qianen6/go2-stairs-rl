"""Public-checkpoint PPO fine-tuning on an audited, height-weighted native mixture."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np

PROFILE = 'PUBLIC_MIXED20_FINETUNE_V1'
PUBLIC_SHA = 'f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2'


def profile_path():
    return Path(__file__).resolve().parents[2] / 'configs/public-mixed20-finetune-v1.json'


def profile_digest():
    return hashlib.sha256(profile_path().read_bytes()).hexdigest()


def mixture_specs():
    result = []
    for family in ['pyramid_stairs', 'pyramid_stairs_inv']:
        for suffix, fraction, height in [('broad', .08, None), ('h18', .04, .18), ('h20', .06, .20), ('h22', .02, .22)]:
            result.append(dict(name=family+'_'+suffix, family=family, proportion=fraction, fixed_height=height))
    for family, fraction in [('boxes', .2), ('random_rough', .2), ('hf_pyramid_slope', .1), ('hf_pyramid_slope_inv', .1)]:
        result.append(dict(name=family, family=family, proportion=fraction, fixed_height=None))
    return result


def column_labels(specs, count):
    weights = np.array([x['proportion'] for x in specs], dtype=float)
    if count < len(specs) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('invalid terrain proportions or column count')
    cumulative = np.cumsum(weights / weights.sum())
    return [specs[int(np.searchsorted(cumulative, i/count+.001, side='right'))]['name'] for i in range(count)]


def warm_start_public(runner, state, learning_rate):
    import torch
    runner.alg.policy.load_state_dict(state['model_state_dict'], strict=True)
    runner.alg.optimizer = torch.optim.Adam(runner.alg.policy.parameters(), lr=learning_rate)
    runner.alg.learning_rate = learning_rate
    runner.current_learning_iteration = 0
    if not all(torch.equal(value.cpu(), state['model_state_dict'][name].cpu())
               for name, value in runner.alg.policy.state_dict().items()):
        raise ValueError('public initialization differs from source weights')


def current_tile_labels(xy, specs, rows, cols, size):
    import torch
    origin = torch.tensor([rows*size/2, cols*size/2], device=xy.device)
    index = torch.floor((xy+origin)/size).long()
    inside = (index[:, 0] >= 0) & (index[:, 0] < rows) & (index[:, 1] >= 0) & (index[:, 1] < cols)
    names = [s['name'] for s in specs]
    by_column = torch.tensor([names.index(n) for n in column_labels(specs, cols)], device=xy.device)
    labels = by_column[index[:, 1].clamp(0, cols-1)]
    labels = torch.where(inside, labels, len(specs))
    local = xy - ((index+.5)*size-origin)
    radius = local.abs().amax(1)
    fixed = torch.tensor([s['fixed_height'] is not None for s in specs]+[False], device=xy.device)
    riser_zone = inside & fixed[labels] & (radius >= 1.1) & (radius <= 3.4)
    return labels, riser_zone


def curriculum_eligible_ids(env_ids, episode_lengths, terrain_types, fixed_columns):
    return env_ids[(episode_lengths[env_ids] > 0) & ~fixed_columns[terrain_types[env_ids]]]


def mixed_terrain_levels(env, env_ids):
    import torch
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    specs = mixture_specs()
    fixed = {s['name'] for s in specs if s['fixed_height'] is not None}
    columns = column_labels(specs, env.cfg.scene.terrain.terrain_generator.num_cols)
    fixed_columns = torch.tensor([x in fixed for x in columns], device=env.device)
    eligible = curriculum_eligible_ids(env_ids, env.episode_length_buf, env.scene.terrain.terrain_types, fixed_columns)
    if eligible.numel():
        mdp.terrain_levels_vel(env, eligible)
    return env.scene.terrain.terrain_levels.float().mean()


def configure_mixture(cfg, seed):
    from .native_factory import restore_release_velocity_limits, restore_release_stabilization
    native = cfg.scene.terrain.terrain_generator.sub_terrains
    updated = {}
    for spec in mixture_specs():
        terrain = copy.deepcopy(native[spec['family']])
        terrain.proportion = spec['proportion']
        if spec['fixed_height'] is not None:
            terrain.step_height_range = (spec['fixed_height'], spec['fixed_height'])
        updated[spec['name']] = terrain
    cfg.scene.terrain.terrain_generator.sub_terrains = updated
    cfg.scene.terrain.terrain_generator.num_cols = 50
    cfg.scene.terrain.terrain_generator.num_rows = 10
    cfg.scene.terrain.terrain_generator.seed = seed
    cfg.scene.terrain.terrain_generator.border_width = 30.
    cfg.scene.terrain.terrain_generator.use_cache = False
    cfg.curriculum.terrain_levels.func = mixed_terrain_levels
    restore_release_velocity_limits(cfg.scene.robot.actuators)
    restore_release_stabilization(cfg.sim)
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    return cfg


def verify_finetune_checkpoint(path):
    import torch
    from .official_rough import validate_checkpoint
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == PUBLIC_SHA:
        return dict(source_kind='NVIDIA_PUBLISHED', sha256=digest, finetune_updates=0, training_seed='unknown')
    state = torch.load(path, map_location='cpu', weights_only=True)
    info = state.get('infos') or {}
    if (info.get('profile') != PROFILE or info.get('origin_sha256') != PUBLIC_SHA
            or info.get('profile_config_sha256') != profile_digest()
            or info.get('finetune_updates') != state.get('iter', -2)+1):
        raise ValueError('checkpoint is not a verified continuation of this frozen fine-tune profile')
    validate_checkpoint(path, state['iter']+1)
    return dict(source_kind='PUBLIC_WEIGHT_FINETUNED', sha256=digest,
                finetune_updates=state['iter']+1, training_seed=info['training_seed'])


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if callable(value):
        return value.__module__ + ':' + value.__qualname__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ExposureTracker:
    def __init__(self, env):
        import torch
        self.env = env
        self.specs = mixture_specs()
        self.names = [s['name'] for s in self.specs] + ['outer_border']
        self.steps = torch.zeros(len(self.names), dtype=torch.long, device=env.device)
        self.riser_steps = torch.zeros_like(self.steps)
        self.starts = torch.zeros_like(self.steps)
        labels = column_labels(self.specs, 50)
        self.by_column = torch.tensor([self.names.index(n) for n in labels], device=env.device)
        self.assigned = self.by_column[env.scene.terrain.terrain_types]
        self.starts += torch.bincount(self.assigned, minlength=len(self.names))
        self.policy_steps = 0

    def before_step(self):
        import torch
        labels, zone = current_tile_labels(self.env.scene['robot'].data.root_pos_w[:, :2], self.specs, 10, 50, 8.)
        self.steps += torch.bincount(labels, minlength=len(self.names))
        self.riser_steps += torch.bincount(labels[zone], minlength=len(self.names))
        self.policy_steps += 1

    def after_step(self, done):
        import torch
        self.starts += torch.bincount(self.assigned[done.bool()], minlength=len(self.names))

    def report(self):
        import torch
        assigned = torch.bincount(self.assigned, minlength=len(self.names)).cpu().tolist()
        steps, zones, starts = self.steps.cpu().tolist(), self.riser_steps.cpu().tolist(), self.starts.cpu().tolist()
        return dict(policy_steps=self.policy_steps, environment_steps=sum(steps), groups={
            name: dict(assigned_envs=assigned[i], actual_tile_samples=steps[i], target_riser_zone_samples=zones[i], episode_starts=starts[i])
            for i, name in enumerate(self.names)})


def audit_scene(env):
    import torch
    from isaaclab.utils.warp import raycast_mesh
    from .reference_reproduction import require_scan_pose_alignment, require_height_hit_coverage
    robot = env.scene['robot']
    scanner = env.scene['height_scanner']
    require_scan_pose_alignment(robot.data.root_pos_w, scanner.data.pos_w)
    require_height_hit_coverage(scanner.data.ray_hits_w[..., 2])
    velocity_limits = robot.root_physx_view.get_dof_max_velocities().cpu()
    if not torch.allclose(velocity_limits, torch.full_like(velocity_limits, 30.)):
        raise ValueError('effective solver limits differ from frozen compatibility profile')
    if env.physics_dt != .005 or env.step_dt != .02 or not env.cfg.sim.physx.enable_stabilization:
        raise ValueError('physics or policy period drift')
    columns = column_labels(mixture_specs(), 50)
    origins = env.scene.terrain.terrain_origins
    checks = []
    for spec in mixture_specs():
        if spec['fixed_height'] is None:
            continue
        column = columns.index(spec['name'])
        origin = origins[0, column]
        starts = origin.repeat(3, 1).clone()
        starts[:, 0] += torch.tensor([1.05, 1.35, 1.65], device=env.device)
        starts[:, 2] = 20.
        dirs = torch.zeros_like(starts); dirs[:, 2] = -1.
        hits = raycast_mesh(starts, dirs, scanner.meshes['/World/ground'])[0]
        heights = hits[:, 2]
        increments = (heights[1:]-heights[:-1]).abs()
        if not torch.isfinite(hits).all() or not torch.allclose(increments, torch.full_like(increments, spec['fixed_height']), atol=2e-5):
            raise ValueError('actual target terrain does not match requested height: '+spec['name'])
        checks.append(dict(group=spec['name'], expected=spec['fixed_height'], measured_risers=increments.cpu().tolist()))
    return dict(physics_dt=env.physics_dt, policy_dt=env.step_dt, solver_limits=velocity_limits[0].tolist(),
                joint_names=robot.joint_names, fixed_height_geometry=checks, sensor_coverage='finite',
                actuator_cfg=canonical(env.cfg.scene.robot.actuators), stabilization=True)


def main():
    import argparse
    from datetime import datetime, timezone
    import math
    import os
    import sys
    import time
    import traceback
    import torch
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wall-seconds', type=float, default=1200.)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.updates <= 100 or not 50 <= args.num_envs <= 2000 or args.num_envs % 50:
        raise ValueError('bounded blocks require1..100 updates and50..2000 envs divisible by50')
    plan = json.loads((profile_path().parent/'reference-first-20cm-plan.json').read_text(encoding='utf-8'))
    if not plan.get('training_allowed_by_plan') or plan.get('authorized_training_profile') != PROFILE:
        raise ValueError('active plan has not authorized this training profile')
    profile = json.loads(profile_path().read_text(encoding='utf-8'))
    parent = args.checkpoint.resolve()
    parent_meta = verify_finetune_checkpoint(parent)
    from .official_rough import source_identity, validate_checkpoint
    from .checkpoints import capture_rng_state, restore_rng_state
    identity = source_identity()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    native = root/'native'; native.mkdir()
    baseline = parent_meta['finetune_updates']
    status = dict(status='STARTING', profile=PROFILE, parent=str(parent), parent_sha256=parent_meta['sha256'],
                  origin_sha256=PUBLIC_SHA, public_checkpoint_iteration=1499, base_finetune_updates=baseline,
                  completed_finetune_updates=baseline, block_updates=args.updates, num_envs=args.num_envs,
                  training_seed=args.seed, source_identity=identity, profile_config_sha256=profile_digest(),
                  training=True, start_utc=datetime.now(timezone.utc).isoformat(), checkpoints=[])
    def save_status():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root/'status.tmp'; temp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(root/'status.json')
    save_status()
    app = env = None; code = 1
    try:
        app = AppLauncher(args).app
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
        from .contracts import FiniteActionMixin
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = UnitreeGo2RoughEnvCfg()
        reward_before = canonical(cfg.rewards.to_dict())
        cfg = configure_mixture(cfg, args.seed)
        if canonical(cfg.rewards.to_dict()) != reward_before:
            raise ValueError('native rewards were changed')
        cfg.seed = args.seed; cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device; cfg.log_dir = str(native)
        env = ManagerBasedRLEnv(cfg)
        class Wrapper(FiniteActionMixin, RslRlVecEnvWrapper):
            tracker = None
            def step(self, actions):
                if self.tracker is not None:
                    self.tracker.before_step()
                result = super().step(actions)
                obs, reward, done, extras = result
                if obs['policy'].shape != (self.num_envs, 235) or not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise ValueError('invalid training observation/reward')
                if not torch.isfinite(env.scene['height_scanner'].data.ray_hits_w).all():
                    raise ValueError('raw height ray misses during training')
                if self.tracker is not None:
                    self.tracker.after_step(done)
                return result
        agent = UnitreeGo2RoughPPORunnerCfg()
        agent.seed = args.seed; agent.device = args.device; agent.logger = 'tensorboard'
        agent.max_iterations = baseline+args.updates; agent.save_interval = 25
        agent.algorithm.learning_rate = profile['learning_rate']; agent.algorithm.schedule = 'fixed'
        wrapper = Wrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(native), device=args.device)
        state = torch.load(parent, map_location='cpu', weights_only=True)
        if baseline == 0:
            warm_start_public(runner, state, profile['learning_rate'])
            status['initialization'] = 'exact actor/critic/std; fresh Adam; fresh physics; fine-tune counter0'
        else:
            runner.alg.policy.load_state_dict(state['model_state_dict'], strict=True)
            runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
            runner.current_learning_iteration = baseline
            if runner.alg.learning_rate != profile['learning_rate']:
                raise ValueError('resume learning rate changed')
            saved_levels = torch.tensor(state['infos']['terrain_levels'], dtype=torch.long, device=env.device)
            if saved_levels.shape != env.scene.terrain.terrain_levels.shape:
                raise ValueError('resume environment count changed')
            env.scene.terrain.terrain_levels[:] = saved_levels
            env.scene.terrain.env_origins[:] = env.scene.terrain.terrain_origins[saved_levels, env.scene.terrain.terrain_types]
            restore_rng_state(state['infos']['rng'])
            env.episode_length_buf.zero_(); env.reset()
            status['initialization'] = 'model/Adam/curriculum/RNG restored; fresh physics episodes'
        initial_weights = {k: v.detach().cpu().clone() for k, v in runner.alg.policy.state_dict().items()}
        if not all(torch.equal(v, state['model_state_dict'][k].cpu()) for k, v in initial_weights.items()):
            raise ValueError('initial weights do not match selected parent')
        tracker = ExposureTracker(env); wrapper.tracker = tracker
        status['runtime_audit'] = audit_scene(env)
        status['exposure'] = tracker.report()
        if any(v['assigned_envs'] == 0 for k, v in status['exposure']['groups'].items() if k != 'outer_border'):
            raise ValueError('a configured terrain group has no assigned environment')
        status['initial_weights_exact_match'] = True
        status['optimizer_loaded_from_public'] = False
        dump_yaml(str(native/'params/env.yaml'), cfg); dump_yaml(str(native/'params/agent.yaml'), agent)
        (native/'params/profile.json').write_bytes(profile_path().read_bytes())
        status['rewards_unchanged'] = True
        status['status'] = 'TRAINING'; save_status()
        original_save = runner.save
        def audited_save(path, infos=None):
            update = runner.current_learning_iteration+1
            info = dict(profile=PROFILE, origin_sha256=PUBLIC_SHA, parent_sha256=parent_meta['sha256'],
                        profile_config_sha256=status['profile_config_sha256'], finetune_updates=update,
                        training_seed=args.seed, num_envs=args.num_envs,
                        terrain_levels=env.scene.terrain.terrain_levels.cpu().tolist(), rng=capture_rng_state())
            original_save(path, infos=info)
            meta = validate_checkpoint(path, update)
            status['checkpoint'] = str(Path(path).resolve()); status['checkpoint_metadata'] = meta
            status['completed_finetune_updates'] = update
            status['exposure'] = tracker.report()
            status['checkpoints'].append(dict(path=str(path), finetune_updates=update, sha256=meta['sha256']))
            save_status()
        runner.save = audited_save
        original_update = runner.alg.update
        update_count = 0; started = time.monotonic()
        def audited_update():
            nonlocal update_count
            losses = original_update(); update_count += 1
            if any(not math.isfinite(float(v)) for v in losses.values()):
                raise ValueError('nonfinite PPO loss')
            weights = runner.alg.policy.state_dict()
            if not all(torch.isfinite(v).all() for v in weights.values()) or not bool((weights['std'] > 0).all()):
                raise ValueError('nonfinite weights or invalid exploration std')
            if update_count == 1:
                deltas = {k: float((v.detach().cpu()-initial_weights[k]).abs().max()) for k, v in weights.items()}
                if not any(v > 0 for k, v in deltas.items() if k.startswith('actor.')):
                    raise ValueError('PPO did not update actor weights')
                status['first_update'] = dict(actor_max_abs_delta=max(v for k, v in deltas.items() if k.startswith('actor.')),
                                              losses={k: float(v) for k, v in losses.items()}, finite=True)
            if update_count == 1 or update_count % 10 == 0:
                status['completed_finetune_updates'] = baseline+update_count
                status['exposure'] = tracker.report(); status['last_losses'] = {k: float(v) for k, v in losses.items()}
                save_status()
            if time.monotonic()-started > args.wall_seconds:
                raise TimeoutError('bounded training wall-time limit')
            return losses
        runner.alg.update = audited_update
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
        if source_identity() != identity or hashlib.sha256(parent.read_bytes()).hexdigest() != parent_meta['sha256'] or profile_digest() != status['profile_config_sha256']:
            raise ValueError('source, profile or parent changed during training')
        if update_count != args.updates:
            raise ValueError('training update counter mismatch')
        status.update(status='TRAINING_BLOCK_COMPLETE_EVALUATION_PENDING', train_exit_code=0,
                      completed_finetune_updates=baseline+update_count, end_utc=datetime.now(timezone.utc).isoformat())
        save_status(); print('TRAINING_BLOCK_COMPLETE', json.dumps(status['checkpoint_metadata']), flush=True); code = 0
    except BaseException as error:
        status.update(status='TRAINING_FAILED', train_exit_code=1, error=repr(error), end_utc=datetime.now(timezone.utc).isoformat())
        save_status(); traceback.print_exc()
    finally:
        if code:
            sys.stdout.flush(); sys.stderr.flush(); os._exit(code)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
