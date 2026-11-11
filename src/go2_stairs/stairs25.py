"""STAIRS25: Go2 stair training with the 2026-10-19 root-cause fixes, driven by a profile file.

The reference profile configs/stairs25-v1.json (STAIRS25_V1) switches on: Unitree actuator limits
(calf 45.43 Nm), head-contact termination, thigh/calf contact and foot-stumble penalties, a stair-heavy
5-28 cm curriculum, forward-biased commands, and critic warm-up plus adaptive-KL PPO from the public
weights. Student profiles are renamed copies (go2_stairs.stairs25_profile) and may add reward or
termination terms implemented in go2_stairs.student_terms. Observations, actions, PD gains and timing
are fixed, so every checkpoint loads into the same evaluator. Each checkpoint embeds its full profile.
Training completion is not stair acceptance; use go2_stairs.stairs25_matrix.

    python -m go2_stairs.stairs25 --profile configs/students/x.json --init finetune \
        --checkpoint <reference model_1499.pt> --output runs/x-seed0 --updates 300 --headless
"""
from pathlib import Path
import copy
import hashlib
import importlib
import json
import numpy as np

from .stairs25_profile import (FAMILY, REFERENCE_PROFILE, STAIR_FAMILIES, ProfileError, actuator_groups,  # noqa: F401
                               column_families, is_stairs25_info, load_profile, profile_digest,
                               profile_from_checkpoint_info, read_profile, reference_path, stair_height_at_level)

PROFILE = REFERENCE_PROFILE


def profile_path():
    return reference_path()


def apply_actuators(robot_cfg, groups):
    """Replace the native single 23.5 Nm group with profile groups; evaluation does the same per checkpoint."""
    from isaaclab.actuators import DCMotorCfg
    if not groups:
        raise ValueError('no actuator groups given')
    robot_cfg.actuators = {name: DCMotorCfg(**g) for name, g in groups.items()}
    return robot_cfg


def stumble_count(forces, ratio, min_force):
    """Feet whose contact force is mostly horizontal, i.e. toes driven into a riser."""
    horizontal = forces[..., :2].norm(dim=-1)
    return ((horizontal > min_force) & (horizontal > ratio*forces[..., 2].abs())).sum(dim=-1).float()


def feet_stumble(env, sensor_cfg, ratio=2., min_force=5.):
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    return stumble_count(forces, ratio, min_force)


def guarded_terrain_levels(env, env_ids):
    """Native velocity curriculum, skipping the construction-time reset where no walking happened."""
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    eligible = env_ids[env.episode_length_buf[env_ids] > 0]
    if eligible.numel():
        mdp.terrain_levels_vel(env, eligible)
    return env.scene.terrain.terrain_levels.float().mean()


def resolve_function(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


def term_params(params):
    """JSON params -> manager params; {"scene_entity": "contact_forces", "body_names": [...]} becomes SceneEntityCfg."""
    from isaaclab.managers import SceneEntityCfg
    out = {}
    for key, value in params.items():
        if isinstance(value, dict) and 'scene_entity' in value:
            extra = {k: value[k] for k in ('body_names', 'joint_names') if k in value}
            out[key] = SceneEntityCfg(value['scene_entity'], **extra)
        else:
            out[key] = value
    return out


def configure_env(cfg, profile, seed):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    fixes = profile['fixes']
    if fixes['actuators']:
        apply_actuators(cfg.scene.robot, actuator_groups(profile))
    generator = cfg.scene.terrain.terrain_generator
    if fixes['terrain']:
        t = profile['terrain']
        native = generator.sub_terrains
        if set(t['proportions']) != set(native):
            raise ValueError('terrain profile must keep exactly the native sub-terrain families')
        updated = {}
        for name, proportion in t['proportions'].items():
            sub = copy.deepcopy(native[name])
            sub.proportion = proportion
            if name in STAIR_FAMILIES:
                sub.step_height_range = tuple(t['stair_step_height_range_m'])
                sub.step_width = t['stair_step_width_m']
            updated[name] = sub
        generator.sub_terrains = updated
        generator.num_rows = t['num_rows']
        generator.num_cols = t['num_cols']
        generator.border_width = t['border_width_m']
    generator.seed = seed
    generator.use_cache = False
    cfg.curriculum.terrain_levels.func = guarded_terrain_levels
    if fixes['commands']:
        ranges = cfg.commands.base_velocity.ranges
        for key in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
            setattr(ranges, key, tuple(profile['commands'][key]))
    rewards = profile['rewards']
    if fixes['contact_penalties']:
        leg = rewards['leg_contact']
        cfg.rewards.undesired_contacts = RewardTermCfg(
            func=mdp.undesired_contacts, weight=leg['weight'],
            params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(leg['bodies'])),
                    'threshold': leg['threshold_N']})
    if fixes['foot_clearance']:
        stumble = rewards['feet_stumble']
        cfg.rewards.feet_air_time.weight = rewards['feet_air_time_weight']
        cfg.rewards.feet_stumble = RewardTermCfg(
            func=feet_stumble, weight=stumble['weight'],
            params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot'),
                    'ratio': stumble['horizontal_to_vertical_ratio'],
                    'min_force': stumble['min_horizontal_force_N']})
    if fixes['head_termination']:
        head = profile['terminations']['head_contact']
        cfg.terminations.head_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=head['bodies']),
                    'threshold': head['threshold_N']})
    for group, terms, make in (('rewards', rewards.get('extra_terms', []), lambda x: RewardTermCfg(func=resolve_function(x['func']), weight=x['weight'], params=term_params(x['params']))),
                               ('terminations', profile['terminations'].get('extra_terms', []), lambda x: TerminationTermCfg(func=resolve_function(x['func']), params=term_params(x['params'])))):
        manager_cfg = getattr(cfg, group)
        for term in terms:
            if hasattr(manager_cfg, term['name']):
                raise ValueError(f'{group}.extra_terms name {term["name"]} collides with an existing term')
            setattr(manager_cfg, term['name'], make(term))
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    return cfg


def set_actor_trainable(policy, trainable):
    """Freeze/unfreeze everything except the critic (actor layers and exploration std)."""
    for name, parameter in policy.named_parameters():
        if not name.startswith('critic'):
            parameter.requires_grad_(trainable)


def warmup_schedule(update_index, warmup_updates):
    """0-based update index within the critic warm-up window -> 'critic_only' or 'full'."""
    return 'critic_only' if update_index < warmup_updates else 'full'


def obs_dim(env):
    return int(env.observation_manager.group_obs_dim['policy'][0])


def audit_env(env, profile):
    """Fail fast if the built environment does not contain the configured profile."""
    import torch
    from .reference_reproduction import require_height_hit_coverage, require_scan_pose_alignment
    robot = env.scene['robot']
    require_scan_pose_alignment(robot.data.root_pos_w, env.scene['height_scanner'].data.pos_w)
    require_height_hit_coverage(env.scene['height_scanner'].data.ray_hits_w[..., 2])
    fixes = profile['fixes']
    effort = {}
    for name, actuator in robot.actuators.items():
        effort[name] = dict(joints=list(actuator.joint_names), effort_limit=float(actuator.effort_limit.max()),
                            velocity_limit=float(actuator.velocity_limit.max()))
    if fixes['actuators']:
        expected = actuator_groups(profile)
        if set(effort) != set(expected):
            raise ValueError('actuator groups differ from profile')
        for name, group in expected.items():
            if (abs(effort[name]['effort_limit']-group['effort_limit']) > 1e-4
                    or abs(effort[name]['velocity_limit']-group['velocity_limit']) > 1e-4):
                raise ValueError(f'effective actuator group {name} differs from profile: {effort[name]}')
        if sorted(j for g in effort.values() for j in g['joints']) != sorted(robot.joint_names):
            raise ValueError('actuator groups do not cover every joint exactly once')
    rewards = env.reward_manager.active_terms
    terminations = env.termination_manager.active_terms
    required_rewards = ((['undesired_contacts'] if fixes['contact_penalties'] else [])
                        + (['feet_stumble'] if fixes['foot_clearance'] else [])
                        + [t['name'] for t in profile['rewards'].get('extra_terms', [])])
    required_terminations = ((['head_contact'] if fixes['head_termination'] else [])
                             + [t['name'] for t in profile['terminations'].get('extra_terms', [])])
    if any(term not in rewards for term in required_rewards) or any(term not in terminations for term in required_terminations):
        raise ValueError('a configured reward or termination term is not active')
    if obs_dim(env) != 235 or env.physics_dt != .005 or env.step_dt != .02:
        raise ValueError('observation or timing contract drift')
    generator = env.cfg.scene.terrain.terrain_generator
    stairs = {n: list(generator.sub_terrains[n].step_height_range) for n in STAIR_FAMILIES}
    if fixes['terrain'] and any(v != list(profile['terrain']['stair_step_height_range_m']) for v in stairs.values()):
        raise ValueError('stair height range differs from profile')
    if not torch.isfinite(robot.data.root_pos_w).all():
        raise ValueError('nonfinite initial robot state')
    return dict(actuators=effort, reward_terms=list(rewards), termination_terms=list(terminations),
                stair_step_height_range_m=stairs, physics_dt=env.physics_dt, policy_dt=env.step_dt,
                commands={k: list(getattr(env.cfg.commands.base_velocity.ranges, k)) for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')})


def family_levels(env):
    """Mean curriculum level and implied stair height per terrain family (for status reporting).

    Robots that clear the top row are re-sampled to random rows, so the mean understates ability.
    """
    generator = env.cfg.scene.terrain.terrain_generator
    proportions = {n: s.proportion for n, s in generator.sub_terrains.items()}
    columns = column_families(proportions, generator.num_cols)
    types = env.scene.terrain.terrain_types.cpu().numpy()
    levels = env.scene.terrain.terrain_levels.cpu().numpy()
    result = {}
    for name in proportions:
        mask = np.array([columns[t] == name for t in types])
        if not mask.any():
            continue
        mean = float(levels[mask].mean())
        entry = dict(envs=int(mask.sum()), mean_level=mean, top_row_fraction=float((levels[mask] == generator.num_rows-1).mean()))
        if name in STAIR_FAMILIES:
            entry['mean_riser_m'] = stair_height_at_level(mean, generator.sub_terrains[name].step_height_range, generator.num_rows)
        result[name] = entry
    return result


def verify_checkpoint(path):
    """Verify a STAIRS25-family checkpoint and return evaluation metadata (incl. its actuator groups)."""
    import torch
    from .official_rough import validate_checkpoint
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    info = state.get('infos') or {}
    if not is_stairs25_info(info):
        raise ValueError('not a STAIRS25 training checkpoint')
    try:
        profile, digest, _ = profile_from_checkpoint_info(info)
    except ProfileError as error:
        raise ValueError(f'checkpoint profile cannot be verified: {error}') from error
    if info.get('updates') != state.get('iter', -2)+1:
        raise ValueError('checkpoint update counter mismatch')
    meta = validate_checkpoint(path, state['iter']+1)
    return dict(source_kind='STAIRS25_TRAINED', sha256=meta['sha256'], finetune_updates=info['updates'],
                training_seed=info['training_seed'], profile=profile['profile'], profile_config_sha256=digest,
                init_kind=info['init_kind'], origin_sha256=info.get('origin_sha256'), parent_sha256=info.get('parent_sha256'),
                fixes=profile['fixes'], actuators=actuator_groups(profile) if profile['fixes']['actuators'] else None)


def plan_initialization(init, profile, digest, public_sha, parent_info=None, parent_sha=None, num_envs=None):
    """Pure decision for how a run starts: learning rate, warm-up, counters, curriculum and lineage."""
    plan = _plan(init, profile, digest, public_sha, parent_info, parent_sha, num_envs)
    # A resumed run keeps the optimizer's current rate, but after warm-up returns to its start mode's rate.
    plan['post_warmup_learning_rate'] = (initial_learning_rate(profile, plan['init_kind']) if init == 'resume'
                                         else plan['learning_rate'])
    return plan


def initial_learning_rate(profile, init_kind):
    ppo = profile['ppo']
    return {'public': ppo['public_init_learning_rate'], 'scratch': ppo['scratch_learning_rate'],
            'finetune': ppo.get('finetune_learning_rate', ppo['public_init_learning_rate'])}[init_kind]


def _plan(init, profile, digest, public_sha, parent_info, parent_sha, num_envs):
    ppo = profile['ppo']
    if init == 'public':
        return dict(init_kind='public', base_updates=0, learning_rate=ppo['public_init_learning_rate'],
                    critic_warmup=ppo['critic_warmup_updates_public_init'], warmup_offset=0,
                    origin_sha256=public_sha, parent_sha256=public_sha, restore_optimizer=False, restore_curriculum=False)
    if init == 'scratch':
        return dict(init_kind='scratch', base_updates=0, learning_rate=ppo['scratch_learning_rate'], critic_warmup=0,
                    warmup_offset=0, origin_sha256=None, parent_sha256=None, restore_optimizer=False, restore_curriculum=False)
    parent_profile, parent_digest, _ = profile_from_checkpoint_info(parent_info)
    if init == 'resume':
        if parent_profile['profile'] != profile['profile'] or parent_digest != digest:
            raise ValueError('resume requires the same profile file as the parent; use --init finetune for a changed profile')
        if parent_info['num_envs'] != num_envs:
            raise ValueError('resume requires the same number of environments')
        return dict(init_kind=parent_info['init_kind'], base_updates=parent_info['updates'], learning_rate=None,
                    critic_warmup=parent_info['critic_warmup_updates'], warmup_offset=parent_info['updates'],
                    origin_sha256=parent_info.get('origin_sha256'), parent_sha256=parent_sha,
                    restore_optimizer=True, restore_curriculum=True)
    if init == 'finetune':
        same_terrain = (parent_profile['terrain'] == profile['terrain'] and parent_profile['fixes']['terrain'] == profile['fixes']['terrain']
                        and parent_info.get('num_envs') == num_envs)
        return dict(init_kind='finetune', base_updates=0,
                    learning_rate=initial_learning_rate(profile, 'finetune'),
                    critic_warmup=ppo.get('critic_warmup_updates_finetune', 0), warmup_offset=0,
                    origin_sha256=parent_info.get('origin_sha256'), parent_sha256=parent_sha,
                    parent_profile=parent_profile['profile'], restore_optimizer=False, restore_curriculum=same_terrain)
    raise ValueError(f'unknown init {init}')


def main(argv=None):
    import argparse
    from datetime import datetime, timezone
    import math
    import os
    import sys
    import time
    import traceback
    import torch
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', type=Path, default=None, help='profile JSON (default: frozen reference configs/stairs25-v1.json)')
    parser.add_argument('--init', choices=['public', 'scratch', 'resume', 'finetune'], required=True,
                        help='public: NVIDIA weights; scratch: random; resume: continue same profile; finetune: weights of a STAIRS25 checkpoint, new optimizer, any profile')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--critic-warmup', type=int, help='override the warm-up length (recorded in the checkpoint)')
    parser.add_argument('--wall-seconds', type=float, help='default 900 + 25 s per update')
    parser.add_argument('--strict-runtime', action='store_true', help='refuse to run unless Isaac Lab is the clean v2.3.2 reference checkout')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 5000 or not 64 <= args.num_envs <= 4096:
        raise ValueError('updates must be 1..5000 and num-envs 64..4096')
    if (args.init == 'scratch') != (args.checkpoint is None):
        raise ValueError('--checkpoint is required for public/resume/finetune and not allowed for scratch')
    profile_file = (args.profile or reference_path()).resolve()
    profile, digest, profile_text = read_profile(profile_file)
    from .official_rough import validate_checkpoint
    from .checkpoints import capture_rng_state, restore_rng_state
    from .machine import runtime_identity
    from .mixed_finetune import warm_start_public
    identity = runtime_identity(args.strict_runtime)
    parent = args.checkpoint.resolve() if args.checkpoint else None
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest() if parent else None
    parent_state = parent_info = None
    if args.init == 'public':
        if parent_sha != profile['public_sha256']:
            raise ValueError('--init public requires the hash-locked NVIDIA Go2 rough checkpoint')
    elif args.init in ('resume', 'finetune'):
        verify_checkpoint(parent)
        parent_state = torch.load(parent, map_location='cpu', weights_only=True); parent_info = parent_state['infos']
    start = plan_initialization(args.init, profile, digest, profile['public_sha256'], parent_info, parent_sha, args.num_envs)
    if args.critic_warmup is not None:
        if args.init == 'resume' and args.critic_warmup != start['critic_warmup']:
            raise ValueError('resume cannot change the critic warm-up length')
        start['critic_warmup'] = args.critic_warmup
    baseline = start['base_updates']; warmup = start['critic_warmup']; base_lr = start['learning_rate']
    wall = args.wall_seconds or 900.+25.*args.updates
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    native = root/'native'; native.mkdir()
    status = dict(status='STARTING', family=FAMILY, profile=profile['profile'], profile_file=str(profile_file),
                  profile_config_sha256=digest, fixes=profile['fixes'], init=args.init, start=start, parent=str(parent) if parent else None,
                  base_updates=baseline, completed_updates=baseline, budget_updates=args.updates, critic_warmup_updates=warmup,
                  num_envs=args.num_envs, training_seed=args.seed, runtime=identity, stair_skill_verified=False,
                  start_utc=datetime.now(timezone.utc).isoformat(), checkpoints=[])
    if not identity['matches_reference_runtime']:
        status['runtime_warning'] = 'Software stack differs from the reference Isaac Lab v2.3.2 checkout; results may not be comparable.'
    def save_status():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root/'status.tmp'; temp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(root/'status.json')
    save_status()
    app = env = None; code = 1
    try:
        app = AppLauncher(args).app
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
        from .contracts import FiniteActionMixin
        torch.set_num_threads(4)
        cfg = configure_env(UnitreeGo2RoughEnvCfg(), profile, args.seed)
        cfg.seed = args.seed; cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device
        env = ManagerBasedRLEnv(cfg)
        class Wrapper(FiniteActionMixin, RslRlVecEnvWrapper):
            def step(self, actions):
                result = super().step(actions)
                obs, reward = result[0], result[1]
                if obs['policy'].shape != (self.num_envs, 235) or not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                    raise ValueError('invalid training observation/reward')
                return result
        ppo = profile['ppo']
        agent = UnitreeGo2RoughPPORunnerCfg()
        agent.seed = args.seed; agent.device = args.device; agent.logger = 'tensorboard'
        agent.max_iterations = baseline+args.updates; agent.save_interval = 50
        agent.algorithm.schedule = ppo['schedule']; agent.algorithm.desired_kl = ppo['desired_kl']
        if base_lr is not None:
            agent.algorithm.learning_rate = base_lr
        wrapper = Wrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(native), device=args.device)
        if args.init in ('public', 'finetune'):
            state = parent_state or torch.load(parent, map_location='cpu', weights_only=True)
            warm_start_public(runner, state, base_lr)
            status['initialization'] = f'exact {args.init} actor/critic/std; fresh Adam; fresh physics'
        elif args.init == 'resume':
            runner.alg.policy.load_state_dict(parent_state['model_state_dict'], strict=True)
            runner.alg.optimizer.load_state_dict(parent_state['optimizer_state_dict'])
            base_lr = runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
            runner.current_learning_iteration = baseline
            restore_rng_state(parent_info['rng'])
            status['initialization'] = 'model/Adam/curriculum/RNG restored; fresh physics episodes'
        else:
            status['initialization'] = 'native random initialization'
        if start['restore_curriculum']:
            levels = torch.tensor(parent_info['terrain_levels'], dtype=torch.long, device=env.device)
            if levels.shape != env.scene.terrain.terrain_levels.shape:
                raise ValueError('parent curriculum does not match the environment count')
            env.scene.terrain.terrain_levels[:] = levels
            env.scene.terrain.env_origins[:] = env.scene.terrain.terrain_origins[levels, env.scene.terrain.terrain_types]
            env.episode_length_buf.zero_(); env.reset()
            status['curriculum'] = 'restored from parent checkpoint'
        else:
            status['curriculum'] = 'fresh (native initial levels 0..5)'
        status['learning_rate_start'] = base_lr
        status['runtime_audit'] = audit_env(env, profile)
        status['terrain_families'] = family_levels(env)
        dump_yaml(str(native/'params/env.yaml'), cfg); dump_yaml(str(native/'params/agent.yaml'), agent)
        (native/'params/profile.json').write_text(profile_text, encoding='utf-8')
        status['status'] = 'TRAINING'; save_status()
        original_save = runner.save
        def audited_save(path, infos=None):
            update = runner.current_learning_iteration+1
            info = dict(family=FAMILY, profile=profile['profile'], profile_config_sha256=digest, profile_config=profile_text,
                        fixes=profile['fixes'], init_kind=start['init_kind'], origin_sha256=start['origin_sha256'],
                        parent_sha256=start['parent_sha256'], parent_profile=start.get('parent_profile'), updates=update,
                        critic_warmup_updates=warmup, training_seed=args.seed, num_envs=args.num_envs,
                        runtime_matches_reference=identity['matches_reference_runtime'],
                        terrain_levels=env.scene.terrain.terrain_levels.cpu().tolist(), rng=capture_rng_state())
            original_save(path, infos=info)
            meta = validate_checkpoint(path, update)
            status.update(checkpoint=str(Path(path).resolve()), checkpoint_metadata=meta, completed_updates=update,
                          terrain_families=family_levels(env))
            status['checkpoints'].append(dict(path=str(path), updates=update, sha256=meta['sha256']))
            save_status()
        runner.save = audited_save
        original_update = runner.alg.update
        frozen = {}; count = 0; started = time.monotonic()
        def audited_update():
            nonlocal count
            index = start['warmup_offset']+count
            phase = warmup_schedule(index, warmup)
            if phase == 'critic_only':
                if not frozen:
                    frozen.update({k: v.detach().clone() for k, v in runner.alg.policy.state_dict().items() if not k.startswith('critic')})
                set_actor_trainable(runner.alg.policy, False)
                runner.alg.schedule = 'fixed'; runner.alg.learning_rate = ppo['critic_warmup_learning_rate']
                for group in runner.alg.optimizer.param_groups:
                    group['lr'] = runner.alg.learning_rate
            elif index == warmup and warmup > 0:
                current = runner.alg.policy.state_dict()
                if frozen and not all(torch.equal(current[k], v) for k, v in frozen.items()):
                    raise ValueError('actor or std changed during critic-only warm-up')
                set_actor_trainable(runner.alg.policy, True)
                post_lr = start['post_warmup_learning_rate']
                runner.alg.schedule = ppo['schedule']; runner.alg.learning_rate = post_lr
                for group in runner.alg.optimizer.param_groups:
                    group['lr'] = post_lr
                status['critic_warmup'] = dict(updates=warmup, actor_unchanged_verified=bool(frozen), finished_at_update=index)
            losses = original_update(); count += 1
            if any(not math.isfinite(float(v)) for v in losses.values()):
                raise ValueError('nonfinite PPO loss')
            weights = runner.alg.policy.state_dict()
            if not all(torch.isfinite(v).all() for v in weights.values()) or not bool((weights['std'] > 0).all()):
                raise ValueError('nonfinite weights or invalid exploration std')
            if count == 1 or count % 10 == 0 or count == args.updates:
                status.update(completed_updates=baseline+count, phase=phase, learning_rate=float(runner.alg.learning_rate),
                              last_losses={k: float(v) for k, v in losses.items()},
                              action_noise_std_mean=float(weights['std'].mean()), terrain_families=family_levels(env))
                save_status()
            if time.monotonic()-started > wall:
                raise TimeoutError('training wall-time limit; resume from the last saved checkpoint')
            return losses
        runner.alg.update = audited_update
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=not start['restore_curriculum'])
        if runtime_identity() != identity or hashlib.sha256(profile_file.read_bytes()).hexdigest() != digest:
            raise ValueError('runtime or profile file changed during training')
        if count != args.updates:
            raise ValueError('training update counter mismatch')
        status.update(status='TRAINING_COMPLETED_EVALUATION_PENDING', completed_updates=baseline+count,
                      end_utc=datetime.now(timezone.utc).isoformat())
        save_status(); print('TRAINING_COMPLETE', json.dumps(status.get('checkpoint_metadata')), flush=True); code = 0
    except BaseException as error:
        status.update(status='CANCELLED' if isinstance(error, KeyboardInterrupt) else 'TRAINING_FAILED',
                      error=repr(error), end_utc=datetime.now(timezone.utc).isoformat())
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
