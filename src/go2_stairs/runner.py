"""Pinned RSL-RL PPO with complete non-overwriting project checkpoints."""
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from .checkpoints import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from .training_health import TrainingHealth, TrainingHalt

NORMALIZATION = {'mode': 'fixed', 'angular_velocity_scale': .2, 'joint_velocity_scale': .05,
                 'history_frames': 5, 'frame_order': 'oldest_to_newest'}


def runner_config(profile='bounded-v3'):
    if profile not in ('legacy', 'bounded-v3'):
        raise ValueError('unknown policy profile')
    cfg = {
        'num_steps_per_env': 24, 'save_interval': 100, 'logger': 'tensorboard',
        'obs_groups': {'policy': ['policy'], 'critic': ['critic']},
        'policy': {'class_name': 'ActorCritic', 'init_noise_std': 1.,
                   'actor_hidden_dims': [256, 256, 128], 'critic_hidden_dims': [256, 256, 128],
                   'activation': 'elu', 'actor_obs_normalization': False, 'critic_obs_normalization': False},
        'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1., 'use_clipped_value_loss': True,
                      'clip_param': .2, 'entropy_coef': .01, 'num_learning_epochs': 5,
                      'num_mini_batches': 4, 'learning_rate': 1e-3, 'schedule': 'adaptive',
                      'gamma': .99, 'lam': .95, 'desired_kl': .01, 'max_grad_norm': 1.},
    }
    if profile == 'bounded-v3':
        cfg['policy']['class_name'] = 'BoundedBetaActorCritic'
        del cfg['policy']['init_noise_std']
        cfg['algorithm'].update(schedule='fixed', desired_kl=None, learning_rate=3e-4, entropy_coef=.005)
    return cfg


class CheckpointRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir, device, contract_hash, run_config):
        # Upstream pops class_name keys while constructing the algorithm.
        super().__init__(env, copy.deepcopy(train_cfg), str(log_dir), device)
        self.contract_hash = contract_hash
        self.run_config = copy.deepcopy(run_config)
        self.run_config['ppo'] = copy.deepcopy(train_cfg)
        if not isinstance(self.run_config.get('asset_bundle_hash'), str) or len(self.run_config['asset_bundle_hash']) != 64:
            raise ValueError('verified asset bundle hash is required')
        self.curriculum_state = {'stage': run_config['stage'], 'level': 0}
        self._published = set()
        self.last_checkpoint = None
        self.last_metrics = None
        self.health = TrainingHealth()

    def _construct_algorithm(self, obs):
        if self.policy_cfg['class_name'] != 'BoundedBetaActorCritic':
            return super()._construct_algorithm(obs)
        from rsl_rl.algorithms import PPO
        from .policy import BoundedBetaActorCritic
        if (self.alg_cfg.get('schedule') != 'fixed' or self.alg_cfg.get('desired_kl') is not None
                or self.alg_cfg.get('symmetry_cfg') or self.alg_cfg.get('rnd_cfg') or self.is_distributed):
            raise ValueError('Beta profile requires fixed LR, no Gaussian KL, symmetry, RND or distributed options')
        cfg = dict(self.policy_cfg)
        cfg.pop('class_name')
        policy = BoundedBetaActorCritic(obs, self.cfg['obs_groups'], self.env.num_actions, **cfg).to(self.device)
        algorithm = dict(self.alg_cfg)
        if algorithm.pop('class_name') != 'PPO':
            raise ValueError('only upstream PPO is supported')
        ppo = PPO(policy, device=self.device, **algorithm)
        ppo.init_storage('rl', self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return ppo

    def save(self, path, infos=None):
        path = Path(path)
        key = (str(path.resolve()), self.current_learning_iteration)
        # Upstream calls save twice for its final update if it is also a periodic save.
        if key in self._published:
            return
        state = {
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'normalization': dict(NORMALIZATION), 'curriculum': dict(self.curriculum_state),
            'rng': capture_rng_state(), 'iter': self.current_learning_iteration + 1,
            'contract_hash': self.contract_hash, 'config': self.run_config,
            'learning_rate': self.alg.learning_rate, 'total_timesteps': self.tot_timesteps,
            'total_time_s': self.tot_time, 'resume_semantics': 'next_update_with_fresh_episodes',
            'training_health': self.health.to_dict(),
        }
        digest = save_checkpoint(path, state)
        self.last_checkpoint = str(path.resolve())
        self._published.add(key)
        print(f'CHECKPOINT updates={state["iter"]} sha256={digest} path={path}', flush=True)

    def restore(self, path, *, training=True):
        state = load_checkpoint(path, self.contract_hash)
        if state['normalization'] != NORMALIZATION:
            raise ValueError('normalization contract mismatch')
        if state['curriculum']['stage'] != self.run_config['stage']:
            raise ValueError('resume stage mismatch; stage transfer is a separate operation')
        saved_config = state['config']
        if saved_config.get('asset_bundle_hash') != self.run_config['asset_bundle_hash']:
            raise ValueError('checkpoint asset bundle mismatch or missing identity')
        if saved_config.get('task_parameters') != self.run_config.get('task_parameters'):
            raise ValueError('checkpoint task parameters differ; use explicit policy initialization for a new task')
        for key in ('policy', 'algorithm', 'obs_groups', 'num_steps_per_env'):
            if saved_config.get('ppo', {}).get(key) != self.run_config['ppo'][key]:
                raise ValueError(f'checkpoint PPO configuration mismatch: {key}')
        if training and state.get('training_health', {}).get('reason'):
            raise TrainingHalt(state['training_health']['reason'])
        self.alg.policy.load_state_dict(state['model_state_dict'])
        if not training:
            return state
        self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.alg.learning_rate = state['learning_rate']
        self.current_learning_iteration = state['iter']
        self.tot_timesteps = state['total_timesteps']
        self.tot_time = state['total_time_s']
        self.curriculum_state = state['curriculum']
        if 'training_health' in state:
            self.health = TrainingHealth.from_dict(state['training_health'])
        restore_rng_state(state['rng'])
        return state

    def initialize_policy(self, path):
        """Explicit weights-only transfer, NOT an exact training resume.

        Accepts a different physical bundle deliberately; current assets were
        already verified by the environment. Never imports optimizer/RNG/curriculum.
        """
        state = load_checkpoint(path, self.contract_hash)
        if state['normalization'] != NORMALIZATION:
            raise ValueError('normalization contract mismatch')
        for key in ('policy', 'obs_groups'):
            if state['config'].get('ppo', {}).get(key) != self.run_config['ppo'][key]:
                raise ValueError(f'initial policy PPO configuration mismatch: {key}')
        self.alg.policy.load_state_dict(state['model_state_dict'])
        self.run_config['initialization'] = {
            'checkpoint': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'weights_only': True, 'source_updates': state['iter'],
            'source_asset_bundle_hash': state['config'].get('asset_bundle_hash'),
        }

    def log(self, locs, width=80, pad=35):
        metrics = {str(k): float(v) for k, v in locs['loss_dict'].items()}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('nonfinite PPO loss')
        if not all(bool(torch.isfinite(p).all()) for p in self.alg.policy.parameters()):
            raise FloatingPointError('nonfinite model parameter')
        with contextlib.redirect_stdout(io.StringIO()):
            super().log(locs, width, pad)
        health_metrics = {}
        if hasattr(self.env, 'consume_training_metrics'):
            health_metrics = self.env.consume_training_metrics()
            with torch.no_grad():
                deterministic = self.alg.policy.act_inference(self.env.get_observations())
                health_metrics['near_bound_fraction'] = float((deterministic.abs()>.95).float().mean())
                health_metrics['action_std_mean'] = float(self.alg.policy.action_std.mean())
            metrics.update(health_metrics)
        metrics.update(completed_updates=locs['it'] + 1, total_timesteps=self.tot_timesteps,
                       collection_s=locs['collection_time'], learning_s=locs['learn_time'],
                       learning_rate=self.alg.learning_rate,
                       mean_reward=sum(locs['rewbuffer'])/len(locs['rewbuffer']) if locs['rewbuffer'] else None,
                       mean_episode_steps=sum(locs['lenbuffer'])/len(locs['lenbuffer']) if locs['lenbuffer'] else None)
        self.last_metrics = metrics
        reason = None
        if health_metrics and self.run_config['ppo']['policy']['class_name']=='BoundedBetaActorCritic':
            reason = self.health.update(health_metrics)
        if reason:
            metrics['guard_reason'] = reason
            metrics = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k,v in metrics.items()}
        with (Path(self.log_dir)/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        if reason:
            self.save(str(Path(self.log_dir)/f'guard_stop_{locs["it"]}.pt'))
            raise TrainingHalt(reason)
        if locs['it'] % 10 == 0 or locs['it'] == locs['tot_iter'] - 1:
            print('PPO ' + json.dumps(metrics, allow_nan=False), flush=True)
