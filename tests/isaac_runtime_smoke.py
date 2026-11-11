"""Explicit GPU/PhysX runtime test; not a trained policy or baseline acceptance."""

import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=128)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if args.num_envs < 1 or args.steps < 1:
    raise ValueError('environment and step counts must be positive')
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import unitree_rl_lab.tasks  # noqa: F401 - gym registration
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg

env = None
try:
    cfg = RobotEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.robot.spawn.usd_path = 'C:/rl-runtimes/unitree_model/Go2/usd/go2.usd'
    cfg.scene.terrain.visual_material = None
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.height_scanner = None
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.sim.dt = .002
    cfg.decimation = 10
    cfg.sim.render_interval = 10
    cfg.scene.contact_forces.update_period = .002
    cfg.sim.device = args.device
    env = gym.make('Unitree-Go2-Velocity', cfg=cfg)
    obs, info = env.reset()
    actions = torch.zeros((args.num_envs, 12), device=env.unwrapped.device)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(actions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        assert bool(torch.isfinite(obs['policy']).all())
        assert bool(torch.isfinite(reward).all())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    report = dict(status='ISAAC_RUNTIME_SMOKE_PASS', trained_policy=False,
                  device=str(env.unwrapped.device), num_envs=args.num_envs, steps=args.steps,
                  physics_dt=cfg.sim.dt, policy_dt=cfg.sim.dt*cfg.decimation,
                  elapsed_s=elapsed, environment_steps_per_s=args.num_envs*args.steps/elapsed,
                  policy_obs_shape=list(obs['policy'].shape), critic_obs_shape=list(obs['critic'].shape),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  gpu_device_used_mb=(total_gpu-free_gpu)/1024**2,
                  gpu_device_total_mb=total_gpu/1024**2)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
finally:
    if env is not None:
        env.close()
    app.close()
