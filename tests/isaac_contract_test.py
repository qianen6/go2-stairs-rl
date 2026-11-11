"""Run after AppLauncher, against actual GPU articulation (not a mock)."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
app = AppLauncher(args).app
env = None
exit_code = 1
try:
    import torch
    from go2_stairs.env import make_env, JOINT_NAMES, control_manifest, CheckedRslRlVecEnvWrapper
    from go2_stairs.contracts import joint_target

    env = make_env('flat', {'num_envs': 16, 'device': args.device, 'noise': False,
                          'tracking_std': .2, 'reward_profile': 'bounded-v3'})
    assert env.cfg.rewards.track_lin_vel_xy.func.__name__ == 'relative_linear_tracking'
    assert env.cfg.rewards.track_lin_vel_xy.params['std'] == .2
    assert env.cfg.rewards.track_ang_vel_z.params['std'] == .2
    obs, _ = env.reset()
    assert obs['policy'].shape == (16, 225)
    assert obs['critic'].shape == (16, 349)
    assert torch.equal(obs['critic'][:, :225], obs['policy'])
    assert torch.equal(obs['policy'], obs['policy'][:, -45:].repeat(1, 5))
    assert torch.equal(env.observation_manager.compute()['policy'], obs['policy'])
    assert env.physics_dt == .002 and env.step_dt == .02

    action = torch.linspace(-4, 4, 12, device=env.device).repeat(16, 1)
    term = env.action_manager.get_term('joint_position')
    term.process_actions(action)
    manifest = control_manifest(env)
    assert manifest.joint_names == JOINT_NAMES
    assert tuple(term._joint_names) == JOINT_NAMES
    assert torch.equal(term.raw_actions, action.clamp(-1, 1))
    assert torch.allclose(term.processed_actions, joint_target(action, manifest), atol=1e-6)
    assert torch.allclose(term.processed_actions[0, :3], torch.tensor([-.225, .55, -1.75], device=env.device))
    action.zero_()
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs['critic']).all() and torch.isfinite(reward).all()
        assert torch.equal(obs['critic'][:, :225], obs['policy'])
    before = obs['policy'].clone()
    env._reset_idx(torch.tensor([0, 3], device=env.device))
    obs = env.observation_manager.compute()
    assert torch.equal(obs['policy'][0], obs['policy'][0, -45:].repeat(5))
    assert torch.equal(obs['policy'][3], obs['policy'][3, -45:].repeat(5))
    assert torch.equal(before[1], obs['policy'][1])
    # Change privileged linear velocity only: actor history must be unchanged.
    robot = env.scene['robot']
    original = robot.data.root_vel_w.clone()
    velocity = original.clone()
    velocity[:, :3] += .5
    robot.write_root_velocity_to_sim(velocity)
    changed = env.observation_manager.compute()
    assert torch.equal(changed['policy'], obs['policy'])
    assert not torch.equal(changed['critic'][:, 225:228], obs['critic'][:, 225:228])
    robot.write_root_velocity_to_sim(original)
    wrapper = CheckedRslRlVecEnvWrapper(env, clip_actions=1.)
    counter = env.common_step_counter
    for bad in (float('inf'), -float('inf'), float('nan')):
        try:
            wrapper.step(torch.full((16, 12), bad, device=env.device))
        except ValueError:
            pass
        else:
            raise AssertionError('wrapper accepted nonfinite actions')
    assert env.common_step_counter == counter
    assert len(env.asset_bundle['files']) == 4
    report = {'status': 'ISAAC_CONTRACT_PASS', 'checks': 25, 'num_envs': 16,
              'actor': 225, 'critic': 349, 'actions': 12, 'contract': manifest.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ISAAC_CONTRACT_PASS: 225/349/12, history/reset/order/limits/isolation')
    exit_code = 0
except Exception:
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    # Kit's Windows shutdown forcibly exits 0, even during exception unwinding.
    if exit_code:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    app.close()
sys.exit(exit_code)
