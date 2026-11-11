"""Fixed development gate, separate from frozen stair acceptance cases."""
import math


class FlatEpisodeMetrics:
    def __init__(self):
        self.steps = 0
        self.failure = None
        self.move_steps = 0
        self.turn_steps = 0
        self.distance = 0.
        self.requested_distance = 0.
        self.move_error_sq = 0.
        self.turn_error_sq = 0.
        self.stop_streak = 0
        self.max_stop_streak = 0

    def update(self, phase, command, velocity, terminated):
        if phase not in {'stand', 'move', 'turn', 'stop'}:
            raise ValueError('unknown development phase')
        if len(command) != 3 or len(velocity) != 3:
            raise ValueError('expected vx/vy/yaw triples')
        self.steps += 1
        if not all(math.isfinite(float(x)) for x in (*command, *velocity)):
            self.failure = 'nonfinite_state'
            return
        if terminated and self.failure is None:
            self.failure = 'terminated'
        if phase == 'move':
            self.move_steps += 1
            self.distance += velocity[0] * .02
            self.requested_distance += command[0] * .02
            self.move_error_sq += (velocity[0]-command[0])**2 + (velocity[1]-command[1])**2
        elif phase == 'turn':
            self.turn_steps += 1
            self.turn_error_sq += (velocity[2]-command[2])**2
        if phase == 'stop':
            stable = math.hypot(velocity[0], velocity[1]) < .05 and abs(velocity[2]) < .1
            self.stop_streak = self.stop_streak + 1 if stable else 0
            self.max_stop_streak = max(self.max_stop_streak, self.stop_streak)

    def result(self):
        move_rmse = math.sqrt(self.move_error_sq/self.move_steps) if self.move_steps else None
        turn_rmse = math.sqrt(self.turn_error_sq/self.turn_steps) if self.turn_steps else None
        complete = self.steps == 900 and self.move_steps == 500 and self.turn_steps == 150
        passed = bool(complete and self.failure is None and self.stop_streak >= 50
                      and self.requested_distance > 0 and self.distance >= .5*self.requested_distance
                      and move_rmse <= .10 and turn_rmse <= .15)
        return {'passed': passed, 'failure': self.failure, 'complete': complete,
                'steps': self.steps, 'move_rmse_mps': move_rmse, 'turn_rmse_radps': turn_rmse,
                'forward_distance_m': float(self.distance), 'commanded_distance_m': float(self.requested_distance),
                'stable_stop_s': self.stop_streak*.02, 'longest_stop_s': self.max_stop_streak*.02}


def main():
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import traceback
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--tracking-std', type=float)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.video:
        args.enable_cameras = True
    env = None
    app = None
    writer = None
    exit_code = 1
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from .env import make_env, control_manifest, CheckedRslRlVecEnvWrapper
        from .runner import CheckpointRunner, runner_config
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        task = saved['config'].get('task_parameters', {})
        width = task.get('tracking_std', .5) if args.tracking_std is None else args.tracking_std
        profile = task.get('reward_profile', 'legacy')
        cfg = {'stage': 'flat', 'num_envs': 20, 'seed': 2000, 'noise': False, 'device': args.device,
               'render_mode': 'rgb_array' if args.video else None, 'tracking_std': width, 'reward_profile': profile}
        env = make_env('flat', cfg)
        cfg['asset_bundle_hash'] = env.asset_bundle['sha256']
        cfg['task_parameters'] = env.task_parameters
        (args.output/'asset-bundle.json').write_text(json.dumps(env.asset_bundle, indent=2), encoding='utf-8')
        wrapper = CheckedRslRlVecEnvWrapper(env, clip_actions=1. if profile=='legacy' else None)
        runner = CheckpointRunner(wrapper, saved['config']['ppo'], args.output, args.device, control_manifest(env).digest(), cfg)
        runner.restore(args.checkpoint, training=False)
        policy = runner.get_inference_policy()
        obs = wrapper.get_observations()
        trackers = [FlatEpisodeMetrics() for _ in range(20)]
        velocities = torch.tensor([.1, .2]*10, device=env.device)
        command = env.command_manager.get_term('base_velocity')
        trace = []
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.output/'rollout-env0.mp4'), fps=25, codec='libx264', quality=7)
        with torch.inference_mode():
            for step in range(900):
                command.time_left.fill_(1000.)
                command.is_standing_env.fill_(False)
                command.vel_command_b.zero_()
                if 50 <= step < 550:
                    phase = 'move'
                    command.vel_command_b[:, 0] = velocities
                elif 550 <= step < 700:
                    phase = 'turn'
                    command.vel_command_b[:, 2] = .2
                else:
                    phase = 'stop' if step >= 700 else 'stand'
                requested = command.vel_command_b.cpu().numpy().copy()
                # New command is visible in the next frame; the first response has a 20 ms delay.
                action = policy(obs)
                obs, reward, dones, _ = wrapper.step(action)
                data = env.scene['robot'].data
                velocity = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1).cpu().numpy()
                failed = dones.cpu().numpy()
                for index, tracker in enumerate(trackers):
                    tracker.update(phase, requested[index], velocity[index], bool(failed[index]))
                trace.append(torch.cat((data.root_pos_w, data.root_quat_w, data.joint_pos,
                                        data.root_lin_vel_b, data.root_ang_vel_b, action), -1).cpu().numpy())
                if writer and step % 2 == 0:
                    pos = data.root_pos_w[0].cpu().numpy()
                    env.sim.set_camera_view(pos + np.array([1.6, -1.6, .9]), pos)
                    frame = env.render()
                    writer.append_data(frame)
                    if step == 500:
                        imageio.imwrite(args.output/'preview.png', frame)
        results = [dict(case_id=f'flat-dev-{i:02}', **tracker.result()) for i, tracker in enumerate(trackers)]
        np.savez_compressed(args.output/'trajectory.npz', states=np.stack(trace), joint_names=env.scene['robot'].joint_names)
        report = {'status': 'FLAT_DEVELOPMENT_EVALUATED', 'formal_stair_acceptance': False,
                  'checkpoint': str(args.checkpoint.resolve()),
                  'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  'scene_seed': 2000, 'noise': False, 'cases': results,
                  'passed_count': sum(r['passed'] for r in results),
                  'passed': all(r['passed'] for r in results),
                  'protocol': '18s: stand1, forward10(vx=.1/.2), turn3(yaw=.2), stop4; 20 parallel trials',
                  'thresholds': {'no_termination': True, 'stable_stop_s': 1., 'move_rmse_mps': .1,
                                 'turn_rmse_radps': .15, 'minimum_commanded_distance_fraction': .5}}
        actions = np.stack(trace)[:, :, -12:]
        report['action_metrics'] = {'hard_clip_fraction': float((np.abs(actions)>=1).mean()),
                                    'near_bound_fraction': float((np.abs(actions)>.95).mean()),
                                    'max_abs_action': float(np.abs(actions).max())}
        (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'FLAT_DEVELOPMENT_RESULT passed={report["passed_count"]}/20', flush=True)
        exit_code = 0  # completed evaluation, distinct from policy passing its gate
    except Exception:
        traceback.print_exc()
    finally:
        if writer:
            writer.close()
        if env:
            env.close()
        if exit_code:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        if app:
            app.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
