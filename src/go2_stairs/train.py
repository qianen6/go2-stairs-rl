"""Run a bounded flat PPO job. Start from an ASCII working directory on Windows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['flat'], default='flat')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=512)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    load = parser.add_mutually_exclusive_group()
    load.add_argument('--resume', type=Path)
    load.add_argument('--initialize-from', type=Path, help='Explicit weights-only transfer; reset optimizer, RNG and counters')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--tracking-std', type=float, default=.2)
    parser.add_argument('--profile', choices=['legacy','bounded-v3'], default='bounded-v3')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.save_interval) < 1:
        parser.error('counts must be positive')
    if not .05 <= args.tracking_std <= 1.:
        parser.error('tracking-std must be finite and between .05 and 1.0')
    # Refuse existing run directories before starting Kit.
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    report = {'status': 'STARTING', 'stage': args.stage, 'seed': args.seed, 'start_utc': started,
              'requested_updates': args.iterations, 'num_envs': args.num_envs, 'trained_baseline_verified': False}
    def record():
        temp = args.output/'status.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(args.output/'status.json')
    record()
    env = None
    app = None
    exit_code = 1
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from .env import make_env, control_manifest, CheckedRslRlVecEnvWrapper
        from .runner import CheckpointRunner, runner_config

        torch.set_num_threads(8)
        run_config = {'stage': args.stage, 'seed': args.seed, 'num_envs': args.num_envs,
                      'device': args.device, 'noise': True, 'tracking_std': args.tracking_std,
                      'reward_profile': args.profile,
                      'config_version': f'{args.profile}-track-{args.tracking_std:g}'}
        env = make_env(args.stage, run_config)
        run_config['asset_bundle_hash'] = env.asset_bundle['sha256']
        run_config['task_parameters'] = env.task_parameters
        (args.output/'asset-bundle.json').write_text(json.dumps(env.asset_bundle, indent=2), encoding='utf-8')
        manifest = control_manifest(env)
        (args.output/'control-manifest.json').write_text(json.dumps(manifest.to_dict(), indent=2), encoding='utf-8')
        train_cfg = runner_config(args.profile)
        train_cfg['save_interval'] = args.save_interval
        run_config['ppo'] = train_cfg.copy()
        root = Path(__file__).resolve().parents[2]
        run_config['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        run_config['code_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root))
        run_config['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted((root/'src/go2_stairs').glob('*.py'))}
        (args.output/'config.json').write_text(json.dumps(run_config, indent=2), encoding='utf-8')
        dump_yaml(str(args.output/'environment.yaml'), env.cfg)
        wrapper = CheckedRslRlVecEnvWrapper(env, clip_actions=1. if args.profile=='legacy' else None)
        runner = CheckpointRunner(wrapper, train_cfg, args.output, args.device, manifest.digest(), run_config)
        if args.resume:
            state = runner.restore(args.resume)
            report['resumed_updates'] = state['iter']
            report['resume_checkpoint'] = str(args.resume.resolve())
        elif args.initialize_from:
            runner.initialize_policy(args.initialize_from)
            report['initialization'] = runner.run_config['initialization']
        (args.output/'config.json').write_text(json.dumps(runner.run_config, indent=2), encoding='utf-8')
        report.update(status='TRAINING', contract_hash=manifest.digest())
        record()
        before = {k: v.clone() for k, v in runner.alg.policy.state_dict().items()}
        start = time.perf_counter()
        runner.learn(args.iterations, init_at_random_ep_len=True)
        elapsed = time.perf_counter() - start
        changed = any(not torch.equal(before[k], v) for k, v in runner.alg.policy.state_dict().items())
        if not changed:
            raise AssertionError('PPO did not update model parameters')
        if runner.writer:
            runner.writer.flush()
            runner.writer.close()
        free, total = torch.cuda.mem_get_info()
        report.update(status='PPO_JOB_COMPLETED', end_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_s=elapsed, completed_updates=runner.current_learning_iteration + 1,
                      new_environment_steps=args.iterations*24*args.num_envs,
                      total_environment_steps=runner.tot_timesteps,
                      environment_steps_per_s=args.iterations*24*args.num_envs/elapsed,
                      checkpoint=runner.last_checkpoint, parameters_changed=changed,
                      checkpoint_sha256=hashlib.sha256(Path(runner.last_checkpoint).read_bytes()).hexdigest(),
                      last_metrics=runner.last_metrics, gpu_device_used_mib=(total-free)/1024**2,
                      torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        record()
        print('PPO_JOB_COMPLETED ' + json.dumps(report, allow_nan=False), flush=True)
        exit_code = 0
    except Exception as exc:
        report.update(status='GUARD_STOPPED' if hasattr(exc,'guard_reason') else 'FAILED', error=f'{type(exc).__name__}: {exc}',
                      end_utc=datetime.now(timezone.utc).isoformat())
        if hasattr(exc,'guard_reason'):
            exit_code = 2
        record()
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        if exit_code:
            # Windows Kit shutdown exits 0 even while unwinding: preserve actual failure.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        if app is not None:
            app.close()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
