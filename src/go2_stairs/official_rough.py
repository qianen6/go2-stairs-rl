"""Run Isaac Lab's unchanged Go2 Rough entry; never claim stair acceptance.

This native reference has 235 observations including terrain heights and base
linear velocity. It is not the old 225-observation proprioceptive deployment
contract. Target stair geometry is an evaluation set, not a training override.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .reference import ISAAC, adapter_identity, run_reference_child

TASK = 'Isaac-Velocity-Rough-Unitree-Go2-v0'
REVISION = '37ddf626871758333d6ed89cf64ad702aef127d0'


def build_command(output, num_envs=2048, updates=1500, seed=0):
    for value, low, high in ((num_envs,1,4096),(updates,1,1500),(seed,0,2**31-1)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError('invalid official rough budget or seed')
    native = (Path(output).resolve()/'native').as_posix()
    return [sys.executable, str(ISAAC/'scripts/reinforcement_learning/rsl_rl/train.py'),
            '--task', TASK, '--headless', '--num_envs', str(num_envs),
            '--max_iterations', str(updates), '--seed', str(seed),
            '--run_name', 'official_rough', '--logger', 'tensorboard',
            'agent.experiment_name='+json.dumps(native,ensure_ascii=False),
            'env.scene.terrain.visual_material=null',
            'env.scene.sky_light.spawn.texture_file=null']


def load_yaml_data(path):
    """Read saved config as data; never construct Python objects from YAML tags."""
    import yaml
    class DataLoader(yaml.SafeLoader):
        pass
    def unknown(loader,node):
        if isinstance(node,yaml.MappingNode):
            return loader.construct_mapping(node,deep=True)
        if isinstance(node,yaml.SequenceNode):
            return loader.construct_sequence(node,deep=True)
        return loader.construct_scalar(node) or node.tag
    DataLoader.add_constructor(None,unknown)
    return yaml.load(Path(path).read_text(encoding='utf-8'),Loader=DataLoader)


def validate_effective_config(env, agent, num_envs, updates):
    def require(condition):
        if not condition:
            raise ValueError('effective config is not the expected original Go2 Rough task')
    try:
        terrain=env['scene']['terrain']
        require(env['scene']['num_envs']==num_envs)
        require(terrain['terrain_type']=='generator')
        generator=terrain['terrain_generator']
        require(generator['curriculum'] is True)
        for name in ('pyramid_stairs','pyramid_stairs_inv'):
            cfg=generator['sub_terrains'][name]
            require(cfg['proportion']==.2)
            require(list(cfg['step_height_range'])==[.05,.23])
            require(cfg['step_width']==.3)
        require(env['curriculum']['terrain_levels'])
        require(env['observations']['policy']['height_scan'])
        require(env['observations']['policy']['base_lin_vel'])
        require(agent['max_iterations']==updates)
        require(agent['policy']['actor_hidden_dims']==[512,256,128])
        require(agent['policy']['critic_hidden_dims']==[512,256,128])
        require(agent['algorithm']['class_name']=='PPO')
    except (KeyError, TypeError) as exc:
        raise ValueError('effective config is not the expected original Go2 Rough task') from exc
    return dict(task=TASK,stairs_enabled=True,stair_skill_verified=False,
                requires_height_scan=True,requires_base_linear_velocity=True,
                step_height_range_m=[.05,.23],step_width_m=.3,
                terrain_shape='pyramid_and_inverted_pyramid',
                target_stair_evaluation='PENDING; 4/6/8cm are evaluation-only',
                deployment_contract='NATIVE_235_HEIGHT_SCAN_REFERENCE_ONLY')


def validate_checkpoint(path, updates):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    if type(state.get('iter')) is not int or state['iter']!=updates-1:
        raise ValueError('official rough checkpoint iteration mismatch')
    weights=state['model_state_dict']
    expected={'std':(12,)}
    for name,dims in [('actor',[235,512,256,128,12]),('critic',[235,512,256,128,1])]:
        for i,(a,b) in enumerate(zip(dims,dims[1:])):
            expected[f'{name}.{i*2}.weight']=(b,a)
            expected[f'{name}.{i*2}.bias']=(b,)
    if set(weights)!=set(expected) or not isinstance(state.get('optimizer_state_dict'),dict):
        raise ValueError('unexpected official rough checkpoint structure')
    for key,shape in expected.items():
        value=weights[key]
        if value.shape!=shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f'invalid rough checkpoint tensor: {key}')
    if not bool((weights['std']>0).all()):
        raise ValueError('invalid Gaussian exploration standard deviation')
    return dict(actor_obs=235,critic_obs=235,actions=12,completed_updates=updates,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                action_noise_std_mean=float(weights['std'].mean()))


def source_identity():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ISAAC,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ISAAC,text=True).strip()
    if head!=REVISION or dirty:
        raise ValueError('Isaac Lab source differs from the locked clean revision')
    script=ISAAC/'scripts/reinforcement_learning/rsl_rl/train.py'
    return dict(isaaclab_revision=head,train_py_sha256=hashlib.sha256(script.read_bytes()).hexdigest())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=1500)
    parser.add_argument('--num-envs',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args(argv)
    root=args.output.resolve()
    command=build_command(root,args.num_envs,args.updates,args.seed)
    identity=source_identity()
    adapter=adapter_identity()
    root.mkdir(parents=True,exist_ok=False)
    status=dict(status='TRAINING_OFFICIAL_ROUGH',task=TASK,command=command,
                seed=args.seed,num_envs=args.num_envs,budget_updates=args.updates,
                source_identity=identity,adapter_sha256=adapter,
                stair_skill_verified=False,target_stair_evaluation='PENDING',
                deployment_contract='NATIVE_235_HEIGHT_SCAN_REFERENCE_ONLY',
                asset_integrity='Native official asset URL; dependency hashes not yet frozen',
                start_utc=datetime.now(timezone.utc).isoformat())
    def record():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=root/'status.tmp'
        tmp.write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.replace(root/'status.json')
    record()
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    try:
        code=run_reference_child(command,root/'train.log',300+15*args.updates)
        status['train_exit_code']=code
        if code:
            raise RuntimeError(f'official rough training exited {code}')
        if source_identity()!=identity or adapter_identity()!=adapter:
            raise ValueError('source changed during official rough training')
        checkpoints=list((root/'native').glob(f'*/model_{args.updates-1}.pt'))
        if len(checkpoints)!=1:
            raise ValueError('missing or ambiguous final checkpoint; exit zero alone is not completion')
        checkpoint=checkpoints[0]
        params=checkpoint.parent/'params'
        config=validate_effective_config(load_yaml_data(params/'env.yaml'),
                                         load_yaml_data(params/'agent.yaml'),args.num_envs,args.updates)
        metadata=validate_checkpoint(checkpoint,args.updates)
        status.update(status='TRAINING_COMPLETED_EVALUATION_PENDING',checkpoint=str(checkpoint),
                      checkpoint_metadata=metadata,effective_task=config,
                      effective_config_sha256={n:hashlib.sha256((params/n).read_bytes()).hexdigest()
                                               for n in ('env.yaml','agent.yaml')})
        status['end_utc']=datetime.now(timezone.utc).isoformat()
        record()
        print(json.dumps(status,ensure_ascii=False),flush=True)
        return 0
    except BaseException as exc:
        status.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                      error=f'{type(exc).__name__}: {exc}',end_utc=datetime.now(timezone.utc).isoformat())
        record()
        raise


if __name__=='__main__':
    raise SystemExit(main())
