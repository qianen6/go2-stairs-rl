"""Bounded diagnostic run of the unmodified pinned Unitree training entry.

This is a 45/60-observation upstream reference, NOT the custom stair baseline
or its deployment contract. Native checkpoints do not claim full RNG resume.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM = Path('C:/rl-runtimes/unitree_rl_lab')
ISAAC = Path('C:/rl-runtimes/IsaacLab')
USD = Path('C:/rl-runtimes/unitree_model/Go2/usd/go2.usd')


def build_command(output, num_envs, updates, seed, *, forward_cue=False):
    if type(updates) is not int or not 1<=updates<=300 or not 1<=num_envs<=4096:
        raise ValueError('reference budget must be 1..300 updates and 1..4096 environments')
    native=(Path(output).resolve()/'native').as_posix()
    command=[sys.executable,str(UPSTREAM/'scripts/rsl_rl/train.py'),'--task','Unitree-Go2-Velocity',
            '--headless','--num_envs',str(num_envs),'--max_iterations',str(updates),'--seed',str(seed),
            '--run_name','reference','--logger','tensorboard',
            'agent.experiment_name='+json.dumps(native,ensure_ascii=False),
            'env.scene.robot.spawn.usd_path='+json.dumps(USD.as_posix()),
            'env.scene.terrain.visual_material=null','env.scene.sky_light.spawn.texture_file=null']
    if forward_cue:
        command.append('env.commands.base_velocity.ranges.lin_vel_x=[0.6,0.8]')
    return command


def validate_native_checkpoint(path, updates):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    if type(state.get('iter')) is not int or state['iter']!=updates-1:
        raise ValueError('upstream checkpoint iteration mismatch')
    weights=state['model_state_dict']
    expected={'std':(12,)}
    for name,dims in [('actor',[45,512,256,128,12]),('critic',[60,512,256,128,1])]:
        for index,(a,b) in enumerate(zip(dims,dims[1:])):
            expected[f'{name}.{index*2}.weight']=(b,a)
            expected[f'{name}.{index*2}.bias']=(b,)
    if set(weights)!=set(expected) or not isinstance(state.get('optimizer_state_dict'),dict):
        raise ValueError('unexpected native checkpoint structure')
    for key,shape in expected.items():
        value=weights[key]
        if value.shape!=shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f'bad native tensor: {key}')
    if not bool((weights['std']>0).all()):
        raise ValueError('invalid native Gaussian standard deviation')
    return dict(actor_obs=45,critic_obs=60,actions=12,completed_updates=updates,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                action_noise_std_mean=float(weights['std'].mean()))


def reference_pass_count(report):
    from .training_health import _validated_physical_development
    return _validated_physical_development(report)[1]


def adapter_identity():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def validate_reference_origin(checkpoint,metadata,identity,assets):
    checkpoint=Path(checkpoint).resolve()
    record=json.loads((checkpoint.parents[2]/'status.json').read_text(encoding='utf-8'))
    if (record['profile']!='UPSTREAM_REFERENCE_ONLY' or record['checkpoint_metadata']['sha256']!=metadata['sha256']
            or record['source_identity']!=identity or record['asset_bundle']['sha256']!=assets['sha256']):
        raise ValueError('reference source/asset/checkpoint provenance mismatch')
    for name in ('env.yaml','agent.yaml'):
        actual=hashlib.sha256((checkpoint.parent/'params'/name).read_bytes()).hexdigest()
        if record['effective_config_sha256'][name]!=actual:
            raise ValueError('reference effective configuration changed')
    return record


def verify_reference_state(checkpoint,metadata,identity,assets,adapter=None):
    from .assets import locked_go2_assets
    if source_identity()!=identity or locked_go2_assets(USD)!=assets:
        raise ValueError('reference source/assets changed during replay')
    if adapter is not None and adapter_identity()!=adapter:
        raise ValueError('reference adapter changed during replay')
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=metadata['sha256']:
        raise ValueError('reference checkpoint changed during replay')
    validate_reference_origin(checkpoint,metadata,identity,assets)


def terminate_child_tree(proc):
    """Bounded cleanup of only this launched process and its current descendants."""
    import psutil
    try:
        parent=psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        proc.wait(timeout=3)
        return
    family=parent.children(recursive=True)+[parent]
    for item in reversed(family):
        try:item.terminate()
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:pass  # retry/escalate below and report survivors
    _,alive=psutil.wait_procs(family,timeout=3)
    for item in alive:
        try:item.kill()
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:pass
    _,alive=psutil.wait_procs(alive,timeout=3)
    if alive:
        raise RuntimeError('surviving child PIDs: '+','.join(str(p.pid) for p in alive))
    proc.wait(timeout=3)


def run_reference_child(command,log,timeout):
    environment=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with Path(log).open('w',encoding='utf-8') as output:
        # Do not use Popen's context manager: exceptional __exit__ can wait indefinitely.
        proc=subprocess.Popen(command,cwd='C:/rl-runtimes',env=environment,stdout=output,
                              stderr=subprocess.STDOUT,creationflags=flags)
        try:
            return proc.wait(timeout=timeout)
        except BaseException as cause:
            try:
                terminate_child_tree(proc)
            except BaseException as cleanup:
                raise RuntimeError(f'CHILD_CLEANUP_FAILED pid={proc.pid}: {cleanup}') from cause
            raise


def source_identity():
    expected={UPSTREAM:'4960b84732b0c2ec593dccbfe963fda1bcd7b1e3',ISAAC:'37ddf626871758333d6ed89cf64ad702aef127d0'}
    result={}
    for root,revision in expected.items():
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
        if head!=revision or dirty:
            raise ValueError(f'upstream source changed: {root}')
        result[str(root)]=head
    result['train_py_sha256']=hashlib.sha256((UPSTREAM/'scripts/rsl_rl/train.py').read_bytes()).hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=300)
    parser.add_argument('--num-envs',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--skip-replay',action='store_true',help='Use only for installation smoke checks')
    parser.add_argument('--forward-cue',action='store_true',help='Explicit one-field ablation: initial forward commands .6.. .8 m/s')
    args=parser.parse_args()
    root=args.output.resolve()
    command=build_command(root,args.num_envs,args.updates,args.seed,forward_cue=args.forward_cue)
    identity=source_identity()
    from .assets import locked_go2_assets
    assets=locked_go2_assets(USD)
    root.mkdir(parents=True,exist_ok=False)
    adapter=adapter_identity()
    status=dict(status='STARTING',profile='UPSTREAM_REFERENCE_ONLY',custom_baseline_verified=False,
                command_regime='FORWARD_CUE_ABLATION' if args.forward_cue else 'OFFICIAL_DEFAULTS',
                seed=args.seed,num_envs=args.num_envs,updates=args.updates,command=command,
                source_identity=identity,adapter_sha256=adapter,asset_bundle=assets,start_utc=datetime.now(timezone.utc).isoformat())
    def record():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=root/'status.tmp';temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(root/'status.json')
    def run(cmd,log,timeout):
        if adapter_identity()!=adapter:
            raise RuntimeError('reference adapter changed during the job')
        result=run_reference_child(cmd,log,timeout)
        if adapter_identity()!=adapter:
            raise RuntimeError('reference adapter changed during the job')
        return result
    record()
    try:
        status['status']='TRAINING_REFERENCE';record()
        code=run(command,root/'train.log',300+15*args.updates)
        status['train_exit_code']=code
        if code:
            raise RuntimeError(f'official training exited {code}; inspect train.log')
        if source_identity()!=identity or locked_go2_assets(USD)!=assets:
            raise ValueError('source or assets changed during the reference job')
        candidates=list((root/'native').glob('*/model_'+str(args.updates-1)+'.pt'))
        if len(candidates)!=1:
            raise ValueError('missing or ambiguous completed native checkpoint')
        checkpoint=candidates[0]
        metadata=validate_native_checkpoint(checkpoint,args.updates)
        for filename in ('env.yaml','agent.yaml'):
            if not (checkpoint.parent/'params'/filename).is_file():
                raise ValueError('official effective configuration was not saved')
        config_hashes={name:hashlib.sha256((checkpoint.parent/'params'/name).read_bytes()).hexdigest()
                       for name in ('env.yaml','agent.yaml')}
        status.update(status='REFERENCE_TRAINING_COMPLETED',checkpoint=str(checkpoint),checkpoint_metadata=metadata,
                      effective_config_sha256=config_hashes)
        record()
        if not args.skip_replay:
            status['status']='EVALUATING_REFERENCE';record()
            code=run([sys.executable,'-m','go2_stairs.reference_eval','--headless','--checkpoint',str(checkpoint),
                      '--output',str(root/'development')],root/'development.log',900)
            if code:
                raise RuntimeError(f'reference replay exited {code}')
            verify_reference_state(checkpoint,metadata,identity,assets,adapter)
            report=json.loads((root/'development/summary.json').read_text(encoding='utf-8'))
            if report['checkpoint_sha256']!=metadata['sha256']:
                raise ValueError('replay checkpoint identity mismatch')
            status.update(status='REFERENCE_EVALUATED',passed_count=reference_pass_count(report),
                          physical_flat_gate_passed=report['passed_count']==20)
        status['end_utc']=datetime.now(timezone.utc).isoformat();record()
        print(json.dumps(status),flush=True)
        return 0
    except BaseException as exc:
        outcome='REFERENCE_CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'REFERENCE_FAILED'
        if str(exc).startswith('CHILD_CLEANUP_FAILED'):
            outcome='REFERENCE_CLEANUP_FAILED'
        status.update(status=outcome,error=f'{type(exc).__name__}: {exc}',end_utc=datetime.now(timezone.utc).isoformat())
        record()
        raise


if __name__=='__main__':
    raise SystemExit(main())
