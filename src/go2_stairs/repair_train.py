"""Low-speed adaptation of the native Go2 Rough policy; frozen tests unchanged.

Resume model and Adam, but start fresh physics episodes/RNG. This is a task-
adapted continuation, NOT an unchanged upstream reproduction or accepted skill.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import torch

PROFILE='LOW_SPEED_REPAIR_V1'


def sample_repair_commands(count,device):
    selector=torch.rand(count,device=device)
    regime=torch.where(selector<.6,0,torch.where(selector<.85,1,2))
    command=torch.rand(count,3,device=device)*2-1
    low=regime==0;stop=regime==2
    command[low,0]=.08+torch.rand(int(low.sum()),device=device)*.17
    command[low,1:]=0;command[stop]=0
    return command,regime


def repair_tracking(command,velocity):
    from .training_health import relative_tracking_reward
    return relative_tracking_reward(command,velocity,.5)


def eligible_curriculum_ids(episode_lengths,env_ids):
    return env_ids[episode_lengths[env_ids]>0]


def repair_terrain_levels(env,env_ids):
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    eligible=eligible_curriculum_ids(env.episode_length_buf,env_ids)
    # Before the very first reset, robot poses need not yet be at terrain origins.
    # A large spawn/origin distance is not evidence of completed stair traversal.
    if eligible.numel():return mdp.terrain_levels_vel(env,eligible)
    return env.scene.terrain.terrain_levels.float().mean()


def command_metrics(command,velocity):
    low=(command[:,0]>.05)&(command[:,0]<.3)&(command[:,1:].abs().amax(1)<1e-6)
    count=low.sum();denominator=count.clamp_min(1)
    error=(command[:,:2]-velocity).norm(dim=-1)
    stalled=(velocity[:,0]<command[:,0]*.2).float()
    return dict(low_count=count,low_tracking_error=(error*low).sum()/denominator,
                low_stall_fraction=(stalled*low).sum()/denominator)


def tracking_reward(env,command_name):
    command=env.command_manager.get_command(command_name)
    velocity=env.scene['robot'].data.root_lin_vel_b[:,:2]
    for name,value in command_metrics(command,velocity).items():
        env.extras.setdefault('log',{})['Repair/'+name]=value.detach()
    terrain=env.scene.terrain
    for name,mask in [('positive_stairs',(terrain.terrain_types<4)),
                       ('inverted_stairs',(terrain.terrain_types>=4)&(terrain.terrain_types<8))]:
        if bool(mask.any()):env.extras['log']['Repair/'+name+'_terrain_level']=terrain.terrain_levels[mask].float().mean()
    return repair_tracking(command[:,:2],velocity)


def restore_runner(runner,checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if type(state.get('iter')) is not int or state['iter']<0:raise ValueError('invalid resume counter')
    runner.alg.policy.load_state_dict(state['model_state_dict'])
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    # RSL3.1.2 stores the LAST zero-based update; next update must not repeat it.
    runner.current_learning_iteration=state['iter']+1
    runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
    for name,value in runner.alg.policy.state_dict().items():
        if not torch.equal(value.cpu(),state['model_state_dict'][name].cpu()):
            raise ValueError('resume weight mismatch: '+name)
    return dict(base_completed_updates=state['iter']+1,optimizer_restored=True,
                learning_rate=runner.alg.learning_rate,initial_weights_exact_match=True,
                physics_and_rng='fresh episodes; not bitwise simulator resume')


def configure_repair(cfg):
    from isaaclab.envs.mdp.commands import UniformVelocityCommand
    from isaaclab.managers import RewardTermCfg,SceneEntityCfg
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    class RepairVelocityCommand(UniformVelocityCommand):
        def _resample_command(self,env_ids):
            values,regimes=sample_repair_commands(len(env_ids),self.device)
            self.vel_command_b[env_ids]=values
            self.is_heading_env[env_ids]=False
            self.is_standing_env[env_ids]=regimes==2
    command=cfg.commands.base_velocity
    command.class_type=RepairVelocityCommand
    command.heading_command=False;command.rel_heading_envs=0.;command.rel_standing_envs=.15
    command.resampling_time_range=(30.,50.)
    command.debug_vis=False
    cfg.episode_length_s=60.
    cfg.scene.terrain.max_init_terrain_level=1
    cfg.curriculum.terrain_levels.func=repair_terrain_levels
    cfg.rewards.track_lin_vel_xy_exp.func=tracking_reward
    cfg.rewards.track_lin_vel_xy_exp.params={'command_name':'base_velocity'}
    cfg.rewards.undesired_contacts=RewardTermCfg(func=mdp.undesired_contacts,weight=-1.,
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names='.*_(thigh|calf)'), 'threshold':1.})
    return cfg


def validate_repair_config(env,agent,num_envs,total_updates):
    from .official_rough import validate_effective_config
    result=validate_effective_config(env,agent,num_envs,total_updates)
    c=env['commands']['base_velocity'];r=env['rewards']
    required=(env['episode_length_s']==60.,env['scene']['terrain']['max_init_terrain_level']==1,
              c['heading_command'] is False,c['rel_standing_envs']==.15,
              'repair_terrain_levels' in str(env['curriculum']['terrain_levels']['func']),
              list(c['resampling_time_range'])==[30.,50.],
              'RepairVelocityCommand' in str(c['class_type']),
              'tracking_reward' in str(r['track_lin_vel_xy_exp']['func']),
              r['track_lin_vel_xy_exp']['weight']==1.5,
              r['undesired_contacts']['weight']==-1.,r['undesired_contacts']['params']['threshold']==1.)
    if not all(required):raise ValueError('repair effective configuration mismatch')
    return dict(result,profile=PROFILE,original_rewards=False,command_mixture='60% .08.. .25 straight;25% general;15% stop')


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--seed',type=int,default=0)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if not 1<=args.updates<=1500 or not 1<=args.num_envs<=4096:raise ValueError('invalid bounded repair budget')
    from .stair_eval import verify_training
    from .official_rough import source_identity,validate_checkpoint,load_yaml_data
    from .reference import adapter_identity
    parent=args.checkpoint.resolve();base=verify_training(parent)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    native=root/'native'/'repair';native.mkdir(parents=True)
    identity=source_identity();adapter=adapter_identity()
    total=base['completed_updates']+args.updates
    status=dict(status='STARTING_REPAIR',profile=PROFILE,task='Isaac-Velocity-Rough-Unitree-Go2-v0',
                seed=args.seed,num_envs=args.num_envs,budget_updates=total,additional_budget_updates=args.updates,
                base_completed_updates=base['completed_updates'],completed_updates=base['completed_updates'],
                parent_checkpoint=str(parent),parent_sha256=base['sha256'],source_identity=identity,
                adapter_sha256=adapter,trainer_entry='go2_stairs.repair_train',stair_skill_verified=False,
                start_utc=datetime.now(timezone.utc).isoformat(),checkpoints=[])
    def record():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=root/'status.tmp';temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(root/'status.json')
    record();app=None;env=None;code=1
    try:
        app=AppLauncher(args).app
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
        from .contracts import FiniteActionMixin
        class Wrapper(FiniteActionMixin,RslRlVecEnvWrapper):pass
        cfg=configure_repair(UnitreeGo2RoughEnvCfg());cfg.seed=args.seed;cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device
        cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
        cfg.log_dir=str(native)
        agent=UnitreeGo2RoughPPORunnerCfg();agent.seed=args.seed;agent.device=args.device
        agent.max_iterations=total;agent.save_interval=50;agent.logger='tensorboard'
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        env=ManagerBasedRLEnv(cfg);wrapper=Wrapper(env,clip_actions=agent.clip_actions)
        runner=OnPolicyRunner(wrapper,agent.to_dict(),log_dir=str(native),device=agent.device)
        resumed=restore_runner(runner,parent)
        if resumed['base_completed_updates']!=base['completed_updates']:raise ValueError('resume count mismatch')
        dump_yaml(str(native/'params/env.yaml'),cfg);dump_yaml(str(native/'params/agent.yaml'),agent)
        effective=validate_repair_config(load_yaml_data(native/'params/env.yaml'),load_yaml_data(native/'params/agent.yaml'),args.num_envs,total)
        status.update(status='TRAINING_REPAIR',resume=resumed,effective_task=effective,
            effective_config_sha256={n:hashlib.sha256((native/'params'/n).read_bytes()).hexdigest() for n in ('env.yaml','agent.yaml')})
        record();completed=0
        while completed<args.updates:
            block=min(100,args.updates-completed)
            runner.current_learning_iteration=base['completed_updates']+completed
            runner.learn(num_learning_iterations=block,init_at_random_ep_len=completed==0)
            completed+=block
            checkpoint=native/f'model_{base["completed_updates"]+completed-1}.pt'
            metadata=validate_checkpoint(checkpoint,base['completed_updates']+completed)
            if source_identity()!=identity or adapter_identity()!=adapter:raise ValueError('source changed during repair')
            if hashlib.sha256(parent.read_bytes()).hexdigest()!=base['sha256']:raise ValueError('parent model changed')
            for name,digest in status['effective_config_sha256'].items():
                if hashlib.sha256((native/'params'/name).read_bytes()).hexdigest()!=digest:raise ValueError('effective config changed')
            status.update(completed_updates=base['completed_updates']+completed,additional_completed_updates=completed,
                          checkpoint=str(checkpoint),checkpoint_metadata=metadata)
            status['checkpoints'].append(dict(additional_updates=completed,**metadata));record()
        status.update(status='TRAINING_COMPLETED_EVALUATION_PENDING',end_utc=datetime.now(timezone.utc).isoformat(),train_exit_code=0)
        record();print(json.dumps(status),flush=True);code=0
    except BaseException as exc:
        status.update(status='REPAIR_FAILED',error=repr(exc),train_exit_code=1,end_utc=datetime.now(timezone.utc).isoformat());record()
        traceback.print_exc()
    finally:
        if code:sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if env is not None:env.close()
        if app is not None:app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
