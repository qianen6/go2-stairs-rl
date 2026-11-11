"""Same physical development protocol, native 45/60 reference policy only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback


def reference_policy_step(wrapper,policy,step,speed):
    if not 0<=step<900:
        raise ValueError('step outside fixed reference protocol')
    command=wrapper.unwrapped.command_manager.get_term('base_velocity')
    command.time_left.fill_(1000.);command.is_standing_env.fill_(False)
    command.vel_command_b.zero_()
    if 50<=step<550:
        phase='move';command.vel_command_b[:,0]=speed
    elif 550<=step<700:
        phase='turn';command.vel_command_b[:,2]=.2
    else:
        phase='stop' if step>=700 else 'stand'
    requested=command.vel_command_b.clone()
    # Native single-frame observations can be refreshed without advancing physics/history.
    action=policy.act_inference(wrapper.get_observations())
    return phase,requested,action,wrapper.step(action)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    app=None;env=None;code=1
    try:
        from .reference import (source_identity,validate_native_checkpoint,validate_reference_origin,
                                verify_reference_state,adapter_identity,reference_pass_count,USD)
        from .assets import locked_go2_assets
        identity=source_identity();assets=locked_go2_assets(USD);adapter=adapter_identity()
        match=re.fullmatch(r'model_(\d+)\.pt',args.checkpoint.name)
        if match is None:
            raise ValueError('not a native checkpoint filename')
        meta=validate_native_checkpoint(args.checkpoint,int(match[1])+1)
        validate_reference_origin(args.checkpoint,meta,identity,assets)
        import yaml
        native=args.checkpoint.parent
        with (native/'params/env.yaml').open(encoding='utf-8') as stream:
            saved_env=yaml.load(stream,Loader=yaml.BaseLoader)
        with (native/'params/agent.yaml').open(encoding='utf-8') as stream:
            saved_agent=yaml.load(stream,Loader=yaml.BaseLoader)
        if float(saved_env['sim']['dt'])!=.005 or int(saved_env['decimation'])!=4 or saved_agent['policy']['activation']!='elu':
            raise ValueError('unexpected upstream control/policy settings')
        app=AppLauncher(args).app
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules import ActorCritic
        from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
        from .contracts import FiniteActionMixin
        from .flat_development import FlatEpisodeMetrics
        class Wrapper(FiniteActionMixin,RslRlVecEnvWrapper):
            pass
        cfg=RobotEnvCfg();cfg.seed=2000;cfg.scene.num_envs=20;cfg.sim.device=args.device
        cfg.scene.robot.spawn.usd_path=str(USD)
        cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
        cfg.observations.policy.enable_corruption=False
        cfg.commands.base_velocity.debug_vis=False
        # Development replay uses nominal dynamics, like the custom flat gate.
        cfg.events.physics_material=None;cfg.events.add_base_mass=None;cfg.events.push_robot=None
        cfg.curriculum.terrain_levels=None;cfg.curriculum.lin_vel_cmd_levels=None
        env=ManagerBasedRLEnv(cfg)
        wrapper=Wrapper(env,clip_actions=None)
        obs=wrapper.get_observations()
        assert obs['policy'].shape==(20,45) and obs['critic'].shape==(20,60)
        policy=ActorCritic(obs,{'policy':['policy'],'critic':['critic']},12,
                           actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',
                           actor_obs_normalization=False,critic_obs_normalization=False).to(env.device)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        policy.load_state_dict(saved['model_state_dict']);policy.eval()
        tracker=[FlatEpisodeMetrics() for _ in range(20)];trace=[]
        speed=torch.tensor([.1,.2]*10,device=env.device)
        with torch.inference_mode():
            for step in range(900):
                phase,requested,action,result=reference_policy_step(wrapper,policy,step,speed)
                requested=requested.cpu().numpy()
                obs,reward,done,_=result
                data=env.scene['robot'].data
                velocity=torch.cat((data.root_lin_vel_b[:,:2],data.root_ang_vel_b[:,2:3]),-1).cpu().numpy()
                done=done.cpu().numpy()
                for index,item in enumerate(tracker):
                    item.update(phase,requested[index],velocity[index],bool(done[index]))
                trace.append(torch.cat((data.root_pos_w,data.root_quat_w,data.joint_pos,
                                        data.root_lin_vel_b,data.root_ang_vel_b,action),-1).cpu().numpy())
        cases=[dict(case_id=f'flat-dev-{i:02}',**item.result()) for i,item in enumerate(tracker)]
        states=np.stack(trace)
        np.savez_compressed(args.output/'trajectory.npz',states=states,joint_names=env.scene['robot'].joint_names)
        report=dict(status='FLAT_DEVELOPMENT_EVALUATED',profile='UPSTREAM_REFERENCE_ONLY',
                    formal_stair_acceptance=False,custom_control_contract_verified=False,
                    actor_obs=45,critic_obs=60,physics_dt=.005,policy_dt=.02,
                    checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=meta['sha256'],
                    scene_seed=2000,noise=False,cases=cases,passed_count=sum(c['passed'] for c in cases),
                    passed=all(c['passed'] for c in cases),
                    protocol='18s: stand1, forward10(vx=.1/.2), turn3(yaw=.2), stop4; 20 parallel trials',
                    thresholds=dict(no_termination=True,stable_stop_s=1.,move_rmse_mps=.1,
                                    turn_rmse_radps=.15,minimum_commanded_distance_fraction=.5),
                    native_action_metrics=dict(raw_abs_gt_one_fraction=float((abs(states[:,:,-12:])>1).mean()),
                                               max_abs_action=float(abs(states[:,:,-12:]).max())),
                    action_note='Native Gaussian raw action is not clipped to [-1,1]; do not apply the custom bound gate.',
                    source_identity=identity,asset_bundle=assets,evaluation_adapter_sha256=adapter)
        assert reference_pass_count(report)==report['passed_count']
        verify_reference_state(args.checkpoint,meta,identity,assets,adapter)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(f'REFERENCE_PHYSICAL_REPLAY passed={report["passed_count"]}/20; custom_contract_verified=false',flush=True)
        code=0
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps({'status':'FAILED','error':f'{type(exc).__name__}: {exc}'}),encoding='utf-8')
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        if code:
            sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if app is not None:
            app.close()
    return code


if __name__=='__main__':
    raise SystemExit(main())
