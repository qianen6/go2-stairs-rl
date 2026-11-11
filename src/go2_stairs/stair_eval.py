"""Frozen straight-stair evaluation of the native235-observation Rough policy.

No optimization, auto-reset, heading correction, or policy changes. Physics-step
measurements drive the existing EpisodeJudge; saved state traces drive videos.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import numpy as np

from .evaluate import build_core_suite, EpisodeJudge, EpisodeSample, summarize
from .terrain import StairSpec, build_boxes, height_at


def spec_for(case):
    return StairSpec(case['height'],case['depth'],case['steps'],case['direction'])


def case_setup(case,index):
    rng=np.random.default_rng(case['seed'])
    return dict(offset=[float(index%12*8),float(index//12*8),0.],start_x=.5,
                start_z=.4+spec_for(case).start_height,lateral=float(rng.uniform(-.05,.05)),
                yaw=float(rng.uniform(-math.pi/36,math.pi/36)))


def frozen_mesh(cases):
    import trimesh
    meshes=[];offsets=[];origins=[]
    for i,case in enumerate(cases):
        spec=spec_for(case);setup=case_setup(case,i);offset=np.array(setup['offset'])
        offsets.append(offset);origins.append(offset+np.array([.5,0,spec.start_height]))
        floor=trimesh.creation.box(extents=(8.,8.,.1))
        floor.apply_translation(offset+np.array([2.5,0.,-.05]));meshes.append(floor)
        for box in build_boxes(spec):
            mesh=trimesh.creation.box(extents=np.array(box.half_size)*2)
            mesh.apply_translation(offset+np.array(box.center));meshes.append(mesh)
    return trimesh.util.concatenate(meshes),np.array(offsets),np.array(origins)[None,:,:]


def mesh_height(mesh,x,y):
    """Test helper for our axis-aligned box mesh (not a general ray tracer)."""
    triangles=mesh.triangles
    mask=(mesh.face_normals[:,2]>.99)&(triangles[:,:,0].min(1)<=x)&(triangles[:,:,0].max(1)>=x)
    mask&=(triangles[:,:,1].min(1)<=y)&(triangles[:,:,1].max(1)>=y)
    return float(triangles[mask,:,2].max())


def on_finish(body,feet,spec):
    return all(spec.finish_x+.15<=p[0]<=spec.length and abs(p[1])<=spec.width/2
               for p in (body,*feet))


def sample_from_row(time,row):
    return EpisodeSample(time,tuple(row[:3]),tuple(map(tuple,row[3:15].reshape(4,3))),
                         float(row[15]),float(row[16]),float(row[17]),tuple(row[18:21]),
                         float(row[21]),float(row[22]))


def judge_trace(case,trace,dt):
    judge=EpisodeJudge(spec_for(case),physics_dt=dt)
    result={'done':False,'success':False,'reason':'incomplete_trace','elapsed_s':0.}
    for i,row in enumerate(trace):
        result=judge.step(sample_from_row((i+1)*dt,row))
        if result['done']:return result
    return dict(done=True,success=False,reason='incomplete_trace',elapsed_s=len(trace)*dt)


def select_replays(rows):
    selected=[]
    for direction in ('up','down'):
        for success in (True,False):
            match=next((r['case_id'] for r in rows if r['direction']==direction and r['success']==success),None)
            if match is not None:selected.append(match)
    return selected


def ensure_finite(tensor,label):
    import torch
    if not bool(torch.isfinite(tensor).all()):raise ValueError('nonfinite '+label)


def persist_invalid_run(out,cases,partial,elapsed,exc,traces,states,physics_dt,policy_dt,metadata):
    """Even a global/terminal-row fault retains the complete frozen denominator."""
    results=[]
    for case,result in zip(cases,partial):
        if result is None:
            result=dict(done=True,success=False,reason='runtime_error',elapsed_s=float(elapsed))
        results.append(dict(**case,**result))
    summary=summarize(cases,results)
    summary['passed']=False  # incomplete/invalid execution can never certify acceptance
    report=dict(status='EVALUATION_INVALID',evaluation_valid=False,error=repr(exc),
                results=results,summary=summary,checkpoint_metadata=metadata,
                training_seed=metadata.get('training_seed'),formal_three_seed_baseline_verified=False)
    np.savez_compressed(out/'physics_trace.partial.npz',measurements=np.asarray(traces),dt=physics_dt)
    np.savez_compressed(out/'state_trace.partial.npz',states=np.asarray(states),dt=policy_dt)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def verify_training(checkpoint):
    from .official_rough import source_identity,validate_checkpoint,validate_effective_config,load_yaml_data
    p=Path(checkpoint).resolve();record=json.loads((p.parents[2]/'status.json').read_text(encoding='utf-8'))
    if record['status']!='TRAINING_COMPLETED_EVALUATION_PENDING' or str(p)!=record['checkpoint']:
        raise ValueError('not the completed frozen final training checkpoint')
    metadata=validate_checkpoint(p,record['budget_updates'])
    if metadata['sha256']!=record['checkpoint_metadata']['sha256'] or source_identity()!=record['source_identity']:
        raise ValueError('training checkpoint/source identity mismatch')
    for name in ('env.yaml','agent.yaml'):
        if hashlib.sha256((p.parent/'params'/name).read_bytes()).hexdigest()!=record['effective_config_sha256'][name]:
            raise ValueError('training configuration changed')
    validator=validate_effective_config
    if record.get('profile')=='LOW_SPEED_REPAIR_V1':
        from .repair_train import validate_repair_config
        validator=validate_repair_config
    validator(load_yaml_data(p.parent/'params/env.yaml'),load_yaml_data(p.parent/'params/agent.yaml'),
              record['num_envs'],record['budget_updates'])
    return dict(metadata,training_seed=record['seed'],training_profile=record.get('profile','ORIGINAL_GO2_ROUGH'))


def make_scene(cases,device,render=False):
    """Keep native robot/control/observation definitions, replace only evaluation conditions."""
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
    mesh,offsets,origins=frozen_mesh(cases)
    class FrozenGenerator:
        def __init__(self,cfg,device):
            self.terrain_mesh=mesh;self.terrain_origins=origins;self.flat_patches={}
    cfg=UnitreeGo2RoughEnvCfg();cfg.seed=424242;cfg.scene.num_envs=len(cases);cfg.sim.device=device
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(class_type=FrozenGenerator,size=(8.,8.),
        num_rows=1,num_cols=len(cases),sub_terrains={},curriculum=False)
    cfg.scene.terrain.max_init_terrain_level=0
    cfg.scene.terrain.visual_material=None;cfg.scene.sky_light.spawn.texture_file=None
    if render:
        from isaaclab.sim import PreviewSurfaceCfg
        cfg.scene.terrain.visual_material=PreviewSurfaceCfg(diffuse_color=(.22,.28,.35),roughness=.8)
    cfg.curriculum.terrain_levels=None
    cfg.observations.policy.enable_corruption=False
    # Nominal friction remains the original fixed .8/.6 startup material; no mass/pose randomization.
    for name in ('add_base_mass','base_com','base_external_force_torque','reset_base','reset_robot_joints','push_robot'):
        setattr(cfg.events,name,None)
    cfg.terminations.time_out=None;cfg.terminations.base_contact=None
    cfg.commands.base_velocity.debug_vis=False
    cfg.viewer.resolution=(1280,720)
    env=ManagerBasedRLEnv(cfg,render_mode='rgb_array' if render else None)
    env.reset()
    return env,offsets


def physics_step(env):
    env._sim_step_counter+=1
    env.action_manager.apply_action();env.scene.write_data_to_sim()
    env.sim.step(render=False);env.scene.update(dt=env.physics_dt)


def initial_pose(env,cases,offsets):
    import torch
    robot=env.scene['robot'];state=robot.data.default_root_state.clone()
    for i,c in enumerate(cases):
        s=case_setup(c,i)
        state[i,:3]=torch.tensor(offsets[i]+[s['start_x'],s['lateral'],s['start_z']],device=env.device)
        state[i,3:7]=torch.tensor([math.cos(s['yaw']/2),0,0,math.sin(s['yaw']/2)],device=env.device)
    state[:,7:]=0
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
    env.action_manager.process_action(torch.zeros(env.num_envs,12,device=env.device))
    # Explicit initial condition only: settle native nominal joint targets for one second.
    for _ in range(round(1/env.physics_dt)):physics_step(env)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    meta=verify_training(args.checkpoint)
    from .reference import adapter_identity
    source=adapter_identity();app=None;env=None;code=1
    cases=build_core_suite()
    if args.smoke:cases=[cases[i] for i in (0,1,5,6)]
    out=args.output.resolve()
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    results=[None]*len(cases);traces=[];states=[];physics_count=0
    try:
        app=AppLauncher(args).app
        import torch
        from tensordict import TensorDict
        from rsl_rl.modules import ActorCritic
        env,offsets=make_scene(cases,args.device)
        initial_pose(env,cases,offsets)
        robot=env.scene['robot'];sensor=env.scene['contact_forces']
        feet,feet_names=robot.find_bodies('.*_foot',preserve_order=True)
        torso,torso_names=sensor.find_bodies('base|Head.*',preserve_order=True)
        legs,leg_names=sensor.find_bodies('.*_(hip|thigh|calf)',preserve_order=True)
        if len(feet)!=4 or not torso or not legs:raise ValueError('unexpected contact/body mapping')
        command=env.command_manager.get_term('base_velocity');command.vel_command_b.zero_()
        obs=TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])
        if obs['policy'].shape!=(env.num_envs,235):raise ValueError('native observation mismatch')
        policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},12,
            actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',
            actor_obs_normalization=False,critic_obs_normalization=False).to(env.device)
        policy.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']);policy.eval()
        offset=torch.tensor(offsets,dtype=torch.float32,device=env.device)
        def measurements():
            d=robot.data;force=sensor.data.net_forces_w.norm(dim=-1)
            return torch.cat((d.root_pos_w-offset,(d.body_pos_w[:,feet]-offset[:,None,:]).flatten(1),
                d.root_lin_vel_w[:,:2].norm(dim=-1,keepdim=True),d.root_ang_vel_b[:,2:3],
                torch.acos((-d.projected_gravity_b[:,2:3]).clamp(-1,1)),command.vel_command_b,
                force[:,torso].amax(1,keepdim=True),force[:,legs].amax(1,keepdim=True)),dim=1).cpu().numpy()
        # Raycaster sees the same solid mesh as contacts. Verify sampled surfaces before scoring.
        hits=env.scene['height_scanner'].data.ray_hits_w.cpu().numpy()
        errors=[]
        for i,c in enumerate(cases):
            for hit in hits[i]:
                x,y,z=hit-offsets[i]
                errors.append(abs(float(z)-height_at(float(x),float(y),spec_for(c))))
        if not np.isfinite(hits).all() or max(errors)>.003:
            raise ValueError(f'height raycaster mismatch: {max(errors)}')
        initial=measurements();results=[None]*len(cases);crossed=[False]*len(cases)
        judges=[EpisodeJudge(spec_for(c),physics_dt=env.physics_dt) for c in cases]
        for i,row in enumerate(initial):
            if not np.isfinite(row).all() or row[15]>.1 or row[17]>.35 or row[2]-spec_for(cases[i]).start_height<.18:
                results[i]=dict(done=True,success=False,reason='initialization_failed',elapsed_s=0.)
        traces=[];states=[torch.cat((robot.data.root_state_w,robot.data.joint_pos,robot.data.joint_vel),dim=1).cpu().numpy().copy()];physics_count=0
        max_steps=20 if args.smoke else round(60/env.step_dt)
        with torch.inference_mode():
            for step in range(max_steps):
                command.vel_command_b.zero_()
                for i,c in enumerate(cases):
                    if results[i] is None and not crossed[i]:command.vel_command_b[i,0]=c['command_vx']
                obs=TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])
                ensure_finite(obs['policy'],'observation')
                action=policy.act_inference(obs)
                ensure_finite(action,'action')
                env.action_manager.process_action(action)
                for _ in range(env.cfg.decimation):
                    physics_step(env);physics_count+=1;rows=measurements();traces.append(rows.copy())
                    for i,row in enumerate(rows):
                        if results[i] is not None:continue
                        sample=sample_from_row(physics_count*env.physics_dt,row)
                        result=judges[i].step(sample)
                        if on_finish(sample.body_position,sample.foot_positions,spec_for(cases[i])):crossed[i]=True
                        if result['done']:results[i]=result
                states.append(torch.cat((robot.data.root_state_w,robot.data.joint_pos,robot.data.joint_vel),dim=1).cpu().numpy().copy())
                if step%250==0:
                    print(f'EVAL_SECONDS={(step+1)*env.step_dt:.2f} TERMINAL={sum(r is not None for r in results)}/{len(cases)}',flush=True)
                if all(r is not None for r in results):break
        trace=np.asarray(traces,dtype=np.float32)
        np.savez_compressed(out/'physics_trace.npz',measurements=trace,dt=env.physics_dt)
        np.savez_compressed(out/'state_trace.npz',states=np.asarray(states),offsets=offsets,dt=env.step_dt,
                            joint_names=np.array(robot.joint_names))
        for i,c in enumerate(cases):
            if results[i] is None:results[i]=dict(done=True,success=False,reason='incomplete_trace',elapsed_s=physics_count*env.physics_dt)
            if results[i]['reason']!='initialization_failed':
                replay=judge_trace(c,trace[:,i],env.physics_dt)
                if replay!=results[i]:raise ValueError(f'offline judge mismatch: {c["case_id"]}: {replay} != {results[i]}')
        rows=[dict(**c,**r,fully_crossed=bool(crossed[i])) for i,(c,r) in enumerate(zip(cases,results))]
        if verify_training(args.checkpoint)!=meta or adapter_identity()!=source:raise ValueError('model/source changed during evaluation')
        report=dict(status='ENGINE_SMOKE_OK' if args.smoke else 'SINGLE_SEED_CORE_EVALUATED',evaluation_valid=not args.smoke,
            checkpoint=str(args.checkpoint.resolve()),checkpoint_metadata=meta,training_seed=meta['training_seed'],training_seeds_evaluated=1,
            formal_three_seed_baseline_verified=False,results=rows,summary=summarize(cases,rows),
            replay_case_ids=select_replays(rows),physics_dt=env.physics_dt,policy_dt=env.step_dt,
            trace_sha256={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('physics_trace.npz','state_trace.npz')},
            height_probe_max_error_m=max(errors),contact_body_names=dict(feet=feet_names,torso=torso_names,legs=leg_names),
            setup=[case_setup(c,i) for i,c in enumerate(cases)],adapter_sha256=source,
            protocol='Native235 policy, deterministic inference; original PD/dynamics and fixed .8/.6 friction. No mass randomization or observation noise. One-second nominal-PD initial settling, then0.1/0.2m/s body-forward, zero lateral/yaw; latch zero command after full crossing;60s timeout; judge every physics step. No auto-resets. Native asset dependencies remain unpinned cloud contents.')
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'status':report['status'],'summary':report['summary']}),flush=True);code=0
    except BaseException as exc:
        dt=env.physics_dt if env is not None else .005
        step_dt=env.step_dt if env is not None else .02
        persist_invalid_run(out,cases,results,physics_count*dt,exc,traces,states,dt,step_dt,meta)
        (out/'ERROR.json').write_text(json.dumps({'status':'EVALUATION_INVALID','error':repr(exc),'checkpoint':str(args.checkpoint)}),encoding='utf-8')
        traceback.print_exc()
    finally:
        if code:sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if env is not None:env.close()
        if app is not None:app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
