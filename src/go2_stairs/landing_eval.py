"""Frozen-weight runtime contrasts and corrected stair evaluation, never training."""
import argparse,hashlib,json,math,os,sys,time,traceback,types
from pathlib import Path
import numpy as np
from .reference_reproduction import ordered_body_indices
from .guarded_finetune import verify_finetune_checkpoint as verify_public_checkpoint
from .fine_eval_factory import make_scene,native_cases,validate_native_batch,FrozenActor


def build_retest_cases(suite):
    rows=[]
    configs=[(.2,[.8,1.],10)] if suite=='evaluation' else [(.2,[.5,.8,1.],2),(.1,[.8],2)]
    for height,speeds,repeats in configs:
        for speed in speeds:
            for direction in ['up','down']:
                for repeat in range(repeats):
                    i=len(rows);rows.append(dict(case_id=f'{suite}-{i:03}',phase='stairs',height=height,speed=speed,direction=direction,seed=(53000 if suite=='evaluation' else 52000)+i,repeat=repeat))
    return rows


def foot_riser_event(previous,current,edge,start_height,riser_height):
    return dict(crossed_plane=bool(previous[0]<edge<=current[0]),height_above_start=float(current[2]-start_height),
                clearance_above_step=float(current[2]-start_height-riser_height))


from .landing_protocol import InteriorLandingJudge as RetestJudge


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--suite',choices=['diagnostic','evaluation'],default='diagnostic');p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=16)
    p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--riser-height',type=float,default=.2);p.add_argument('--direction',choices=['up','down','flat'],required=True)
    p.add_argument('--duration',type=float,default=30.);p.add_argument('--settle',type=float,default=0.)
    p.add_argument('--initial-from',type=Path);p.add_argument('--legacy-motor',action='store_true')
    p.add_argument('--release-stabilization',action='store_true');p.add_argument('--release-velocity-limits',action='store_true');p.add_argument('--physics-dt',type=float);p.add_argument('--solver-iterations',type=int,nargs=2)
    p.add_argument('--heading-offset',type=float,default=0.,help='declared diagnostic approach angle in radians; default0')
    p.add_argument('--record',default='');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    cases=native_cases(args.riser_height,args.direction,args.count);record=[int(x) for x in args.record.split(',') if x]
    if not 1<=len(cases)<=32 or len(record)>4 or any(i>=len(cases) or i<0 for i in record):raise ValueError('bad batch/video selection')
    for case in cases:
        case['heading_offset']=args.heading_offset
        case['seed']+=args.seed_offset
        if args.seed_offset:case['case_id']+='-offset'+str(args.seed_offset)
    meta=verify_public_checkpoint(args.checkpoint)
    actuator_groups=meta.get('actuators')
    if meta.get('source_kind')=='STAIRS25_TRAINED' and (args.release_velocity_limits or args.release_stabilization or args.legacy_motor):
        raise ValueError('STAIRS25 checkpoints are evaluated with their own training physics; release/legacy flags would mismatch')
    args.output.mkdir(parents=True,exist_ok=False);out=args.output
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    save('manifest.json',dict(command=sys.argv,training=False,checkpoint=meta,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='RUNNING'))
    save('cases.json',cases)
    app=None;env=None;writers={};code=1;judges=[];measurements=[];body_trace=[];force_trace=[];policy_trace=[];obs_trace=[];action_trace=[]
    try:
        args.enable_cameras=bool(record);app=AppLauncher(args).app
        import torch,carb
        torch.set_num_threads(4)
        from isaaclab.utils.math import matrix_from_quat
        from isaaclab.utils.io import dump_yaml
        env,offsets,geometry=make_scene(cases,args.device,bool(record),context_border=6.,physics_dt=args.physics_dt,solver_iterations=args.solver_iterations,release_velocity_limits=args.release_velocity_limits,release_stabilization=args.release_stabilization,actuator_groups=actuator_groups)
        robot=env.scene['robot'];sensor=env.scene['contact_forces'];scanner=env.scene['height_scanner'];cmd=env.command_manager.get_term('base_velocity')
        ids=torch.arange(len(cases),dtype=torch.int32,device='cpu')
        if args.initial_from:
            a=np.load(args.initial_from)
            if a['masses'].shape!=tuple(robot.root_physx_view.get_masses().shape):raise ValueError('initial mass shape mismatch')
            if list(a['case_ids'])!=[c['case_id'] for c in cases]:raise ValueError('initial case order mismatch')
            robot.root_physx_view.set_masses(torch.tensor(a['masses'],device='cpu'),ids)
            robot.root_physx_view.set_inertias(torch.tensor(a['inertias'],device='cpu'),ids)
            robot.write_root_state_to_sim(torch.tensor(a['root'],device=env.device))
            robot.write_joint_state_to_sim(torch.tensor(a['q'],device=env.device),torch.zeros_like(robot.data.joint_vel))
            for ss in env.scene.sensors.values():ss.reset()
            env.scene.update(0.)
        masses=robot.root_physx_view.get_masses().cpu().numpy().copy();inertias=robot.root_physx_view.get_inertias().cpu().numpy().copy()
        np.savez_compressed(out/'initial.npz',masses=masses,inertias=inertias,case_ids=[c['case_id'] for c in cases],root=robot.data.root_state_w.cpu().numpy(),q=robot.data.joint_pos.cpu().numpy())
        if args.legacy_motor:
            def old_clip(self,effort):
                hi=torch.clip(self._saturation_effort*(1-self._joint_vel/self.velocity_limit),min=self._zeros_effort,max=self.effort_limit)
                lo=torch.clip(self._saturation_effort*(-1-self._joint_vel/self.velocity_limit),min=-self.effort_limit,max=self._zeros_effort)
                return torch.clip(effort,min=lo,max=hi)
            for act in robot.actuators.values():act._clip_effort=types.MethodType(old_clip,act)
        feet,foot_names=robot.find_bodies('.*_foot',preserve_order=True);sf=ordered_body_indices(foot_names,sensor.body_names)
        torso,_=sensor.find_bodies('base|Head.*',preserve_order=True);legs,_=sensor.find_bodies('.*_(hip|thigh|calf)',preserve_order=True)
        body_to_sensor=ordered_body_indices(robot.body_names,sensor.body_names)
        off=torch.tensor(offsets,dtype=torch.float32,device=env.device);heading=robot.data.heading_w.clone()
        judges=[RetestJudge(g['last_edge_x'],g['last_edge_x']+3,g['ground_min'],args.duration) for g in geometry]
        obs=env.observation_manager.compute()
        policy=FrozenActor(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state_dict'],env.device)
        if obs['policy'].shape!=(len(cases),235):raise ValueError('observation contract mismatch')
        dump_yaml(str(out/'env.yaml'),env.cfg)
        for layer in env.sim.stage.GetUsedLayers():
            if layer.identifier.endswith('/Go2/go2.usd'):
                (out/'go2-root.usda').write_text(layer.ExportToString(),encoding='utf-8')
        import importlib.metadata
        runtime=dict(package_versions={n:importlib.metadata.version(n) for n in ['isaacsim','torch','numpy','isaaclab']},native_generator=True,native_reset=True,device=env.device,legacy_motor=args.legacy_motor,settle_s=args.settle,heading_offset=args.heading_offset,
            collision_approximate_cylinders=carb.settings.get_settings().get('/physics/collisionApproximateCylinders'),
            mass_sha256=hashlib.sha256(masses.tobytes()).hexdigest(),inertia_sha256=hashlib.sha256(inertias.tobytes()).hexdigest(),
            joint_names=robot.joint_names,body_names=robot.body_names,foot_indices=feet,foot_names=foot_names,
            release_velocity_limits=args.release_velocity_limits,enable_stabilization=env.cfg.sim.physx.enable_stabilization,
            physx_max_velocity=robot.root_physx_view.get_dof_max_velocities().cpu().tolist(),
            physx_max_effort=robot.root_physx_view.get_dof_max_forces().cpu().tolist(),
            physx_drive_stiffness=robot.root_physx_view.get_dof_stiffnesses().cpu().tolist(),physx_drive_damping=robot.root_physx_view.get_dof_dampings().cpu().tolist(),
            actuator_cfg=str(env.cfg.scene.robot.actuators),step_dt=env.step_dt,physics_dt=env.physics_dt,
            solver_iterations=[env.cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count,env.cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count],
            initial_root=robot.data.root_state_w.cpu().tolist())
        save('runtime.json',runtime);save('geometry.json',geometry)
        physics_steps=0;missing_counts=np.zeros(len(cases),dtype=int);events=[];previous_feet=None
        def measure():
            d=robot.data;force=sensor.data.net_forces_w;norm=force.norm(dim=-1);pos=d.root_pos_w-off;links=d.body_pos_w-off[:,None]
            ext=torch.abs(matrix_from_quat(d.root_quat_w))@torch.tensor([.4,.16,.15],device=env.device)
            lower=torch.minimum(pos-ext,links.amin(1)-.05);upper=torch.maximum(pos+ext,links.amax(1)+.05)
            rows=torch.cat((pos,(links[:,feet]).flatten(1),d.root_lin_vel_w[:,:2].norm(dim=-1,keepdim=True),d.root_ang_vel_b[:,2:3],torch.acos((-d.projected_gravity_b[:,2:3]).clamp(-1,1)),
                cmd.vel_command_b,norm[:,torso].amax(1,keepdim=True),norm[:,legs].amax(1,keepdim=True),d.root_lin_vel_b[:,:2],d.heading_w[:,None],cmd.heading_target[:,None],
                d.body_lin_vel_w[:,feet,:2].flatten(1),force[:,sf,2],d.applied_torque,lower[:,0:1],upper[:,0:1],lower[:,1:2],upper[:,1:2]),1).cpu().numpy().copy()
            return rows,links.cpu().numpy().copy(),force[:,body_to_sensor].cpu().numpy().copy()
        old_update=env.scene.update
        def observe(dt):
            nonlocal physics_steps,previous_feet
            old_update(dt)
            if abs(dt-env.physics_dt)>1e-9:return
            physics_steps+=1;t=physics_steps*dt-args.settle;rows,bodies,forces=measure()
            measurements.append(rows);body_trace.append(bodies);force_trace.append(forces)
            current_feet=rows[:,3:15].reshape(len(cases),4,3)
            if t>0:
                for i,j in enumerate(judges):
                    if previous_feet is not None and j.result is None and cases[i]['direction']=='up':
                        g=geometry[i]
                        for f,name in enumerate(foot_names):
                            e=foot_riser_event(previous_feet[i,f],current_feet[i,f],g['first_edge_x'],g['start_height'],cases[i]['height'])
                            if e['crossed_plane']:events.append(dict(case_id=cases[i]['case_id'],time_s=t,foot=name,**e))
                    j.step(t,rows[i])
            previous_feet=current_feet.copy()
        env.scene.update=observe
        def no_reset(env_ids):
            t=max(0.,physics_steps*env.physics_dt-args.settle)
            for i in env_ids.cpu().tolist():judges[i].native_termination(t)
        env._reset_idx=no_reset
        frame_counts={i:0 for i in record}
        if record:
            import imageio.v2 as imageio
            from PIL import Image,ImageDraw
            for i in record:writers[i]=imageio.get_writer(str(out/(cases[i]['case_id']+'.mp4')),fps=25,codec='libx264',quality=8,macro_block_size=16)
            for _ in range(4):env.render()
        start=time.monotonic();nsteps=round((args.duration+args.settle)/env.step_dt)
        with torch.inference_mode():
            for k in range(nsteps):
                t=k*env.step_dt-args.settle;cmd.time_left.fill_(1000.);cmd.vel_command_b.zero_();cmd.is_heading_env.fill_(True);cmd.is_standing_env.fill_(False)
                for i,c in enumerate(cases):
                    cmd.vel_command_b[i,0]=c['speed'];cmd.heading_target[i]=heading[i]
                    cmd.is_standing_env[i]=(t<0 or judges[i].stop_requested or judges[i].result is not None)
                cmd._update_command();obs=env.observation_manager.compute()
                raw_valid=torch.isfinite(scanner.data.ray_hits_w).all(dim=-1).all(dim=-1).cpu().numpy()
                for i,valid in enumerate(raw_valid):
                    if not valid and judges[i].result is None:missing_counts[i]+=1;judges[i].invalidate(max(0.,t),'height_coverage')
                active=torch.tensor([j.result is None for j in judges],device=env.device)
                if not torch.isfinite(obs['policy'][active]).all():raise ValueError('nonfinite active observation')
                obs['policy'][~active]=0;action=policy.act_inference(obs);action[~active]=0
                obs_trace.append(obs['policy'].cpu().numpy().copy());action_trace.append(action.cpu().numpy().copy())
                env.step(action)
                d=robot.data;target=env.action_manager.get_term('joint_pos').processed_actions
                policy_trace.append(torch.cat((d.root_state_w,d.joint_pos,d.joint_vel,target),1).cpu().numpy().copy())
                if record and k%2==1:
                    for i,w in writers.items():
                        j=judges[i];now=physics_steps*env.physics_dt-args.settle
                        if j.result and now>j.result['elapsed_s']+.5:continue
                        pos=d.root_pos_w[i].cpu().numpy();env.sim.set_camera_view(pos+np.array([.3,-3.5,1.7]),pos+np.array([.5,0.,0.]))
                        im=Image.fromarray(env.render());draw=ImageDraw.Draw(im);c=cases[i]
                        line=f"Go2 WEIGHT={meta['source_kind']} | {c['case_id']} | {c['direction']} | EACH RISER {c['height']*100:.0f}cm | {c['speed']:.1f}m/s"
                        line+=f"\nt={now:.2f}s | PHYSICS ROLLOUT | {args.device} | "+(j.result['reason'] if j.result else 'running')
                        draw.rectangle((0,0,960,50),fill=(12,18,24));draw.text((12,8),line,fill='white');w.append_data(np.asarray(im));frame_counts[i]+=1
                if k%250==0:print('RETEST',k,'done',sum(j.result is not None for j in judges),flush=True)
                if all(j.result is not None for j in judges) and (not record or t>max(j.result['elapsed_s'] for j in judges)+.5):break
                if time.monotonic()-start>850:raise TimeoutError('run wall budget')
        results=[]
        for i,(c,j) in enumerate(zip(cases,judges)):
            if j.result is None:j.finish(args.duration,'timeout')
            results.append(dict(**c,**j.result,crossed=j.crossed,cross_time=j.cross_time,stop_time=j.stop_time,missing_observation_steps=int(missing_counts[i])))
        groups={}
        for x in results:
            key=f"{x['height']:.2f}_{x['direction']}_{x['speed']:.1f}";g=groups.setdefault(key,dict(n=0,completed=0,clean=0,crossed=0,invalid=0,reasons={}))
            g['n']+=1;g['completed']+=int(x['completed']);g['clean']+=int(x['clean']);g['crossed']+=int(x['crossed']);g['invalid']+=int(not x['valid']);g['reasons'][x['reason']]=g['reasons'].get(x['reason'],0)+1
        for w in writers.values():w.close()
        writers={}
        np.savez_compressed(out/'trace.npz',measurements=np.asarray(measurements),body_positions=np.asarray(body_trace),body_forces=np.asarray(force_trace),policy_states=np.asarray(policy_trace),observations=np.asarray(obs_trace),actions=np.asarray(action_trace),offsets=offsets,physics_dt=env.physics_dt,policy_dt=env.step_dt,settle=args.settle)
        save('summary.json',dict(status='RETEST_COMPLETE',evaluation_protocol='INTERIOR_STOP_V1',stop_trigger_margin_m=.35,acceptance_margin_m=.15,training=False,checkpoint=meta,runtime=runtime,groups=groups,results=results,foot_riser_events=events,wall_seconds=time.monotonic()-start))
        save('videos.json',[dict(path=str(out/(cases[i]['case_id']+'.mp4')),case_id=cases[i]['case_id'],frames=frame_counts[i],fps=25,kind='PHYSICS_ROLLOUT') for i in record])
        print('RESULT',json.dumps(groups),flush=True);code=0
    except BaseException as e:
        traceback.print_exc();save('summary.json',dict(status='RETEST_INVALID',training=False,error=repr(e),cases=cases))
    finally:
        for w in writers.values():w.close()
        if code:sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if env is not None:env.close()
        if app is not None:app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
