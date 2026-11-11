"""Inference-only public Go2 reference measurements. No optimizer or training entry.

Native robot/observation/action/PD and env.step are retained. Controlled evaluation
changes terrain, reset pose, commands and episode horizon; native resets are
intercepted AFTER their failure is recorded, so failures never become new trials.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

PUBLIC_SHA='f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2'


def require_height_hit_coverage(hit_heights):
    """Do not let +/-infinity misses masquerade as finite clipped observations."""
    import torch
    if not torch.isfinite(hit_heights).all():
        raise ValueError('height-scan coverage invalid: rays miss the terrain; extend scene context before evaluation')


def require_scan_pose_alignment(robot_positions,sensor_positions,tolerance=1e-4):
    """Fail before policy inference when a scan belongs to stale/different poses."""
    import torch
    if robot_positions.shape!=sensor_positions.shape or robot_positions.ndim!=2 or robot_positions.shape[1]!=3:
        raise ValueError('invalid robot/height-scanner pose shapes')
    if not torch.isfinite(robot_positions).all() or not torch.isfinite(sensor_positions).all():
        raise ValueError('nonfinite robot/height-scanner pose')
    error=float((robot_positions-sensor_positions).abs().max())
    if error>tolerance:raise ValueError(f'stale/misaligned initial height scan: {error}m')
    return error


def classify_stair_profile(z):
    z=np.asarray(z);d=np.diff(z);jumps=abs(d)>.03;steps=d[jumps]
    valid=np.isfinite(z).all() and len(steps)>=5 and np.allclose(steps,steps[0],atol=.002) and np.all(abs(d[~jumps])<.001)
    return dict(direction=('up' if steps[0]>0 else 'down') if valid else None,
                risers=int(len(steps)),height=float(abs(np.mean(steps))) if len(steps) else None)


def choose_native_stair_actors(origins,directions):
    origins=np.asarray(origins)
    candidates=[np.flatnonzero(np.asarray(directions,dtype=object)==name) for name in ['up','down']]
    if any(not len(ids) for ids in candidates):raise ValueError('both native stair origin classes required')
    distances=np.linalg.norm(origins[:,None,:2]-origins[None,:,:2],axis=-1)
    np.fill_diagonal(distances,np.inf);separation=distances.min(1)
    return [int(ids[np.argmax(separation[ids])]) for ids in candidates]


def ordered_body_indices(names,available):
    if len(set(available))!=len(available):raise ValueError('ambiguous body names')
    return [list(available).index(name) for name in names]


def verify_public_checkpoint(path):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest!=PUBLIC_SHA:raise ValueError('not the frozen published checkpoint')
    return dict(source_kind='NVIDIA_PUBLISHED',sha256=digest,training_seed='unknown')


def build_cases(phase):
    if phase=='flat':return [dict(case_id=f'walk-{i:03}',seed=42000+i,phase='flat') for i in range(20)]
    cases=[]
    for h in (.1,.15,.2):
        for v in (.5,.8):
            for direction in ('up','down'):
                for repeat in range(10):
                    i=len(cases)
                    cases.append(dict(case_id=f'stair-{i:03}',seed=43000+i,phase='stairs',height=h,
                                      speed=v,direction=direction,repeat=repeat))
    return cases


def walk_command(t):
    if t<2:return 0.,0.,True
    if t<12:return .5,0.,False
    if t<22:return .8,0.,False
    if t<27:return 0.,math.pi/2,False
    if t<37:return .5,math.pi/2,False
    return 0.,math.pi/2,True


def profile_geometry(x,z,direction,requested_height):
    x=np.asarray(x);z=np.asarray(z);ids=np.flatnonzero(abs(np.diff(z))>1e-5)
    changes=np.diff(z)[ids]
    sign=1 if direction=='up' else -1
    if not len(ids) or not np.allclose(changes,sign*requested_height,atol=1e-5):
        raise ValueError('mesh risers disagree with requested height/direction')
    return dict(riser_count=len(ids),single_riser_heights=abs(changes).tolist(),
                total_height=float(abs(z[-1]-z[0])),first_edge_x=float((x[ids[0]]+x[ids[0]+1])/2),
                last_edge_x=float((x[ids[-1]]+x[ids[-1]+1])/2),start_height=float(z[0]),
                finish_height=float(z[-1]),ground_min=float(min(z)))


class ReferenceJudge:
    def __init__(self,phase,finish_x=3.,end_x=4.,ground_min=0.):
        self.phase=phase;self.finish_x=finish_x;self.end_x=end_x;self.ground_min=ground_min
        self.result=None;self.crossed=False;self.last=0.;self.durations=np.zeros(4)

    def finish(self,t,reason):
        if self.result is None:self.result=dict(success=reason=='success',reason=reason,elapsed_s=float(t))
        return self.result

    def native_termination(self,t):
        # Native terminations are computed after four physics substeps. A success
        # in the same policy step is provisional until that check has completed.
        if self.result and self.result['success'] and 0<=t-self.result['elapsed_s']<=.020001:
            self.result=None
        return self.finish(t,'native_base_contact')

    def step(self,t,r):
        if self.result is not None:return self.result
        dt=t-self.last;self.last=t
        if not np.isfinite(r).all() or dt<=0 or dt>.00751:return self.finish(t,'runtime_error')
        points=np.vstack((r[:3],r[3:15].reshape(4,3)))
        width=20. if self.phase=='flat' else 4.
        if abs(r[1])>width or (self.phase=='flat' and abs(r[0])>20):return self.finish(t,'boundary_exit')
        if r[2]<self.ground_min-.1:return self.finish(t,'fell_below_ground')
        # Conservative whole-body horizontal envelope, not just root and feet.
        body_inside=len(r)>=55 and r[51]>=self.finish_x+.15 and r[52]<=self.end_x and r[53]>=-4 and r[54]<=4
        on_finish=bool(body_inside and np.all((points[:,0]>=self.finish_x+.15)&(points[:,0]<=self.end_x)&(abs(points[:,1])<=4)))
        self.crossed=self.crossed or (self.phase=='stairs' and on_finish)
        stopped=np.max(abs(r[18:21]))<=1e-6 and r[15]<.05 and abs(r[16])<.1
        conditions=[r[21]>20,r[22]>0,r[17]>math.pi/3,self.phase=='stairs' and on_finish and stopped]
        self.durations=np.where(conditions,self.durations+dt,0.)
        for d,limit,reason in zip(self.durations,[.04,.2,.1,1.],['body_contact','leg_contact','tilt','success']):
            if d>=limit-1e-9:return self.finish(t,reason)
        if self.phase=='stairs' and t>=60-1e-9:return self.finish(t,'timeout')
        return None


def flat_metrics(rows,dt):
    a=np.asarray(rows);n=len(a)
    if n<round(42/dt) or not np.isfinite(a).all():return dict(passed=False,reason='incomplete_trace')
    t=np.arange(1,n+1)*dt
    mask=((t>3)&(t<=12))|((t>13)&(t<=22))|((t>28)&(t<=37))
    rmse=float(np.sqrt(np.mean(np.sum((a[mask,23:25]-a[mask,18:20])**2,axis=1))))
    errors=[]
    for end in (12,22,27,37):
        k=round(end/dt)-1;e=(a[k,25]-a[k,26]+np.pi)%(2*np.pi)-np.pi;errors.append(float(abs(e)))
    stopped=bool(np.all(a[-round(1/dt):,15]<.05) and np.all(abs(a[-round(1/dt):,16])<.1)
                 and np.all(abs(a[-round(1/dt):,18:21])<=1e-6))
    age=np.zeros(4);stance_count=0;slip_count=0
    for r in a:
        age=np.where(r[35:39]>5,age+dt,0.)
        valid=age>.04+1e-9;slip=np.linalg.norm(r[27:35].reshape(4,2),axis=1)>.1
        stance_count+=int(valid.sum());slip_count+=int((valid&slip).sum())
    slip_fraction=slip_count/stance_count if stance_count else 1.
    peak_torque=float(np.max(abs(a[:,39:51])))
    # Record continuous near-saturation events instead of mistaking raw actions >1 for illegal.
    events=[]
    for joint in range(12):
        start=None
        for k,near in enumerate(abs(a[:,39+joint])>=.95*23.5):
            if near and start is None:start=k
            if start is not None and (not near or k==n-1):
                end=k if not near else k+1
                if (end-start)*dt>=.5:events.append(dict(joint=joint,start_s=start*dt,duration_s=(end-start)*dt))
                start=None
    passed=rmse<=.15 and max(errors)<=np.deg2rad(10) and stopped and slip_fraction<=.1 and peak_torque<=23.5+1e-3
    return dict(passed=bool(passed),velocity_rmse=rmse,heading_end_errors_deg=np.rad2deg(errors).tolist(),
                final_stop=stopped,stance_slip_fraction=slip_fraction,stance_samples=stance_count,
                peak_applied_torque=peak_torque,near_saturation_events=events)


def aggregate(cases,results):
    byid={r['case_id']:r for r in results};groups={};failures=Counter()
    if len(byid)!=len(results):raise ValueError('duplicate result')
    for c in cases:
        r=byid.get(c['case_id'],dict(success=False,reason='missing_result',fully_crossed=False))
        key='flat' if c['phase']=='flat' else f"{c['height']:.2f}m_{c['speed']:.1f}mps_{c['direction']}"
        g=groups.setdefault(key,dict(episodes=0,successes=0,fully_crossed=0,failure_reasons={}))
        g['episodes']+=1;g['successes']+=int(r['success']);g['fully_crossed']+=int(r.get('fully_crossed',False))
        if not r['success']:
            failures[r['reason']]+=1;g['failure_reasons'][r['reason']]=g['failure_reasons'].get(r['reason'],0)+1
    return dict(episodes=len(cases),successes=sum(g['successes'] for g in groups.values()),groups=groups,failure_reasons=dict(failures))


def physics_decimation(physics_dt,policy_dt=.02):
    if not math.isfinite(physics_dt) or physics_dt<=0:raise ValueError('invalid physics step')
    n=round(policy_dt/physics_dt)
    if n<1 or not math.isclose(n*physics_dt,policy_dt,abs_tol=1e-10):raise ValueError('physics step must divide policy period')
    return n


def sample_initial_pose(case):
    rng=np.random.default_rng(case['seed'])
    return float(rng.uniform(-.05,.05)),float(rng.uniform(-np.pi/36,np.pi/36)+case.get('heading_offset',0.))


def make_scene(cases,device,render,context_border=0.,physics_dt=None,solver_iterations=None):
    import trimesh
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainGeneratorCfg,MeshPlaneTerrainCfg,MeshPyramidStairsTerrainCfg,MeshInvertedPyramidStairsTerrainCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg_PLAY
    from .stair_eval import mesh_height
    meshes=[];origins=[];offsets=[];geometry=[]
    for i,c in enumerate(cases):
        off=np.array([i%6*70.,i//6*70.,0.]);offsets.append(off)
        if c['phase']=='flat':
            tc=MeshPlaneTerrainCfg(size=(60.,60.));parts,origin=tc.function(0.,tc)
            m=trimesh.util.concatenate(parts);m.apply_translation([-30,-30,0]);geometry.append(dict(ground_min=0.))
        else:
            cls=MeshInvertedPyramidStairsTerrainCfg if c['direction']=='up' else MeshPyramidStairsTerrainCfg
            tc=cls(size=(8.,8.),border_width=1.,platform_width=3.,step_width=.3,step_height_range=(c['height'],c['height']),holes=False)
            parts,origin=tc.function(0.,tc);m=trimesh.util.concatenate(parts);m.apply_translation([-4,-4,0])
            xs=np.linspace(.001,3.999,3999);zs=np.array([mesh_height(m,x,0.) for x in xs])
            g=profile_geometry(xs,zs,c['direction'],c['height']);g['profile_x']=xs[::50].tolist();g['profile_z']=zs[::50].tolist();geometry.append(g)
        if context_border>0 and c['phase']!='flat':
            from isaaclab.terrains.trimesh.utils import make_border
            apron=make_border((8+2*context_border,8+2*context_border),(8.,8.),.1,(0.,0.,-.05))
            m=trimesh.util.concatenate([m,*apron])
            geometry[-1]['context_border_m']=float(context_border)
        origin=np.array([0.,0.,origin[2]]);origins.append(off+origin)
        m.apply_translation(off);meshes.append(m)
    mesh=trimesh.util.concatenate(meshes);origin_array=np.array(origins)[None]
    class FrozenNativeMesh:
        def __init__(self,cfg,device):
            self.terrain_mesh=mesh;self.terrain_origins=origin_array;self.flat_patches={}
    cfg=UnitreeGo2RoughEnvCfg_PLAY();cfg.seed=cases[0]['seed'];cfg.scene.num_envs=len(cases);cfg.sim.device=device
    if physics_dt is not None:
        cfg.decimation=physics_decimation(physics_dt);cfg.sim.dt=physics_dt;cfg.sim.render_interval=cfg.decimation
        cfg.scene.contact_forces.update_period=physics_dt
    if solver_iterations is not None:
        position,velocity=solver_iterations
        if position<1 or velocity<0:raise ValueError('invalid solver iterations')
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count=position
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count=velocity
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(class_type=FrozenNativeMesh,size=(60.,60.),num_rows=1,num_cols=len(cases),sub_terrains={},curriculum=False)
    cfg.scene.terrain.max_init_terrain_level=0;cfg.curriculum.terrain_levels=None
    from isaaclab.sim import PreviewSurfaceCfg
    cfg.scene.terrain.visual_material=PreviewSurfaceCfg(diffuse_color=(.18,.23,.29),roughness=.9) if render else None
    cfg.scene.sky_light.spawn.texture_file=None
    cfg.commands.base_velocity.debug_vis=False;cfg.commands.base_velocity.resampling_time_range=(1000.,1000.)
    cfg.episode_length_s=65.;cfg.viewer.resolution=(960,640)
    # Physical dynamics, startup mass variation and fixed .8/.6 friction remain native.
    # Reset coordinates are prescribed for this measured trial rather than sampled by the task.
    cfg.events.reset_base=None;cfg.events.reset_robot_joints=None
    env=ManagerBasedRLEnv(cfg,render_mode='rgb_array' if render else None);env.reset()
    robot=env.scene['robot'];state=robot.data.default_root_state.clone()
    import torch
    for i,c in enumerate(cases):
        dy,yaw=sample_initial_pose(c)
        state[i,:3]=torch.tensor(origins[i]+[0,dy,.4],device=env.device)
        state[i,3:7]=torch.tensor([np.cos(yaw/2),0,0,np.sin(yaw/2)],device=env.device)
        geometry[i]['initial_yaw']=yaw;geometry[i]['initial_lateral']=dy
    state[:,7:]=0;robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
    # env.reset() already computed observations at the old poses. Writing a new
    # root state does not invalidate sensor caches, and update(0) alone cannot
    # advance their sampling clocks. Explicitly invalidate all pose-dependent
    # sensors before the very first policy observation, without a physics step.
    for sensor in env.scene.sensors.values():sensor.reset()
    env.scene.update(0.)
    require_scan_pose_alignment(robot.data.root_pos_w,env.scene['height_scanner'].data.pos_w)
    require_height_hit_coverage(env.scene['height_scanner'].data.ray_hits_w[...,2])
    return env,np.array(offsets),geometry


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['flat','stairs'],required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--record',default='',help='comma-separated local case indices, maximum four')
    parser.add_argument('--max-steps',type=int,default=0,help='smoke only; incomplete never passes')
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    record=[int(x) for x in args.record.split(',') if x]
    cases=build_cases(args.phase)[args.start:args.start+args.count]
    if not 1<=len(cases)<=32 or len(record)>4 or any(i<0 or i>=len(cases) for i in record):raise ValueError('invalid batch/video selection')
    metadata=verify_public_checkpoint(args.checkpoint);args.output.mkdir(parents=True,exist_ok=False)
    out=args.output;app=None;env=None;writers={};traces=[];states=[];action_trace=[];code=1;judges=[];geometries=[]
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    save('cases.json',cases);save('manifest.json',dict(checkpoint=metadata,training=False,command=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='RUNNING'))
    try:
        args.enable_cameras=bool(record);app=AppLauncher(args).app
        import torch
        from tensordict import TensorDict
        from rsl_rl.modules import ActorCritic
        from isaaclab.utils.io import dump_yaml
        env,offsets,geometries=make_scene(cases,args.device,bool(record));save('geometry.json',geometries)
        dump_yaml(str(out/'env.yaml'),env.cfg)
        robot=env.scene['robot'];sensor=env.scene['contact_forces'];cmd=env.command_manager.get_term('base_velocity')
        feet,foot_names=robot.find_bodies('.*_foot',preserve_order=True)
        sensor_feet=ordered_body_indices(foot_names,sensor.body_names)
        torso,_=sensor.find_bodies('base|Head.*',preserve_order=True);legs,_=sensor.find_bodies('.*_(hip|thigh|calf)',preserve_order=True)
        if len(feet)!=4 or not torso or not legs:raise ValueError('invalid body mapping')
        off=torch.tensor(offsets,dtype=torch.float32,device=env.device)
        targets=torch.tensor([g['initial_yaw'] for g in geometries],device=env.device)
        obs=TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])
        if obs['policy'].shape!=(len(cases),235):raise ValueError('wrong observation contract')
        policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},12,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128],activation='elu',actor_obs_normalization=False,critic_obs_normalization=False).to(env.device)
        policy.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']);policy.eval()
        for c,g in zip(cases,geometries):judges.append(ReferenceJudge(c['phase'],finish_x=g.get('last_edge_x',3.),ground_min=g['ground_min']))
        # Assets currently consumed by USD; distinguish content hashes from byte-identical file hashes.
        layers=[]
        for layer in env.sim.stage.GetUsedLayers():
            layers.append(dict(identifier=layer.identifier,resolved_path=layer.realPath,
                               composed_layer_text_sha256=hashlib.sha256(layer.ExportToString().encode()).hexdigest()))
        save('assets.json',dict(status='ASSET_CLOSURE_PARTIAL',used_layers=layers,root_asset=env.cfg.scene.robot.spawn.usd_path))
        save('initial_conditions.json',dict(offsets=offsets.tolist(),masses=robot.root_physx_view.get_masses().cpu().tolist(),foot_names=foot_names,sensor_foot_indices=sensor_feet,sensor_body_names=sensor.body_names,robot_foot_indices=feet,joint_names=robot.joint_names,
                                          body_envelope='root-oriented 0.80x0.32x0.30m box union all link origins +/-0.05m; conservative qualification margin, not exact collision mesh',
                                          physics_dt=env.physics_dt,policy_dt=env.step_dt,reset_policy='native termination latched; no auto-reset'))
        physics_count=0
        def measurement():
            d=robot.data;forces=sensor.data.net_forces_w;norm=forces.norm(dim=-1)
            from isaaclab.utils.math import matrix_from_quat
            extent=torch.abs(matrix_from_quat(d.root_quat_w))@torch.tensor([.4,.16,.15],device=env.device)
            center=d.root_pos_w-off;links=d.body_pos_w-off[:,None]
            lower=torch.minimum(center-extent,links.amin(1)-.05);upper=torch.maximum(center+extent,links.amax(1)+.05)
            return torch.cat((center,(d.body_pos_w[:,feet]-off[:,None]).flatten(1),
                d.root_lin_vel_w[:,:2].norm(dim=-1,keepdim=True),d.root_ang_vel_b[:,2:3],torch.acos((-d.projected_gravity_b[:,2:3]).clamp(-1,1)),
                cmd.vel_command_b,norm[:,torso].amax(1,keepdim=True),norm[:,legs].amax(1,keepdim=True),
                d.root_lin_vel_b[:,:2],d.heading_w[:,None],cmd.heading_target[:,None],
                d.body_lin_vel_w[:,feet,:2].flatten(1),forces[:,sensor_feet,2],d.applied_torque,
                lower[:,0:1],upper[:,0:1],lower[:,1:2],upper[:,1:2]),1).cpu().numpy().copy()
        original_update=env.scene.update
        def observed_update(dt):
            nonlocal physics_count
            original_update(dt)
            if abs(dt-env.physics_dt)<1e-9:
                physics_count+=1;rows=measurement();traces.append(rows)
                for i,j in enumerate(judges):j.step(physics_count*env.physics_dt,rows[i])
        env.scene.update=observed_update
        def terminal_no_reset(ids):
            for i in ids.tolist():judges[i].native_termination(physics_count*env.physics_dt)
        env._reset_idx=terminal_no_reset
        frame_counts={i:0 for i in record};video_paths={}
        if record:
            import imageio.v2 as imageio
            from PIL import Image,ImageDraw
            for i in record:
                path=out/(cases[i]['case_id']+'.mp4');video_paths[i]=str(path)
                writers[i]=imageio.get_writer(str(path),fps=25,codec='libx264',quality=7,macro_block_size=16)
        def capture(step):
            for i,w in writers.items():
                t=physics_count*env.physics_dt;j=judges[i]
                if j.result and t>j.result['elapsed_s']+.5:continue
                pos=robot.data.root_pos_w[i].cpu().numpy()
                env.sim.set_camera_view(pos+np.array([0.,-3.5,1.7]),pos+np.array([.5,0.,0.]))
                rgb=env.render()
                if rgb is None or rgb.size==0:raise ValueError('empty video frame')
                im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);c=cases[i]
                text=f"PUBLIC Go2 | {c['case_id']} | t={t:.2f}s | PHYSICS ROLLOUT"
                text+='\n'+('flat: stand / 0.5 / 0.8 / turn / walk / stop' if c['phase']=='flat' else f"{c['direction']} | each riser={c['height']*100:.0f}cm | vx={c['speed']}m/s | {geometries[i]['riser_count']} risers")
                if j.result:text+='\nRESULT: '+j.result['reason']
                draw.rectangle((0,0,960,64),fill=(20,20,20));draw.text((12,8),text,fill='white')
                w.append_data(np.asarray(im));frame_counts[i]+=1
        if record:
            for _ in range(4):env.render()  # RGB annotator warm-up, no physics or policy stepping.
        start=time.monotonic();max_steps=args.max_steps or round((42 if args.phase=='flat' else 60)/env.step_dt)
        with torch.inference_mode():
            for step in range(max_steps):
                cmd.time_left.fill_(1000.);cmd.is_heading_env.fill_(True);cmd.is_standing_env.fill_(False);cmd.vel_command_b.zero_()
                for i,c in enumerate(cases):
                    if c['phase']=='flat':vx,delta,standing=walk_command(step*env.step_dt)
                    else:vx,delta,standing=c['speed'],0.,judges[i].crossed
                    cmd.vel_command_b[i,0]=vx;cmd.heading_target[i]=targets[i]+delta
                    cmd.is_standing_env[i]=standing or judges[i].result is not None
                cmd._update_command()
                obs=TensorDict(env.observation_manager.compute(),batch_size=[env.num_envs])
                active=torch.tensor([j.result is None for j in judges],device=env.device)
                require_height_hit_coverage(env.scene['height_scanner'].data.ray_hits_w[...,2][active])
                if not torch.isfinite(obs['policy']).all():raise ValueError('nonfinite observation')
                actions=policy.act_inference(obs)
                if not torch.isfinite(actions).all():raise ValueError('nonfinite action')
                action_trace.append(actions.cpu().numpy().copy())
                env.step(actions)
                states.append(torch.cat((robot.data.root_state_w,robot.data.joint_pos,robot.data.joint_vel),1).cpu().numpy().copy())
                if record and step%2==1:capture(step)
                if step%250==0:print('REFERENCE',step,'terminal',sum(j.result is not None for j in judges),flush=True)
                if all(j.result is not None for j in judges) and not record:break
                if all(j.result is not None for j in judges) and physics_count*env.physics_dt>max(j.result['elapsed_s'] for j in judges)+.5:break
                if time.monotonic()-start>850:raise TimeoutError('per-run wall budget')
        data=np.asarray(traces);results=[]
        for i,(c,j) in enumerate(zip(cases,judges)):
            metrics=flat_metrics(data[:,i],env.physics_dt) if c['phase']=='flat' else {}
            if j.result is None:
                reason=('success' if metrics.get('passed') else 'gait_metrics_failed') if c['phase']=='flat' and len(data)>=8400 else 'incomplete_trace'
                j.finish(physics_count*env.physics_dt,reason)
            results.append(dict(**c,**j.result,fully_crossed=j.crossed,metrics=metrics))
        for w in writers.values():w.close()
        writers={}
        np.savez_compressed(out/'trace.npz',physics=data,policy_states=np.asarray(states),actions=np.asarray(action_trace),physics_dt=env.physics_dt,policy_dt=env.step_dt,offsets=offsets)
        save('summary.json',dict(status='INFERENCE_EVALUATED',training=False,checkpoint=metadata,results=results,summary=aggregate(cases,results),wall_seconds=time.monotonic()-start))
        save('videos.json',[dict(case_id=cases[i]['case_id'],path=p,kind='PHYSICS_ROLLOUT',frames=frame_counts[i],fps=25,seconds=frame_counts[i]/25,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for i,p in video_paths.items()])
        print('RESULT',json.dumps(aggregate(cases,results)),flush=True);code=0
    except BaseException as exc:
        traceback.print_exc();results=[]
        for i,c in enumerate(cases):
            prior=judges[i].result if i<len(judges) else None
            results.append(dict(**c,**(prior or dict(success=False,reason='runtime_error',elapsed_s=0.))))
        save('summary.json',dict(status='INVALID_RUNTIME',error=repr(exc),training=False,results=results,summary=aggregate(cases,results)))
        if traces:np.savez_compressed(out/'trace.partial.npz',physics=np.asarray(traces))
    finally:
        for w in writers.values():w.close()
        if code:sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if env is not None:env.close()
        if app is not None:app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
