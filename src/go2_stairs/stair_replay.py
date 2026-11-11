"""Render recorded evaluation states, not a new physics trial or new policy run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np


def relocate_state(state,old_offset,new_offset):
    result=np.array(state,copy=True)
    result[:3]+=np.asarray(new_offset)-np.asarray(old_offset)
    return result


def relocate_batch(states,indices,old_offsets,new_offsets):
    return np.array([relocate_state(states[index],old_offsets[index],new_offsets[i])
                     for i,index in enumerate(indices)])


def verify_trace_files(root,report):
    for name in ('state_trace.npz','physics_trace.npz'):
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=report['trace_sha256'][name]:
            raise ValueError('evaluation trace hash mismatch: '+name)


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    root=args.evaluation.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if report['status']!='SINGLE_SEED_CORE_EVALUATED' or report['evaluation_valid'] is not True:
        raise ValueError('replay requires completed evaluation')
    verify_trace_files(root,report)
    from .stair_eval import make_scene,spec_for,select_replays,judge_trace
    rows=report['results'];selection=select_replays(rows)
    selected=[r for r in rows if r['case_id'] in selection]
    saved=np.load(root/'state_trace.npz',allow_pickle=False)
    measurements=np.load(root/'physics_trace.npz',allow_pickle=False)
    # Independently verify all displayed labels against the original physics-step measurements.
    for row in selected:
        i=next(i for i,r in enumerate(rows) if r['case_id']==row['case_id'])
        if row['reason']!='initialization_failed':
            result=judge_trace(row,measurements['measurements'][:,i],float(measurements['dt']))
            if any(result[k]!=row[k] for k in ('success','reason','elapsed_s')):raise ValueError('video label differs from measured result')
    app=None;env=None;code=1
    try:
        app=AppLauncher(args).app
        import torch
        import imageio.v2 as imageio
        from PIL import Image
        import isaaclab.sim as sim_utils
        env,offsets=make_scene(selected,args.device,render=True)
        robot=env.scene['robot']
        if list(saved['joint_names'])!=list(robot.joint_names):raise ValueError('recorded joint ordering changed')
        # reset events are disabled in the scoring scene; explicitly place EVERY
        # replay actor so unselected actors do not remain overlapping at origin.
        indices=[next(i for i,r in enumerate(rows) if r['case_id']==row['case_id']) for row in selected]
        initial=torch.tensor(relocate_batch(saved['states'][0],indices,saved['offsets'],offsets),device=env.device)
        robot.write_root_state_to_sim(initial[:,:13])
        robot.write_joint_state_to_sim(initial[:,13:25],initial[:,25:37])
        env.scene.write_data_to_sim();env.sim.forward()
        light=sim_utils.DistantLightCfg(intensity=1200.,color=(1.,.97,.92))
        light.func('/World/replay_sun',light,orientation=(.9238795,.3826834,0.,0.))
        videos=[]
        for display,row in enumerate(selected):
            index=next(i for i,r in enumerate(rows) if r['case_id']==row['case_id'])
            spec=spec_for(row);o=offsets[display]
            env.sim.set_camera_view(eye=o+np.array([spec.length/2+1.,-4.2,2.7]),
                                    target=o+np.array([spec.length/2,0.,spec.height*spec.steps/2+.18]))
            stop=min(len(saved['states'])-1,int(np.ceil(row['elapsed_s']/float(saved['dt']))))
            frames=list(range(0,stop+1,2))
            if frames[-1]!=stop:frames.append(stop)
            path=out/f"{row['direction']}_{row['case_id']}_{row['reason']}.mp4"
            writer=imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=None,
                                      ffmpeg_params=['-pix_fmt','yuv420p'])
            try:
                for f,frame in enumerate(frames):
                    state=relocate_state(saved['states'][frame,index],saved['offsets'][index],o)
                    ids=torch.tensor([display],device=env.device)
                    robot.write_root_state_to_sim(torch.tensor(state[:13],device=env.device)[None],env_ids=ids)
                    robot.write_joint_state_to_sim(torch.tensor(state[13:25],device=env.device)[None],
                                                   torch.tensor(state[25:37],device=env.device)[None],env_ids=ids)
                    env.scene.write_data_to_sim();env.sim.forward()
                    for _ in range(24 if f==0 else 2):rgb=env.render()
                    if rgb is None or rgb.shape!=(720,1280,3) or np.std(rgb)<2:raise ValueError('invalid rendered frame')
                    writer.append_data(rgb)
                    if f==min(50,len(frames)//2):Image.fromarray(rgb).save(path.with_suffix('.png'))
            finally:writer.close()
            reader=imageio.get_reader(path)
            try:
                actual=reader.count_frames();fps=float(reader.get_meta_data()['fps']);reader.get_data(actual//2)
            finally:reader.close()
            if actual!=len(frames) or fps!=25.:raise ValueError('encoded replay verification failed')
            videos.append(dict(case_id=row['case_id'],direction=row['direction'],success=row['success'],reason=row['reason'],
                               path=str(path),frames=actual,fps=fps,duration_s=actual/fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print('VIDEO_VERIFIED='+str(path),flush=True)
        verify_trace_files(root,report)
        (out/'VIDEOS.json').write_text(json.dumps(dict(kind='RECORDED_EVALUATION_STATE_REPLAY_NOT_RESIMULATION',
            evaluation=str(root),checkpoint_sha256=report['checkpoint_metadata']['sha256'],videos=videos),indent=2),encoding='utf-8')
        code=0
    except BaseException:
        traceback.print_exc()
    finally:
        if code:sys.stdout.flush();sys.stderr.flush();os._exit(code)
        if env is not None:env.close()
        if app is not None:app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
