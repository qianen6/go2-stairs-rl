"""Offline Go2 stair-scene smoke test. This is not a learned policy evaluator."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contracts import ControlManifest
from .terrain import StairSpec, build_boxes


def scene_xml(model_dir, spec):
    model_dir = Path(model_dir).resolve()
    root = ET.fromstring((model_dir / 'go2.xml').read_text(encoding='utf-8'))
    compiler = root.find('compiler')
    compiler.set('meshdir', (model_dir / 'assets').as_posix())
    option = root.find('option')
    option.set('timestep', '0.002')
    option.set('integrator', 'implicitfast')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='baseline_floor', type='plane', size='0 0 .1',
                  rgba='.19 .23 .28 1', friction='.8 .02 .01')
    ET.SubElement(world, 'light', pos='0 -2 5', dir='0 0 -1', directional='true')
    for i, box in enumerate(build_boxes(spec)):
        ET.SubElement(world, 'geom', name=box.name, type='box',
                      pos=' '.join(map(str, box.center)), size=' '.join(map(str, box.half_size)),
                      rgba='.85 .52 .22 1' if i % 2 == 0 else '.64 .69 .74 1', friction='.8 .02 .01')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.7 .7 .7')
    return ET.tostring(root, encoding='unicode')


def load_scene(model_dir, spec):
    model_dir = Path(model_dir).resolve()
    xml = scene_xml(model_dir, spec)
    assets = {p.name: p.read_bytes() for p in (model_dir / 'assets').iterdir() if p.is_file()}
    # VFS avoids native Windows fopen failures on Chinese paths.
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def initialize(model, spec):
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key_id < 0:
        raise ValueError('Go2 model requires a home keyframe')
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[0] = .5
    data.qpos[1] = 0.
    data.qpos[2] += spec.start_height
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)
    return data


def manifest_from_model(model):
    if model.nu != 12:
        raise ValueError('expected twelve Go2 actuators')
    runtime_ids = [int(j) for j in model.actuator_trnid[:, 0]]
    runtime_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in runtime_ids)
    canonical = tuple(f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                      for joint in ('hip', 'thigh', 'calf'))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in canonical]
    if any(j < 0 for j in ids):
        raise ValueError('Go2 model joint names do not match contract')
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise ValueError('Go2 model requires a home keyframe')
    nominal = tuple(float(model.key_qpos[home, model.jnt_qposadr[j]]) for j in ids)
    limits = tuple(float(max(abs(model.actuator_ctrlrange[runtime_ids.index(j)]))) for j in ids)
    return ControlManifest(joint_names=canonical, runtime_joint_names=runtime_names,
                           q_nominal=nominal, action_scale=(.125, .25, .25) * 4,
                           lower=tuple(float(model.jnt_range[j, 0]) for j in ids),
                           upper=tuple(float(model.jnt_range[j, 1]) for j in ids),
                           kp=(20.,) * 12, kd=(.5,) * 12, torque_limits=limits)


def _save_summary(path, report):
    payload = json.dumps(report, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                     dir=path.parent, prefix='.summary-', suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def run_smoke(model_dir, spec, duration, output_dir, render=False):
    dt = .002
    if not math.isfinite(duration) or not 0 < duration <= 10 or not math.isclose(duration/dt, round(duration/dt), abs_tol=1e-7):
        raise ValueError('smoke duration must be a multiple of .002 in (0,10] seconds')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    model = load_scene(model_dir, spec)
    data = initialize(model, spec)
    manifest = manifest_from_model(model)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'scene.xml').write_text(scene_xml(model_dir, spec), encoding='utf-8')
    (output_dir / 'control_manifest.json').write_text(json.dumps(manifest.to_dict(), indent=2), encoding='utf-8')
    joints = model.actuator_trnid[:, 0]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    target = data.qpos[qadr].copy()
    finite = True
    count = 0
    failure_reason = None
    for _ in range(round(duration / dt)):
        # Fixed standing PD at the initial platform; no learned policy or walking command.
        torque = 20. * (target - data.qpos[qadr]) - .5 * data.qvel[vadr]
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        previous_time = float(data.time)
        mujoco.mj_step(model, data)
        count += 1
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and math.isfinite(data.time)):
            finite = False
            failure_reason = 'nonfinite_state'
            break
        if bool(np.any(data.warning.number > 0)):
            failure_reason = 'simulation_instability'
            break
        if not math.isclose(float(data.time), previous_time + dt, rel_tol=1e-9, abs_tol=1e-10):
            failure_reason = 'simulation_time_discontinuity'
            break
    complete = count == round(duration / dt) and math.isclose(float(data.time), duration, rel_tol=1e-9, abs_tol=1e-10)
    if failure_reason is None and not complete:
        failure_reason = 'incomplete_run'
    w, x, y, z = data.qpos[3:7]
    tilt = math.acos(float(np.clip(1 - 2*(x*x+y*y), -1., 1.))) if finite else math.pi
    standing = bool(finite and data.qpos[2] > spec.start_height+.12 and tilt < math.radians(30))
    report = dict(controller='PD_STAND_SMOKE_NOT_LEARNED', trained_policy_evaluated=False,
                  backend='mujoco', mujoco_version=mujoco.__version__, stair=asdict(spec),
                  physics_dt=dt, physics_steps=count, sim_time_s=_finite_or_none(data.time), finite_state=finite,
                  standing_end=standing, smoke_passed=finite and standing and complete and failure_reason is None,
                  failure_reason=failure_reason,
                  warnings={mujoco.mjtWarning(i).name: int(n) for i, n in enumerate(data.warning.number) if n},
                  final_position=[_finite_or_none(v) for v in data.qpos[:3]], tilt_degrees=math.degrees(tilt),
                  control_manifest_sha256=manifest.digest(), render_status='pending' if render else 'not_requested')
    summary_path = output_dir / 'summary.json'
    # Persist physics evidence before starting the optional graphics subsystem.
    _save_summary(summary_path, report)
    if render:
        try:
            from PIL import Image
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = (spec.length/2, 0., spec.height*spec.steps/2)
            camera.distance = spec.length * 1.25
            camera.azimuth = 135
            camera.elevation = -24
            with mujoco.Renderer(model, height=640, width=960) as renderer:
                renderer.update_scene(data, camera=camera)
                Image.fromarray(renderer.render()).save(output_dir / 'preview.png')
            report['preview'] = 'preview.png'
            report['render_status'] = 'ok'
        except Exception as exc:
            report['render_status'] = 'failed'
            report['render_error'] = f'{type(exc).__name__}: {exc}'
        _save_summary(summary_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--direction', choices=('up', 'down'), required=True)
    parser.add_argument('--height', type=float, default=.06)
    parser.add_argument('--depth', type=float, default=.30)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    report = run_smoke(args.model_dir, StairSpec(args.height, args.depth, args.steps, args.direction),
                       args.seconds, args.output, args.render)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report['smoke_passed'] and report['render_status'] != 'failed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
