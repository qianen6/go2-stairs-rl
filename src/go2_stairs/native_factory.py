"""Native registered Go2 PLAY task; narrow, explicit controlled-evaluation overrides."""
import copy
import math
import numpy as np
import torch


def native_cases(height, direction, count):
    if height not in (.05, .1, .15, .2, .25, .3) or direction not in ('up', 'down') or count < 2 or count % 2:
        raise ValueError('use 5/10/15/20/25/30cm, up/down and an even case count')
    if height in (.1, .2):
        seed = 60000 + round(height * 100) * 100 + (1000 if direction == 'down' else 0)
    else:
        # STAIRS25 heights use a separate range so no seed overlaps the historical 10/20cm cases.
        seed = 80000 + round(height * 100) * 100 + (5000 if direction == 'down' else 0)
    return [dict(case_id=f'native-{round(height*100):02}-{direction}-{i:03}', phase='stairs',
                 height=height, direction=direction, speed=.8 if i < count // 2 else 1.,
                 seed=seed + i, repeat=i) for i in range(count)]


def validate_native_batch(cases):
    keys = {(x['height'], x['direction']) for x in cases}
    if len(keys) != 1:
        raise ValueError('native generator requires one terrain type/height in each batch')
    return next(iter(keys))


class FrozenActor(torch.nn.Module):
    """Exact ELU feedforward actor, with no optimizer or observation normalizer."""
    def __init__(self, model_state, device):
        super().__init__()
        self.actor = torch.nn.Sequential(torch.nn.Linear(235, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(), torch.nn.Linear(256, 128),
            torch.nn.ELU(), torch.nn.Linear(128, 12))
        self.actor.load_state_dict({k[6:]: v for k, v in model_state.items() if k.startswith('actor.')}, strict=True)
        self.to(device).eval().requires_grad_(False)

    def act_inference(self, obs):
        return self.actor(obs['policy'])


def restore_release_velocity_limits(actuators):
    for actuator in actuators.values():
        if hasattr(actuator, 'velocity_limit_sim'):
            actuator.velocity_limit_sim = actuator.velocity_limit


def restore_release_stabilization(sim):
    sim.physx.enable_stabilization = True


def make_scene(cases, device, render, context_border=6., physics_dt=None, solver_iterations=None, release_velocity_limits=False, release_stabilization=False, actuator_groups=None):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401: register the upstream task
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg_PLAY
    from .reference_reproduction import profile_geometry, physics_decimation, require_scan_pose_alignment, require_height_hit_coverage
    from .stair_eval import mesh_height
    height, direction = validate_native_batch(cases)
    cfg = UnitreeGo2RoughEnvCfg_PLAY()
    cfg.seed = cases[0]['seed']
    cfg.scene.num_envs = len(cases)
    cfg.sim.device = device
    if actuator_groups is not None:
        from .stairs25 import apply_actuators
        apply_actuators(cfg.scene.robot, actuator_groups)
    if release_velocity_limits:
        restore_release_velocity_limits(cfg.scene.robot.actuators)
    if release_stabilization:
        restore_release_stabilization(cfg.sim)
    # Keep the native TerrainGenerator class, native 8m tiles and native 20m outer border.
    generator = cfg.scene.terrain.terrain_generator
    terrain_name = 'pyramid_stairs_inv' if direction == 'up' else 'pyramid_stairs'
    terrain = copy.deepcopy(generator.sub_terrains[terrain_name])
    terrain.proportion = 1.
    terrain.step_height_range = (height, height)
    generator.sub_terrains = {terrain_name: terrain}
    generator.num_rows = 1
    generator.num_cols = len(cases)
    generator.curriculum = False
    generator.use_cache = False
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    # Native reset event remains active; constrain only its evaluation approach distribution.
    cfg.events.reset_base.params['pose_range'] = {'x': (0., 0.), 'y': (-.05, .05), 'yaw': (-math.pi/36, math.pi/36)}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.episode_length_s = 65.
    cfg.viewer.resolution = (960, 640)
    cfg.scene.sky_light.spawn.texture_file = None
    if render:
        from isaaclab.sim import PreviewSurfaceCfg
        cfg.scene.terrain.visual_material = PreviewSurfaceCfg(diffuse_color=(.18, .23, .29), roughness=.9)
    else:
        cfg.scene.terrain.visual_material = None
    if physics_dt is not None:
        cfg.sim.dt = physics_dt
        cfg.decimation = physics_decimation(physics_dt)
        cfg.scene.contact_forces.update_period = physics_dt
    if solver_iterations is not None:
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = solver_iterations[0]
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = solver_iterations[1]
    env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-Play-v0', cfg=cfg,
                   render_mode='rgb_array' if render else None).unwrapped
    env.reset()
    # This path does not reconstruct meshes or manually write initial poses.
    robot = env.scene['robot']
    offsets = env.scene.env_origins.cpu().numpy().copy()
    offsets[:, 2] = 0.
    # Read the actually spawned collision geometry from USD. Newer Lab versions no
    # longer retain TerrainImporter.meshes; do not regenerate a surrogate mesh.
    from pxr import Usd, UsdGeom
    import trimesh
    meshes = [p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/ground')) if p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(f'expected one native generated ground mesh, found {len(meshes)}')
    usd_mesh = UsdGeom.Mesh(meshes[0])
    counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError('native ground must be triangulated')
    transform = UsdGeom.Xformable(meshes[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points = np.asarray([transform.Transform(p) for p in usd_mesh.GetPointsAttr().Get()])
    faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    xs = np.linspace(.001, 3.999, 3999)
    zs = np.array([mesh_height(mesh, float(x + offsets[0, 0]), float(offsets[0, 1])) for x in xs])
    shape = profile_geometry(xs, zs, direction, height)
    geometry = []
    for i in range(len(cases)):
        g = copy.deepcopy(shape)
        g.update(initial_yaw=float(robot.data.heading_w[i]), initial_lateral=float(robot.data.root_pos_w[i, 1] - offsets[i, 1]),
                 native_generator=True, native_reset=True, context_border_m=float(generator.border_width))
        geometry.append(g)
    require_scan_pose_alignment(robot.data.root_pos_w, env.scene['height_scanner'].data.pos_w)
    require_height_hit_coverage(env.scene['height_scanner'].data.ray_hits_w[..., 2])
    return env, offsets, geometry
