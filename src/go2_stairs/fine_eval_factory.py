"""Development-only evaluation cases for the frozen mixed20 fine-tune profile."""
import math
import numpy as np
from .native_factory import FrozenActor, validate_native_batch
from .native_factory import native_cases as stair_cases, make_scene as stair_scene


def native_cases(height, direction, count):
    if direction == 'flat':
        if height != 0 or count < 2 or count % 2:
            raise ValueError('flat evaluation requires height0 and an even case count')
        return [dict(case_id=f'dev-flat-{i:03}', phase='flat', height=0., direction='flat',
                     speed=.8 if i < count//2 else 1., seed=71000+i, repeat=i) for i in range(count)]
    rows = stair_cases(height, direction, count)
    for row in rows:
        row['seed'] += 10000
        row['case_id'] = 'dev-'+row['case_id']
    return rows


def make_scene(cases, device, render, **options):
    if cases[0]['phase'] != 'flat':
        return stair_scene(cases, device, render, **options)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab.terrains import MeshPlaneTerrainCfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg_PLAY
    from .native_factory import restore_release_velocity_limits, restore_release_stabilization
    from .reference_reproduction import require_scan_pose_alignment, require_height_hit_coverage
    cfg = UnitreeGo2RoughEnvCfg_PLAY()
    cfg.seed = cases[0]['seed']; cfg.scene.num_envs = len(cases); cfg.sim.device = device
    if options.get('actuator_groups') is not None:
        from .stairs25 import apply_actuators
        apply_actuators(cfg.scene.robot, options['actuator_groups'])
    if options.get('release_velocity_limits'):
        restore_release_velocity_limits(cfg.scene.robot.actuators)
    if options.get('release_stabilization'):
        restore_release_stabilization(cfg.sim)
    if options.get('physics_dt') is not None or options.get('solver_iterations') is not None:
        raise ValueError('development flat profile has fixed native timestep/solver')
    generator = cfg.scene.terrain.terrain_generator
    generator.sub_terrains = {'flat': MeshPlaneTerrainCfg(proportion=1.)}
    generator.num_rows = 1; generator.num_cols = len(cases); generator.curriculum = False
    generator.seed = cases[0]['seed']; generator.use_cache = False
    cfg.scene.terrain.max_init_terrain_level = None; cfg.curriculum.terrain_levels = None
    cfg.events.reset_base.params['pose_range'] = {'x': (0., 0.), 'y': (-.05, .05), 'yaw': (-math.pi/36, math.pi/36)}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.episode_length_s = 65.; cfg.viewer.resolution = (960, 640)
    cfg.scene.sky_light.spawn.texture_file = None; cfg.scene.terrain.visual_material = None
    env = gym.make('Isaac-Velocity-Rough-Unitree-Go2-Play-v0', cfg=cfg,
                   render_mode='rgb_array' if render else None).unwrapped
    env.reset(); robot = env.scene['robot']
    offsets = env.scene.env_origins.cpu().numpy().copy(); offsets[:, 2] = 0.
    geometry = [dict(riser_count=0, first_edge_x=3., last_edge_x=3., start_height=0., finish_height=0.,
                     ground_min=0., initial_yaw=float(robot.data.heading_w[i]), native_generator=True,
                     native_reset=True, context_border_m=20.) for i in range(len(cases))]
    require_scan_pose_alignment(robot.data.root_pos_w, env.scene['height_scanner'].data.pos_w)
    require_height_hit_coverage(env.scene['height_scanner'].data.ray_hits_w[..., 2])
    return env, offsets, geometry
