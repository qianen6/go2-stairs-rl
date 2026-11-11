"""Example STAIRS25 extra terms. Every function receives the Isaac Lab env first, then the profile params.

Reward functions return one value per environment (shape [num_envs]); the manager multiplies it by the
term weight and the policy step (0.02 s). Use a negative weight for penalties. Termination functions
return a bool per environment. Params written as {"scene_entity": "contact_forces", "body_names": [...]}
arrive as a resolved SceneEntityCfg (with .body_ids).

Profile snippet:

    "rewards": {..., "extra_terms": [
        {"name": "leg_contact_force", "func": "go2_stairs.student_terms.examples:contact_force_excess",
         "weight": -0.01, "params": {"sensor_cfg": {"scene_entity": "contact_forces", "body_names": [".*_thigh", ".*_calf"]},
                                     "threshold": 5.0}}]}
"""
import torch


def contact_force_excess(env, sensor_cfg, threshold=5.):
    """Sum over the selected bodies of contact force above `threshold` (N). Graded, unlike a contact count."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    return (forces-threshold).clamp(min=0.).sum(dim=-1)


def base_height_below_target(env, target=.3, sensor_name='height_scanner', asset_name='robot'):
    """Squared shortfall of base height above the terrain under the robot (mean of the height scan)."""
    hits = env.scene.sensors[sensor_name].data.ray_hits_w[..., 2]
    ground = torch.nan_to_num(hits, nan=0., posinf=0., neginf=0.).mean(dim=-1)
    height = env.scene[asset_name].data.root_pos_w[:, 2]-ground
    return (target-height).clamp(min=0.).square()


def any_contact_above(env, sensor_cfg, threshold=1.):
    """Termination: any selected body's contact force exceeds `threshold` (N)."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1)
    return (forces.max(dim=1)[0] > threshold).any(dim=1)
