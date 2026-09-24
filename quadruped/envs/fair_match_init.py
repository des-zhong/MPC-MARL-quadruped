"""Symmetric kickoff geometry for head-to-head policy evaluation."""

import torch


def symmetric_kickoff_roots(robots, ball, origins):
    """Rotate team A by pi to initialize team B; center and stop the ball.

    Inputs are copies of root states [matches, robots, 13], [matches, 13],
    and environment origins [matches, 3]. Quaternion order is xyzw.
    """
    team_size = robots.shape[1] // 2
    if team_size < 1 or robots.shape[1] != 2 * team_size:
        raise ValueError("Symmetric kickoff requires two equally sized teams")
    own = robots[:, :team_size]
    opponent = own.clone()
    opponent[..., :2] = 2 * origins[:, None, :2] - own[..., :2]
    # Left-multiply orientation by quaternion (0, 0, 1, 0).
    x, y, z, w = own[..., 3:7].unbind(-1)
    opponent[..., 3:7] = torch.stack((-y, x, w, -z), dim=-1)
    robots[:, team_size:] = opponent
    robots[..., 7:13] = 0
    ball[:, :2] = origins[:, :2]
    ball[:, 3:7] = ball.new_tensor([0, 0, 0, 1])
    ball[:, 7:13] = 0
    return robots, ball


def configure_fair_match(cfg):
    """Apply after saved training settings so resets cannot override fairness."""
    cfg.env.fair_match_init = True
    cfg.env.randomize_match_init = True
    cfg.env.high_level_near_ball_init_probability = 0.0
    cfg.env.soccer_curriculum = False
    cfg.env.high_level_joint_reset_noise = 0.0
    cfg.env.num_static_opponents = 0
    if hasattr(cfg, "terrain"):
        cfg.terrain.mesh_type = "plane"
        cfg.terrain.curriculum = False
    # Keep each team in its own half with room around the center ball.
    half_length = float(cfg.env.field_length) / 2
    half_width = float(cfg.env.field_width) / 2
    for prefix in ("robot", "teammate"):
        setattr(cfg.env, prefix + "_init_x_range", [-0.8 * half_length, -0.25 * half_length])
        setattr(cfg.env, prefix + "_init_y_range", [-0.7 * half_width, 0.7 * half_width])
    for name in dir(cfg.domain_rand):
        if name.startswith("randomize_") or name == "push_robots":
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.lag_timesteps = 0
    if hasattr(cfg, "noise"):
        cfg.noise.add_noise = False
