"""Multi-robot AS2 football environment.

The physics implementation is shared with the Go1 entry point through
``TwoRobotLeggedRobot``. Robot assets, body names, torque limits, and control
mode are selected by ``Cfg.robot.name`` and ``config_as2``.
"""

from dribblebot.envs.go1.two_robot_velocity_tracking import (
    TwoRobotVelocityTrackingEasyEnv as _SharedTwoRobotVelocityTrackingEasyEnv,
)
from dribblebot.envs.base.legged_robot_config import Cfg


class TwoRobotVelocityTrackingEasyEnv(_SharedTwoRobotVelocityTrackingEasyEnv):
    """Vectorized multi-AS2 football environment with the shared task API."""

    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        prone=False,
        deploy=False,
        cfg: Cfg = None,
        eval_cfg: Cfg = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
    ):
        if cfg is None:
            raise ValueError("AS2 multi-robot environment requires cfg")
        robot_name = str(getattr(cfg.robot, "name", ""))
        if robot_name != "as2":
            raise ValueError(
                "AS2 multi-robot environment requires cfg.robot.name='as2'; "
                f"got {robot_name!r}. Apply config_as2/configure_high_level_cfg first."
            )
        super().__init__(
            sim_device=sim_device,
            headless=headless,
            num_envs=num_envs,
            prone=prone,
            deploy=deploy,
            cfg=cfg,
            eval_cfg=eval_cfg,
            initial_dynamics_dict=initial_dynamics_dict,
            physics_engine=physics_engine,
        )
