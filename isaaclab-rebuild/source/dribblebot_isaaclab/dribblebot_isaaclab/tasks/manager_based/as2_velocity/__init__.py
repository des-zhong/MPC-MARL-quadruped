"""Register AS2 velocity-tracking environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-DribbleBot-AS2-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:AS2VelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2VelocityPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:AS2VelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2VelocityPPORunnerCfg",
    },
)
