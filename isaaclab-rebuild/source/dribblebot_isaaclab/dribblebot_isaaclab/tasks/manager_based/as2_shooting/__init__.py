"""Register AS2 ball-shooting environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-DribbleBot-AS2-Shooting-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shooting_env_cfg:AS2ShootingFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2ShootingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Shooting-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shooting_env_cfg:AS2ShootingFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2ShootingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Shooting-Frozen-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shooting_env_cfg:AS2FrozenShootingFlatEnvCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Shooting-Frozen-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shooting_env_cfg:AS2FrozenShootingFlatEnvCfg_PLAY",
    },
)
