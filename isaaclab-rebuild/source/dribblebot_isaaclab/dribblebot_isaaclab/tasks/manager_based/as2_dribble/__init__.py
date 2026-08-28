"""Register AS2 ball-dribbling environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-DribbleBot-AS2-Dribble-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:AS2DribbleFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2DribblePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Dribble-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:AS2DribbleFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AS2DribblePPORunnerCfg",
    },
)
