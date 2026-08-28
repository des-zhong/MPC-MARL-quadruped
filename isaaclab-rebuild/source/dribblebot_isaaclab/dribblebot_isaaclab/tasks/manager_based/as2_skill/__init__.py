"""Register frozen low-level skill-composition smoke environments."""

import gymnasium as gym


gym.register(
    id="Isaac-DribbleBot-AS2-Skill-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skill_env_cfg:AS2SkillFlatEnvCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Skill-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skill_env_cfg:AS2SkillFlatEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Coordinator-Macro-Flat-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.macro:make_manager_based_macro_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skill_env_cfg:AS2CoordinatorFlatEnvCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Coordinator-Macro-Flat-Play-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.macro:make_manager_based_macro_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skill_env_cfg:AS2CoordinatorFlatEnvCfg_PLAY",
    },
)
