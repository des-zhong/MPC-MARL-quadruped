"""Four-AS2 manager-based football match staging tasks."""

import gymnasium as gym

gym.register(
    id="Isaac-DribbleBot-AS2-Match-Macro-Flat-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.macro:make_manager_based_macro_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.match_env_cfg:AS2MatchFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AS2MatchSelfPlayPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.macro:make_manager_based_macro_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.match_env_cfg:AS2MatchFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AS2MatchSelfPlayPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.self_play:make_match_self_play_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.match_env_cfg:AS2MatchFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AS2MatchSelfPlayPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0",
    entry_point="dribblebot_isaaclab.tasks.manager_based.football.self_play:make_match_self_play_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.match_env_cfg:AS2MatchFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:AS2MatchSelfPlayPPORunnerCfg",
    },
)
