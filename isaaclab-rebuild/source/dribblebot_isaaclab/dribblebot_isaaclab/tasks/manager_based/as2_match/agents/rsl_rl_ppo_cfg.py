"""Static-opponent PPO baseline for the manager-based match wrapper."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AS2MatchSelfPlayPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dribblebot_as2_match_self_play"
    clip_actions = 10.0
    # RSL-RL 3.x names the actor input set ``policy`` (``actor`` was used by
    # older releases).  Keeping the canonical key avoids implicit resolution
    # warnings and makes the 136D history contract explicit.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="HybridActorCritic",
        init_noise_std=0.35,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # Extra fields are consumed by the project HybridActorCritic constructor.
    # Keep parameter exploration bounded and give categorical skill selection
    # a stronger entropy signal than the three continuous command values.
    policy.min_parameter_std = 0.15
    policy.max_parameter_std = 0.6
    policy.skill_entropy_scale = 1.0
    policy.parameter_entropy_scale = 0.0
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        # The standard RSL-RL adaptive KL assumes every action dimension is
        # Gaussian. Skill index is categorical, so the hybrid policy uses the
        # PPO objective without that invalid Gaussian KL adaptation.
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
    )
