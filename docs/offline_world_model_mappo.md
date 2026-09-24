# Offline world-model teacher for MAPPO

This workflow fits dynamics once from collected complete episodes. During MAPPO
self-play, CEM searches the frozen ensemble and supplies skill/command targets to
the existing teacher replay. The actor receives PPO updates on real rollouts plus
bounded teacher distillation updates. Deployment uses the actor alone. The model
parameters and normalization stay fixed; no terminal-value model is trained in
this variant. CEM still runs during policy training, so this saves online model
fitting, not planner inference cost.

Run from the repository root:

```bash
# Optional: collect more data from your policy checkpoints.
OUTPUT_ROOT=data/offline_teacher NUM_ENVS=3 ROLLOUTS=20 DEVICE=cuda:7 \
  bash collect_world_model_data.bash /path/to/high_level_policy 0 400 latest

# Train from all checkpoint_* datasets under this root (or pass one dataset).
bash train_world_model.bash \
  --dataset data/offline_teacher \
  --output checkpoints/offline_teacher \
  --device cuda:7 --epochs 200 --batch-size 1024

# Train MAPPO with the pretrained, frozen teacher.
bash train_high_level_frozen_world_model.bash \
  --world-model-checkpoint checkpoints/offline_teacher/best.pt \
  --cuda 7 --num-envs 256 --iterations 3000 \
  --checkpoint-dir outputs/mappo_frozen_teacher
```

For existing data, replace `data/offline_teacher` with, for example,
`data/world_model_ppo_eval_20260914_210955`. Three episodes are the minimum for
nonempty train/validation/test splits, not sufficient evidence of useful dynamics.
Collect varied policies and enough rare outcomes to train a reliable teacher.
`ROLLOUTS` is episodes per environment per checkpoint. `NUM_ENVS` now works without
the previous duplicate command-line overrides.

Offline training creates `dataset/` inside its output directory, preserving the
sources. It checks schemas, skill metadata, timing and shard checksums, removes
identical episodes, assigns unique episode IDs, and splits whole episodes with
approximately 80/10/10 proportions (at least one episode per held-out split).
Normalization uses the training split only. The former collector label
`held_out_ppo_evaluation` does not mean data remain held out once used here: use the
prepared test split for this model's evaluation. Outputs include `best.pt`,
`latest.pt`, `final.pt`, and per-epoch `history.json`; this entry point logs to
stdout and local files.

```bash
# Resume in the same output directory; epochs is the total target epoch count.
bash train_world_model.bash --dataset data/offline_teacher \
  --output checkpoints/offline_teacher --device cuda:7 --epochs 300 \
  --resume checkpoints/offline_teacher/latest.pt

# Evaluate on the untouched test episodes.
/home/zhz/anaconda3/envs/legged_env/bin/python scripts/evaluate_world_model.py \
  --checkpoint checkpoints/offline_teacher/best.pt \
  --dataset checkpoints/offline_teacher/dataset --split test

# Inspect the policy training command without starting Isaac Gym.
bash train_high_level_frozen_world_model.bash \
  --world-model-checkpoint checkpoints/offline_teacher/best.pt --dry-run
```

Use the same low-level walk/dribble/shoot artifacts, robot count, action bounds,
state schema and macro-action interval for collection and training. Mismatched or
unknown skill hashes and incompatible schemas/timing fail instead of silently
refreshing the model. The launcher uses the same shooting-option goal profile as
`train_high_level.bash`; data collected with another execution profile may require
matching environment flags or recollection.

The default teacher uses horizon 4, 64 candidates, 2 CEM iterations and replans
every 8 decisions, with at most 128 matches queried. PPO runs 5 epochs; teacher
replay runs one update per rollout. Override these using the existing
`--mpc-*` options. Guidance is supplied through distillation; reward shaping is
zero by default. The model-error, predicted-improvement and real-rollout guards
remain enabled. Monitor `mpc_quality/model_gate_open`,
`mpc_quality/accepted_fraction`, `mpc_replay/` update metrics and
`world_model/frozen`. If the frozen model is inaccurate on new policy states,
the guard can stop teacher updates while PPO continues; recollect and retrain
explicitly. World-model update settings cannot unfreeze this launcher.

Policy saves include `world_model_online_<iteration>.pt` and
`world_model_online_latest.pt` for compatibility with existing evaluation tools;
these are exact copies of the offline checkpoint. Teacher and transition replay
can still be saved, but transitions never update frozen dynamics. The existing
online MPC launcher retains its online updates.
