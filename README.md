# MPC-MARL Quadruped Soccer

This repository develops a hierarchical multi-agent reinforcement-learning
system for quadruped robot soccer in NVIDIA Isaac Gym. AS2 quadrupeds learn
walking, dribbling, and shooting as separate low-level skills. A shared
high-level policy selects those skills for every robot, trains against an older
frozen copy of itself through self-play, and can receive privileged guidance
from world-model MPC during training.

The main pipeline supports:

- independent PPO training and validation of walking, dribbling, and shooting;
- shared-parameter multi-agent high-level training for any team size;
- two-team self-play with periodically updated frozen opponents;
- joint world-model data collection over both teams' states and actions;
- hybrid CEM MPC over discrete skills and continuous skill commands;
- privileged MPC teacher rewards for high-level policy training.

This project is derived from
[DribbleBot](https://github.com/Improbable-AI/dribblebot) and
[Walk These Ways](https://github.com/Improbable-AI/walk-these-ways). See
[Acknowledgements and license](#acknowledgements-and-license).

## Requirements

- Linux with an NVIDIA GPU
- NVIDIA Isaac Gym Preview 4
- Python 3.8
- PyTorch 1.10 with CUDA 11.3

The default simulator workloads require a CUDA-capable GPU. Reduce
`--num-envs` when GPU memory is limited.

## Installation

Create the environment and install PyTorch:

```bash
conda create -n legged_env python=3.8
conda activate legged_env

pip install torch==1.10.0+cu113 \
  torchvision==0.11.1+cu113 \
  torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Download Isaac Gym Preview 4, then install it from its extracted directory:

```bash
cd isaacgym/python
pip install -e .
python examples/1080_balls_of_solitude.py
```

Install this repository:

```bash
cd MPC-MARL-quadruped
pip install -e .
```

Install Git LFS and download the curated reproduction checkpoints:

```bash
git lfs install
git lfs pull
./checkpoints/reproduction/assemble_world_model.bash
```

The complete bundle contract and file hashes are documented in
[checkpoints/reproduction/README.md](checkpoints/reproduction/README.md). The
top-level launchers use these checkpoint paths by default.

The launcher scripts default to
`/home/zhz/anaconda3/envs/legged_env/bin/python`. Override that without editing
the scripts when necessary:

```bash
export DRIBBLEBOT_PYTHON="$(command -v python)"
```

Weights & Biases is used by the trainers unless a script exposes and receives
an offline/disabled logging option. Authenticate before starting an online
run:

```bash
wandb login
```

## 1. Train low-level skills

The AS2 low-level policies are trained independently. Each command accepts
`--device`, `--num-envs`, and `--iterations` overrides.

```bash
# Walking
python scripts/train_walking.py \
  --device cuda:0 \
  --headless

# Ball dribbling
python scripts/train_dribbling.py \
  --device cuda:0 \
  --checkpoint-dir tmp/legged_data/dribble \
  --headless

# Shooting
python scripts/train_shooting.py \
  --device cuda:0 \
  --checkpoint-dir tmp/legged_data/shoot \
  --headless
```

Checkpoints and run data under `tmp/` and `wandb/` are deliberately excluded
from Git.

## 2. Validate low-level skills

Use `validate_robot_abilities.py` to evaluate one skill or all three. Supply
the directories containing the exported local policy files:

```bash
python scripts/validate_robot_abilities.py \
  --ability all \
  --skill-policy-source local \
  --walk-policy-dir /path/to/walk/checkpoint-directory \
  --dribble-policy-dir /path/to/dribble/checkpoint-directory \
  --shoot-policy-dir /path/to/shoot/checkpoint-directory \
  --headless
```

Add `--fail-on-threshold` for a non-zero exit status when a validation target
is missed. Results are written under `outputs/ability_validation/`. The
[validate_skill.bash](validate_skill.bash) launcher is a local example; update
its checkpoint paths before using it.

## 3. Train the high-level multi-agent policy

High-level training runs two equal AS2 teams. `--num-robots` is the number of
robots **per team**, and every learning-team robot uses the same actor
parameters. The opponent is a frozen older snapshot updated at
`--self-play-update-interval` PPO iterations.

```bash
python scripts/train_high_level.py \
  --num-robots 2 \
  --self-play-update-interval 500 \
  --skill-policy-source local \
  --walk-policy-dir /path/to/walk/checkpoint-directory \
  --dribble-policy-dir /path/to/dribble/checkpoint-directory \
  --shoot-policy-dir /path/to/shoot/checkpoint-directory \
  --checkpoint-dir tmp/legged_data/high_level \
  --device cuda:0 \
  --headless
```

[train_high_level.bash](train_high_level.bash) provides the same workflow with
the local paths used during development.

## 4. Validate the high-level policy

Evaluate one match with a local high-level checkpoint and the same three
low-level skill policies:

```bash
python scripts/play_high_level.py \
  --num-robots 2 \
  --high-level-policy-source local \
  --high-level-policy-dir /path/to/high-level/checkpoint-directory \
  --skill-policy-source local \
  --walk-policy-dir /path/to/walk/checkpoint-directory \
  --dribble-policy-dir /path/to/dribble/checkpoint-directory \
  --shoot-policy-dir /path/to/shoot/checkpoint-directory \
  --headless
```

Videos, plots, and CSV metrics are saved under `outputs/` by default. See
[validate_high_level.bash](validate_high_level.bash) for a local launcher
example.

## 5. Collect joint world-model data

The collector uses two equal teams of real AS2 robot actors—there are no
static obstacle actors in this dataset. It records the global joint state,
joint action, reward, termination, and event labels. Both teams draw actions
from the same random-valid distribution.

```bash
python scripts/collect_world_model_data.py \
  --config configs/world_model_as2.yaml \
  --output data/world_model_as2 \
  --num-robots 2 \
  --num-episodes 20000 \
  --skill-policy-source local \
  --walk-policy-dir /path/to/walk/checkpoint-directory \
  --dribble-policy-dir /path/to/dribble/checkpoint-directory \
  --shoot-policy-dir /path/to/shoot/checkpoint-directory \
  --device cuda:0
```

Here `--num-robots 2` means two robots per team and therefore four robot slots
in the saved world-model schema. [collect.bash](collect.bash) is the equivalent
development launcher.

## 6. Train and evaluate the world model

Training settings live in
[configs/world_model_as2.yaml](configs/world_model_as2.yaml). In particular,
change `training.max_epochs` there to control the maximum epoch count.

```bash
python scripts/train_world_model.py \
  --config configs/world_model_as2.yaml \
  --dataset data/world_model_as2 \
  --output checkpoints/world_model_as2 \
  --num-robots 2
```

The same defaults are available through [train_world_model.bash](train_world_model.bash).

Evaluate the trained ensemble on a held-out split:

```bash
python scripts/evaluate_world_model.py \
  --checkpoint checkpoints/world_model_as2/best.pt \
  --dataset data/world_model_as2 \
  --split test \
  --output outputs/world_model_as2_evaluation \
  --device cuda:0
```

## 7. Train with privileged MPC teacher guidance

After training the joint world model, MPC can guide self-play PPO. The student
policy still receives decentralized observations. MPC alone sees the joint
state, plans learning-team actions against the frozen opponent-policy forecast,
and supplies a dense action-agreement reward.

```bash
python scripts/train_high_level_with_mpc_teacher.py \
  --world-model-checkpoint checkpoints/world_model_as2/best.pt \
  --mpc-config configs/mpc_joint_teams.yaml \
  --mpc-profile teacher_training \
  --teacher-reward-coefficient 1.0 \
  --num-robots 2 \
  --skill-policy-source local \
  --walk-policy-dir /path/to/walk/checkpoint-directory \
  --dribble-policy-dir /path/to/dribble/checkpoint-directory \
  --shoot-policy-dir /path/to/shoot/checkpoint-directory \
  --checkpoint-dir tmp/legged_data/high_level_mpc_teacher \
  --device cuda:0 \
  --headless
```

The convenience launcher is
[train_high_level_with_mpc_teacher.bash](train_high_level_with_mpc_teacher.bash).

## Repository layout

```text
configs/                    World-model and MPC configurations
dribblebot/envs/as2/        AS2 simulator environments
dribblebot/envs/wrappers/   Skill, self-play, and teacher wrappers
dribblebot/world_model/     Joint dynamics model and dataset components
dribblebot/mpc/             Hybrid CEM MPC and teacher tooling
dribblebot_learn/           PPO implementation
scripts/                    Training, validation, collection, and analysis tools
tests/                      Unit and contract tests
```

## Acknowledgements and license

This work builds on the original DribbleBot implementation by Yandong Ji,
Gabriel B. Margolis, and Pulkit Agrawal, as well as Walk These Ways and NVIDIA
Isaac Gym. Redistributed upstream components retain their original licenses.
See [LICENSE](LICENSE) and [LICENSES/](LICENSES/) for details.
