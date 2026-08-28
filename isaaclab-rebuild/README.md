# DribbleBot IsaacLab 迁移线

本文档是 `isaaclab-rebuild/` 的中文运行说明。该目录将原 Isaac Gym Preview 4
实现迁移到 IsaacLab 的 manager-based 环境，当前重点是 AS2 四足机器人的
walking、dribbling、shooting 低层技能，以及四机器人足球 match/self-play 环境。

## 1. 版本与当前状态

生产迁移目标固定为：

- Isaac Sim 5.1.0；
- IsaacLab v2.3.2；
- PhysX；
- CUDA GPU（已在 RTX 4090、driver 570.169 上验证）。

本项目已经创建独立的 Python 3.11 环境，不与 manipulation、ROS 或其他项目共享：

```text
/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python
```

当前版本为 `isaacsim==5.1.0.0`、IsaacLab pip `2.3.2.post1`（source
`0.54.2`）、`torch==2.7.0+cu126`、`rsl_rl==3.0.1`。NVIDIA EULA 已由用户确认，
`pip check` 无依赖冲突。

已完成或通过 smoke 的内容：

- AS2 velocity/dribble 的 GPU physics 与 legacy policy-contract parity；
- 四台 AS2 articulation + 一个足球的 match scene；
- 24D manager action、`(4, 34)` match observation、self-play 的 `(2, 34)` /
  `(2, 136)` 接口；
- 10 个低层 tick 的 macro 聚合、invalid-skill geometric fallback；
- canonical recorder snapshot 与 128D world-model state 编码；
- 四机器人 GUI scripted match 视频。
- 四机器人 CUDA PPO lifecycle：4-step rollout、1 iteration、checkpoint 与
  TensorBoard event 写出均通过；另已从 `model_450.pt` 恢复并连续训练到
  iteration 500，默认 500-iteration detached opponent snapshot 刷新通过。

仍未完成的正式 gate：高质量 shooting 成功率、历史 opponent pool 的长期行为、
完整足球 reward parity，以及 live world-model/MPC
接入。当前 match reward 是打通训练链路的最小 baseline，不应解读为成熟的足球博弈目标。

## 2. 已注册环境

| 环境 ID | 说明 |
|---|---|
| `Isaac-DribbleBot-AS2-Velocity-Flat-v0` | AS2 walking 训练环境 |
| `Isaac-DribbleBot-AS2-Dribble-Flat-v0` | 带动态足球的 dribbling 训练环境 |
| `Isaac-DribbleBot-AS2-Shooting-Flat-v0` | command-relative shooting 环境 |
| `Isaac-DribbleBot-AS2-Skill-Flat-v0` | 单机器人三技能冻结策略集成 smoke |
| `Isaac-DribbleBot-AS2-Match-Macro-Flat-v0` | 四机器人、24D manager action、macro wrapper |
| `Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0` | 2v2 self-play，外部只暴露学习队两个 agent |
| 各 ID 的 `-Play-v0` | 小规模可视化/smoke 版本 |

match 的基础参数：4 台机器人（`robot_0..3`）、1 个足球、5 ms physics step、
4 步 decimation（20 ms manager step）、10 步 coordinator macro（200 ms），默认
128 个并行 match，PLAY 配置默认 2 个并行 match。

## 3. 创建独立 IsaacLab 环境并安装

Isaac Sim 5.1 要求 Python 3.11。推荐直接在本项目的 `isaaclab-rebuild/.venv`
建立环境，这样 Isaac Sim、IsaacLab、PyTorch、RSL-RL 与其他工程完全隔离。

### 3.1 创建环境

```bash
cd /home/xander/Code/01_Locomotion/MPC-MARL-quadruped

python3.11 -m venv isaaclab-rebuild/.venv
source isaaclab-rebuild/.venv/bin/activate
python -m pip install --upgrade pip
```

### 3.2 安装精确版本

IsaacLab v2.3.2 的官方 pip release 名称为 `2.3.2.post1`。本项目把已验证的最小
版本集合保存在 `requirements-runtime.txt`。先固定 PyTorch，再安装 IsaacLab，
否则 pip 可能把 `torch>=2.7` 解析成更新但未经验证的版本：

```bash
python -m pip install \
  setuptools==81.0.0 wheel==0.42.0 \
  numpy==1.26.0 torch==2.7.0 torchvision==0.22.0

python -m pip install \
  "isaaclab[isaacsim,all]==2.3.2.post1" \
  --extra-index-url https://pypi.nvidia.com
```

也可以一次使用 pin 文件：

```bash
python -m pip install \
  -r isaaclab-rebuild/requirements-runtime.txt \
  --extra-index-url https://pypi.nvidia.com
```

Isaac Sim 5.1 精确要求 `numpy==1.26.0`。当前实际安装的 PyTorch 是
`2.7.0+cu126`，宿主机 driver 570.169 上已确认能访问 RTX 4090。Isaac Sim/
extension cache 下载体积很大，独立 venv 安装后约 17 GB，首次安装需要预留足够
磁盘空间和时间。

### 3.3 安装本项目 extension

```bash
cd /home/xander/Code/01_Locomotion/MPC-MARL-quadruped

ISAAC_PY=$PWD/isaaclab-rebuild/.venv/bin/python

$ISAAC_PY -m pip install --no-build-isolation -e \
  ./isaaclab-rebuild/source/dribblebot_isaaclab
```

IsaacLab pip wheel 会把 `isaaclab_tasks`、`isaaclab_rl` 等 source extension 打包在
wheel 内，但不会自动把它们作为顶层 Python 包暴露。本项目当前使用的是 wheel 内的
2.3.2 source 快照，放在 `isaaclab-rebuild/vendor/isaaclab-2.3.2/source/`，需要一并
editable 安装：

```bash
for pkg in isaaclab isaaclab_assets isaaclab_contrib isaaclab_tasks isaaclab_rl; do
  $ISAAC_PY -m pip install --no-deps --no-build-isolation -e \
    "./isaaclab-rebuild/vendor/isaaclab-2.3.2/source/$pkg"
done
```

该 source 快照还需要 `isaaclab-rebuild/vendor/isaaclab-2.3.2/apps/` 下的
`rendering_modes/*.kit`。这个 vendor 目录是本机安装产物，不应提交为训练数据；若
重新创建环境，可从已安装的 `isaaclab` wheel 的 `source/` 和 `apps/` 目录恢复。
安装过程中的 `isaaclab`/`isaaclab_tasks` 版本分别会显示为 `0.54.2`/`0.11.12`，
这是 v2.3.2 wheel 内 source extension 的版本号，不是 IsaacLab release tag。

开发时也可以不做 editable install，临时设置：

```bash
export PYTHONPATH=/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/source/dribblebot_isaaclab:${PYTHONPATH:-}
```

当前 pip runtime 使用：

```bash
export DRIBBLEBOT_EXPERIENCE=isaacsim.exp.base.python.kit
```

这个 experience 包含当前 GUI/viewport 所需扩展。`DRIBBLEBOT_ASSET_ROOT` 可用于
替换默认 `resources/` 资产根目录；目录应包含 `robots/`、`objects/` 和 `textures/`。

安装后建议先核对版本：

```bash
$ISAAC_PY -m pip show isaaclab isaacsim torch rsl-rl-lib dribblebot-isaaclab
$ISAAC_PY -m pip check
```

第一次启动 Isaac Sim 会显示 NVIDIA Omniverse EULA。请本人阅读链接中的协议并在
终端输入 `Yes` 或 `No`；不要在无人确认的脚本中自动接受许可协议。完成首次确认后，
再运行后续 registration/GPU smoke。

## 4. 低层三技能 checkpoint

### 4.1 是否真的在 match 中调用了三个低层 skill ckpt？

是。match 配置中的每个机器人都有一个独立的
`FrozenSkillPolicyAction` action term。初始化时，它会加载同一套冻结的三个
TorchScript skill：

```text
skill id 0 -> walk
skill id 1 -> dribble
skill id 2 -> shoot
```

coordinator 每台机器人输出 6D：

```text
[walk_logit, dribble_logit, shoot_logit, command_x, command_y, command_yaw]
```

最大 logit 选择 skill，后三个值经过 `tanh` 和对应 skill 的 command scale 后送入
低层策略。低层策略根据 15 帧 legacy history 产生 12D AS2 joint-position action，
再转换为关节目标。

需要区分两件事：

1. 每个机器人 action term 都会加载 walk、dribble、shoot 三个 ckpt；
2. 每个时间步只执行当前 `skill_id` 对应的一个 ckpt，不会同时运行三个 policy。

因此当 coordinator 一直选择 walk 时，dribble/shoot ckpt 已加载但不会被调用。
当前的 scripted GUI runner 也主要请求 walk，并在接近足球时请求 shoot；是否实际
进入 shoot 还要满足距离、前向和横向 strikeability 条件。

### 4.2 ckpt 保存位置

默认根目录是：

```text
/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/checkpoints/reproduction/
```

目录结构：

```text
checkpoints/reproduction/
├── walk/
│   ├── body_latest.jit
│   ├── adaptation_module_latest.jit
│   ├── ac_weights_latest.pt
│   └── config.yaml
├── dribble/
│   ├── body_latest.jit
│   ├── adaptation_module_latest.jit
│   ├── ac_weights_latest.pt
│   └── config.yaml
├── shoot/
│   ├── body_latest.jit
│   ├── adaptation_module_latest.jit
│   ├── ac_weights_latest.pt
│   └── config.yaml
└── high_level/
    ├── body_latest.jit
    ├── adaptation_module_latest.jit
    ├── ac_weights_latest.pt
    ├── opponent_ac_weights_latest.pt
    └── config.yaml
```

运行时使用的是每个 skill 目录中的 `body_latest.jit` 和
`adaptation_module_latest.jit`；`ac_weights_latest.pt` 是训练权重归档，当前
冻结 TorchScript 推理路径不直接读取它。

路径由 `policies/paths.py` 统一解析。默认使用仓库的 `checkpoints/reproduction`，
也可以替换为另一套完整 bundle：

```bash
export DRIBBLEBOT_CHECKPOINT_ROOT=/path/to/reproduction
```

替代目录必须同时包含 `walk/`、`dribble/`、`shoot/`，且每个目录都包含上述两个
`.jit` 文件。当前 manifest 校验结果：三套低层 skill 的 `.jit`、`.pt` 和配置文件
均为 `OK`，详见 `checkpoints/reproduction/manifest.sha256`。

### 4.3 低层 policy 的输入输出 contract

- 单帧 walk observation：72D；dribble/shoot：75D；
- history：15 帧零填充，分别为 1080D/1125D；
- 输出：12D，关节顺序固定为 `FL/FR/RL/RR`，每条腿为 hip/thigh/calf；
- joint target scale：hip 0.125，thigh/calf 0.25；
- match action term 负责 skill 路由、command 注入、history 更新和非法请求降级。

非法请求会记录 `requested_skill_id`、最终 `skill_id` 和 `invalid_skill`：shoot
不可达但 dribble 可达时降为 dribble；两者都不可达时降为 walk。

当前验证结论要分两层理解：

- **加载与路由正常**：三个 TorchScript bundle 都存在且 checksum 正确，
  walk/dribble/shoot/switch 的 10-tick policy contract 已通过；
- **比赛行为质量未全部验收**：现有四机器人 match contract 重点验证了“不合法
  shoot 请求安全降级为 walk”，GUI 视频主要展示 walk/approach。成功 dribble 和
  shoot 在完整 2v2 match 中还没有逐项形成正式 gate；独立 frozen-shooting smoke
  虽然 8/8 都触发了 launch，但成功率仍为 0/8。

所以可以确认“每台机器人确实接入并能够路由到三套低层 ckpt”，但暂时不能声称
三项技能在 2v2 比赛中的策略质量都已经通过。

## 5. Match MDP 摘要

### 5.1 Observation

raw manager 观测为 `(num_matches, 4, 34)`，self-play 对外为学习队两个 agent 的
`(num_matches × 2, 34)`，actor 使用 4 帧 136D history。34D 包含自身位置/速度、
朝向、球相对位置和速度、进攻球门方向、最近队友/对手、球的 dribble/shoot
affordance、当前 skill one-hot 和 skill command。

team 1 在 world frame 攻击 `-x`，观测会旋转到共享策略的 canonical `+x` frame；
walk command 保持 body-frame，dribble/shoot 平面 command 在送入 action term 前反向
旋转。

### 5.2 Reward

match 当前只注册四项：

| term | raw value | weight |
|---|---|---:|
| `alive` | 1 | `+1.0` |
| `goal` | 学习队进球的 0/1 | `+20.0` |
| `opponent_goal` | 对手进球的 0/1 | `-20.0` |
| `possession` | `exp(-2 d_min²)`，`d_min` 是四台机器人到球的最小距离 | `+0.5` |

IsaacLab `RewardManager` 会再乘 manager step `dt=0.02`：

```text
r_t = 0.02 * (1 + 20*I_goal - 20*I_opponent_goal
              + 0.5*exp(-2*d_min^2))
```

每个 macro step 累加最多 10 个低层 reward，self-play 再把 match-level reward 复制
给学习队两个 agent。当前 possession 不区分球队，且没有推进、传球、射门质量、
协作 credit 或实体球门碰撞，因此只适合作为 baseline/smoke reward。

低层 walk/dribble/shoot 任务中的 locomotion reward 不会自动继承到 match；低层
policy 在 match 中是冻结推理模块。

### 5.3 Termination 与 reset

- time out：30 s；
- goal：`ball_x >= 4.0` 且 `|ball_y| <= 1.0`；
- opponent goal：`ball_x <= -4.0` 且 `|ball_y| <= 1.0`；
- ball out：`|x| > 4.0` 或 `|y| > 2.5`；
- robot fallen：任意 base height `< 0.20 m`。

reset 时四台机器人位于 `(-1,±0.65)` 和 `(+1,±0.65)`，后两台 yaw 为 π，球在
原点附近；root velocity、关节状态、低层 history、coordinator history 和 skill
metadata 都会清零。

## 6. 如何开始训练

### 6.1 GUI 训练 smoke（推荐先运行）

GUI 模式不要传 `--headless`。先使用 1 个 match、20 个 iteration 验证训练链路：

```bash
cd /home/xander/Code/01_Locomotion/MPC-MARL-quadruped

ISAAC_PY=$PWD/isaaclab-rebuild/.venv/bin/python
export PYTHONPATH=$PWD/isaaclab-rebuild/source/dribblebot_isaaclab:${PYTHONPATH:-}
export DRIBBLEBOT_EXPERIENCE=isaacsim.exp.base.python.kit
unset HEADLESS

$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 \
  --num_envs 1 \
  --max_iterations 20 \
  --run_name gui_smoke \
  --rendering_mode performance
```

如果从 SSH/Codex 等终端启动，需要显式指定当前 Xorg 会话：

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 1 --max_iterations 20 \
  --run_name gui_smoke --rendering_mode performance
```

确认 smoke 正常后，可保留 GUI 使用 4–8 个 match：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 8 --max_iterations 5000 \
  --run_name match_selfplay_gui --rendering_mode performance
```

GUI 会降低吞吐率；正式吞吐训练建议使用 `--headless --num_envs 128`。

### 6.2 使用已有 high-level opponent checkpoint

低层三技能路径由 `DRIBBLEBOT_CHECKPOINT_ROOT` 控制；对手 coordinator 使用单独
的 `--opponent_checkpoint_root`，它应直接指向 `high_level/`：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 1 --max_iterations 100 \
  --run_name gui_archived_opponent \
  --opponent_checkpoint_root "$PWD/checkpoints/reproduction/high_level" \
  --opponent_policy_device cpu \
  --rendering_mode performance
```

不指定时，新训练从零对手（zero opponent）开始，避免 Isaac Sim 启动阶段同步复制
CUDA actor；PPO 更新后按默认 500 iteration cadence 生成 detached opponent snapshot。
从 checkpoint 恢复时，脚本会在当前 iteration 立即生成恢复 actor 的 detached snapshot，
避免恢复后临时退回 zero opponent。指定 `--opponent_checkpoint_root` 时则直接加载归档对手。

最小端到端 smoke（包含一次有效 PPO 更新）可使用 4-step rollout：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 1 --num_steps_per_env 4 \
  --max_iterations 1 --headless --run_name final_smoke
```

2026-08-27 的 cadence/resume gate 使用 8 个 physical match，从 `model_450.pt`
恢复后产生了 51 个连续 TensorBoard update（450..500），并写出
`logs/rsl_rl/dribblebot_as2_match_self_play/2026-08-27_20-17-40_resume_probe_501/model_500.pt`。
该运行跨过 iteration 500 的同步 opponent snapshot 更新且正常退出，验证了 rollout
后 actor cached distribution 的 detached deepcopy 修复。它不验证历史 opponent pool
采样或策略质量。

从 RSL-RL checkpoint 恢复时，`--max_iterations` 表示最终目标 iteration；脚本会恢复
actor、optimizer 和 iteration，并只运行剩余部分：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 8 --num_steps_per_env 24 \
  --max_iterations 5000 --headless --run_name resumed \
  --resume_checkpoint /absolute/path/to/model_500.pt
```

### 6.3 TensorBoard

训练日志默认写到：

```text
isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play/
```

另开终端：

```bash
/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/tensorboard \
  --logdir /home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play \
  --port 6006
```

浏览器打开 `http://localhost:6006`。runner 默认每 50 iteration 保存一次模型。

### 6.4 单机器人三技能 inference smoke

这个环境验证三个低层 ckpt 的加载、路由和 12D joint target，不是 coordinator PPO
训练任务：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/random_agent.py \
  --task Isaac-DribbleBot-AS2-Skill-Flat-Play-v0 \
  --device cuda:0 --steps 200 --headless
```

## 7. 验证、视频和 recorder

列出注册环境：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/list_envs.py
```

验证 match action/fallback/observation/snapshot：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/validate_match_contract.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0 \
  --num_envs 1 \
  --output isaaclab-rebuild/outputs/macro-validation/match-contract.json \
  --headless
```

运行 GUI scripted match：

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
$ISAAC_PY isaaclab-rebuild/scripts/run_match_screen.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0 \
  --num_envs 1 --steps 300
```

已验收视频：

[四机器人 match 视频](outputs/videos/match-selfplay-screen.mp4)

已有验证报告：

- `outputs/host-validation/20260825T034900Z-policy-contract/`：velocity/dribble 的
  物理和 legacy observation/history parity；
- `outputs/skill-behavior/20260825T042000Z/`：walk/dribble/shoot/switch 的冻结
  policy contract；
- `outputs/macro-validation/match-contract.json`：24D action、fallback、34D/136D
  observation 和 128D world-model state；
- `outputs/macro-validation/match-selfplay-snapshot2.json`：reset + 10 ticks 的
  canonical recorder snapshot。

## 8. 关键源码

| 内容 | 文件 |
|---|---|
| match scene、reward、termination 配置 | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/as2_match/match_env_cfg.py` |
| match reward/termination 函数 | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/match.py` |
| 34D observation | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/observations.py` |
| frozen skill action/router | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/actions.py` |
| walk/dribble/shoot ckpt 配置 | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/frozen_skill_cfg.py` |
| ckpt 路径解析 | `source/dribblebot_isaaclab/dribblebot_isaaclab/policies/paths.py` |
| deterministic match reset | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/events.py` |
| macro rate adapter | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/macro.py` |
| self-play adapter | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/self_play.py` |
| RSL-RL PPO adapter | `source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/rsl.py` |
| self-play training launcher | `scripts/train_self_play.py` |
