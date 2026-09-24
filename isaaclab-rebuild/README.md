# DribbleBot IsaacLab 迁移线

本文档是 `isaaclab-rebuild/` 的中文运行说明。该目录将原 Isaac Gym Preview 4
实现迁移到 IsaacLab 的 manager-based 环境。**主目标是在 IsaacLab 中从零训练
上层共享参数 MARL policy**；walking、dribbling、shooting 是冻结的环境内部执行器，
不是本迁移线要重新训练的 policy。Isaac Gym 用来提供足球任务的行为契约，不要求
逐行或逐 API 复制其实现。

### 迁移范围

从 Isaac Gym 必须迁移、并作为 MARL 训练 gate 的内容：

- 2v2 场景、物理时间尺度、球场/球门/边界以及会改变比赛状态转移的物理参数；
- 每个 agent 的混合 4D skill action（1 个离散 `skill_index` + 3 个连续参数）、
  34D 局部观测、4 帧历史和两队 canonical frame；
- 三个冻结低层 checkpoint 的推理输入、命令坐标系、技能切换、非法动作降级和
  20 ms/200 ms 控制频率；
- 比赛 reset 分布、goal/out-of-bounds/fall/time-out 终止语义；
- 上层 reward、每机器人 credit assignment、rule-based opponent、self-play pool、
  checkpoint resume 和并行 PPO 接口；
- 训练可用性指标：有限 observation/reward/action、有效技能覆盖率、episode 统计、
  TensorBoard/checkpoint、吞吐量以及一段 from-scratch learning curve。

以下内容不阻塞上层 MARL from-scratch 训练，按需后续迁移：

- Isaac Gym 的 simulator lifecycle、tensor API 和旧训练入口；
- walk/dribble/shoot 各自的低层训练 reward、command curriculum 和低层 PPO 配置；
- Gym 与 Lab 的逐时刻轨迹完全一致、完全相同的 learning curve；
- world-model collector、MPC teacher/runtime、版本化 USD 和视频展示增强。

低层 policy contract 仍必须保持兼容，因为它直接决定 MARL action 的真实执行结果。
但验收重点是三个技能在随机比赛状态下可用、稳定且可区分，而不是重新复现它们的
低层训练过程。

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
- 16D manager action、`(4, 34)` match observation、self-play 的 `(2, 34)` /
  `(2, 136)` 接口；
- 10 个低层 tick 的 macro 聚合、invalid-skill geometric fallback；
- canonical recorder snapshot 与 128D world-model state 编码；
- 四机器人 GUI scripted match 视频。
- 四机器人 CUDA PPO lifecycle：4-step rollout、1 iteration、checkpoint 与
  TensorBoard event 写出均通过；另已从 `model_450.pt` 恢复并连续训练到
  iteration 500，默认 500-iteration detached opponent snapshot 刷新通过。
- 2026-09-02 hybrid-action smoke 已验证 `Categorical(3) × Normal(3)` PPO、16D
  manager action、一次 optimizer update、TensorBoard 和 3D parameter-std checkpoint。

仍未完成的正式 gate：高质量 shooting 成功率、历史 opponent pool 的长期行为，
以及 live world-model/MPC 接入。2026-09-01
已同步当前 Isaac Gym 的球技能命令坐标系、随机比赛开局、角色滞回、support deadband、
物理边界墙和主要 dense reward；这是新的训练基线，旧训练曲线不应直接横向比较。

场景资产、接触参数和截图同步的对齐说明及验证命令见 [SCENE_ALIGNMENT.md](docs/SCENE_ALIGNMENT.md)。

## 2. 已注册环境

| 环境 ID | 说明 |
|---|---|
| `Isaac-DribbleBot-AS2-Velocity-Flat-v0` | AS2 walking 训练环境 |
| `Isaac-DribbleBot-AS2-Dribble-Flat-v0` | 带动态足球的 dribbling 训练环境 |
| `Isaac-DribbleBot-AS2-Shooting-Flat-v0` | command-relative shooting 环境 |
| `Isaac-DribbleBot-AS2-Skill-Flat-v0` | 单机器人三技能冻结策略集成 smoke |
| `Isaac-DribbleBot-AS2-Match-Macro-Flat-v0` | 四机器人、16D manager action、macro wrapper |
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

coordinator 每台机器人输出混合 4D：

```text
[skill_index, parameter_x, parameter_y, parameter_yaw]
```

`skill_index ∈ {0,1,2}` 分别表示 walk、dribble、shoot；后三个参数由混合策略的
Gaussian head 采样，经过 `tanh` 和对应 skill 的 command scale 后送入低层策略。
每个机器人只执行所选的一个冻结 ckpt。低层策略根据 15 帧 legacy history 产生
12D AS2 joint-position action，再转换为关节目标。

训练时 RSL-RL 使用 `Categorical(3)` 处理 `skill_index`、`Normal(3)` 处理连续参数，
联合计算 log-prob 和 entropy。环境仍以一个 `(N,4)` GPU tensor 传递混合动作，第一列
必须是整数值；这只是 vectorized transport，不把离散 skill 当作连续物理量学习。

这是新的 learner action contract。已有旧版 6D Gaussian coordinator checkpoint
不能继续作为同一 PPO 优化器恢复训练；如需观看旧 checkpoint，播放/对手 adapter 会
把其三段 logits 转成 `argmax skill_index`，但新训练应从新的 hybrid policy 初始化。

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

当前 Isaac Gym 的新 dribble/shoot 训练约定为：coordinator 仍输出 canonical/world
field-frame 命令，每个低层 tick 按机器人实时 yaw 转为 body-frame，再交给球技能。
IsaacLab 已同步这个 adapter，并明确区分两项配置：

- `ball_command_input_frame`：上层传入语义；match 固定为 `world`；
- `ball_skill_command_frame`：冻结 checkpoint 实际训练时使用的命令坐标系。

仓库现有 `checkpoints/reproduction/{dribble,shoot}/config.yaml` 没有
`Cfg.commands.ball_xy_frame` metadata，因此为防止静默改变旧 checkpoint 行为，默认
仍按 legacy `world` contract 加载。安装明确包含 `ball_xy_frame: body` 的新球技能
bundle 后，启动前设置：

```bash
export DRIBBLEBOT_BALL_SKILL_COMMAND_FRAME=body
```

不能把旧 world-frame 与新 body-frame 的 dribble/shoot checkpoint 混在同一个
bundle 中。walk 命令始终是 body-frame，不受这个变量影响。

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
affordance、attacker/support role bit、当前 skill one-hot 和 skill command。role bit
复用了旧 observation 中恒为 1 的 teammate-presence 槽，因此维度仍保持 34；每队
离球最近者先成为 attacker，只有另一台机器人至少近 0.15 m 时才切换角色。

team 1 在 world frame 攻击 `-x`，观测会旋转到共享策略的 canonical `+x` frame；
walk command 保持 body-frame，dribble/shoot 平面 command 在送入 action term 前反向
旋转。

### 5.2 Reward

match 使用 18 个 team-level terms；reward 权重和 reset 分布按训练阶段动态调度：

| term | raw value | weight |
|---|---|---:|
| `goal` | 学习队进球事件 | `+3000.0`（成熟阶段） |
| `accidental_termination` | 对手进球、球出界或机器人跌倒 | `-180.0`（成熟阶段） |
| `timeout` | 到达时间上限的终止惩罚 | `-180.0`（成熟阶段） |
| `time_pressure` | 随 episode 时间增加的轻微惩罚 | `+0.3`（函数为负） |
| `ball_goal_progress` | 球速在对方球门方向的投影，clip 到 `[-1,1]` | `+1.0` |
| `ball_position_progress` | reset-safe 的球位置增量（朝球门方向） | `+6.0` |
| `robot_spacing` | 仅惩罚队友重叠和 support 抢球 | `-0.5`（成熟阶段） |
| `robot_collision` | 当前/未来 0.25 s 内、0.65 m 阈值的最坏 pair overlap² | `-1.5`（成熟阶段） |
| `invalid_skill` | 学习队任一机器人请求非法 skill（fallback/local credit 已另行处理） | `-0.5`（成熟阶段） |
| `aggressive_command` | walk/dribble 超出稳定速度上限 | `-1.0`（成熟阶段） |
| `pass_ball` | shoot 后球朝可接应队友运动 | `+2.0` |
| `approach_ball` | attacker 使用 walk 时朝球的速度投影 | `+0.75` |
| `walk_command_alignment` | attacker walk command 与球方向一致性 | `+0.2` |
| `face_ball_while_approaching` | attacker 实际朝球运动且面向足球 | `+0.1` |
| `face_goal_while_moving` | attacker 同时朝球门方向和机身前向移动 | `+0.5` |
| `dribble_ball_control` | 近球 dribble 且球速朝向球门 | `+3.5` |
| `shoot_setup` | 近球 shoot 时机器人—球—球门对齐 | `+14.0` |
| `shoot_launch` | 有效 shoot 使球沿 command 突增到发射速度 | `+30.0` |

IsaacLab `RewardManager` 会再乘 manager step `dt=0.02`：

```text
r_t = 0.02 * sum(weight_i * term_i)
```

每个 macro step 累加最多 10 个低层 reward，self-play 再把 match-level reward 复制
给学习队两个 agent。attacker/support local credit 会再按 agent 单独叠加；pass 和
shoot-launch 已同步，其中 launch term 持有 reset-safe 的前一拍球速/距离状态。

Curriculum 分三个阶段：前约 500 个 PPO iteration 只在较小场地、我方近球起点和较轻
终止惩罚下学习稳定控球；约 500–2000 iteration 扩大起始分布并增加 shoot/goal 权重；
之后使用完整随机场地和成熟 reward。每次 reset 会写入
`Curriculum/training_difficulty/{phase,near_ball_probability,difficulty}`。
分段训练恢复时，`train_self_play.py` 会把 checkpoint iteration 换算为
`curriculum_step_offset`，因此每次 eval 后重建 Isaac Sim 不会将 curriculum 重置到 phase 0。
Hybrid PPO 的连续参数 std 被限制在 `[0.15,0.6]`；entropy bonus 只用于 categorical
skill，避免上一轮出现参数噪声变大、随后 skill entropy 塌缩的现象。

训练同时记录 `Episode_Action/requested_*`、`Episode_Action/executed_*`、
`Episode_Action/fallback_rate` 和 `Episode_Action/invalid_rate`，用于确认策略是否
真的选择了 dribble/shoot，而不是被几何 fallback 改写。

低层 walk/dribble/shoot 任务中的 locomotion reward 不会自动继承到 match；低层
policy 在 match 中是冻结推理模块。

### 5.3 Termination 与 reset

- time out：30 s；
- goal：`ball_x >= 4.0` 且 `|ball_y| <= 1.0`；
- opponent goal：`ball_x <= -4.0` 且 `|ball_y| <= 1.0`；
- ball out：`|x| > 4.0` 或 `|y| > 2.5`；
- robot fallen：任意 base height `< 0.20 m`。

训练 reset 使用 8 m × 5 m 球场内的随机布局：学习队在 `x∈[-3.4,0]`，对手在
`x∈[0,3.4]`，所有机器人 `y∈[-1.9,1.9]`、yaw 为 `[-π,π]`，机器人和球默认保持
0.75 m clearance；成熟阶段球范围为 `x∈[-3.2,2.8]`、`y∈[-1.7,1.7]`。
near-ball reset 比例按 curriculum 从 90%（仅我方）降到 70%（仅我方），最后为
50%（双方机器人）。PLAY 配置仍使用固定的
`(-1,±0.65)` / `(+1,±0.65)` 布局，便于复现和录制。

场地外围包含 6 段 0.5 m 高碰撞墙，两端在 `|y|≤1.0` 留球门开口；两侧有白色
立柱/横梁，地面使用 `resources/textures/field.png` 的无碰撞 UV 贴图，物理接触仍由
TerrainImporter 单独负责。终场、长边和球门检测使用 8 m × 5 m canonical field。
reset 同时清零 root velocity、关节
状态、低层/coordinator history、skill metadata 和 attacker 滞回状态。

## 6. 如何开始训练

### 6.0 一键训练脚本

默认使用 128 个并行 match（256 个学习 agent 样本）、24-step rollout、
5000 iterations 和 `cuda:0`：

```bash
./isaaclab-rebuild/train_self_play.sh
```

常用参数通过环境变量覆盖，不需要修改脚本：

```bash
# GUI 小规模训练
HEADLESS=0 NUM_ENVS=1 RUN_NAME=gui_debug \
  ./isaaclab-rebuild/train_self_play.sh

# 从 checkpoint 继续到目标 iteration（MAX_ITERATIONS 是最终目标值）
RESUME_CHECKPOINT=/absolute/path/to/model_1000.pt \
MAX_ITERATIONS=5000 RUN_NAME=resumed \
  ./isaaclab-rebuild/train_self_play.sh

# 换 GPU 或只打印最终命令
DEVICE=cuda:1 DRY_RUN=1 ./isaaclab-rebuild/train_self_play.sh
```

支持的环境变量包括 `ISAAC_PYTHON`、`TASK`、`DEVICE`、`NUM_ENVS`、
`NUM_STEPS_PER_ENV`、`MAX_ITERATIONS`、`SEED`、`RUN_NAME`、`HEADLESS`、
`RESUME_CHECKPOINT`、`OPPONENT_CHECKPOINT_ROOT`、`OPPONENT_POLICY_DEVICE`、
`OPPONENT_MODE`、`OPPONENT_POOL_SIZE`、`OPPONENT_LATEST_PROBABILITY`、`DRIBBLEBOT_CHECKPOINT_ROOT`
和 `DRIBBLEBOT_BALL_SKILL_COMMAND_FRAME`。
额外 CLI 参数会原样追加到 Python 训练入口。

### 6.1 定期 eval 与视频

推荐使用分段编排脚本。它每训练 `EVAL_INTERVAL` 个 iteration 就关闭训练用的
Isaac Sim，启动独立的评估仿真，保存 JSON 指标和 MP4，然后从 checkpoint 继续；
这样不会同时运行两个 Kit/PhysX 进程。

```bash
EVAL_INTERVAL=500 \
EVAL_EPISODES=1 \
EVAL_OPPONENT=self \
EVAL_VIDEO=1 \
MAX_ITERATIONS=5000 \
RUN_NAME=hybrid_marl_eval \
./isaaclab-rebuild/train_self_play_eval.sh
```

产物位于：

```text
isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play/<RUN_NAME>/
├── model_*.pt
└── eval/
    ├── iter_499.json
    ├── iter_499.mp4
    └── ...
```

`Eval/mean_return`、`Eval/mean_length` 和 walk/dribble/shoot 选择比例会写入同一
TensorBoard run。`EVAL_VIDEO=0` 可只保存指标；评估默认只使用 1 个 match，避免
视频渲染显著拖慢训练。

评估脚本的 `--video_output` 会创建独立的场景相机并写出 H.264 MP4（不依赖 Kit
viewport）。例如只评估已有 checkpoint：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/play_self_play.py \
  --checkpoint /absolute/path/to/model_499.pt \
  --episodes 1 --num_envs 1 --opponent self --headless \
  --video_output isaaclab-rebuild/outputs/videos/eval-499.mp4 \
  --metrics_output isaaclab-rebuild/outputs/videos/eval-499.json
```

视频评估会自动使用 USD 同步模式和非渲染 reset；在无 X display 的服务器上也可运行，
但每次评估会比纯 metrics 模式慢一些。周期性流程会在每个 checkpoint 下生成
`eval/iter_<N>.mp4` 与同名 JSON。

### 6.2 GUI 训练 smoke（推荐先运行）

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

### 6.3 使用已有 high-level opponent checkpoint

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
运行时 pool 最多保留 8 个 snapshot（初始 anchor 不淘汰），每个 match/reset 独立采样；
默认以 50% 概率选择最新 snapshot，其余概率均分给历史版本。
learned pool 会写入 RSL-RL checkpoint 的 `infos.dribblebot_opponent_pool` 并在 resume
时重建；保存内容仅含 actor MLP/normalizer，不重复保存 critic。旧的完整 ActorCritic
pool state 也能兼容读取。
从 checkpoint 恢复时，脚本会在当前 iteration 立即生成恢复 actor 的 detached snapshot，
避免恢复后临时退回 zero opponent。指定 `--opponent_checkpoint_root` 时则直接加载归档对手。

训练时也可以使用无需 checkpoint 的规则对手：最近球的机器人以 0.15 m 滞回担任
attacker，按距离选择 walk/dribble/shoot；另一台机器人封堵 learner-to-ball 通道，
两者都带短距碰撞规避：

```bash
OPPONENT_MODE=rule_based ./isaaclab-rebuild/train_self_play.sh
```

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
后 actor cached distribution 的 detached deepcopy 修复。运行时 pool 的分组推理、
reset 重采样、checkpoint 写入均已接入；长期策略质量仍需完整训练验证。

从 RSL-RL checkpoint 恢复时，`--max_iterations` 表示最终目标 iteration；脚本会恢复
actor、optimizer 和 iteration，并只运行剩余部分：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --device cuda:0 --num_envs 8 --num_steps_per_env 24 \
  --max_iterations 5000 --headless --run_name resumed \
  --resume_checkpoint /absolute/path/to/model_500.pt
```

### 6.4 TensorBoard

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

### 6.5 单机器人三技能 inference smoke

这个环境验证三个低层 ckpt 的加载、路由和 12D joint target，不是 coordinator PPO
训练任务：

```bash
$ISAAC_PY isaaclab-rebuild/scripts/random_agent.py \
  --task Isaac-DribbleBot-AS2-Skill-Flat-Play-v0 \
  --device cuda:0 --steps 200 --headless
```

### 6.6 可视化训练 checkpoint

默认自动加载最近修改的 `model_*.pt`，在 Isaac Sim GUI 中观看 3 个 episode；
学习队和对手使用同一个冻结 checkpoint：

```bash
./isaaclab-rebuild/play_self_play.sh
```

常用覆盖方式：

```bash
# 指定最终 checkpoint，观看 5 个 episode
CHECKPOINT=$PWD/isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play/\
2026-08-27_20-24-44_match_selfplay_resumed_20260827/model_4999.pt \
EPISODES=5 ./isaaclab-rebuild/play_self_play.sh

# 和静止的 zero opponent 对战；FRAME_DELAY 控制肉眼观看速度
OPPONENT=zero FRAME_DELAY=0.08 EPISODES=3 \
  ./isaaclab-rebuild/play_self_play.sh

# 和无需 checkpoint 的规则对手对战
OPPONENT=rule_based EPISODES=3 ./isaaclab-rebuild/play_self_play.sh
```

每个 episode 结束后，终端会输出累计 return、长度，以及四台机器人实际执行的
walk/dribble/shoot 次数。关闭 Isaac Sim 窗口可以提前结束。

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
- `outputs/macro-validation/match-contract.json`：混合 action、fallback、34D/136D
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
