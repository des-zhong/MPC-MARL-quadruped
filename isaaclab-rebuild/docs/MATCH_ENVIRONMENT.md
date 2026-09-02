# IsaacLab 四足足球 Match 环境：MDP 与 Reward 说明

更新时间：2026-09-02

本文说明 `isaaclab-rebuild` 中四台 AS2 机器人、两支球队和一个足球组成的
IsaacLab manager-based match 环境。重点是当前代码实际采用的 MDP、reward、
termination、reset 和 self-play 接口，而不是计划中的最终足球任务。

> 当前结论：四机器人场景、分层动作、观测、宏动作和 self-play wrapper 已能在
> Isaac Sim 5.1.0 + PhysX + RTX 4090 上运行；现有 match reward 仍是用于打通训练
> 链路的最小 baseline，不能视为完整足球博弈 reward。

## 1. 环境入口与基本参数

主要 Gym ID：

| Gym ID | 用途 |
|---|---|
| `Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0` | 默认训练环境，外部只暴露学习队的两个 agent |
| `Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0` | 小规模可视化和 smoke 环境 |
| `Isaac-DribbleBot-AS2-Match-Macro-Flat-v0` | 暴露整场 16D 动作的 macro 环境 |
| `Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0` | macro 环境的可视化版本 |

基础配置：

| 项目 | 当前值 |
|---|---:|
| 机器人 | 4 台 AS2，`robot_0` 至 `robot_3` |
| 球队 | team 0：`robot_0/1`；team 1：`robot_2/3` |
| 足球 | 1 个动态刚体 |
| 物理时间步 | 0.005 s，200 Hz |
| manager/低层策略步 | 4 个物理步，即 0.02 s、50 Hz |
| coordinator macro 步 | 10 个低层步，即 0.2 s、5 Hz |
| episode 时长 | 30 s |
| 最大低层步数 | 1500 |
| 最大 macro 步数 | 150 |
| 逻辑球场范围 | `x ∈ [-4, 4]`，`y ∈ [-2.5, 2.5]` |
| 默认并行 match 数 | 128 |
| PLAY 并行 match 数 | 2，可被 `--num_envs` 覆盖 |

当前 scene 使用 8 m × 5 m 平坦 terrain，包含实体边界墙和球门 primitive；终止与
进球仍由 MDP 坐标条件判定。

## 2. 分层 MDP

环境是分层控制系统：high-level coordinator 不直接输出 12 个关节目标，而是选择
一个冻结的低层技能并给出技能命令。

```text
learning-team coordinator: 2 × 4D hybrid action
opponent coordinator:      2 × 4D hybrid action
                   │
                   ▼
        four-robot manager action: 16D
                   │
                   ▼
    walk / dribble / shoot frozen TorchScript router
                   │
                   ▼
         4 × 12D joint-position targets
                   │
                   ▼
         PhysX: 4 × 5 ms per manager step
```

从 coordinator 角度，可将当前任务写成：

- agent：学习队的两台机器人，共享策略参数；
- observation：每个 agent 当前 34D 局部观测和 4 帧历史；
- action：每个 agent 4D hybrid action `[skill_index, p_x, p_y, p_yaw]`；
- transition：冻结低层技能执行 10 个 20 ms manager step；
- reward：一个 match-level 标量，复制给学习队两个 agent；
- opponent：另一个共享 coordinator callable，定期用训练策略快照更新。

严格来说，单帧 34D 不是完整 Markov state。训练 actor 实际消费 4 帧、136D 历史，
以补充速度趋势和技能切换上下文。

## 3. Coordinator action

### 3.1 外部与内部 shape

Self-play 环境的外部 action shape 为：

```text
(num_matches × 2 learning agents, 4)
```

wrapper 为对手队生成另外两个 4D action，再拼成 manager 接收的：

```text
(num_matches, 4 robots × 4) = (num_matches, 16)
```

每台机器人动作定义为：

```text
[skill_index, parameter_x, parameter_y, parameter_yaw]
```

其中 `skill_index` 是离散动作，取值 `0/1/2`；后三项是连续 skill parameters。
RSL-RL actor 使用 `Categorical(3) × Normal(3)` 联合分布，环境 transport tensor 的
第一列保存采样得到的整数 skill index。动作解码为：

```math
s = \operatorname{round}(a_0) \in \{0,1,2\}
```

```math
c = \tanh(a_{1:4}) \odot C_s
```

其中 command scale 为：

| skill id | skill | `C_s = (x, y, yaw)` |
|---:|---|---|
| 0 | walk | `(1.5, 1.5, 1.0)` |
| 1 | dribble | `(1.5, 1.5, 1.0)` |
| 2 | shoot | `(3.0, 3.0, 0.0)` |

shoot 的第三个命令分量总是强制为 0。walk 的平面命令按机器人 body frame 解释；
dribble/shoot 的平面命令按球场 frame 解释。

### 3.2 几何合法性与技能降级

每个 `FrozenSkillPolicyAction` 在执行前检查球相对机器人是否满足技能几何条件：

- dribble：机器人到球的平面距离不超过 1.0 m；
- shoot：距离不超过 0.75 m，且球在机器人局部坐标中满足
  `x >= -0.1 m`、`|y| <= 0.45 m`。

不合法请求按以下规则安全降级：

| 请求 | 当前几何状态 | 实际执行 |
|---|---|---|
| dribble | 球超过 1.0 m | walk |
| shoot | 不能 shoot，但球在 1.0 m 内 | dribble |
| shoot | 既不能 shoot，也不能 dribble | walk |

强制 walk 时，控制器以 0.9 m/s 向球靠近，并根据球的局部方位生成 yaw 命令。
原始请求、最终技能和 `invalid_skill` mask 都保存在 action term 和 recorder 中。

### 3.3 冻结低层策略

walk、dribble、shoot 都使用已有 TorchScript body + adaptation module，不在 match
训练中更新。低层策略输入遵循 legacy checkpoint contract：

| 75D 单帧分量 | 维度 |
|---|---:|
| 球在机器人 body frame 中的位置 | 3 |
| projected gravity | 3 |
| 缩放后的速度和 gait command | 15 |
| 关节位置偏移 | 12 |
| 缩放后的关节速度 | 12 |
| 当前低层 action | 12 |
| 前一低层 action | 12 |
| 四腿 gait clock | 4 |
| heading | 1 |
| gait phase | 1 |
| 合计 | 75 |

每个低层策略使用 15 帧零填充历史。dribble/shoot 输入宽度为
`15 × 75 = 1125`；walk policy 会去掉每帧的 3D 球传感器，得到
`15 × 72 = 1080`。低层输出为 12D action，经关节默认角和不同关节组的 scale
转换为 position target：hip scale 0.125，thigh/calf scale 0.25。

## 4. 34D match observation

raw manager observation shape 是 `(num_matches, 4, 34)`。Self-play wrapper 只返回
team 0 的两个机器人，因此对外为 `(num_matches × 2, 34)`；4 帧历史为 136D。

当前 `policy` 和 `critic` 使用同一个 34D observation，尚未给 critic 增加 privileged
global state。

| index | 分量 | 维度 | 归一化或含义 |
|---|---|---:|---|
| 0–1 | own position | 2 | 除以 `(4.0, 2.5)` |
| 2–3 | robot forward vector | 2 | canonical team frame |
| 4–5 | own planar velocity | 2 | 除以 3.0 |
| 6 | own yaw rate | 1 | 除以 3.0 |
| 7–8 | robot-to-ball vector | 2 | world delta，除以 `(4.0, 2.5)` |
| 9–10 | ball planar velocity | 2 | 除以 5.0 |
| 11 | ball distance | 1 | 除以球场半长宽的对角线 |
| 12–13 | robot-to-attacking-goal vector | 2 | 除以 `(4.0, 2.5)` |
| 14–15 | nearest teammate relative position | 2 | canonical team frame |
| 16 | teammate valid mask | 1 | 当前 2v2 中为 1 |
| 17–18 | nearest opponent relative position | 2 | canonical team frame |
| 19–20 | nearest opponent velocity | 2 | 除以 3.0 |
| 21 | opponent valid mask | 1 | 当前 2v2 中为 1 |
| 22–23 | ball delta in robot-local frame | 2 | 除以 `(4.0, 2.5)` |
| 24 | ball distance | 1 | 与 index 11 相同的归一化距离 |
| 25 | can dribble | 1 | 0/1 affordance |
| 26 | can shoot | 1 | 0/1 affordance |
| 27 | behind-ball alignment | 1 | `[-1, 1]` |
| 28–30 | current executed skill | 3 | one-hot |
| 31–33 | current skill command | 3 | 归一化并转换到 canonical team frame |

team 0 在 world frame 中向 `+x` 进攻。team 1 向 `-x` 进攻，其 position、velocity、
forward、球和其他机器人相对量会旋转 `π`，让两队共享的 coordinator 始终看到
“向 canonical `+x` 进攻”的观测。对手 policy 输出的 dribble/shoot 平面命令随后
再旋转回 world frame；walk 命令保持 body-frame 语义，不做该镜像。

## 5. 当前 match reward

### 5.1 Reward terms

match 只计算上层 MARL objective，不继承低层 walk/dribble/shoot 的训练 reward。
当前注册 14 项 team-level terms：

| term | 作用 | weight |
|---|---|---:|
| `goal` | 学习队进球事件 | `+500.0` |
| `accidental_termination` | 对手进球、球出界或机器人跌倒 | `-200.0` |
| `ball_goal_progress` | 球速朝攻击球门方向的投影 | `+2.0` |
| `robot_spacing` | 队友有效间距与 support crowding | `+0.75` |
| `robot_collision` | 当前/0.25 s 预测机器人碰撞 | `-2.0` |
| `invalid_skill` | 非法 skill 请求 | `-3.0` |
| `pass_ball` | 射门后球朝可接应队友运动 | `+2.0` |
| `approach_ball` | attacker 用 walk 接近球 | `+1.0` |
| `walk_command_alignment` | walk command 朝向球 | `+0.5` |
| `face_ball_while_approaching` | 接近球时机身朝向球 | `+0.5` |
| `face_goal_while_moving` | 朝球门且沿机身前向移动 | `+0.75` |
| `dribble_ball_control` | dribble 时球速与 command 一致 | `+2.0` |
| `shoot_setup` | 合法、对齐的射门选择 | `+5.0` |
| `shoot_launch` | 射门导致球速沿 command 突增 | `+10.0` |

IsaacLab `RewardManager` 会把每个 term 的 weight 再乘 manager step 的 `dt=0.02 s`。
`MacroActionWrapper` 最多执行 10 个低层 step，并将活动行的 reward 求和；随后
self-play wrapper 将 team return 加上按 attacker/support 分配的 local role credit，
复制给学习队两个 agent。`goal` 和终止惩罚是稀疏团队目标，其余项只作为 shaping，
不能替代进球目标。

`MacroActionWrapper` 最多执行 10 个低层 step，并只累计尚未结束的行：

```math
R_k^{macro} = \sum_{j=0}^{n_k-1} r_{k,j}, \qquad 1 \le n_k \le 10
```

Self-play wrapper 随后把同一个 match reward 复制给学习队的两个 agent。当前没有
per-agent reward，也没有按球权贡献拆分 credit。

### 5.2 Goal、失球和 role credit 的精确定义

- 学习队进球：`ball_x >= 4.0` 且 `|ball_y| <= 1.0`；
- 对手进球：`ball_x <= -4.0` 且 `|ball_y| <= 1.0`；
- attacker 由每队离球最近机器人决定，并带 0.15 m hysteresis；support 不抢球，
  使用带 0.08 m deadband 的支撑命令。
- local role credit 只奖励学习队实际承担的 attacker/support 行为；终止行不重复
  叠加状态差分 shaping。

### 5.3 当前 reward 的适用范围与缺口

这套 reward 已覆盖 from-scratch MARL 的最小进攻、协作和安全信号，但仍需用训练
统计校准权重。当前尚未把 world-model/MPC teacher 信号混入 objective，也不要求
低层技能 reward/curriculum 进入 match。修改 reward 时应保持 goal 的稀疏团队目标，
并分别检查 reward scale、`dt` 缩放和 macro 累加，避免把 50 Hz term weight 误当成
5 Hz coordinator reward。

## 6. Termination 与 reset

### 6.1 Termination

| term | 条件 | 类型 |
|---|---|---|
| `time_out` | episode 达到 30 s | truncated/time-out |
| `goal` | `x >= 4.0` 且 `|y| <= 1.0` | terminated |
| `opponent_goal` | `x <= -4.0` 且 `|y| <= 1.0` | terminated |
| `ball_out_of_bounds` | `|x| > 4.0` 或 `|y| > 2.5` | terminated |
| `robot_fallen` | 任意一台机器人 base height `< 0.20 m` | terminated |

Macro step 中一旦某个 match 结束，wrapper 会保存 termination/truncation，之后的
子步只给该行传零动作，也不再累计 reward。`elapsed_low_level_steps` 记录实际执行步数。

### 6.2 Match reset

PLAY/smoke 配置相对每个 environment origin 使用固定布局：

| robot | position `(x, y, z)` | yaw |
|---|---|---:|
| `robot_0` | `(-1.0, -0.65, 0.34)` | 0 |
| `robot_1` | `(-1.0, +0.65, 0.34)` | 0 |
| `robot_2` | `(+1.0, -0.65, 0.34)` | π |
| `robot_3` | `(+1.0, +0.65, 0.34)` | π |
| ball | `(0.0, 0.0, 0.10)` | — |

训练配置改用随机布局：学习队 `x∈[-3.4,0]`，对手队 `x∈[0,3.4]`，机器人
`y∈[-1.9,1.9]`、yaw∈`[-π,π]`，球 `x∈[-3.2,2.8]`、`y∈[-1.7,1.7]`，并保持
最小 0.75 m clearance；40% reset 将球放到随机机器人前方 0.4–0.95 m。所有 reset
都会清零 root velocity、关节状态、低层/coordinator history、skill metadata 和
attacker 滞回状态。

## 7. Self-play 与 PPO 接口

`MatchSelfPlayWrapper` 将一个物理 match 展开成两个学习样本：

```text
physical num_envs = number of matches
training num_envs = number of matches × team_size(2)
```

如果没有安装 opponent callable，对手输出全零 action。训练入口会安装当前 actor 的
detached deterministic snapshot，并默认每 500 个 PPO iteration 更新一次。对手队
输入使用自己的 136D canonical history，输出经过 team-frame 逆变换后送入 manager。

当前 PPO baseline：

- rollout：每个 agent 24 个 macro step；
- actor/critic MLP：`[512, 256, 128]`，ELU；
- `gamma=0.99`，`lambda=0.95`；
- learning rate：`1e-4`；
- action clip：10（连续 transport 值；skill index 由 categorical sample 产生）；
- actor 输入：136D history；
- critic 输入：当前 34D observation。

GPU PPO lifecycle 已通过 1-iteration smoke，并已从 `model_450.pt` 恢复训练到
iteration 500；默认 cadence 的 detached opponent snapshot 刷新和 `model_500.pt`
写出均成功。历史 opponent pool 的持久化/采样和训练质量仍需正式 gate。

新的 hybrid action 在 RTX4090 上完成了独立 4-step/1-iteration smoke；actor 的
六个网络输出解释为三项 categorical logits 和三项 parameter means，环境只接收
采样后的 4D action。checkpoint 中只保存三项 parameter std。旧 6D Gaussian
checkpoint 可以经 adapter 做确定性推理，但不能恢复新 hybrid PPO optimizer。

## 8. Recorder 与 world-model state

`RecorderManager` 每个低层 tick 可记录：

- 四台机器人的 root state、12D joint position/velocity/target；
- 足球 13D root state；
- 四台机器人的 requested/executed skill id；
- 四台机器人的 3D skill command 和 `invalid_skill`。

这些 snapshot 可经 `IsaacLabFootballWorldModelStateAdapter` 编码为现有四机器人
128D canonical world-model state。world model 和 MPC 应消费该 simulator-neutral
snapshot，不应直接读取 IsaacLab manager 内部 tensor。

## 9. 运行与验证

推荐使用本项目独立的 Python 3.11 环境：

```bash
/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python
```

完整创建和安装步骤见 [`README.md`](README.md)。此前 GPU/GUI smoke 临时复用了
`manifold_manip/.venv`，但正式训练不再推荐跨项目共享该环境。

运行 match contract smoke：

```bash
PYTHON=/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python
$PYTHON isaaclab-rebuild/scripts/validate_match_contract.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0 \
  --num_envs 1 \
  --output isaaclab-rebuild/outputs/macro-validation/match-contract.json \
  --headless
```

运行可视化 scripted match：

```bash
PYTHON=/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python
$PYTHON isaaclab-rebuild/scripts/run_match_screen.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0 \
  --num_envs 1 \
  --steps 300
```

训练 self-play baseline：

```bash
PYTHON=/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python
$PYTHON isaaclab-rebuild/scripts/train_self_play.py \
  --task Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0 \
  --num_envs 128 \
  --headless
```

已有证据：

- [match contract](outputs/macro-validation/match-contract.json)
- [macro/self-play smoke](outputs/macro-validation/match-selfplay.json)
- [recorder snapshot smoke](outputs/macro-validation/match-selfplay-snapshot2.json)
- [四机器人可视化视频](outputs/videos/match-selfplay-screen.mp4)

## 10. 关键源码

| 内容 | 文件 |
|---|---|
| scene、reward、termination 和 manager config | [`match_env_cfg.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/as2_match/match_env_cfg.py) |
| match reward/termination functions | [`mdp/match.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/match.py) |
| 34D observation | [`mdp/observations.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/observations.py) |
| frozen skill action router | [`mdp/actions.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/actions.py) |
| deterministic reset | [`mdp/events.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/events.py) |
| coordinator-rate aggregation | [`macro.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/macro.py) |
| self-play/team expansion | [`self_play.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/self_play.py) |
| team-frame transform | [`team_frame.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/team_frame.py) |
| legacy low-level observation contract | [`legacy_contract.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/mdp/legacy_contract.py) |
| PPO/RSL-RL adapter | [`rsl.py`](source/dribblebot_isaaclab/dribblebot_isaaclab/tasks/manager_based/football/rsl.py) |
