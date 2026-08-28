# IsaacLab 四足足球 Match 环境：MDP 与 Reward 说明

更新时间：2026-08-27

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
| `Isaac-DribbleBot-AS2-Match-Macro-Flat-v0` | 暴露整场 24D 动作的 macro 环境 |
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

当前 scene 使用 8 m × 8 m 平坦 terrain。球门和边线由 MDP 中的坐标条件定义，
并没有对应的实体球门、围栏或场地标线。

## 2. 分层 MDP

环境是分层控制系统：high-level coordinator 不直接输出 12 个关节目标，而是选择
一个冻结的低层技能并给出技能命令。

```text
learning-team coordinator: 2 × 6D action
opponent coordinator:      2 × 6D action
                   │
                   ▼
        four-robot manager action: 24D
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
- action：每个 agent 6D skill action；
- transition：冻结低层技能执行 10 个 20 ms manager step；
- reward：一个 match-level 标量，复制给学习队两个 agent；
- opponent：另一个共享 coordinator callable，定期用训练策略快照更新。

严格来说，单帧 34D 不是完整 Markov state。训练 actor 实际消费 4 帧、136D 历史，
以补充速度趋势和技能切换上下文。

## 3. Coordinator action

### 3.1 外部与内部 shape

Self-play 环境的外部 action shape 为：

```text
(num_matches × 2 learning agents, 6)
```

wrapper 为对手队生成另外两个 6D action，再拼成 manager 接收的：

```text
(num_matches, 4 robots × 6) = (num_matches, 24)
```

每台机器人动作定义为：

```text
[walk_logit, dribble_logit, shoot_logit, command_0, command_1, command_2]
```

技能选择和命令解码为：

```math
s = \operatorname*{argmax}_{i \in \{0,1,2\}} a_i
```

```math
c = \tanh(a_{3:6}) \odot C_s
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

当前 match 环境只注册四个 reward term：

| term | raw value | weight |
|---|---|---:|
| `alive` | 恒为 1 | `+1.0` |
| `goal` | 学习队进球 termination 的 0/1 指示量 | `+20.0` |
| `opponent_goal` | 对手进球 termination 的 0/1 指示量 | `-20.0` |
| `possession` | `exp(-2 d_min²)` | `+0.5` |

注意：`AS2MatchRewardsCfg` 没有继承或重新注册 AS2 velocity/dribble 任务中的
`track_lin_vel_xy_exp`、`track_ang_vel_z_exp`、`dof_torques_l2`、`action_rate_l2`、
`ball_setup_position_exp` 等低层 reward。那些 reward 只在独立的 walk/dribble/shoot
技能训练环境中使用；match 运行时低层策略被冻结，match manager 只计算上表四项。

其中：

```math
d_{min} = \min_{i \in \{0,1,2,3\}} \|p_i^{robot} - p^{ball}\|_2
```

IsaacLab `RewardManager` 会把每个 term 的 `weight` 再乘 manager step 的
`dt = 0.02 s`。所以一个低层 step 的总 reward 是：

```math
r_t = 0.02 \left[
1 + 20 I_{goal} - 20 I_{opponent\_goal}
+ 0.5 \exp(-2d_{min}^2)
\right]
```

这解释了 smoke 日志里未进球时约 `0.0206` 的单步 reward：主要来自 alive，
再叠加一个很小的 proximity/possession bonus。

`MacroActionWrapper` 最多执行 10 个低层 step，并只累计尚未结束的行：

```math
R_k^{macro} = \sum_{j=0}^{n_k-1} r_{k,j}, \qquad 1 \le n_k \le 10
```

Self-play wrapper 随后把同一个 match reward 复制给学习队的两个 agent。当前没有
per-agent reward，也没有按球权贡献拆分 credit。

### 5.2 Goal、失球和 possession 的精确定义

- 学习队进球：`ball_x >= 4.0` 且 `|ball_y| <= 1.0`；
- 对手进球：`ball_x <= -4.0` 且 `|ball_y| <= 1.0`；
- possession：只看四台机器人中离球最近的距离，不区分球队，也不判断控球朝向、
  接触状态或球速。

因此 `possession` 这个名字目前比实现语义更强。它实际是“任意机器人接近球”的
全局 shaping。无论靠近球的是学习队还是对手队，该项都会增加学习队收到的 reward。

### 5.3 当前 reward 的适用范围与缺口

这套 reward 足以验证 manager、低层技能路由、macro 聚合和 PPO 数据流，但不适合
直接作为成熟的 2v2 足球目标。主要缺口包括：

- possession 没有 team sign，不能奖励我方控球、惩罚对方控球；
- 没有球向对方球门推进、射门质量或防守阻挡 reward；
- 没有 pass、接球、空间占位、角色分工或协作 credit；
- 没有碰撞、跌倒前兆、技能切换和动作平滑成本；
- alive 是持续正奖励，可能鼓励拖延而非进攻；
- 两个学习 agent 收到完全相同的 team reward，尚未处理 individual credit assignment；
- 进球完全由球坐标判断，没有实体球门碰撞或穿门平面事件。

修改 reward 时，应优先保持 goal 作为稀疏团队目标，再添加带 team sign 的球权和
推进 shaping，并分别检查 reward scale、`dt` 缩放和 macro 累加，避免把 50 Hz
term weight 误当成 5 Hz coordinator reward。

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

### 6.2 Deterministic match reset

相对每个 environment origin，四台机器人被放置为：

| robot | position `(x, y, z)` | yaw |
|---|---|---:|
| `robot_0` | `(-1.0, -0.65, 0.34)` | 0 |
| `robot_1` | `(-1.0, +0.65, 0.34)` | 0 |
| `robot_2` | `(+1.0, -0.65, 0.34)` | π |
| `robot_3` | `(+1.0, +0.65, 0.34)` | π |
| ball | `(0.0, 0.0, 0.10)` | — |

reset 时机器人和球的 root velocity 清零，机器人关节回到默认位置。低层 15 帧历史、
coordinator 4 帧历史、前次 action、skill id 和 invalid mask 也会重置。

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
- action clip：10；
- actor 输入：136D history；
- critic 输入：当前 34D observation。

GPU PPO lifecycle 已通过 1-iteration smoke，并已从 `model_450.pt` 恢复训练到
iteration 500；默认 cadence 的 detached opponent snapshot 刷新和 `model_500.pt`
写出均成功。历史 opponent pool 的持久化/采样和训练质量仍需正式 gate。

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
