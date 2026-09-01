# MPC-MARL Quadruped Soccer — Current Context

更新时间：2026-08-27

## 项目目的

本仓库实现四足机器人足球的分层多智能体强化学习系统，当前主线仍是 Isaac Gym Preview 4，迁移线位于 `isaaclab-rebuild/`。

系统包含：

- AS2/Go1 四足机器人的 walking、dribbling、shooting 低层技能；
- 共享参数的 high-level skill coordinator；
- 两队 self-play 与历史 opponent pool；
- joint world model 数据采集和训练；
- hybrid CEM MPC、terminal-value model 和 MPC teacher guidance。

核心领域对象是：机器人、足球、球场、球门、球队、技能、技能命令、比赛状态、世界模型状态和 MPC rollout。

## 稳定领域约定

- 坐标：学习队攻击 `+x` 方向；对手队在 world frame 攻击 `-x`，输入到共享策略前旋转 `pi`。
- 物理步长：5 ms。
- 低层 policy/environment 步长：20 ms，即 4 个物理步。
- high-level coordinator 步长：10 个低层步，即 200 ms。
- AS2 legacy joint order：`FL/FR/RL/RR`，每条腿依次 hip、thigh、calf，共 12 个 actuated joints。
- 低层 legacy policy observation：walking 72D，带球技能 75D。
- 低层 legacy history：15 帧，分别为 1080D/1125D，首个外部 reset 使用零填充。
- match agent observation：34D；4 帧 coordinator history：136D。
- 四机器人 world-model canonical state 当前编码为 128D。
- invalid shoot request 必须安全降级为 walk，并暴露 `invalid_skill` 标记。

## 当前实现的两条代码线

### Legacy Isaac Gym implementation

主要模块：

- `quadruped/envs/base/base_task.py`
- `quadruped/envs/base/legged_robot.py`
- `quadruped/envs/base/legged_robot_walk.py`
- `quadruped/envs/base/legged_robot_two.py`
- `quadruped/sensors/`
- `quadruped/rewards/`
- `quadruped/terrains/`
- `quadruped/envs/wrappers/`

这些模块把 simulator lifecycle、Gym tensor layout、actor indices、reset、reward、observation 和 wrapper 逻辑集中在几个很大的、较 shallow 的 module 中。训练脚本仍直接创建 Isaac Gym 环境。

### Isaac Lab manager-based migration

迁移实现位于 `isaaclab-rebuild/source/dribblebot_isaaclab/`，任务位于：

- `tasks/manager_based/as2_velocity/`
- `tasks/manager_based/as2_dribble/`
- `tasks/manager_based/as2_shooting/`
- `tasks/manager_based/as2_skill/`
- `tasks/manager_based/as2_match/`
- `tasks/manager_based/football/`

manager-based 结构如下：

- `InteractiveSceneCfg`：机器人、足球、地面和环境实例；
- `ActionManager`：关节位置控制、冻结低层策略和 skill router；
- `ObservationManager`：普通 observation、legacy policy observation 和 stateful history；
- `RewardManager`：运动、控球、射门和比赛奖励 terms；
- `TerminationManager`：跌倒、出界、进球和射门 phase termination；
- `EventManager`：reset、COM/collider 设置、domain randomization；
- `CommandManager`：速度、步态和比赛命令；
- `RecorderManager`：canonical football snapshots。

## 已建立的架构 seam

这里使用架构术语：module、interface、implementation、depth、deep、shallow、seam、adapter、leverage、locality。

1. **Manager-based environment seam**

   Isaac Lab 的 `ManagerBasedRLEnv` 隐藏 simulator lifecycle；任务差异通过 config 和 manager terms 表达。物理状态访问应集中在 manager term 的 implementation 内，不向训练脚本泄漏。

2. **Legacy policy contract seam**

   `legacy_policy` 和 `legacy_history` 保留旧 checkpoint 所需的 observation 顺序、缩放、噪声、action history 和 reset 语义。不要用 Isaac Lab 默认 circular history 替换零填充 contract。

3. **Coordinator rate seam**

   `football/macro.py` 中的 `MacroActionWrapper` 位于 manager environment 外部。manager 每次仍只执行一个 20 ms low-level step，wrapper 负责 10 步聚合 reward、done 和 `elapsed_low_level_steps`。

4. **Match/self-play seam**

   `football/self_play.py` 中的 `MatchSelfPlayWrapper` 把四机器人 match 映射为学习队的两个 agent，并隔离 team-frame、opponent callable、history 和 snapshot iteration。RSL-RL 的 agent-expanded adapter 不应把 match count 错当成 agent count。

5. **Canonical snapshot seam**

   `football/snapshot.py` 和 `world_model_adapter.py` 定义 simulator-neutral snapshot。world model、dataset 和 MPC 应只依赖这个 interface；Isaac Gym 和 Isaac Lab 各自提供 adapter。

## 当前验证状态

### 已通过或已完成 smoke

- Isaac Sim 5.1.0 + PhysX + RTX 4090 上，AS2 velocity/dribble 的 zero/sine physics 与 policy-contract parity archive 通过 4/4。
- `Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0` 可以创建 4 个 AS2 articulation 和 1 个足球。
- match manager action 为 24D，即 4 个机器人各 6D skill action。
- manager observation 为 `(4, 34)`；self-play wrapper 对外提供 `(2, 34)` observation 和 `(2, 136)` history。
- coordinator macro smoke 能执行 10 个 low-level ticks，reward finite，done contract 正常。
- invalid shooting request 的 geometric fallback 会执行 walk 并标记 invalid。
- RecorderManager 能记录 reset plus 10 个 low-level ticks 的机器人、关节、球、技能和命令状态。
- canonical snapshot 能编码为现有四机器人 128D world-model state。
- frozen skill behavior matrix 的 walk/dribble/shoot/switch interface contract 已通过 10-tick gate。
- 四机器人 CUDA PPO 已从 `model_450.pt` 恢复并完成 51 个连续 update（450..500）；
  iteration 500 的 detached opponent snapshot 同步刷新和 `model_500.pt` 写出均通过。

主要证据：

- `isaaclab-rebuild/outputs/host-validation/20260825T034900Z-policy-contract/`
- `isaaclab-rebuild/outputs/skill-behavior/20260825T042000Z/`
- `isaaclab-rebuild/outputs/macro-validation/match-contract.json`
- `isaaclab-rebuild/outputs/macro-validation/match-selfplay-snapshot2.json`
- `isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play/2026-08-27_20-17-40_resume_probe_501/`
- `isaaclab-rebuild/docs/MIGRATION.md`

### 未完成或不能过度解读的部分

- frozen shooting smoke 已有 8/8 launch，但成功率为 0/8；射门行为质量仍未通过。
- 历史 opponent pool 的持久化、采样和长期行为仍需验证；当前只验证了单个 detached
  snapshot 在默认 iteration 500 cadence 的替换。
- match-specific goal、possession、pass、collision 和 accidental termination 奖励尚未与 Isaac Gym 精确 parity。
- world-model collector/MPC 的 live end-to-end 接入尚未完成。
- command curriculum 尚未完全迁移。
- URDF 仍是当前运行资产来源，版本化 USD asset package 尚未完成。
- Isaac Gym Preview 4 的 Torch 1.10/cu113 runtime 在 RTX 4090 上触发 NVRTC `sm_89` 不兼容，因此不能直接用该机器完成新的 Gym-vs-Lab behavior 结论。
- 最近的可视化尝试可以初始化 Isaac Sim、RTX 4090、四机器人 scene 和所有 manager，但 viewport RGB frame 没有成功写入有效 MP4；视频 capture seam 仍需单独修复，不能把该次尝试当作视频验收通过。

## 版本与运行约束

生产迁移目标固定为：

- Isaac Sim 5.1.0；
- Isaac Lab v2.3.2；
- PhysX；
- CUDA GPU。

当前可用的本机运行解释器是：

`/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/isaaclab-rebuild/.venv/bin/python`

该独立环境报告 `isaacsim==5.1.0.0`、IsaacLab pip `2.3.2.post1`、
`torch==2.7.0+cu126` 和 `rsl_rl==3.0.1`。

当前经验文件默认使用 `isaacsim.exp.base.python.kit`，原因是某些 Isaac Sim 5.1 pip runtime 的 headless experience 缺少 viewport extension。AS2 URDF importer 可能只有 2.4.30，迁移 package 提供 runtime-only compatibility shim；不要修改外部 Isaac Lab checkout。

## 下一步优先级

1. 修复并验证 Isaac Lab viewport/RGB video capture，保存一段四机器人 match MP4，并进行 standing pose/self-collision 可视检查。
2. 对 frozen shooting 做 launch timing、post-strike decay、success distance/speed 和 reset distribution parity。
3. 验证 match-specific reward parity，并实现历史 opponent pool 的持久化和采样。
4. 将 `RecorderManager` transitions 接入现有 world-model dataset writer，再接入 MPC runtime/controller。
5. 将 AS2、足球、场地和球门转换为版本化 USD，保存 asset identity/property manifest。
6. 通过 clean Isaac Lab v2.3.2 release environment 后，逐步废弃 Isaac Gym training entry points。

## 工作规则

- 在 parity gate 完成前，不比较 Isaac Gym 与 Isaac Lab 的 learning curves。
- 不让训练脚本、world model 或 MPC 直接读取 simulator-specific tensor；访问必须经过 manager term 或 adapter seam。
- 保持 checkpoint observation/action/history contract 的 shape、顺序、缩放和 reset 语义。
- `isaaclab-rebuild/` 当前被根目录 `.gitignore` 忽略；继续开发前应取消整目录 ignore，只忽略生成的 outputs、缓存和大型数据。
- Isaac Lab 3.0/Newton 不属于当前生产迁移目标，应另开实验分支，不能与 PhysX parity 工作混合。
