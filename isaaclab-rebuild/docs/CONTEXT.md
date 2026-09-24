# MPC-MARL Quadruped Soccer — Current Context

更新时间：2026-09-02

## 项目目的

本仓库实现四足机器人足球的分层多智能体强化学习系统。`isaaclab-rebuild/` 的生产
目标是在 IsaacLab 中 **from scratch 训练上层共享参数 MARL policy**。Isaac Gym
Preview 4 是任务语义和已有低层技能的参考实现，不再是要求全量逐项复制的主线。

系统包含：

- AS2/Go1 四足机器人的 walking、dribbling、shooting 低层技能；
- 共享参数的 high-level skill coordinator；
- 两队 self-play 与历史 opponent pool；
- joint world model 数据采集和训练；
- hybrid CEM MPC、terminal-value model 和 MPC teacher guidance。

核心领域对象是：机器人、足球、球场、球门、球队、技能、技能命令、比赛状态、世界模型状态和 MPC rollout。

## Isaac Gym 迁移选择原则

只迁移会影响上层 MARL 学习问题定义或训练稳定性的内容：

1. **MDP contract**：每个 agent 的 observation/action、时间尺度、reward、termination、
   reset distribution 和 per-agent credit assignment。
2. **Action consequence**：冻结 walk/dribble/shoot checkpoint 的输入/history、命令 frame、
   skill affordance/fallback、切换语义和真实物理执行结果。
3. **Multi-agent contract**：2v2 team frame、共享策略展开、对手接口、self-play pool、
   snapshot/resume 和每个 match 的独立采样。
4. **Training operability**：GPU vectorization、有限值检查、episode/reward/skill 统计、
   TensorBoard、checkpoint 和可复现实验配置。

不以其为当前阻塞项：Isaac Gym tensor/API 结构、低层技能从零训练的 reward/curriculum、
旧 runner 的完整兼容、逐时刻全轨迹相等、world-model/MPC、视频增强和 USD 资产打包。
物理 parity archive 保留为回归证据，但新增迁移需求必须先回答“它是否改变上层 MARL
的学习信号或状态转移”；答案为否时默认不迁移。

## 稳定领域约定

- 坐标：学习队攻击 `+x` 方向；对手队在 world frame 攻击 `-x`，输入到共享策略前旋转 `pi`。
- 物理步长：5 ms。
- 低层 policy/environment 步长：20 ms，即 4 个物理步。
- high-level coordinator 步长：10 个低层步，即 200 ms。
- AS2 legacy joint order：`FL/FR/RL/RR`，每条腿依次 hip、thigh、calf，共 12 个 actuated joints。
- 低层 legacy policy observation：walking 72D，带球技能 75D。
- 低层 legacy history：15 帧，分别为 1080D/1125D，首个外部 reset 使用零填充。
- match agent observation：34D；4 帧 coordinator history：136D。
- MARL action：每个机器人 4D `[skill_index, parameter_x, parameter_y, parameter_yaw]`；
  `skill_index` 为离散 `{0,1,2}`，后三项为连续 skill parameters。
- 四机器人 world-model canonical state 当前编码为 128D。
- invalid shoot request 必须安全降级为 walk，并暴露 `invalid_skill` 标记。
- coordinator 的 ball command 使用 canonical/world field frame；新 dribble/shoot
  checkpoint 使用 body frame，IsaacLab action adapter 必须根据 `wxyz` root yaw 在每个
  low-level tick 转换。无 `ball_xy_frame` metadata 的旧 bundle 保持 world-frame。

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
- match manager action 为 16D，即 4 个机器人各 4D hybrid skill action。
- manager observation 为 `(4, 34)`；self-play wrapper 对外提供 `(2, 34)` observation 和 `(2, 136)` history。
- coordinator macro smoke 能执行 10 个 low-level ticks，reward finite，done contract 正常。
- invalid shooting request 的 geometric fallback 会执行 walk 并标记 invalid。
- RecorderManager 能记录 reset plus 10 个 low-level ticks 的机器人、关节、球、技能和命令状态。
- canonical snapshot 能编码为现有四机器人 128D world-model state。
- frozen skill behavior matrix 的 walk/dribble/shoot/switch interface contract 已通过 10-tick gate。
- 四机器人 CUDA PPO 已从 `model_450.pt` 恢复并完成 51 个连续 update（450..500）；
  iteration 500 的 detached opponent snapshot 同步刷新和 `model_500.pt` 写出均通过。
- 2026-09-01 已从 Isaac Gym 同步 ball-skill frame adapter、随机/近球 reset、物理边界墙、
  attacker 0.15 m 滞回、support 0.08 m deadband，以及 goal/progress/spacing/collision/
  invalid/pass/approach/walk alignment/face-ball/face-goal/dribble control/shoot setup/
  shoot launch rewards、
  local role credit、rule-based opponent、场地贴图和球门 primitive。同步后尚未完成新的
  长训练，因此旧 iteration 500 checkpoint 只代表同步前 baseline。
- opponent pool 已接入 RSL checkpoint `infos` 持久化；compact actor-only state 和旧
  ActorCritic state 的 simulator-free round-trip 均通过，checkpoint 写入已确认。
- 2026-09-02 将 learner action 从三 logits + 三参数改为真正的 hybrid action：每个
  agent 4D tensor transport、Categorical skill 和三维 Gaussian parameters。RTX4090
  contract smoke 与一次 PPO optimizer update 已通过，日志位于
  `logs/rsl_rl/dribblebot_as2_match_self_play/2026-09-02_01-28-47_hybrid_action_smoke_20260902/`。
- 已增加 `train_self_play_eval.sh` 分段训练/eval 编排和 `play_self_play.py` 的
  `--metrics_output`/`--video_output` 接口；eval 在训练 chunk 完整退出后单独启动，避免
  两个 Isaac Sim/OmniClient 进程同时抢占 GPU 资源。视频模式使用独立 TiledCamera、USD
  同步和非渲染 reset；RTX4090 上已验证可生成有效 H.264 MP4。
- 新的 reward-rebalanced run（`reward_rebalanced_marl_v2_20260903`）从零开始，不恢复旧
  optimizer：goal/timeout、位置进展、time pressure 和“只惩罚拥挤”的 spacing 已启用；
  hybrid PPO 的参数 std 限制在 `[0.15, 0.6]`，entropy bonus 只作用于 categorical
  skill，并单独记录 skill/parameter entropy、
  requested/executed/fallback action 统计。
- 该 run 的指标表明固定难度仍会把策略推向高风险追球（摔倒/出界上升），因此新增
  `AS2MatchCurriculumCfg`：按约 500/2000 PPO iteration 分三阶段调整 near-ball 起点、
  场地范围、终止惩罚和 shoot/goal shaping；PLAY 配置关闭 curriculum，保持固定布局。
- `curriculum_marl_v3_20260904` 暴露了分段训练的时间基准问题：每次 eval 后重建环境会
  清零 `common_step_counter`，导致 5000 iterations 全部停在 phase 0。训练入口现根据
  `model_<iteration>.pt` 和 rollout/macro 长度恢复 `curriculum_step_offset`；从
  `model_499.pt` 恢复的 GPU gate 已记录 phase 1/global step 120000。修复后的正式 run 为
  `curriculum_marl_v4_20260905`。

主要证据：

- `isaaclab-rebuild/outputs/host-validation/20260825T034900Z-policy-contract/`
- `isaaclab-rebuild/outputs/skill-behavior/20260825T042000Z/`
- `isaaclab-rebuild/outputs/macro-validation/match-contract.json`
- `isaaclab-rebuild/outputs/macro-validation/match-selfplay-snapshot2.json`
- `isaaclab-rebuild/logs/rsl_rl/dribblebot_as2_match_self_play/2026-08-27_20-17-40_resume_probe_501/`
- `isaaclab-rebuild/docs/MIGRATION.md`

### 未完成或不能过度解读的部分

- frozen shooting smoke 已有 8/8 launch，但成功率为 0/8；射门行为质量仍未通过。
- opponent pool 已支持最多 8 个 detached snapshot、保留初始 anchor、每个 match/reset
  重采样、50% latest 概率和 checkpoint 持久化；长期行为仍需验证。
- 2026-09-01 最终 pool GPU restore gate 受到宿主机 inotify/OmniClient 资源耗尽导致的
  Isaac Sim native crash 干扰；payload 写入和 actor-only/legacy state 重建已分别验证。
- world-model collector/MPC 的 live end-to-end 接入尚未完成。
- velocity command curriculum 尚未迁移；match reset/reward curriculum 已接入。
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

1. 启动新的 from-scratch MARL 基线，检查 reward/termination/skill-selection 分布、PPO
   数值稳定性、吞吐量和早期 learning signal。
2. 在随机 match reset 分布中分别验证 walk、dribble、shoot 的可达性、成功率和动作后果；
   shooting 是上层动作之一，因此“可用”是训练 gate，但无需逐拍复制 Gym 轨迹。
3. 根据训练统计调整 reward credit、reset/curriculum 和 opponent 难度，并验证长期
   opponent pool 行为以及 GPU resume。
4. 固化一套 MARL 回归 gate：observation/action shape、frame、macro rate、有限值、全部
   termination、reward term 覆盖率和 checkpoint round-trip。
5. 完成以上主线后再接 world-model/MPC、版本化 USD 和视频增强。

## 工作规则

- 评估 IsaacLab from-scratch MARL 的绝对训练质量；不要求它复现 Isaac Gym 的相同
  learning curve。
- 不让训练脚本、world model 或 MPC 直接读取 simulator-specific tensor；访问必须经过 manager term 或 adapter seam。
- 保持冻结低层 checkpoint observation/action/history contract 的 shape、顺序、缩放和
  reset 语义，因为它是上层 action consequence 的一部分。
- 只有影响 MARL MDP、低层 action consequence、self-play 分布或训练可靠性的 Gym
  更新才进入 IsaacLab 主线；其余更新记录为 optional/deferred。
- `isaaclab-rebuild/` 当前被根目录 `.gitignore` 忽略；继续开发前应取消整目录 ignore，只忽略生成的 outputs、缓存和大型数据。
- Isaac Lab 3.0/Newton 不属于当前生产迁移目标，应另开实验分支，不能与 PhysX parity 工作混合。
