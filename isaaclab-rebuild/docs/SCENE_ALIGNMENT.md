# Isaac Gym / Isaac Lab 场景对齐

对齐基准：仓库当前 `scripts/train_high_level.py` 的默认 8 m × 5 m、2v2、有围墙比赛，
以及 `quadruped/envs/base/legged_robot_two.py` 实际创建的资产。Isaac Gym 使用 URDF/mesh
与程序化形状；Isaac Lab 在运行时生成 USD，不存在两套同名 USD 文件可直接比较。

## 资产与接触

| 对象 | Lab 对齐方式 |
| --- | --- |
| 并行原点 | 禁用单块地形的共享 reset 原点，使用与场景克隆一致的间距网格；避免所有 match 的机器人/球重置在世界零点 |
| 4 台 AS2 | 引用同一份 `resources/robots/as2/urdf/as2.urdf`；学习队保留 URDF 材质，对手全部 link 为 `(0.85, 0.10, 0.10)` |
| 球场 | 同一张 `field.png`，按球场 URDF 的 `8.5333336 × 5.533679` 外缘铺贴；可踢区域为 8 × 5 m |
| 草坪碰撞 | 恢复 Gym 的 0.01 m 薄盒，中心 z=-0.003 m、顶面 z=0.002 m；下方仍有 z=0 地面 |
| 球门 | 读取原 `goalpost_dual.stl`，应用 URDF 中相同的非均匀缩放、旋转和位移；碰撞仍为 URDF 定义的 6 个 box，不将球网凸化成障碍 |
| 围墙 | 6 段，厚 0.12 m、高 0.50 m、距场线 0.05 m；两端保留 2 m 球门开口；颜色 `(0.92, 0.92, 0.88)` |
| 球 | 黄色，半径 0.0889 m；创建时每 env 一次采样 `0.318 × U(0.5, 0.8)` kg，同时缩放球惯量 |
| 材料 | Preview 4 的 average 组合方式；地面/门柱摩擦 1、回弹 0，墙摩擦 0.35、回弹 0.85，球摩擦 1、回弹 0.85 |
| 接触 | robot、ball、墙、门柱和草坪显式 contact offset 0.01 m、rest offset 0 |
| 重力 | Gym 实际运行时 `_randomize_gravity()` 使用的 `(0, 0, -9.8)` |
| 球阻力 | 世界 XY 二次阻力 `F=-c*v*abs(v)`，`c ~ U(0.1, 0.8)`，每 15 s 全局重采样；每 20 ms 更新、每 5 ms 施加 |
| 场外判定 | 有墙比赛不因穿过场线立即终止；进球判定继续独立生效 |

球阻力使用零维 ActionTerm，策略仍输出四机器人合计 16 维动作。
这些物理覆盖仅作用于 match，单机器人 velocity/dribble 的固定参数 parity 配置保留。

## 渲染

录像入口启用 Fabric，让 GPU 中的实际机器人位姿同步给 RTX。相机初始 offset 在
场景创建前写入；只在初始化后修改 USD camera pose 会导致 Fabric 下相机不同步。
这修复了旧入口画面停在初始位置、四台机器人看似重叠的问题。

## 验证

```bash
isaaclab-rebuild/.venv/bin/python isaaclab-rebuild/scripts/validate_scene_alignment.py \
  --headless --num_envs 2 \
  --output isaaclab-rebuild/outputs/validation/scene-alignment.json
```

验证直接读取实际 USD/PhysX 状态，与 Gym URDF 和独立加载的墙体布局函数比较，
并执行有限 observation/reward、撞墙回弹、球门通行、二次阻力和重采样检查。
以生成报告中的 `passed: true` 为成功依据。

本次对齐资产、场景接触和球动力学，不回滚 Lab 已有的奖励重平衡、训练课程、reset
分布课程或混合动作 PPO。两代引擎的接触求解、光照与渲染不同，不保证逐帧像素或
逐时刻动力学轨迹完全相等；此验证也不能替代低层技能成功率和长训练验收。

## 本次本机验证结果

- 双 env 实际 USD/PhysX 检查通过：6 个球门碰撞体、6 面墙、草坪尺寸/高度、
  独立场景原点、球队材质、球质量范围与惯量。
- 以 2 m/s 撞向侧墙，两个 env 的回弹速度均约 -1.70 m/s。
- 球从 x=3.65 m 向球门运动，两个 env 均能穿过门口到 x≈4.221 m。
- 空中初速 1 m/s 的球经过 0.1 s 二次阻力后分别为 0.833/0.803 m/s；
  episode reset 不重采样、15 s 周期重采样均通过。
- 现有 checkpoint 的单 env 视频通过；6 帧之间存在实际画面变化。
- 2 个 match、每次 4 个 rollout step、1 次 PPO iteration 完成，16 个 agent timestep，
  无跌倒/出界终止。该短测试仅验证训练管线，不代表训练质量收敛。

报告：`outputs/validation/scene-alignment.json`。短训练输出位于
`outputs/validation/scene-alignment-train-smoke/`，未写入正式训练 run 目录。
最终截图：`outputs/screenshots/aligned-single-env.png`。
