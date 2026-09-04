# DWBC Isaac Lab Migration Design

## Goal

将 `Deep-Whole-Body-Control` 中的自定义 `widowGo1` 全身控制任务迁移为可在 Isaac Lab 训练、可复现实验、可支撑后续研究的独立工程；保留项目的自定义双奖励、双价值及相关训练算法，并以定量对齐证明迁移未改变任务语义。

## Scope and Non-goals

本设计只覆盖从旧 Isaac Gym / Legged Gym 工程到新 Isaac Lab 的迁移基线。

- 保留 WidowGo1 的本体、机械臂、目标物、地形、随机化、控制、观测、奖励和终止语义。
- 保留自定义 RSL-RL 算法：腿部与机械臂的独立奖励路径、双价值头、混合优势/回报逻辑、历史与特权编码器正则、可选力矩监督。
- 新仓库为 `/home/xxs/research/dwbc-isaaclab`；旧仓库仅作为基线来源，不在本迁移中修改。
- 首个可运行版本采用 Isaac Lab `DirectRLEnv`，不在首轮改造成 manager-based 任务。
- 不把“加载旧 checkpoint”作为首个迁移目标。只有数据契约和网络结构验证一致后，才作为额外实验尝试。
- 不承诺不同 PhysX 版本的逐时间步长轨迹或训练曲线完全相同；验收采用可解释的逐层数值容差与多种子统计比较。

## Platform Boundary

运行栈固定为：Isaac Lab 负责任务和训练接口，Isaac Sim 负责 USD 资产、运行时和可视化，PhysX 负责刚体、关节、碰撞和接触动力学。训练算法保持为项目自有包，通过适配层连接 Isaac Lab 环境；不得直接修改 Isaac Lab 或上游 RSL-RL 的源代码。

新仓库初期需固定 Isaac Lab 版本、Isaac Sim 版本、Python 版本和 RSL-RL 兼容版本，并将精确版本写入安装文档和锁定文件。旧项目的 Conda 环境 `dwbc` 仅用于导出基线轨迹和运行旧版对照，不作为新 Isaac Lab 环境的依赖约束。

## Repository Layout

```text
dwbc-isaaclab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── assets/widow_go1/
│   ├── source_urdf/
│   ├── usd/
│   └── import_config/
├── source/
│   ├── dwbc_isaaclab/tasks/widow_go1/
│   │   ├── widow_go1_env.py
│   │   ├── widow_go1_env_cfg.py
│   │   ├── observations.py
│   │   ├── rewards.py
│   │   ├── resets.py
│   │   └── agents/
│   └── dwbc_rsl_rl/
│       ├── algorithms/
│       ├── modules/
│       ├── runners/
│       └── storage/
├── scripts/
├── tools/
├── tests/
├── configs/
└── docs/
```

`dwbc_isaaclab` 只承担场景、动力学、任务逻辑和 Isaac Lab 接口；`dwbc_rsl_rl` 只承担自定义学习算法。两者通过一个明确的过渡数据契约交互，避免算法代码依赖 Isaac Lab 的内部实现。

## Environment API Mapping

旧环境到 DirectRLEnv 的映射如下：

| 旧 Legged Gym 职责 | Isaac Lab DirectRLEnv 职责 |
| --- | --- |
| `create_sim`、actor 创建、地形创建 | `_setup_scene` |
| `step` 中动作预处理 | `_pre_physics_step` |
| 施加 PD/力矩控制 | `_apply_action` |
| 重置环境和采样随机化 | `_reset_idx` |
| 计算观测 | `_get_observations` |
| 计算奖励 | `_get_rewards` |
| 超时、跌倒和任务失败判定 | `_get_dones` |

所有刚体、关节和执行器必须按名称解析，不能依据旧 Isaac Gym 的索引位置。转换层应显式保存“旧动作/观测顺序 → Isaac Lab 关节名 → Isaac Lab 索引”的双向表，并在启动时校验无缺失、无重复。

## Frozen Data Contract

动作维度固定为 18，顺序固定如下：

```text
FR hip/thigh/calf, FL hip/thigh/calf, RR hip/thigh/calf, RL hip/thigh/calf,
widow_waist, widow_shoulder, widow_elbow, widow_forearm_roll,
widow_wrist_angle, widow_wrist_rotate
```

夹爪两个自由度保留在 20 维关节状态中，但不是策略动作。该规则必须由测试强制执行。

本体观测维度固定为 76，按顺序由下列字段组成：投影重力 2、基座角速度 3、20 个关节位置、20 个关节速度、前一动作 18、4 个足端接触、命令 3、末端目标位置 3、末端姿态差 3。

历史长度固定为 10，特权观测固定为 24 维。因此策略/评论家使用的完整旧布局固定为 `76 × 11 + 24 = 860`。Isaac Lab 环境可返回 policy/critic 字典，但适配层必须能无损重建此布局，以供旧算法结构和对齐工具使用。

四元数和坐标变换必须在边界显式标注约定。默认采用 Isaac Lab 的 `xyzw` 表示；旧代码与新接口相接处必须执行单元测试，禁止依赖隐式约定。

## Reward and Transition Contract

环境返回的总奖励用于标准环境接口；同时每次 transition 必须保留以下命名字段：`leg_reward`、`arm_reward`、每项奖励分量、`torque_target`（若启用）、`terminated`、`time_outs` 与结构化终止原因。

适配层输入为 Isaac Lab 观测、动作和 extras，输出为自定义 PPO 所需的过渡记录。自定义 PPO 的两条奖励、双价值、优势/回报混合、特权正则与力矩监督的计算公式、权重、调度和日志名称均从旧实现逐项迁移，不得简化为单奖励标准 PPO。

## Asset and Physics Parity

原始 URDF 与导入配置存档，以保证 USD 可追溯。USD 导入后必须审计并测试以下属性：刚体/关节名称、20 个关节的限位、质量、惯量、碰撞体、关节驱动类型、刚度、阻尼、最大力矩、接触和摩擦相关配置。

地形、目标物、重力、仿真步长、控制抽样频率、求解器设置和随机化范围在配置中显式定义。任何旧版/新版单位、驱动含义或坐标轴差异都由对齐报告记录，不允许仅凭视觉效果判定正确。

## Alignment Protocol

迁移按以下门槛顺序推进，失败时回到最早失败层定位，而不继续训练：

| Gate | 输入 | 比较输出 | 成功条件 |
| --- | --- | --- | --- |
| A 资产契约 | 导入后的 USD | 名称、数量、限位、质量、惯量、驱动和碰撞 | 离散项完全一致；连续参数在预先声明的换算容差内 |
| B 重置契约 | 固定随机种子 | 初态、命令、目标、随机化样本 | 同一归一化坐标系内逐字段可比 |
| C 动力学轨迹 | 固定初态、固定动作序列 | 关节、基座、末端、接触和目标轨迹 | 短时误差受控，长时趋势和稳定性一致 |
| D 任务逻辑 | B/C 的录制状态 | 观测、奖励分量、总奖励、终止原因 | 维度和语义一致，数值满足每项声明容差 |
| E 算法过渡 | 固定录制 rollout | returns、advantages、各损失、一次参数更新 | 同一输入下自定义算法数值一致 |
| F 训练统计 | 至少三个固定种子 | 回报、成功率、课程进度、稳定性 | 中位数、方差及达到目标阈值的迭代数落入旧基线统计区间 |

容差不能在验收失败后临时放宽。每项连续量的绝对/相对容差、采样长度、种子和运行命令必须在 `configs/alignment/` 中预先登记。对 PhysX 版本导致的长期混沌发散，只允许以短时状态比对和长时统计指标判定。

## Execution Stages

1. 冻结旧版基线：记录 Git 提交、依赖、资产哈希、配置、种子、checkpoint 与固定 trace。
2. 初始化新工程：固定 Isaac Lab 安装方案，建立可安装包、环境检查和无仿真的契约测试。
3. 导入并审计资产：生成 USD、完成 Gate A。
4. 实现 DirectRLEnv：依次实现 reset、控制、观测、奖励、终止和随机化，完成 Gate B–D。
5. 迁移自定义 RSL-RL：先以录制 rollout 验证，再接入环境，完成 Gate E。
6. 基准与训练：在 3070 Ti 上测量显存和吞吐，从小并行数逐级扩大；执行三种子 Gate F。
7. 研究扩展：只从通过 Gate F 的 `main` 创建实验分支，任何行为变化均附带对比实验。

## Version-control Policy

`main` 必须始终可安装并通过最小 smoke test。`baseline-freeze` 固定旧版对照元数据；`env-parity`、`algo-parity`、`train-parity` 分别对应上述阶段。每个科研想法从通过验收的 main 创建独立分支，并附配置、随机种子、代码版本及比较结果。

## Hardware Constraint

目标机器为 RTX 3070 Ti（8 GB）。Isaac Sim/Lab 的运行时显存开销可能高于旧 Isaac Gym，不能将旧版 512 环境数视作保证。每次正式训练前先执行容量和吞吐基准；环境数量、渲染开关、地形复杂度均必须配置化。

## Acceptance Criteria

首个迁移完成版本须满足：可在固定 Isaac Lab 环境启动无头 smoke test；可使用项目自定义算法训练；全部 Gate A–E 通过；Gate F 具有至少三种子的完整报告；文档能使新研究者复现环境、基线与验收过程。
