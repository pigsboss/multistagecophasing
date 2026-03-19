# Multi-stage Co-Phasing Control (MCPC)

本项目是一个专门为 **“多级共相位控制”** 任务打造的工业级航天器动力学与控制仿真基准框架。MCPC 旨在解决分布式航天器阵列在深空环境下的高精度协同问题，通过从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，支撑空间分布式合成孔径干涉成像任务（如“觅音计划”等）。

当前状态：**v1.1 (Level 1: 基于真实物理量纲的 LQR 绝对轨道维持已交付)**

## 🎯 项目愿景与多级控制目标

本项目采用敏捷开发模式，随 Level 演进逐步覆盖多级控制维度：
* **轨道级 (Orbital Stage - L1 已实现)**: 解决基于 CRTBP 模型的平动点绝对轨道维持。已实现抗科氏力耦合的 LQR 最优控制。
* **编队级 (Formation Stage - L2~L3 进行中)**: 解决多星相对运动学协同与厘米/毫米级编队重构，引入 LVLH 相对运动坐标系。
* **相位级 (Phasing Stage - L4~L5 愿景)**: 解决亚微米级光学延迟线补偿与刚体姿轨耦合波前控制。

## ✨ 核心架构特性

* **强坐标系契约 (Strict Coordinate Contract)**: 引入全局枚举 `CoordinateFrame` 和 `Telecommand` 数据类。任何跨模块的数据流（引力、推力、遥测、指令）均强制进行坐标系标签核对，从底层规避参考系混淆导致的计算灾难。
* **物理与信息域解耦 (Domain Decoupling)**: 
    * **物理域 (Physical Domain)**: 仿真真实的客观宇宙。`Spacecraft` 仅作为受力积分的容器，绝对不直接读取算法输出。
    * **信息域 (Information Domain)**: `GNC_Subsystem` 模拟星载 OBC 处理逻辑，独立进行导航滤波与控制律计算，通过发送带契约的指令影响物理域。
* **真实物理量纲的最优控制 (Real-dimension Optimal Control)**: 摒弃了传统的无量纲化 CRTBP 模型，在 LQR 最优控制中引入真实的日地角速度（$\omega \approx 1.99 \times 10^{-7} \text{ rad/s}$）与引力梯度，彻底解决了离散化仿真中的高频振荡问题，实现极低燃料消耗下的平滑收敛。
* **高性能数据流转**: 基于 `h5py` 实现增量式 HDF5 记录器 (`HDF5Logger`) 防 OOM，并将计算与可视化渲染完全分离。

## 📂 目录结构与模块映射

```text
mission_sim/
├── docs/                             # 架构演进文档 (PlantUML)
│   ├── architecture_global_v2.puml   # 全局总体静态架构 (引入 L2 规划)
│   └── architecture_L1_v2.puml       # Level 1 实现细节图 (引入 LQR 与 Telecommand)
├── core/                             # 核心领域层 (各 Level 演进主战场)
│   ├── types.py                      # 全局契约基石 (定义 CoordinateFrame, Telecommand)
│   ├── spacecraft.py                 # 物理域：航天器本体与变质量动力学模型
│   ├── environment.py                # 物理域：CRTBP 引力场与离心力/科氏力环境
│   ├── ground_station.py             # 信息域：地面测控网模拟与指令上行
│   └── gnc_subsystem.py              # 信息域：星载制导导航与控制大脑
├── utils/                            # 基础设施层 (跨级复用工具)
│   ├── math_tools.py                 # 数学工具箱 (LQR 代数 Riccati 方程求解器等)
│   ├── loggers.py                    # 高性能 HDF5 增量记录器
│   └── visualizer.py                 # 离线数据渲染与 3D 动画引擎
├── tests/                            # 测试驱动开发 (TDD) 模块
│   └── integration_L1_closed_loop.py # L1 级物理闭环与 LQR 最优控制收敛性集成测试
├── main.py                           # 仿真场景主入口
└── requirements.txt                  # 依赖清单