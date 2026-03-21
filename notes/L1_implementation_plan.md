# MCPC L1 级开发实现计划

## 1. 项目目标

本计划旨在完成 MCPC 框架 L1 级的全部功能补全与应用挖掘，确保 L1 级作为通用数字底座，能够支持多种轨道类型（LEO/GEO、日地平动点、地月空间等），具备可插拔的力学模型、盲区外推能力、精确燃料统计及全面的性能分析工具。

**核心产出**：
- 完整的多态轨道生成器工厂（开普勒、J2、Halo 等）
- 可扩展的力学注册表（CRTBP、J2、SRP、大气阻力）
- 盲区外推器集成（简单线性、二体）
- 燃料账单自动生成与分析工具
- 全套单元测试与性能分析脚本

## 2. 任务分解与优先级

| ID | 任务名称 | 优先级 | 依赖 | 验收标准 | 涉及文件 |
|----|----------|--------|------|----------|----------|
| **T1** | 重构 trajectory 模块（拆分子包） | **高** | 无 | 原有功能不受影响；工厂函数正常工作；导入路径全部修正；单元测试通过 | 新建 `generators/` 子包，移动 `KeplerianGenerator`, `J2KeplerianGenerator`, `HaloDifferentialCorrector`，删除旧文件 |
| **T2** | 修复 Halo 微分修正集成 | **高** | T1 | 生成轨道的周期闭合性优于 1e-6（无量纲）；雅可比常数标准差小于 1e-6 | `generators/halo.py`, `differential_correction.py` |
| **T3** | 集成星上外推器到 GNC | **高** | 无 | 盲区时导航状态随时间变化（非冻结）；可通过配置启用/禁用；外推器可插拔 | `gnc_subsystem.py`, `propagator.py`, `main_L1_runner.py` |
| **T4** | 支持多轨道类型选择和力学模型自动注册 | **高** | T1, T3 | 配置 `orbit_type` 可生成对应星历并自动注册力模型；无硬编码依赖 | `main_L1_runner.py`, `generators/__init__.py`, `environment.py` |
| **T5** | 注册太阳光压模型（SRP） | **中** | T4 | 启用 SRP 后加速度包含光压分量；可通过日志验证 | `main_L1_runner.py`, `srp.py` |
| **T6** | 实现大气阻力模型 | **低** | T4 | LEO 仿真中阻力加速度随高度变化明显 | `models/atmospheric_drag.py`, `main_L1_runner.py` |
| **T7** | 自动生成燃料账单文件 | **中** | 无 | 每次仿真后 `data/` 下生成 CSV/JSON 文件，内容完整 | `main_L1_runner.py` |
| **T8** | 补充单元测试 | **中** | T1-T7 | 所有测试通过，代码覆盖率不低于 80% | `tests/test_trajectory_generators.py`, `tests/test_physics_models.py`, `tests/test_gnc_propagator.py` 等 |
| **T9** | Halo 轨道生成能力深度分析 | **中** | T2, T4 | 生成报告，包含不同振幅轨道的闭合误差、雅可比常数变化、周期等图表 | `analysis/halo_orbit_analysis.py` |
| **T10** | 控制鲁棒性蒙特卡洛分析 | **中** | T3-T5 | 脚本可正常运行，输出统计图表和报告 | `analysis/control_robustness_analysis.py` |
| **T11** | 燃料开销分析工具可用性验证 | **中** | T4, T7 | 脚本可正确读取 HDF5 数据并生成燃料账单对比图 | `analysis/fuel_analysis.py` |
| **T12** | 测控弧段与采样率影响分析 | **中** | T3, T4 | 生成测控间隔与导航精度、ΔV 的关系曲线 | `analysis/visibility_analysis.py` |
| **T13** | 积分步长与数值稳定性分析 | **低** | T4 | 生成不同步长下的能量误差、控制收敛情况图表 | `analysis/integration_analysis.py` |
| **T14** | 控制参数敏感性分析 | **低** | T4 | 生成 LQR 权重与性能的响应面图和敏感度报告 | `analysis/gain_sensitivity.py` |
| **T15** | 生成 L1 级综合评估报告 | **低** | T9-T14 | 整合所有分析结果，形成 PDF/Markdown 报告 | `analysis/L1_evaluation_report.py` 或手动文档 |

## 3. 实施顺序与阶段划分

### 阶段 1：基础重构（1-2 天）
- **T1**：重构 trajectory 模块，拆分子包，建立工厂函数。
- **T3**：集成外推器到 GNC（可并行）。

### 阶段 2：核心功能补全（3-5 天）
- **T2**：修复 Halo 微分修正。
- **T4**：多轨道类型支持和力学模型自动注册。
- **T5**：注册 SRP 模型。
- **T7**：燃料账单文件生成。

### 阶段 3：可选模型与测试（2-3 天）
- **T6**：大气阻力模型（若需要 LEO 场景）。
- **T8**：补充单元测试，覆盖新功能。

### 阶段 4：应用挖掘与分析（3-5 天）
- **T9**：Halo 轨道深度分析。
- **T10**：控制鲁棒性分析（验证并增强）。
- **T11**：燃料开销分析（验证并增强）。
- **T12**：测控弧段影响分析。
- **T13**：积分步长分析。
- **T14**：控制参数敏感性分析。

### 阶段 5：报告与文档（1 天）
- **T15**：生成综合评估报告。
- 更新 UML 图（`L1_activities.puml`, `L1_architecture.puml`, `L1_classes.puml`），与代码同步。

## 4. 目录结构规划
```text
mission_sim/
├── core/
│ ├── trajectory/
│ │ ├── init.py
│ │ ├── ephemeris.py
│ │ └── generators/ # 新增子包
│ │ ├── init.py
│ │ ├── base.py # BaseTrajectoryGenerator
│ │ ├── keplerian.py # KeplerianGenerator
│ │ ├── j2_keplerian.py # J2KeplerianGenerator
│ │ └── halo.py # HaloDifferentialCorrector
│ ├── physics/
│ │ ├── models/
│ │ │ ├── gravity_crtbp.py
│ │ │ ├── j2_gravity.py
│ │ │ ├── srp.py # 已实现
│ │ │ └── atmospheric_drag.py # 待实现
│ │ ├── environment.py
│ │ └── spacecraft.py
│ └── gnc/
│ ├── gnc_subsystem.py # 增加外推器集成
│ ├── propagator.py # 已实现
│ └── ground_station.py
├── utils/
│ ├── differential_correction.py
│ ├── logger.py
│ └── visualizer_*.py
├── analysis/ # 新增分析目录
│ ├── halo_orbit_analysis.py
│ ├── control_robustness_analysis.py
│ ├── fuel_analysis.py
│ ├── visibility_analysis.py
│ ├── integration_analysis.py
│ ├── gain_sensitivity.py
│ └── L1_evaluation_report.md
├── tests/
│ ├── test_trajectory_generators.py
│ ├── test_physics_models.py
│ └── test_gnc_propagator.py
├── main_L1_runner.py # 更新以支持多轨道类型
└── docs/
├── L1_activities.puml
├── L1_architecture.puml
├── L1_classes.puml
└── L1_implementation_plan.md # 本文件
```

## 5. 关键技术决策

- **轨道生成器工厂**：通过 `create_generator(orbit_type, **kwargs)` 返回相应生成器，解耦主程序与具体类。
- **力学模型注册**：`CelestialEnvironment` 保持为纯注册表，所有力模型（包括中心引力）均通过 `IForceModel` 接口注册，便于扩展。
- **外推器接口**：`Propagator` 抽象基类，便于后续添加高精度外推器。
- **燃料账单**：每次仿真后生成独立文件，便于批量对比分析。
- **测试覆盖**：使用 `pytest`，重点测试边界条件和异常处理。

## 6. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| Halo 微分修正不收敛 | T2 延迟 | 保留备用轨道生成（当前已实现），继续优化迭代算法或增加初值搜索 |
| 多轨道类型导致配置复杂 | T4 可读性下降 | 提供示例配置文件（`config/halo.yaml`, `config/leo_j2.yaml`）作为模板 |
| 外推器与 GNC 集成引入新 bug | T3 稳定性 | 增加单元测试，逐步集成，保持向后兼容 |
| 分析脚本依赖大量数据 | T9-T14 运行时间长 | 使用采样策略，支持断点续存，将结果缓存 |

## 7. 验收标准

- **功能完整性**：所有 T1-T8 任务完成，满足各自验收标准。
- **分析可用性**：T9-T15 脚本可运行，输出报告清晰合理。
- **文档同步**：UML 图与代码一致，开发计划与实施相符。
- **测试通过率**：单元测试 100% 通过，代码覆盖率 ≥80%。

## 8. 后续工作（L2 级衔接）

完成 L1 级后，L2 级将直接复用 L1 的绝对轨道基准和力学环境，重点开发：
- 编队相对动力学与控制
- 星间链路模拟
- 编队构型重构与燃料优化

本计划为 L2 开发提供了可靠的数据基础和已验证的底层模块。

---

**版本历史**：
- 2025-03-21：初始版本，基于 L1 级开发计划创建。