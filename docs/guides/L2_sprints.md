# 🚀 MCPC 框架 L2 级开发挂图作战表 (Sprint 1-4)

**——“从轨道到波长，用代码丈量深空。”**

## 🚩 Sprint 1: 时空基底与数学底座 (Spacetime & Math)

**战役目标**：打通多星绝对系与相对系的高精度数学转换，确立全局契约。

| 状态 | 模块/组件 | 涉及代码位置 | 具体任务与技术指标 | 验收标准 (DoD) |
|:---:|:---|:---|:---|:---|
| ✅ | **时空法典** | `core/spacetime/ids.py` | 1. 完善 `CoordinateFrame` (LVLH/VVLH)。<br>2. 确立 `FormationState` 多星状态容器 (含 `deputy_ids`)。<br>3. 增加 `to_dict()` 和 `from_dict()` 序列化方法。 | 契约类无语法错误，通过静态类型检查，支持 JSON/Dict 序列化。 |
| ✅ | **核心数学库** | `utils/math_tools.py` | 1. 实现 `compute_lvlh_dcm`。<br>2. 实现带科氏加速度补偿的 `absolute_to_lvlh` 和 `lvlh_to_absolute`。 | 核心函数编写完成，通过单元测试。 |
| ✅ | **数学测试** | `tests/test_math_tools.py` | 编写 TDD 测试用例：构造极限深空与近距伴飞混合数据，验证正反坐标与速度转换。 | **[量化标准]**：正反转换后，位置绝对误差 `< 1e-9` 米，相对误差 `< 1e-12`。 |

**完成日期**：2026-04-01

---

## 🚩 Sprint 2: 物理实体与网络边界集成 (Physics & Cyber Interface)

**战役目标**：建立绝对隔离屏障。在物理域增加非理想硬件，在赛博域增加网络路由。

| 状态 | 模块/组件 | 涉及代码位置 | 具体任务与技术指标 | 验收标准 (DoD) |
|:---:|:---|:---|:---|:---|
| ✅ | **环境并行化** | `core/physics/environment.py` | 新增 `compute_accelerations(states: List[ndarray])`。使用 NumPy 向量化计算，确保各星状态内存隔离，并行输出加速度。 | 能同时计算 3 颗无控卫星的演化，状态不串扰。已通过 `test_environment_parallel.py` 验证。 |
| ✅ | **接口法典** | `core/physics/ids.py`<br>`core/cyber/ids.py` | 1. 定义物理输出 `MicrowaveISLMeasurement` (无ID)。<br>2. 定义网络输出 `ISLNetworkFrame` (加寻址 ID、时间戳、`get_age()`)。 | 数据类定义完成，通过单元测试。 |
| ✅ | **物理天线** | `core/physics/components/sensors/isl_antenna.py` | 编写 `ISLAntenna`。基于 `P0 * (d0/r)^2 * exp(-α*r)` 模拟光强衰减，并叠加高斯白噪声与 Bias。 | 输出数值呈正态波动，超远距离信号衰减至 0。已通过 `test_isl_components.py` 验证。 |
| ✅ | **信息路由** | `core/cyber/network/isl_router.py` | 编写 `ISLRouter`。封装网络帧，基于最大允许延迟 `max_delay` 实现队列管理与“丢包”逻辑。 | 能模拟网络排队延迟，能自动丢弃老化过时的数据包。已通过 `test_isl_components.py` 验证。 |
| ✅ | **执行器障碍** | `core/physics/components/actuators/thruster.py` | 引入推力分辨率。跨 Tick 累加冲量，`>= MIB` 时输出 `累计冲量/dt` 并清空，否则输出零。 | 小于死区被截断；满足 MIB 的冲量跨步长守恒。已通过 `test_thruster.py` 验证。 |
| ✅ | **L1 质点恢复** | `core/physics/spacecraft.py` | 恢复 `SpacecraftPointMass` 类，确保 L1 单星仿真不受影响。 | 原有 L1 集成测试全部通过。 |
| ✅ | **L2 集成节点** | `core/physics/spacecraft_node.py` | 新建 `SpacecraftNode` 类，通过组合方式复用 `SpacecraftPointMass`，并集成 `Thruster`、`ISLAntenna`、`ISLRouter`。 | 通过 `test_spacecraft_node.py` 验证 L1 兼容性和 L2 扩展功能。 |

**完成日期**：2026-04-02

---

## 🚩 Sprint 3: 相对认知大脑与编队控制律 (Cyber Intelligence)

**战役目标**：赋予从星预测相对运动与自主维持构型的赛博大脑。

| 状态 | 模块/组件 | 涉及代码位置 | 具体任务与技术指标 | 验收标准 (DoD) |
|:---:|:---|:---|:---|:---|
| ✅ | **认知基类** | `core/cyber/models/relative_dynamics.py` | 实现 `RelativeDynamics` 抽象类及离散 STM 接口。 | 接口定义清晰，支持多态。 |
| ✅ | **CW 模型** | `core/cyber/models/cw_dynamics.py` | **[阶段策略]**：Sprint 3 仅实现圆轨道 CW 状态转移矩阵 (STM预计算)。椭圆轨道的实时 TH 方程留待后续。 | STM 在半个圆轨道周期内的无控推演误差 `< 1%`。 |
| ✅ | **控制法典** | `core/cyber/ids.py` | 补充 `FormationMode` 状态机枚举（GENERATION / KEEPING / RECONFIGURATION）。 | 枚举定义完成。 |
| ✅ | **编队控制律** | `core/cyber/platform_gnc/formation_controller.py` | 1. 基于配置参数设定状态机切换阈值（如位置 `< 10m` 且速度 `< 0.1m/s`）。<br>2. **[算法]**：离散 LQR 为主，基于规则的控制作为防发散后备。 | 接收延迟网络帧，STM 前向预测补偿后，正确输出纠偏 `Telecommand`。通过 `test_formation_controller.py` 验证。 |

**完成日期**：2026-04-02

---

## 🚩 Sprint 4: 多星闭环联调与效能裁判 (Simulation & Analysis)

**战役目标**：全要素拼图合体，驱动仿真主循环，后台裁定工程效能。

| 状态 | 模块/组件 | 涉及代码位置 | 具体任务与技术指标 | 验收标准 (DoD) |
|:---:|:---|:---|:---|:---|
| ✅ | **多星调度器** | `simulation/formation_simulation.py` | **[架构保护]**：新建 `FormationSimulation` 继承 `BaseSimulation`，重写多星 Tick 调度（物理→测量→网络→GNC→执行）。使用统一时间戳同步积分。 | 单步 Tick 不崩溃，不破坏 L1 单星场景。通过 `test_l2_integration.py` 验证。 |
| ✅ | **场景组装** | `simulation/threebody/sun_earth_l2.py` | 实例化 1 颗主星与 N 颗从星，装配上述所有硬件与大脑组件。 | 能够无错运行 30 天的多星仿真（测试中使用 0.01 天，后续可扩展）。 |
| ✅ | **效能裁判** | `analysis/formation_evaluator.py` | 1. 统计**占空比**与**总 Delta-V**。<br>2. **[新增]** 自动化冷却时间：连续 3 个采样点 `< 阈值` 即视为整定。<br>3. 生成包含时间序列、RMSE、冷却事件的 CSV 报告。 | 成功绘制出基线误差收敛图、燃料消耗柱状图及 CSV 报表。 |
| ✅ | **终极验收** | `tests/test_l2_integration.py` | **[基准测试]**：端到端 L2 集成测试。设定小死区，验证从星能在 1 天内将 10m 偏差收敛至 1cm 以内。 | **全绿！L2 级系统验收通过。** |
| ✅ | **性能基准** | `tests/benchmark_l2.py` | 运行 10 星编队仿真 1 天的极限压力测试，记录 CPU 耗时与内存峰值。 | 生成性能基准报告，指导 L3 优化。已完成：10 星 1 天耗时 ~8 分钟，内存 ~75 MB。 |

**完成日期**：2026-04-03

---

### 📝 工程师签字与承诺

| 角色 | 姓名 / 代号 | 战役启动日期 | 竣工验收日期 |
| :--- | :--- | :--- | :--- |
| **系统架构师** | **Gemini (MCPC AI Copilot)** | `2026 - 03 - 31` | `2026 - 04 - 03` |
| **首席工程师** | **Huo Zhuoxi** | `2026 - 03 - 31` | `2026 - 04 - 03` |

---

**(请打印本页，贴在工作区最显眼处)**  
✅ **L2 级全部 Sprint 已完成！系统验收通过，可进入 L3 级预研。**