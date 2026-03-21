# MCPC 框架

**多级共相控制（Multi-stage Co-Phasing Control）** —— 面向分布式航天器阵列高精度协同任务的工业级航天器动力学与控制仿真框架。MCPC 支持从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，为空间分布式合成孔径干涉成像（如“觅音计划”）等任务提供数字底座。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

<div align="center">
  <p><a href="./README.en.md">English</a> | 中文</p>
</div>

---

## 🎯 项目愿景与多级控制目标

MCPC 采用渐进式模型演进策略，按工程保真度逐级逼近真实世界：

| 层级 | 工程定位 | 动力学模型 | 测量与执行 | 状态 |
|------|----------|------------|------------|------|
| **L1** | 基准定标 | 宏观轨道质点，主导摄动 | 理想推力，地基测控，绝对遥测 | ✅ **已完成**（日地 L2 点 Halo 轨道） |
| **L2** | 协同效能 | 多质点相对运动，姿态锁定假设 | 理想星间链路，阵列推力分配 | 🔄 进行中 |
| **L3** | 原理验证 | 平台-载荷双层多体，简化运动学 | 光链路初建，多级机构协同 | 📋 规划中 |
| **L4** | 样机鉴定 | 全 6-DOF 姿轨耦合，硬件非线性 | 工程噪声谱，执行器死区/延迟 | 📋 规划中 |
| **L5** | 数字孪生 | 刚柔液多体拓扑，时变极端环境 | 全息数据融合，在轨辨识 | 📋 规划中 |

每一层级均有严格的动力学、测量、执行与控制假设，您可根据研发阶段选择适合的保真度。

---

## ✨ 核心架构特性

- **正交解耦**  
  按力学本质将任务分为**二体类**（中心引力为主）与**三体类**（多体引力显著），并按工程保真度分为 **L1~L5** 两个维度正交设计，最大化代码复用，同时保持模型粒度精准。

- **强坐标系契约**  
  所有跨模块数据（状态、指令、力）均携带 `CoordinateFrame` 枚举标签，并在接口处强制校验（如 `J2000_ECI`、`SUN_EARTH_ROTATING`、`LVLH`）。不一致的坐标系会立即抛出异常，从根本上杜绝参考系混淆。

- **物理域与信息域解耦**  
  - **物理域**（`core/physics`）模拟客观宇宙：航天器仅受力积分，不直接读取控制输出。  
  - **信息域**（`core/gnc`）模拟星载计算机：独立进行导航滤波与控制律计算，通过带契约的指令影响物理域。

- **真实物理量纲的最优控制**  
  LQR 控制器直接使用 SI 单位制（如日地系统角速度 ~2×10⁻⁷ rad/s），避免无量纲化带来的数值问题，实现平滑收敛。

- **高性能数据管道**  
  `HDF5Logger` 采用内存缓冲与压缩，支持增量写入，即使在长达数月的仿真中也避免内存溢出和 I/O 瓶颈。

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/mcpc.git
cd mcpc
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例仿真（日地 L2 点 Halo 轨道，L1 级）

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 1 --time_step 60
```

或使用 YAML 配置文件（例如 `config/halo_example.yaml`）：

```yaml
mission_name: "Halo L1 Test"
simulation_days: 1
time_step: 60.0
Az: 0.05
```

```bash
python run.py --scene sun_earth_l2 --level 1 --config config/halo_example.yaml
```

### 4. 查看结果

输出文件位于 `data/` 目录：

- **`*.h5`** – HDF5 文件，包含历元、标称/真实/导航状态、跟踪误差、控制力、累计 ΔV。
- **`fuel_bill_*.csv`** – 燃料账单（总 ΔV、平均每天 ΔV、最终误差等）。
- **`*_trajectory.png`**、**`*_errors.png`**、**`*_control.png`** – 可视化图表（若 `enable_visualization` 为 `True`）。

也可生成完整的 HTML 报告：

```bash
python visualize.py data/simulation.h5 --report
```

---

## 📂 目录结构

```
mission_sim/
├── core/                     # 核心领域模型
│   ├── dynamics/             # 运动方程（二体/三体）
│   ├── physics/              # 物理域（环境、航天器、力模型）
│   ├── gnc/                  # 信息域（GNC、地面站、外推器）
│   ├── trajectory/           # 星历与轨道生成器
│   └── types.py              # 全局类型（CoordinateFrame, Telecommand）
├── simulation/               # 仿真控制器（按场景分类）
│   ├── base.py               # 抽象基类（模板方法）
│   ├── threebody/            # 三体场景（日地 L2 等）
│   └── twobody/              # 二体场景（LEO、GEO 等）
├── utils/                    # 基础设施
│   ├── logger.py             # HDF5Logger（缓冲、压缩）
│   ├── math_tools.py         # LQR、LVLH、轨道根数转换
│   ├── differential_correction.py
│   └── visualizer_*.py
├── tests/                    # 单元测试与集成测试
├── analysis/                 # 后处理脚本（蒙特卡洛、燃料分析）
├── config/                   # 示例 YAML 配置
├── run.py                    # 统一仿真入口
├── visualize.py              # 数据可视化工具
└── requirements.txt
```

---

## 🛠️ 扩展与定制

### 添加新场景（例如 LEO）

1. 创建文件 `simulation/twobody/leo.py`。
2. 继承 `BaseSimulation`（或未来的 `TwoBodyBaseSimulation`）。
3. 实现抽象方法：  
   - `_generate_nominal_orbit()` – 使用 `KeplerianGenerator` 或 `J2KeplerianGenerator`。  
   - `_initialize_physical_domain()` – 注册合适的力模型（`J2Gravity`、`AtmosphericDrag`）。  
   - `_initialize_information_domain()` – 创建 `GroundStation` 和 `GNC_Subsystem`。  
   - `_design_control_law()` – 计算反馈增益矩阵。
4. 在 `run.py` 的 `SCENE_MODULE_MAP` 中注册场景名称。

### 添加新力模型

1. 在 `core/physics/models/` 中创建新类，实现 `IForceModel` 接口。
2. 实现 `compute_accel(state, epoch)` 方法。
3. 在仿真控制器的 `_initialize_physical_domain` 中通过 `self.environment.register_force()` 注册。

### 添加新层级

在对应场景子包中添加新的仿真类（如 `SunEarthL2L2Simulation`），继承场景基类并重写相关方法（如 L2 级添加相对运动）。

---

## 📊 应用挖掘示例

`analysis/` 目录提供深度分析脚本：

- **控制鲁棒性蒙特卡洛分析** – `control_robustness_analysis.py`  
  改变初始偏差、测量噪声、控制增益等，统计最终误差、ΔV 消耗、收敛时间等指标，输出统计图表与报告。

- **燃料开销分析** – `fuel_analysis.py`  
  扫描不同轨道类型、控制增益、盲区时长，评估燃料消耗。

示例运行：

```bash
cd analysis
python control_robustness_analysis.py
```

---

## 🤝 贡献指南

欢迎通过 Issue 和 Pull Request 参与贡献。请确保代码遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范，并为新功能添加单元测试。

1. Fork 本仓库。
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)。
3. 提交更改 (`git commit -m 'Add some amazing feature'`)。
4. 推送到分支 (`git push origin feature/amazing-feature`)。
5. 打开 Pull Request。

---

## 📄 许可证

Apache License 2.0（详见 [LICENSE](LICENSE) 文件）。

---

**MCPC – 从轨道到波长，逐级逼近真实。**