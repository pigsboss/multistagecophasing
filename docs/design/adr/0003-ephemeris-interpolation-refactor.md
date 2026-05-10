# ADR-0003: Ephemeris 基类插值方法简化与子类多态设计

## 状态

已接受

## 背景

`Ephemeris` 基类原使用 `scipy.interpolate.CubicSpline` 作为默认插值策略，存在以下缺陷：

1. **物理不正确**：对 Kepler 轨道，在笛卡尔坐标下直接进行三次样条插值无法保持角动量、能量和轨道平面不变，导致虚构的轨道漂移。
2. **强依赖**：引入 `scipy` 作为核心模块的必要依赖，削弱了 MCPC 的最小依赖原则。
3. **缺乏灵活性**：所有子类都强制使用同一种插值方式，无法为不同轨道类型（理想 Kepler、数值积分、CRTBP 等）提供最优方案。

## 决策

### 1. 基类移除 scipy 依赖，提供简单线性插值

`Ephemeris` 基类不再调用 `CubicSpline`，改为内置基于 `numpy` 的线性插值作为默认实现。核心契约不变，但降低了外部依赖。

- `__init__` 中不再创建 scipy 插值器。
- `get_state` 默认实现：查找时间区间，线性插值 6 个状态分量。
- 边界外推：保持警告但允许执行。

**基础实现**已在 `mission_sim/core/spacetime/ephemeris/base.py` 中落地。

### 2. 子类多态实现物理精确插值

通过重写 `get_state`，子类可以提供最优插值策略，而无需修改调用方代码。

- **理想 Kepler 轨道**：`KeplerEphemeris` 存储轨道根数（`a, e, i, Ω, ω, M0` 等），`get_state` 直接求解开普勒方程并转换到笛卡尔坐标，完全消除插值误差。该实现位于 `mission_sim/core/spacetime/ephemeris/kepler_ephemeris.py`。
- **表格插值（保留 spline）**：若仍需要三次样条插值，可在 `SplineEphemeris` 子类中按需引入 `scipy`（未强制要求）。
- **特殊轨道**：如 CRTBP 周期轨道的辛插值或 Hamilton 保持插值，可创建对应子类实现。

### 3. 生成器适配

`KeplerianGenerator.generate()` 现在直接返回 `KeplerEphemeris` 实例，不再生成纯粹的表格再交给样条插值。这既准确又高效。

## 后果

- **正面**：
  - 核心模块依赖最小化（仅 numpy），便于分发和嵌入式部署。
  - 物理精度提升：Kepler 轨道不再受非物理插值影响。
  - 架构更加灵活，新轨道类型可通过多态无缝集成。
  - 与 ADR-0002 中星历‑姿态对称性架构相呼应，都通过重写查询方法实现多态。

- **负面**：
  - 线性插值在精度要求高的原型开发中可能略显粗糙，但可通过切换到 `SplineEphemeris` 解决。
  - 现有代码中若直接依赖 `Ephemeris` 的三次样条行为（如依赖 C² 连续性），可能需要显式切换到 `SplineEphemeris`。

- **注意事项**：
  - 确保所有现有测试仍然通过；必要时为 `SplineEphemeris` 编写新测试。
  - 文档中需明确说明各子类的精度特性及适用场景。
