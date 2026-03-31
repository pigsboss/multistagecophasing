# MCPC 框架 L2 级（多星编队协同）架构设计与开发准则

**文档状态**：实施版 (Execution Draft)
**目标层级**：L2 (Formation Flying & Cooperative Control)
**核心主旨**：在“3+2 域”系统工程架构下，规范多星编队的相对状态计算、星间测量物理层、信息路由网络层与离散控制逻辑。

---

## 🏛️ 一、 核心架构定律 (The Iron Laws of Architecture)

在 L2 级开发中，任何代码的合入必须无条件遵守以下三条“铁律”。**代码审查 (Code Review) 将以此为最高标准。**

### 定律 1：物理真值与赛博认知的绝对屏障
* **物理域 (`physics/`)** 是盲目的客观世界：它只负责根据牛顿定律积分绝对状态，根据光学规律叠加测量噪声。**严禁在物理域中包含任何 `if-else` 的控制策略或数据包排队逻辑。**
* **赛博域 (`cyber/`)** 是戴着镣铐跳舞的大脑：它**绝对不被允许**直接读取 `environment.py` 或 `spacecraft.py` 里的绝对真值。它只能消费网络层发来的带噪 `ISLNetworkFrame`。

### 定律 2：跨域数据流必须契约化 (IDS Contractualization)
所有在 `spacetime`、`physics`、`cyber` 之间流转的数据，必须是 `ids.py` 中定义的 `@dataclass` 实例。
* **物理层输出**：必须是 `ISLPhysicalMeasurement`（仅含物理噪声与距离/角度标量）。
* **网络层输出**：必须是 `ISLNetworkFrame`（封装了物理测量，附加网络时间戳、路由 ID 与数据老化标志）。
* **控制层输出**：必须是 `Telecommand`（受制于 LVLH 标架，受制于 `duration_s` 持续时间）。

### 定律 3：时钟分离与冲量等效原则
物理环境的全局积分步长 (如 $dt = 1s$) 与推力器的最小脉宽 MIB (如 $dt = 10ms$) 存在跨尺度冲突。
* **准则**：严禁为了配合推力器而将全局积分步长缩短至毫秒级（会导致性能崩溃）。必须在物理执行器入口处实现 **冲量平滑器 (Impulse Smoother)**，将离散的高频控制指令等效为低频的平均推力。

---

## ⚙️ 二、 “前台三域”组件功能规范

### 1. 🌌 时空域 (Spacetime Domain)
* **坐标转换精度**：`math_tools.py` 中的 `absolute_to_lvlh` 与 `lvlh_to_absolute` 必须使用 `np.float64`。考虑到深空背景（$10^{11}$ 米）与编队基线（$10^2$ 米）的数量级差异，必须通过严密的运动学方程（考虑 $\vec{\omega} \times \Delta\vec{r}$）来处理速度矢量，**严禁简单地仅乘旋转矩阵**。
* **数据总线**：`FormationState` 是 L2 级唯一合法的多体状态容器。

### 2. 🪐 物理域 (Physics Domain)
* **环境场隔离**：`environment.py` 中的 `compute_accelerations` 必须支持多星状态矩阵的并行计算，禁止不同实例的状态发生内存越界。
* **传感组件 (`sensors/isl_antenna.py`)**：负责生成高斯白噪声、常数偏置 (Bias)。这是唯一的误差源头。
* **执行组件 (`actuators/thruster.py`)**：必须拦截并过滤掉不满足 **死区 (Deadband)** 和 **MIB** 的推力指令，输出阶梯状的物理量化推力。

### 3. 🧠 赛博控制域 (Cyber Domain)
* **通信网络 (`networks/isl_router.py`)**：引入队列 (Queue) 模拟数据传输。支持配置网络延迟 $T_{delay}$ 和丢包率 $P_{loss}$。
* **相对认知 (`models/relative_dynamics.py`)**：必须提供离散化状态转移矩阵 (Discrete STM)，供控制器在网络延迟存在的情况下进行**前向预测补偿**。
* **编队 GNC (`platform_gnc/formation_controller.py`)**：必须实现三段式状态机（编队生成、稳态保持、重构机动）。控制算法推荐采用 LQR 或 MPC。

---

## 🎬 三、 “后台两域”调度与效能评估

### 1. 编排域 (Simulation Domain) 时序流转规范
在 `simulation/base.py` 的每个 Tick 中，必须严格按以下时序进行，**不可乱序**：
1. **[物理]** 环境力积分，更新所有主从星绝对真值 (`t_current`)。
2. **[物理]** `ISL_Antenna` 根据真值生成物理测量。
3. **[赛博]** `ISL_Router` 打包测量数据，压入网络队列。
4. **[赛博]** 各从星 GNC 获取队列数据（可能有延迟），解算推力指令。
5. **[时钟]** 时钟步进至 `t_next = t_current + dt`。

### 2. 分析域 (Analysis Domain) 核心指标定义
* **冷却整定时间 (Settling Time)**：指令下发后，直至基线误差进入并稳定在标称观测死区（如 $\pm 10\mu m$）内的时间。
* **稳态均方根误差 (RMSE)**：在排除冷却期后的稳定观测期内，计算三轴相对漂移的 RMSE。
* **联合燃料账单 ($\Delta V$ Budget)**：累加各星推力器实际喷射物理冲量的等效速度增量。