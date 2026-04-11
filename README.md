# 🌌 MCPC: Multi-stage Co-Phasing Control Simulation Framework

# 多级共相控制仿真框架

**多级共相控制（Multi-stage Co-Phasing Control）** —— 面向分布式航天器阵列高精度协同任务的工业级航天器动力学与控制全数字仿真框架。

MCPC 支持从轨道级（公里级）到波长级（纳米级）的多级嵌套控制，为空间分布式合成孔径干涉成像（如“觅音计划”）、引力波探测等前沿空间科学任务提供全生命周期的数字底座。

本框架严格遵循航天系统工程范式，采用 **“前台三域 (Spacetime, Physics, Cyber) + 后台两域”** 的解耦架构，实现了物理世界客观规律与赛博空间主观智能的物理级断开与信息级隔离。


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


-----
<div align="center">
<p><a href="./README.en.md">English</a> | 中文</p>
</div>

-----

## 🎯 核心使命 (Mission Statement)

从 L1 到 L5 的全系统级仿真与数字孪生：

  * **[L1] 基石阶段 (Foundation)**：构建高逼真度的单星动力学环境（光压、高阶重力场、多体摄动），打通组件级闭环。
  * **[L2] 编队协同 (Formation)**：模拟多星相对导航与精准构型保持。
  * **[L3] 载荷闭环 (Payload)**：将平台大姿态机动与光学载荷的微米/纳米级位移控制联调。
  * **[L4] 柔性摄动 (Flexibility)**：引入大型太阳帆板及天线等附件的低频振动耦合模型。
  * **[L5] 效能评估 (Evaluation)**：基于蒙特卡洛打靶，综合输出编队在寿命末期的系统级效能报告。

-----

## 🏛️ 系统架构与目录地图 (Architecture & Directory Map)

为保障代码的严谨性，MCPC 将核心业务逻辑划分为独立的领域模型，所有跨域交互均通过 `ids.py` (Interface Definition Specification, 接口定义规范) 定义的契约完成：

```text
mcpc/
├── run.py                          # [入口] 仿真启动主脚本
├── visualize.py                    # [入口] 数据可视化与图表生成
│
├── mission_sim/                    # ================= 仿真主包 =================
│   ├── config/                     # YAML 任务配置文件目录
│   │
│   ├── core/                       # 🌟 【前台三域】：系统核心业务逻辑
│   │   ├── spacetime/              # 🌌 [时空域] 绝对的时空基底与星历标架
│   │   │   ├── ids.py              # 📜 跨域大法典：CoordinateFrame, Telecommand, FormationState
│   │   │   ├── ephemeris/          # 星历引擎：天体绝对位置真值
│   │   │   └── generators/         # 标称基准：Halo 等参考轨道生成器
│   │   │
│   │   ├── physics/                # 🪐 [物理域] 航天器受力与客观规律
│   │   │   ├── ids.py              # 📜 物理法典：物理常量、单位制、硬件故障枚举
│   │   │   ├── environment.py      # 环境工厂：引力场、光压场聚合计算
│   │   │   ├── spacecraft.py       # 系统级实体：质量、受力积分接口
│   │   │   └── components/         # 组件级模型 (含死区、摩擦、底噪，无控制算法)
│   │   │       ├── actuators/      # 执行组件 (推力器、飞轮)
│   │   │       ├── sensors/        # 敏感组件 (星敏、ISL天线)
│   │   │       └── mechanisms/     # 机构组件 (快摆镜、延迟线)
│   │   │
│   │   └── cyber/                  # 🧠 [赛博域] 承载感通存算的主观智能大脑
│   │       ├── ids.py              # 📜 赛博法典：控制状态机、通信协议帧
│   │       ├── models/             # 认知模型：CW方程、STM 等用于预测的数学模型
│   │       ├── networks/           # 通信协议：ISL 测距/测角数据流与路由延迟
│   │       └── platform_gnc/       # 控制大脑：导航滤波、姿轨控算法、模式切换
│   │
│   ├── simulation/                 # 🎬 【后台编排域】：上帝视角的装配车间与时钟
│   │   ├── base.py                 # 仿真主循环与事件驱动引擎
│   │   ├── threebody/              # 深空场景组装工厂 (日地 L2 等)
│   │   └── twobody/                # 近地场景组装工厂 (LEO/GEO 等)
│   │
│   ├── analysis/                   # ⚖️ 【后台分析域】：系统工程裁判
│   │   └── fuel_analysis.py        # 离线效能评估与燃料账单统计
│   │
│   └── utils/                      # 🛠️ 【基础设施层】
│       ├── math_tools.py           # 坐标系旋转、控制律求解等核心数学库
│       └── logger.py               # HDF5 高频增量日志数据总线
│
└── tests/                          # 🛡️ 契约驱动的单元测试套件
```

**架构原则：**
- **时空域**：提供绝对的时空基底，包括天体星历、坐标框架、标称轨道生成。自然天体之间的相互作用（如地球绕太阳的公转）通过星历隐含提供。
- **物理域**：负责计算航天器受到的所有力（引力、光压、大气阻力等），使用时空域提供的天体位置作为输入，不计算自然天体之间的相互作用。
- **赛博域**：实现航天器的感知、通信与控制智能。

-----

## ⚙️ 快速开始 (Quick Start)

### 1\. 环境准备

```bash
# 克隆仓库
git clone https://github.com/your-username/multistagecophasing.git
cd multistagecophasing

# 创建虚拟环境并激活
python -m venv venv
source venv/bin/activate  # Windows 用户使用 venv\Scripts\activate

# 安装依赖
pip install -r mission_sim/requirements.txt
```

### 2\. 运行首次仿真 (Run First Simulation)

例如，运行基于 CRTBP 模型的日地 L2 Halo 轨道维持仿真 (L1 级)：

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 30
```

### 3\. 数据可视化 (Visualization)

仿真执行完毕后，全生命周期的状态将被刻录为 HDF5 文件。执行以下命令生成图表：

```bash
python visualize.py --input data/logs/simulation_xxx.h5
```

-----

## 🛡️ 测试驱动 (Test-Driven Engineering)

MCPC 坚持严苛的验证标准。在提交代码前，请确保全流程测试通过：

```bash
pytest tests/
```

## 📋 开发规范 (Development Standards)

- **编码规范**: [CODING_STANDARDS.md](docs/development/CODING_STANDARDS.md)
  - 文件编码: UTF-8
  - 注释/文档: 优先中文，英文也可
  - 运行时输出 (stdout/logging/HDF5): **英文**
  - 可视化标签: **英文**

-----

<div align="center">
<b>MCPC – 从轨道到波长，用代码丈量深空。</b>
</div>
