# MCPC 共享文献库（LIT 索引）

> 维护约定：检索（`tools/openalex_search.py`）→ 下载至 `pdfs/` → MinerU 自动解析入库 → 在本表登记 LIT 编号。
> **三方一切文献引用以 LIT 编号为准**；检索/精读统一入口：`mineru search "<词>"`、`mineru read doc:<id>/tier:standard/page:N`。
> 更新日期：2026-10-09

## 登记表（已入库）

| 编号 | 年份 | 文献 | 主题 / 用途 | 关联级别 | 来源 | 入库状态 |
|---|---|---|---|---|---|---|
| LIT-001 | 2019 | Monnier et al., *A Realistic Roadmap to Formation Flying Space Interferometry*（Astro2020 白皮书） | 编队干涉测量路线图：技术就绪度、关键使能技术、历史任务教训 | L1–L4 总体 | arXiv:1907.09583 | ✅ 已入库（doc:9b3fed1，standard 全篇 1-17） |
| LIT-002 | 2024 | *STARI: STarlight Acquisition and Reflection toward Interferometry* | 干涉测量技术验证星方案（含编队/光学闭环设计） | L2–L3 | arXiv:2408.03925 | ✅ 已入库（doc:46e896a，flash 1-10 + standard 全篇 1-13） |
| LIT-003 | 2026 | *Stellar Observation Scheduling Optimization for Distributed Space Interferometry Missions* | 分布式干涉测量任务的观测调度与队形优化 | L2 | arXiv:2607.10888 | ✅ 已入库（doc:ca82e45，flash 1-10 + standard 全篇 1-22） |
| LIT-004 | 2025 | *SILVIA: Ultra-precision formation flying demonstration for space-based interferometry*（PASJ） | 超精密编队飞行在轨演示（亚微米级相对控制） | L2–L3 | DOI:10.1093/pasj/psaf086 | ⏳ 网页有 Cloudflare 人机校验，自动下载受阻；可手动下载后放入 `pdfs/`（自动入库） |

## 候选清单（检索命中、暂未入库）

| 候选 | 年份 | 说明 | 链接 |
|---|---|---|---|
| NASA STARI Mission: A Formation-Flying Pathfinder for Space Interferometry | 2026 | 小卫星编队干涉路径验证任务（SmallSat 会议论文） | digitalcommons.usu.edu（Cloudflare，同上） |
| Space and lunar interferometry: emerging concepts and pathways | 2026 | 空间/月基干涉测量新概念综述 | arXiv:2607.07746 |
| Adaptive sliding mode formation control for space interferometer missions | 2026 | 干涉测量编队滑模控制（L2 控制律对比参考） | DOI:10.1016/j.conengprac.2026.107025（暂无 OA PDF） |
| Range Emulator: A Compact Paraxial Optical System to Emulate Long-Distance Space Interferometry | 2026 | 地面光程模拟器（L4 地试对照参考） | arXiv:2602.20538 |
| Space Optical Interferometry for High-Resolution Coherent Imaging（MIT 博士论文） | 2025 | 空间光学干涉测量系统综述 | dspace.mit.edu（云防护，需手动） |

## 明细

### LIT-001《A Realistic Roadmap to Formation Flying Space Interferometry》

- PDF：`pdfs/2019_Monnier_FormationFlyingSpaceInterferometry_arXiv1907.09583.pdf`
- 精读入口：`mineru read doc:9b3fed1/tier:standard/page:6`（Technology Drivers 章节）
- 与本项目相关的要点：
  1. **技术成熟度阶梯**：把编队干涉测量拆为"地基干涉成熟 → 小卫星技术演示 → 编队飞行 → 成像任务"的渐进路线，并主张先做小规模端到端演示（目标 2030 年前首次星光干涉条纹）。与本框架 L1–L5 逐级验收策略同构。
  2. **关键使能技术清单**（原文 Technology Drivers）：轨道与发射、星间通信（km 级 5G 局域网）、星间计量（LIDAR/无线电测距）、**控制与机动/编队刚性**：指出当时在轨编队指标（±0.5 m 相对保持、~100 µm/s 漂移）**不足以支撑干涉测量**，核心难点是维持光程差（OPD）实现相位稳定长曝光；以及重定向/加注/交会对接的燃料问题。
  3. **历史教训**：SIM（天体测量，4 µas 目标）、TPFI/DARWIN（红外消零）因技术延期与成本取消；相关技术随后由 LISA 引力波干涉仪与小卫星项目承接——佐证"分级验证、控制成本"的工程路线。
  4. 对 L1–L4 的直接启示：L2 的编队保持指标要向"相位稳定 OPD"看齐；L3/L4 必须显式建模 OPD 与指向稳定度；燃料预算（重定向 + 观测保持）是系统级寿命瓶颈。

### LIT-002《STARI: Starlight Acquisition and Reflection toward Interferometry》

- PDF：`pdfs/2024_STARI_STarlight_Acquisition_and_Reflection_toward_Interferometry.pdf`
- 精读入口：`mineru read doc:46e896a/tier:standard/page:4`
- 要点：
  1. 面向中红外类地行星探测的空间干涉测量**技术验证星**概念（小卫星平台），给出使命目标与体系结构的子系统级拆解；
  2. 关键子系统：**LED 信标精密相对导航**、单模光纤注入、星光/参考激光共准直——与 L2（相对导航）和 L3（共相/共准直）直接对应；
  3. 可作为 L2 相对测量链路与 L3 OPD 链路的工程实现参照。

### LIT-003《Stellar Observation Scheduling Optimization for Distributed Space Interferometry Missions》

- PDF：`pdfs/2026_Stellar_Observation_Scheduling_Optimization_for_Distributed_Space_Interferometry.pdf`
- 精读入口：`mineru read doc:ca82e45/tier:standard/page:2`
- 要点：
  1. 以绝对/相对轨道根数建立分布式干涉测量的**轨道设计与观测约束**框架（太阳规避、地/日掩星、基线振荡、冻结相对轨道约束）；
  2. 面向"观测调度 + 队形约束"的任务规划方法——直接支撑 L2 的任务规划与构型保持指标设计；
  3. 冻结相对轨道约束可作为 L2 编队目标状态的候选设计依据（待精读确认）。
