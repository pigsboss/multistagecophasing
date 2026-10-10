# MCPC 工作台面（L1–L4 研制过程）

> 本目录为三方角色（设计开发 / 评审专家 / 质量管理）的工作台面，依据约定建立：
> "采用测试驱动开发的方式，设置设计开发、评审专家和质量管理三个角色，为三个角色分别创建工作台面，
> 你轮流扮演三个角色……" 另含共享区（文献资料），三方不必重复开展文献调研。
> 建立日期：2026-10-09。

## 目录结构

```text
workbench/
├── README.md                      # 本文件：过程章程与当前状态
├── shared/                        # 共享区（三方公用）
│   ├── literature/
│   │   ├── library.md             # 文献索引与登记表（LIT 编号，正式引用出处）
│   │   ├── pdfs/                  # 原始论文 PDF（MinerU 监控目录：新文件自动解析入库）
│   │   └── notes/                 # 文献批注（三方各自留痕）
│   └── tools/
│       └── openalex_search.py     # OpenAlex 检索工具（检索/筛 OA/下载，支持断点续传）
├── design_dev/                    # 设计开发工作台面（DD-xxx）
├── review_expert/                 # 评审专家工作台面（RE-xxx）
└── quality_mgmt/                  # 质量管理工作台面（QM-xxx）
```

## 角色与流程（按约定）

> **角色称呼（自 2026-10-10 起统一）**：设计组 / 专家组 / 质量组。
> 历史文件中的【设计开发】【评审专家】【质量管理】署名与之一一对应。

- **设计组**：编制《系统方案设计》《工作策划》《测试大纲》→ 评审修订后生效 → 按 TDD 推进 → 里程碑提交《工作报告》+《测试报告》。
- **专家组**：对三份文件给出"能否生效"意见；里程碑节点给出"通过 / 不通过"意见；不通过 → 技术归零后重审。
- **质量组**：全过程监督设计组与专家组是否履职，未充分履职可叫停；依据 PDCA 对双方提出改进意见。

## 文件编号规则

| 前缀 | 含义 | 存放位置 |
|---|---|---|
| DD-xxx | 设计组文件（方案 / 策划 / 大纲 / 报告 / 归零） | `design_dev/` |
| RE-xxx | 专家组文件（评审意见书 / 评审结论） | `review_expert/` |
| QM-xxx | 质量组文件（监督记录 / 检查单 / 改进单 / 叫停） | `quality_mgmt/` |
| LIT-xxx | 共享文献编号（一切文献引用以此为准） | `shared/literature/library.md` |

## 共享文献区使用

1. **检索**：`python3 workbench/shared/tools/openalex_search.py "<关键词>" --oa --limit 10`
2. **下载**：追加 `--download-dir workbench/shared/literature/pdfs --download-limit N`
3. **自动解析**：PDF 放入 `pdfs/` 后由 MinerU 监控自动入库；亦可手动 `mineru scan workbench/shared/literature/pdfs`
4. **检索/精读已入库文献**：`mineru search "<词>"`、`mineru read doc:<id>/tier:standard/page:N`
5. **登记**：在 `library.md` 登记 LIT 编号后方可作为正式引用。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 方案生效 | 三份文件 v1 → 修订 v1.1 → 生效（M0） | ✅ 完成（RE-002 签署） |
| P1 L1 修复与验证 | 红→绿 12/12；全量 253/253（M1） | ✅ 完成（RE-003 通过） |
| P2 L2 接通与验收 | 红→绿 10/10；全量 263/263（M2） | ✅ 完成（RE-004 通过）——L2 完成 |
| P3 L3 原理原型 | 红→绿 6/6；全量 269/269（M3） | ✅ 完成（RE-005 通过）——L3 完成 |
| P4 L4 工程化原型 | 红→绿 7/7；全量 276/276（M4） | ✅ 完成（RE-006 通过）——**L1–L4 全部完成** |
| LS 专题（excursion） | LUNAR-SWING：严格周期解 + 30 天摄动演化 + 能力沉淀（问题牵引；含 1 次技术归零） | ✅ 技术完成（M1/M2 通过；LS-004/005 + RE-LS-002/003 + QM-LS-002） |
| AS 专题（excursion） | 低近地点共振摆动：族域勘测（负/部分结果）+ 能量账本实验 + 加速内核（CR-AS-01） | ✅ 技术完成（M1/M2 通过；AS-004/005 + RE-AS-001/001A/002/003 + QM-AS-001/002） |

最近报告：`design_dev/DD-012_L4工作报告_v1.md`、`DD-013_L4测试报告_v1.md`；评审与监督：`review_expert/RE-006`、`quality_mgmt/QM-007`。
lunar_swing 专题（2026-10-10）：立项 `design_dev/LS-001/002/003`；报告 `LS-004/005`；评审 `review_expert/RE-LS-001/002/003`；监督 `quality_mgmt/QM-LS-001/002`；证据 `design_dev/evidence/LS/`。
研究总结报告（v1.1，LaTeX→PDF，66 页，含 10 张图解）：`docs/excursion/lunar_swing/report/LUNAR-SWING_研究总结报告_v1.1.pdf`（源码与构建：同目录 `book.yaml`、`sources/`、`make_figures.py`；评审 `review_expert/RE-LS-004`）。
电子附件动画（4 段 MP4：机制示意 / 严格解双视角 / LS 30 天 / AS 30 天）：`docs/excursion/lunar_swing/report/animations/`（清单与复现见该目录 `动画清单.md`、`make_animations.py`）。
AS 专题（2026-10-10）：立项 `design_dev/AS-001/002/003`、变更 `AS-CR-01`；报告 `AS-004/005`；评审 `review_expert/RE-AS-001/001A/002/003`；监督 `quality_mgmt/QM-AS-001/002`；证据 `design_dev/evidence/AS/`；实验数据 `docs/excursion/lunar_swing/results/as_lowperigee/`。
