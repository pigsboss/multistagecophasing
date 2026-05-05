# ADR-001: SPICE 内核加载顺序与 ET 时间单位使用范围

- **Status**: Accepted
- **Date**: 2025-01-XX
- **Deciders**: MCPC Architecture Team
- **Related**: `spice_interface.py`, `high_precision.py`, `diagnose_spice_kernels.py`

---

## Context

在使用 NASA NAIF SPICE 工具包进行高精度星历计算时，我们遇到了两类工程化问题：

1. **内核加载顺序导致的时标转换失败**  
   在诊断脚本 `diagnose_spice_kernels.py` 中，为了展示 SPK 文件内容，直接在未加载 LSK（闰秒内核）的情况下调用 `spiceypy.et2utc()`。SPICE 内核池缺少 `DELTET/DELTA_T_A`、`DELTET/K` 等参数，抛出 `SPICE(MISSINGTIMEINFO)` 异常。虽然 SPK 数据本身完好，但 UTC 显示功能完全失效。这说明 SPICE 各内核之间存在隐式依赖，必须按正确顺序加载。

2. **ET 作为裸露数值在模块间传递**  
   SPICE 内部使用 Ephemeris Time（ET，秒 past J2000，本质为 TDB）进行所有几何计算。在排查问题时，诊断脚本曾将原始 ET 秒数（如 `831211269.185`）直接暴露给终端用户和测试报告。ET 是机器内部计算量，对人类不可读，也不应在模块 API 边界外传播。

---

## Decision

### 1. SPICE 内核加载顺序（强制）

加载必须遵循以下严格顺序，后续内核可能依赖前面内核定义的池变量：

| 顺序 | 内核类型 | 作用 | 说明 |
|:---|:---|:---|:---|
| 1 | **LSK** | 闰秒内核 | 提供 `DELTET/*` 参数，是 **所有** 时间转换（ET↔UTC）的前提 |
| 2 | **PCK** | 行星常数 | 包含天体质量、半径、自转模型；部分 PCK 依赖时间模型 |
| 3 | **SPK** | 星历数据 | DE440/441/442 等；状态查询依赖 LSK 进行光行时计算 |
| 4 | **FK/IK/CK** | 坐标系/仪器/姿态 | 定义参考架和传感器参数，通常最后加载 |

**实施规则：**
- `SPICEKernelManager.initialize()` 必须首先调用 `_load_kernel_type('lsk')`，不允许延迟或跳过。
- 任何需要在加载前/后检查 SPK 文件内容的工具（如诊断脚本），如果涉及 UTC 显示，**必须先完成 LSK 加载**，或在无 LSK 时回退到纯 ET 秒显示并明确标注“raw ET”。
- 禁止在 `SPICEKernelManager` 之外手动调用 `spice.furnsh()` 绕过加载顺序。

### 2. ET 使用范围约定（强制）

| 层级 | 允许使用 ET | 必须使用 UTC/ISO 字符串 |
|:---|:---|:---|
| **内部动力学计算** | ✅ `spacetime` 模块内部、SPICE 接口底层、数值积分器 | ❌ |
| **模块间 API 参数** | ❌ | ✅ 所有 `get_state()` 等公开接口应接受 UTC 字符串或 datetime，内部再转换 |
| **日志/stdout/stderr** | ❌ | ✅ 人类可读的 ISO 8601（如 `2026-05-05T00:01:09.185Z`） |
| **测试报告/诊断输出** | ❌ | ✅ 终端展示、警告信息、HDF5 元数据 |
| **可视化标签** | ❌ | ✅ 场景时间戳、轴标签、标题 |
| **持久化存储** | ❌ | ✅ UTC 字符串或带明确时标的 datetime 对象 |

**实施规则：**
- `SPICEInterface` 与 `HighPrecisionEphemeris` 对外暴露的时间参数统一为 UTC 字符串；ET 转换仅在类内部完成。
- 诊断与测试脚本中，`epoch` 变量若需打印，必须经 `et2utc` 转换。
- 若因性能需要缓存 ET，缓存必须封装在 `spacetime` 内部，不得泄漏到调用方。

---

## Consequences

### Positive
- 消除 `MISSINGTIMEINFO` 类低级错误，提升内核加载可靠性。
- 统一时间表示，降低跨模块集成时的歧义。
- 符合 MCPC 国际化政策（stdout/stderr/日志使用英文且人类可读）。
- 便于非航天背景开发者理解接口（无需理解 J2000/ET 概念）。

### Negative
- 内部高频计算场景下，UTC↔ET 转换带来微小开销。可通过内部批量预转换缓解，但不得破坏接口约定。
- 现有代码中若存在裸露 ET 参数，需要逐步重构。

---

## Compliance Checklist

在审查涉及 SPICE 或时间处理的 PR 时，确认：

- [ ] 新增内核加载代码是否将 LSK 作为第一步？
- [ ] 新增 stdout / 日志 / 异常信息中是否出现裸露 ET 秒数？
- [ ] 新增 API 是否接受 UTC 字符串而非强制调用方传入 ET？
- [ ] 新增 HDF5 / 元数据中的时间戳是否为 UTC？
- [ ] 诊断/测试工具在显示 SPK 时间区间前是否已确保 LSK 加载？

---

## References

- [NAIF SPICE Documentation: Kernel Loading](https://naif.jpl.nasa.gov/naif/utilities_PC_Windows_Vista.html)
- MCPC Coding Standards & Internationalization Policy (`docs/guides/CODING_STANDARDS.md`)
- 相关 Issue: SPICE `MISSINGTIMEINFO` on `et2utc` before LSK load
- 相关 Commit: `spice_interface.py` `_load_kernel_type` LSK-first guarantee
