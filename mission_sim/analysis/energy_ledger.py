# mission_sim/analysis/energy_ledger.py
# -*- coding: utf-8 -*-
"""
能量账本工具（G5④ 沉淀件）
============================
核算：机械能变化 ΔE = Σ_k ∫ (a_k · v) dt 的闭合性。
E = ½|v|² − μ/|r|（近似机械能，二体参考）；a_k 为各力加速度（中央引力/阻力/第三体…）。
账本闭合性是一种独立验证范式（与势梯度/跨积分器互补）。
"""
import numpy as np


def ledger_closure(t, states, accel_parts, mu):
    """账本闭合性核算（双口径）。

    ① 动能口径：传入**全部**力（含中心项）→ 检查 `residual_kin_rel`；
    ② 机械能口径（账本）：仅传入**扰动力**（阻力/第三体等）→ 检查 `residual_rel`
       （ΔE_mech 应等于扰动力做功之和）。

    闭合残差下限由**输入轨迹的积分精度**决定（实测：rtol 1e-11 → ~1e-7；
    rtol 1e-12 → ~8.5e-9；与采样密度无关）。实验轨迹建议 rtol ≤ 1e-12。

    Args:
        t: (N,) 采样时刻
        states: (N, 6) 状态（与 mu 同单位制）
        accel_parts: callable(t_i, state_i) -> dict{力名: (3,) 加速度}
        mu: 中心引力常数（与 states 单位制一致）

    Returns: dict(dE, dE_kin, work{name: ...}, work_total, residual, residual_rel,
                  residual_kin, residual_kin_rel)
    """
    t = np.asarray(t, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    r = np.linalg.norm(states[:, :3], axis=1)
    v2 = np.sum(states[:, 3:6] ** 2, axis=1)
    energy = 0.5 * v2 - mu / r
    d_e = float(energy[-1] - energy[0])

    v = states[:, 3:6]
    work_series = {}
    for i in range(t.size):
        parts = accel_parts(float(t[i]), states[i])
        for name, a in parts.items():
            if name not in work_series:
                work_series[name] = np.empty(t.size)
            work_series[name][i] = float(np.dot(np.asarray(a, dtype=np.float64), v[i]))
    work_total_series = np.sum([work_series[k] for k in work_series], axis=0)
    work_totals = {k: float(np.trapezoid(work_series[k], t)) for k in work_series}
    total = float(np.trapezoid(work_total_series, t))
    d_e_kin = float(0.5 * v2[-1] - 0.5 * v2[0])
    residual = d_e - total
    scale = max(abs(d_e), abs(total), 1e-30)
    residual_kin = d_e_kin - total
    scale_kin = max(abs(d_e_kin), abs(total), 1e-30)
    return {
        "dE": d_e,
        "dE_kin": d_e_kin,
        "work": work_totals,
        "work_total": total,
        "residual": residual,
        "residual_rel": abs(residual) / scale,
        "residual_kin": residual_kin,
        "residual_kin_rel": abs(residual_kin) / scale_kin,
    }
