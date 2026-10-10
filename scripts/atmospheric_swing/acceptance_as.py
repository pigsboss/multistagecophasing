#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS 严格验收脚本（判据无后门）
================================
所有判据必须严格通过；脚本内不得存在容忍性分支（静态断言由 T-AS-008 检查）。

验收项：
  1. 加速内核一致性（numba fp64 vs numpy 参考，逐位）
  2. 事件检测样例（近地点计数 / 阈值穿越）
  3. 能量账本闭合（小型算例，校准门限）
  4. 低近地点搜索端到端（缩配：结构完整；如有解则独立复核通过）
  5. 敏感度工具（区间口径）

用法: python3 scripts/atmospheric_swing/acceptance_as.py
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

CHECKS = {}


def main() -> int:
    print("=" * 70)
    print("AS Strict Acceptance (低近地点共振摆动专题)")
    print("=" * 70)

    # ---- 1. 加速内核一致性 ----
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    from mission_sim.utils.dynamics import compiled_kernels as ck
    c = UniversalCRTBP.earth_moon_system()
    mu = c.mu
    x = np.array([0.5, 0.1, 0.0, 0.1, 0.8, 0.0])
    d_rhs = float(np.abs(np.asarray(ck.rhs_crtbp(x, mu))
                         - np.concatenate([x[3:], c._crtbp_acceleration_nd(x)])).max())
    d_jac = float(np.abs(np.asarray(ck.jac_crtbp(x, mu)) - c.variational_matrix_nd(x)).max())
    CHECKS["加速内核逐位一致"] = (d_rhs <= 1e-15 and d_jac <= 1e-15)
    print(f"   [1] RHS/Ψ 差: {d_rhs:.2e} / {d_jac:.2e}")

    # ---- 2. 事件检测 ----
    from mission_sim.utils.dynamics import events
    m_up = 3.986004418e14
    n = np.sqrt(m_up / 1.0e7 ** 3)
    t = np.linspace(0.0, 2 * np.pi / n, 20001)
    r = 1.0e7 * (1.0 - 0.3 * np.cos(n * t))
    states = np.zeros((t.size, 6))
    states[:, 0] = r
    per = events.perigee_events(t, states)
    CHECKS["事件检测（近地点计数≥2）"] = len(per) >= 2
    print(f"   [2] 近地点事件数: {len(per)}")

    # ---- 3. 能量账本闭合 ----
    from scipy.integrate import solve_ivp
    from mission_sim.analysis.energy_ledger import ledger_closure
    r0, mu_p = 7.0e6, 3.986004418e14
    v0 = np.sqrt(mu_p / r0)
    state0 = np.array([r0, 0.0, 0.0, 0.0, v0 * 0.9, 0.0])
    a_t = 1e-4

    def rhs(t_, y):
        rr = np.linalg.norm(y[:3])
        v = y[3:6]
        return np.concatenate([v, -mu_p * y[:3] / rr ** 3 + a_t * v / np.linalg.norm(v)])

    t_end = 2.0 * np.pi * np.sqrt(r0 ** 3 / mu_p) * 0.5
    sol = solve_ivp(rhs, (0, t_end), state0, rtol=1e-12, atol=1e-9,
                    dense_output=True, method="DOP853")
    ts = np.linspace(0, t_end, 20001)
    xs = sol.sol(ts).T

    def parts(_t, st):
        v = st[3:6]
        return {"tangential": a_t * v / np.linalg.norm(v)}

    out = ledger_closure(ts, xs, parts, mu=mu_p)
    CHECKS["账本闭合（≤1e-6，校准）"] = out["residual_rel"] <= 1e-6
    print(f"   [3] 账本 residual_rel: {out['residual_rel']:.2e}")

    # ---- 4. 搜索端到端（缩配） ----
    from mission_sim.core.cyber.algorithms.low_perigee import search_low_perigee
    rep = search_low_perigee(resonance=(2, 1), max_alt_km=300.0, grid=(8, 8),
                             n_steps=8000, refine_iter=30, tol=1e-8,
                             n_candidates=2, refine_steps=20000)
    sv = rep["survey"]
    struct_ok = (sv["n_seeds"] >= 64 and "best_residual" in sv
                 and sv["refine_attempts"] >= min(2, sv["n_candidates"]))
    sol_ok = True
    if rep["found"]:
        s0 = rep["solutions"][0]
        sol_ok = (s0["full_state_residual"] <= 1e-8 and s0["perigee_alt_km"] <= 301.0)
    CHECKS["搜索端到端（结构完整+有解则核验）"] = bool(struct_ok and sol_ok)
    print(f"   [4] 搜索: found={rep['found']} n_seeds={sv['n_seeds']} "
          f"best_residual={sv['best_residual']:.3e}")

    # ---- 5. 敏感度工具 ----
    from mission_sim.analysis.sensitivity import parameter_sweep
    res = parameter_sweep(lambda p: {"metric": 1.0 + 2.0 * p["density_scale"]},
                          parameter="density_scale", values=[0.5, 1.0, 1.5])
    CHECKS["敏感度工具（区间口径）"] = (res["interval"]["min"] == 2.0
                                        and res["interval"]["max"] == 4.0)
    print(f"   [5] 敏感度区间: {res['interval']}")

    # ---- 汇总 ----
    print("-" * 70)
    ok = all(CHECKS.values())
    for name, passed in CHECKS.items():
        print(f"   {'✅' if passed else '❌'} {name}")
    print("=" * 70)
    print("Overall: " + ("🎉 PASSED" if ok else "❌ FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
