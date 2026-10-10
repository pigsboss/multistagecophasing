# mission_sim/core/cyber/algorithms/low_perigee.py
# -*- coding: utf-8 -*-
"""
低近地点共振族搜索（AS W3；CR-AS-02 约定修正版）
==================================================
约定（概念口径，与 LS targeter 不同并在 AS-004 中说明）：
  resonance=(p, q) 表示 p 个**航天器惯性周期** : q 个月球周期
  （概念："航天器每绕地球 2 圈、月球绕 1 圈" ⇒ (2,1) ⇒ T_sc = T_moon/2 = 13.66 d）。
  → 对称截面时刻 t_half = T_sc（半射程）；旋转系闭合周期 = 2·T_sc = T_moon。
  → 二体种子：惯性周期 = T_sc·f（f 为周期因子扫描）；近地点在 ±x 轴。

策略：种子网格 → 编译内核批量粗筛（大网格可选 MLX fp32 筛选层）
      → 候选精细化（对称打靶，线搜索牛顿）→ 严格核验（scipy 独立复核全闭合/近地点高度）。

负结果路径（AS-001 G1）：found=False 时 survey 记录覆盖域、候选数、精修尝试与
最佳残差（O-9 留档；勘测充分性由评审检查）。
"""
import time

import numpy as np

from mission_sim.core.cyber.algorithms.symmetric_shooting import (
    SymmetricShooter, jacobi_constant)
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.utils.dynamics import compiled_kernels as ck
from mission_sim.utils.dynamics import mlx_screen

MU_EARTH = 3.986004418e14      # 地球 GM（物理）
R_EARTH = 6378137.0
CHAR_PERIOD = 4.342 * 86400.0
T_MOON = 27.321661 * 86400.0


def _sc_period_for(resonance):
    """航天器惯性周期（秒）：resonance=(p, q) ⇒ T_sc = (q/p)·T_moon。"""
    p, q = resonance
    return (q / p) * T_MOON


def _seed_state(r_p_m, p_inertial, side, mu_nd, L, omega):
    """构造旋转系对称初值（y=0, ẋ=0；近地点在 ±x 轴）。

    二体椭圆（惯性周期 p_inertial、近地点半径 r_p_m）→ 旋转系状态：
      x_nd = side·r_p/L − μ；v_nd(+x 侧) = v_p/(Lω) − r_p/L（−x 侧取反）。
    """
    a2 = (MU_EARTH * (p_inertial / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    v_p = np.sqrt(MU_EARTH * (2.0 / r_p_m - 1.0 / a2))
    x_nd = side * (r_p_m / L) - mu_nd
    vy0 = side * (v_p / (L * omega)) - (x_nd + mu_nd)
    s = np.zeros(6)
    s[0] = x_nd
    s[4] = vy0
    return s


def _build_seeds(resonance, min_alt_km, max_alt_km, grid, period_span, mu_nd, L, omega):
    t_sc = _sc_period_for(resonance)
    alts = np.linspace(min_alt_km, max_alt_km, int(grid[0]))
    factors = np.linspace(1.0 - period_span, 1.0 + period_span, int(grid[1]))
    seeds, meta = [], []
    for side in (+1, -1):
        for h_km in alts:
            r_p = R_EARTH + h_km * 1e3
            for f in factors:
                seeds.append(_seed_state(r_p, t_sc * float(f), side, mu_nd, L, omega))
                meta.append((side, float(h_km), float(f)))
    return np.array(seeds), meta, t_sc


def _verify_solution(state, mu, L, shot_period_s):
    """独立复核（scipy DOP853 稠密输出）：全闭合 / 近地点高度 / Jacobi 漂移。"""
    from scipy.integrate import solve_ivp

    t_full = shot_period_s / CHAR_PERIOD

    def rhs(t, y):
        return ck.rhs_crtbp(y, mu)

    sol = solve_ivp(rhs, (0.0, t_full), state, rtol=1e-11, atol=1e-13,
                    dense_output=True, method="DOP853")
    ts = np.linspace(0.0, t_full, 20001)
    xs = sol.sol(ts).T
    r_geo = np.linalg.norm(xs[:, :3] + np.array([mu, 0.0, 0.0]), axis=1)
    per_alt_km = float((r_geo.min() * L - R_EARTH) / 1e3)
    closure = float(np.linalg.norm(sol.y[:, -1] - state))
    c0 = jacobi_constant(state, mu)
    c1 = jacobi_constant(sol.y[:, -1], mu)
    drift = abs(c1 - c0) / max(abs(c0), 1e-30)
    return {"perigee_alt_km": per_alt_km,
            "full_state_residual": closure,
            "jacobi_drift": float(drift),
            "shot_period_days": t_full * CHAR_PERIOD / 86400.0}


def search_low_perigee(resonance=(2, 1), max_alt_km=300.0, grid=(24, 24),
                       n_steps=40000, refine_iter=60, tol=1e-8,
                       min_alt_km=120.0, period_span=0.03, n_candidates=12,
                       refine_steps=60000, verify=True, use_mlx="auto"):
    """低近地点共振周期解搜索（详见模块 docstring）。"""
    t0 = time.perf_counter()
    c = UniversalCRTBP.earth_moon_system()
    mu_nd, L, omega = c.mu, c.distance, c.omega

    seeds, meta, t_sc = _build_seeds(resonance, min_alt_km, max_alt_km,
                                     grid, period_span, mu_nd, L, omega)
    n_seeds = int(seeds.shape[0])
    t_half_nd = t_sc / CHAR_PERIOD          # 对称截面时刻（= 一个 T_sc）
    shot_period_s = 2.0 * t_sc              # 旋转系闭合周期（= 2·T_sc）

    use_mlx = bool(use_mlx) if use_mlx in (True, False) else \
        (n_seeds >= 8192 and mlx_screen.screening_available())
    if use_mlx:
        screen = mlx_screen.batch_half_period_residual_fp32(
            seeds.astype(np.float32), t_half_nd, n_steps, mu_nd)
        screen_path = "mlx-fp32"
    else:
        screen = ck.batch_half_period_residual(seeds, t_half_nd, n_steps, mu_nd)
        screen_path = "numba-fp64"

    resn = np.linalg.norm(screen, axis=1)
    order = np.argsort(resn)
    k = min(int(n_candidates), n_seeds)
    cand_idx = order[:k]
    best_residual = float(resn[order[0]]) if n_seeds else float("inf")

    shooter = SymmetricShooter(mu=mu_nd, period_seconds=shot_period_s, n_steps=refine_steps)
    solutions = []
    refine_results = []
    refine_attempts = 0
    for idx in cand_idx:
        refine_attempts += 1
        s = seeds[idx]
        out = shooter.solve(s[0], s[4], tol=tol, max_iter=refine_iter)
        refine_results.append({"seed": {"side": meta[idx][0], "alt_km": meta[idx][1],
                                        "period_factor": meta[idx][2]},
                               "success": bool(out["success"]),
                               "residual": float(out["residual"]),
                               "iterations": out["iterations"]})
        if not out["success"]:
            continue
        ver = shooter.verify(out["x0"], out["vy0"], n_steps=refine_steps) if verify \
            else {"full_state_residual": float("nan"), "jacobi_drift": float("nan")}
        extra = {}
        if verify:
            extra = _verify_solution(out["state"], mu_nd, L, shot_period_s)
        if extra.get("perigee_alt_km", min_alt_km) > max_alt_km + 1.0:
            continue
        if extra.get("full_state_residual", 0.0) > max(tol, 1e-8):
            continue
        solutions.append({
            "state_nd": out["state"].tolist(),
            "x0": out["x0"], "vy0": out["vy0"],
            "refine_residual": out["residual"],
            "iterations": out["iterations"],
            "seed_meta": {"side": meta[idx][0], "alt_km": meta[idx][1],
                          "period_factor": meta[idx][2]},
            **ver, **extra,
        })

    survey = {
        "resonance": list(resonance),
        "convention": "p 个航天器惯性周期 : q 个月球周期；t_half = T_sc；闭合周期 = 2·T_sc",
        "n_seeds": n_seeds,
        "grid": list(grid),
        "alt_range_km": [float(min_alt_km), float(max_alt_km)],
        "period_span": float(period_span),
        "t_sc_days": t_sc / 86400.0,
        "shot_period_days": shot_period_s / 86400.0,
        "screen_path": screen_path,
        "n_candidates": int(k),
        "refine_attempts": refine_attempts,
        "best_residual": best_residual,
        "best_seed_residuals": resn[order[:min(5, n_seeds)]].tolist(),
        "refine_results": refine_results,
        "runtime_s": time.perf_counter() - t0,
    }
    return {"found": len(solutions) > 0, "solutions": solutions, "survey": survey}
