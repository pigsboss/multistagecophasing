#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存在性试算（非立项交付）：低近地点 2:1 共振周期解
=================================================
背景：原 lunar_swing 概念（01_concept_design.md）要求"低近地点（≤200 km 大气穿越）+
高远地点（接近月球轨道）"的 2:1 共振摆动轨道；LS 专题严格化的是同一共振比的另一成员
（近地点 ~31.6 万 km）。本脚本用已交付的对称求解器，从二体解析种子直接试解低近地点解，
回答"该族是否可及"的存在性问题。

种子（二体 2:1：T=13.66 d、r_p=6578.137 km、a≈241,420 km、v_p≈10.933 km/s）：
  A: 近地点在 +x 侧（三体旋转系，质心原点）
  B: 近地点在 −x 侧

用法: python scripts/lunar_swing/probe_lowperigee_2to1.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.integrate import solve_ivp

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

CHAR = 4.342 * 86400.0
T_MOON = 27.321661 * 86400.0


def main() -> int:
    c = UniversalCRTBP.earth_moon_system()
    L, mu = c.distance, c.mu
    T_nd = (0.5 * T_MOON) / CHAR

    seeds = {
        "+x 侧近地点": (0.0049680, 10.6545),
        "-x 侧近地点": (-0.0292574, -10.6545),
    }
    for tag, (x0g, vy0g) in seeds.items():
        print(f"===== 试算 [{tag}] seed=({x0g:.7f}, {vy0g:.4f}) =====", flush=True)
        tg = LunarSwingTargeter(dynamics_model=c, mu=c.mu,
                                integrator_type="rk4", num_steps=200000)
        r = tg.find_symmetric_periodic_orbit(
            (2, 1), x0g, vy0g, tol=1e-6, max_iter=30,
            char_period=CHAR, num_steps=200000, verbose=True)
        print(f"  success={r.get('success')} iters={r.get('iterations')} "
              f"half={r.get('half_period_residual')}")
        if not r.get("success"):
            print("  → 未收敛（记录为负结果候选）")
            continue
        x0 = np.asarray(r["state"])
        print("  state_nd =", np.array2string(x0, precision=10))

        def rhs(t, x):
            return np.concatenate([x[3:6], c._crtbp_acceleration_nd(x)])

        sol = solve_ivp(rhs, (0.0, T_nd), x0, rtol=1e-11, atol=1e-13,
                        dense_output=True, method="DOP853")
        ts = np.linspace(0, T_nd, 20001)
        X = sol.sol(ts)
        rgeo = np.linalg.norm(X[0:3] + np.array([[mu], [0], [0]]), axis=0) * L
        d_moon = np.linalg.norm(X[0:3] - np.array([[1 - mu], [0], [0]]), axis=0) * L
        print("  地心半径: min=%.1f km (高度 %.1f km)  max=%.1f km" % (
            rgeo.min() / 1e3, rgeo.min() / 1e3 - 6378.137, rgeo.max() / 1e3))
        print("  近月最近距离: %.0f km" % (d_moon.min() / 1e3))
        print("  单周期闭合复核: %.3e" % np.linalg.norm(sol.y[:, -1] - x0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
