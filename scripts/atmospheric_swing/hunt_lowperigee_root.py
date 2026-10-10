#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低近地点族求根攻坚（W3 探索工具）：scipy trust-region（hybr）批量求根
====================================================================
两批种子：
  A. 4D 全状态种子（/tmp/as4d_seeds.npy，27.32 天周期）
  B. 对称谷域种子（(alt, f) 小网格，13.66 天半周期截面）
用法: python3 scripts/atmospheric_swing/hunt_lowperigee_root.py [--n 100000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.optimize import root

from mission_sim.core.cyber.algorithms.low_perigee import CHAR_PERIOD, R_EARTH, MU_EARTH
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.utils.dynamics import compiled_kernels as ck

T_MOON = 27.321661 * 86400.0
T_SC = 0.5 * T_MOON
T_SHOT = 2 * T_SC


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--maxfev", type=int, default=300)
    args = ap.parse_args()

    c = UniversalCRTBP.earth_moon_system()
    mu, L, om = c.mu, c.distance, c.omega
    t_half = T_SC / CHAR_PERIOD          # 对称截面时刻
    t_shot = T_SHOT / CHAR_PERIOD        # 4D 全周期

    solutions = []
    t0 = time.perf_counter()

    # ---------------- A. 4D 种子 ----------------
    try:
        seeds4 = np.load("/tmp/as4d_seeds.npy")
    except FileNotFoundError:
        seeds4 = np.zeros((0, 6))
    print(f"[A] 4D 种子 {len(seeds4)} 个（hybr, n={args.n}）")

    def res4(x4):
        s = np.zeros(6)
        s[0], s[1], s[3], s[4] = x4
        xf = ck.rk4_propagate(s, t_shot, args.n, mu)
        return np.array([xf[0] - s[0], xf[1] - s[1], xf[3] - s[3], xf[4] - s[4]])

    for i, s in enumerate(seeds4):
        x4 = np.array([s[0], s[1], s[3], s[4]])
        try:
            sol = root(res4, x4, method="hybr",
                       options={"maxfev": args.maxfev, "xtol": 1e-12})
        except Exception as e:  # noqa: BLE001
            print(f"  A{i}: 异常 {e}")
            continue
        rn = float(np.linalg.norm(res4(sol.x)))
        if rn < 1e-6:
            print(f"  A{i}: ✓ res={rn:.3e} x4={np.array2string(sol.x, precision=10)}")
            solutions.append({"batch": "4D", "res": rn, "x4": sol.x.tolist()})
        else:
            print(f"  A{i}: res={rn:.3e}")

    # ---------------- B. 对称谷域种子 ----------------
    def make_seed(r_p, p_in, side):
        a2 = (MU_EARTH * (p_in / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        v_p = np.sqrt(MU_EARTH * (2.0 / r_p - 1.0 / a2))
        x_nd = side * (r_p / L) - mu
        vy0 = side * (v_p / (L * om)) - (x_nd + mu)
        return x_nd, vy0

    alts = np.linspace(120.0, 200.0, 9)
    fs = np.linspace(0.9845, 0.9885, 17)
    seedsB = []
    for side in (+1, -1):
        for h in alts:
            for f in fs:
                seedsB.append((side, h, f))
    print(f"[B] 对称谷域种子 {len(seedsB)} 个（hybr, n={args.n}）")

    def res2(x2):
        s = np.zeros(6)
        s[0], s[4] = x2
        xf = ck.rk4_propagate(s, t_half, args.n, mu)
        return np.array([xf[1], xf[3]])

    for side, h, f in seedsB:
        x0, vy0 = make_seed(R_EARTH + h * 1e3, T_SC * f, side)
        try:
            sol = root(res2, np.array([x0, vy0]), method="hybr",
                       options={"maxfev": args.maxfev, "xtol": 1e-12})
        except Exception as e:  # noqa: BLE001
            print(f"  B(side={side:+d},alt={h:.0f},f={f:.4f}): 异常 {e}")
            continue
        rn = float(np.linalg.norm(res2(sol.x)))
        tag = "✓" if rn < 1e-6 else " "
        print(f" {tag} B(side={side:+d},alt={h:.0f},f={f:.4f}): res={rn:.3e} x2={np.array2string(sol.x, precision=10)}")
        if rn < 1e-6:
            solutions.append({"batch": "sym", "res": rn, "x2": sol.x.tolist(),
                              "seed": {"side": side, "alt_km": h, "f": f}})

    print(f"共 {len(solutions)} 个收敛解；用时 {time.perf_counter() - t0:.1f}s")
    out = project_root / "workbench/design_dev/evidence/AS" / "hunt_roots_result.json"
    out.write_text(json.dumps(solutions, indent=2), encoding="utf-8")
    print("结果写入:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
