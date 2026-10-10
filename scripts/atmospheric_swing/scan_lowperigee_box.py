#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低近地点族邮箱扫描 v2（带 r_max 过滤；AS W3 探索工具）
========================================================
背景：残差图被"小轨道相位巧合"（周期性回返）污染，必须用
"半周期弧的最大地心半径"过滤出真正"远地点近月球"的候选。

用法: python3 scripts/atmospheric_swing/scan_lowperigee_box.py [--nx 60 --nv 120 --steps 24000 --refine-top 10]
"""
import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from mission_sim.core.cyber.algorithms.low_perigee import (
    CHAR_PERIOD, R_EARTH, MU_EARTH, _sc_period_for)
from mission_sim.core.cyber.algorithms.symmetric_shooting import SymmetricShooter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.utils.dynamics import compiled_kernels as ck


def build_box(nx, nv):
    # +x 侧：x0 ∈ [0.00498, 0.020]（地心半径 6560–8320 km）；vy0 ∈ [7.7, 10.75]
    xs_p = np.linspace(0.00498, 0.020, nx)
    vs_p = np.linspace(7.7, 10.75, nv)
    # −x 侧：x0 ∈ [−0.0337, −0.0293]（地心半径 6578–8298 km）
    xs_m = np.linspace(-0.0337, -0.0293, nx)
    vs_m = -vs_p
    seeds, meta = [], []
    for x0 in xs_p:
        for vy0 in vs_p:
            s = np.zeros(6)
            s[0] = x0
            s[4] = vy0
            seeds.append(s)
            meta.append((+1, float(x0), float(vy0)))
    for x0 in xs_m:
        for vy0 in vs_m:
            s = np.zeros(6)
            s[0] = x0
            s[4] = vy0
            seeds.append(s)
            meta.append((-1, float(x0), float(vy0)))
    return np.asarray(seeds), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=60)
    ap.add_argument("--nv", type=int, default=120)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--rmax-min", type=float, default=0.9, help="地心 r_max 过滤下限（DU）")
    ap.add_argument("--refine-top", type=int, default=10)
    ap.add_argument("--refine-steps", type=int, default=80000)
    args = ap.parse_args()

    c = UniversalCRTBP.earth_moon_system()
    mu = c.mu
    t_sc = _sc_period_for((2, 1))
    t_half = t_sc / CHAR_PERIOD          # 对称截面时刻 = 一个 T_sc
    shot_period = 2.0 * t_sc

    seeds, meta = build_box(args.nx, args.nv)
    t0 = time.perf_counter()
    ext = ck.batch_half_period_residual_extent(seeds, t_half, args.steps, mu)
    print(f"scan {len(seeds)} seeds x {args.steps} steps: {time.perf_counter() - t0:.1f}s")

    resn = np.hypot(ext[:, 0], ext[:, 1])
    rmax, rmin, rend = ext[:, 3], ext[:, 2], ext[:, 4]
    mask = rmax >= args.rmax_min
    print(f"filtered (r_max_geo >= {args.rmax_min} DU): {int(mask.sum())}/{len(seeds)}")
    idx = np.where(mask)[0]
    if idx.size == 0:
        print("过滤后无候选——扩大扫描域或降低过滤阈值")
        return 1
    order = idx[np.argsort(resn[idx])]
    print("== top 20（过滤后按半周期残差） ==")
    for i in order[:20]:
        side, x0, vy0 = meta[i]
        print("  res=%.4e rmin=%.4f rmax=%.4f rend=%.4f side=%+d x0=%.6f vy0=%.5f"
              % (resn[i], rmin[i], rmax[i], rend[i], side, x0, vy0))

    shooter = SymmetricShooter(mu=mu, period_seconds=shot_period, n_steps=args.refine_steps)
    print(f"== Newton 精修（top {args.refine_top}；{args.refine_steps} 步） ==")
    n_success = 0
    for i in order[:args.refine_top]:
        s = seeds[i]
        out = shooter.solve(s[0], s[4], tol=1e-8, max_iter=50)
        print("  seed(side=%+d, x0=%.6f, vy0=%.5f) → success=%s res=%.3e iters=%d"
              % (meta[i][0], meta[i][1], meta[i][2], out["success"], out["residual"], out["iterations"]))
        if out["success"]:
            n_success += 1
            ver = shooter.verify(out["x0"], out["vy0"], n_steps=args.refine_steps)
            print("     → x0=%.8f vy0=%.8f full_res=%.3e jacobi_drift=%.2e"
                  % (out["x0"], out["vy0"], ver["full_state_residual"], ver["jacobi_drift"]))
    print(f"精修成功 {n_success}/{min(args.refine_top, order.size)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
