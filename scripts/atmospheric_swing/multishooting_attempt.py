#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多重打靶攻坚（W2「分段/自适应打靶」技术验证；W3 探索工具）
===============================================================
把 27.32 天闭合周期切成 k 段（自适应积分），未知数 = k 个节点状态（4k 维），
方程 = k-1 个段间连续 + 1 个闭包；雅可比用逐段 STM（FD 局部化），牛顿 + 线搜索。

用法: python3 scripts/atmospheric_swing/multishooting_attempt.py [--k 8] [--iters 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from mission_sim.core.cyber.algorithms.low_perigee import CHAR_PERIOD, R_EARTH, MU_EARTH
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.utils.dynamics import compiled_kernels as ck

T_MOON = 27.321661 * 86400.0
T_SHOT = T_MOON / CHAR_PERIOD          # 6.2924 TU
SCALE = 5e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=float, default=SCALE)
    args = ap.parse_args()

    c = UniversalCRTBP.earth_moon_system()
    mu, L = c.mu, c.distance
    k = args.k
    tau = T_SHOT / k
    scale = args.scale

    def segprop(y4):
        s = np.array([y4[0], y4[1], 0.0, y4[2], y4[3], 0.0])
        xf, _ = ck.rk4_propagate_adaptive(s, tau, mu, scale=scale,
                                          h_max=4e-3, h_min=2e-7)
        return np.array([xf[0], xf[1], xf[3], xf[4]])

    def seg_jac(y4, h=None):
        if h is None:
            h = np.array([1e-9, 1e-9, 1e-8, 1e-8])
        J = np.zeros((4, 4))
        for j in range(4):
            yp = y4.copy(); yp[j] += h[j]
            ym = y4.copy(); ym[j] -= h[j]
            J[:, j] = (segprop(yp) - segprop(ym)) / (2.0 * h[j])
        return J

    # 候选种子（来自宽域交叉扫描/谷底微扫描的最优近失点）
    seeds_x2 = [
        (0.0049602039, 10.6562923),
        (0.0052427000, 10.3059630),
        (0.0046494000, 10.6944950),
        (0.0052921000, 10.3665140),
        (0.0050449000, 10.4724770),
        (0.0046494000, 10.2353210),
    ]

    all_results = []
    for sx0, svy0 in seeds_x2:
        y0 = np.array([sx0, 0.0, 0.0, svy0])
        # 生成种子节点链：沿近失轨道分段采样
        nodes = np.zeros((k, 4))
        nodes[0] = y0
        ok = True
        for i in range(k - 1):
            nodes[i + 1] = segprop(nodes[i])
        # 闭包误差
        closure = segprop(nodes[k - 1]) - nodes[0]
        print(f"== seed({sx0:.7f},{svy0:.5f}) k={k} 初始闭包误差 |F|= {np.linalg.norm(closure):.3e} ==", flush=True)

        Y = nodes.copy()
        best_norm = np.inf
        for it in range(args.iters):
            F = np.zeros(4 * k)
            Phi = np.zeros((k, 4, 4))
            for i in range(k):
                Phi[i] = seg_jac(Y[i])
                Yn = segprop(Y[i])
                if i < k - 1:
                    F[4 * i:4 * i + 4] = Yn - Y[i + 1]
                else:
                    F[4 * i:4 * i + 4] = Yn - Y[0]
            fn = float(np.linalg.norm(F, np.inf))
            if fn < best_norm:
                best_norm = fn
            if it % 10 == 0 or fn < 1e-9:
                print(f"   it={it:3d} |F|inf={fn:.3e}", flush=True)
            if fn < 1e-9:
                break
            # 组装块雅可比（4k × 4k）
            J = np.zeros((4 * k, 4 * k))
            for i in range(k):
                J[4 * i:4 * i + 4, 4 * i:4 * i + 4] = Phi[i]
                if i < k - 1:
                    J[4 * i:4 * i + 4, 4 * (i + 1):4 * (i + 1) + 4] = -np.eye(4)
                else:
                    J[4 * i:4 * i + 4, 0:4] = -np.eye(4)
            dY = np.linalg.lstsq(J, -F, rcond=None)[0]
            # 线搜索（在节点扰动下重算 F 的无穷范数）
            accepted = False
            for tsc in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
                Yt = (Y + tsc * dY.reshape(k, 4))
                Ft = np.zeros(4 * k)
                for i in range(k):
                    Yn = segprop(Yt[i])
                    if i < k - 1:
                        Ft[4 * i:4 * i + 4] = Yn - Yt[i + 1]
                    else:
                        Ft[4 * i:4 * i + 4] = Yn - Yt[0]
                if np.linalg.norm(Ft, np.inf) < fn:
                    Y = Yt
                    accepted = True
                    break
            if not accepted:
                print(f"   线搜索失败于 it={it}（|F|={fn:.3e}），停止本种子", flush=True)
                break

        # 终检：合并整周期（自适应）与 scipy 复核
        y0b = Y[0]
        s = np.array([y0b[0], y0b[1], 0.0, y0b[2], y0b[3], 0.0])
        xf, _ = ck.rk4_propagate_adaptive(s, T_SHOT, mu, scale=1.5e-4)
        full_res = float(np.linalg.norm(xf - s))
        print(f"   终检: |F|inf={best_norm:.3e}  全周期闭包(adaptive)= {full_res:.3e}", flush=True)
        all_results.append({"seed": [sx0, svy0], "k": k, "F_inf": best_norm,
                            "full_res": full_res, "state_nd": s.tolist()})
        if full_res < 1e-8:
            print("   ★★ 收敛！保存。", flush=True)
            np.save("/tmp/as_multishoot_solution.npy", s)

    out = project_root / "workbench/design_dev/evidence/AS" / f"multishooting_k{k}_result.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("结果:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
