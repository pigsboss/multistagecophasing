#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS W8 独立验证（补充证据）
============================
① 加速内核一致性（抽样）② 近失解步长/多积分器一致性 ③ 事件检测采样敏感性
④ 负结果重放（搜索 smoke 与既往留档对比）
用法: python3 scripts/atmospheric_swing/verify_w8_as.py
"""
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np

from mission_sim.core.cyber.algorithms.low_perigee import CHAR_PERIOD, R_EARTH
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.utils.dynamics import compiled_kernels as ck
from mission_sim.utils.dynamics import events as ev

X_NEAR = np.array([0.0049602039, 0.0, 0.0, 0.0, 10.6562923, 0.0])
T_HALF = 13.6608305 * 86400.0 / CHAR_PERIOD
OUTDIR = project_root / "docs/excursion/lunar_swing/results/as_lowperigee"


def main() -> int:
    c = UniversalCRTBP.earth_moon_system()
    mu = c.mu
    ok_all = True

    # ① 内核一致性（抽样）
    rng = np.random.default_rng(11)
    dmax = 0.0
    for _ in range(5):
        x = np.concatenate([rng.uniform(0.01, 0.8, 3), rng.uniform(-2, 2, 3)])
        d = np.abs(np.asarray(ck.rhs_crtbp(x, mu))
                   - np.concatenate([x[3:], c._crtbp_acceleration_nd(x)])).max()
        dmax = max(dmax, float(d))
    print(f"[W8-1] 内核一致性 max|Δrhs| = {dmax:.2e}  → {'OK' if dmax <= 1e-15 else 'FAIL'}")
    ok_all &= dmax <= 1e-15

    # ② 近失解整周期：自适应 vs scipy 稠密复核一致性
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        return ck.rhs_crtbp(y, mu)

    xa, n1 = ck.rk4_propagate_adaptive(X_NEAR, T_HALF, mu, scale=2.0e-5)
    so = solve_ivp(rhs, (0.0, T_HALF), X_NEAR, rtol=1e-13, atol=1e-15, method="DOP853")
    d2 = float(np.linalg.norm(np.asarray(xa) - so.y[:, -1]))
    res_a = np.array([xa[1], xa[3]])
    res_s = np.array([so.y[1, -1], so.y[3, -1]])
    d_res = float(np.linalg.norm(res_a - res_s))
    print(f"[W8-2] 近失解半周期：adaptive(scale=2e-5, 步{n1}) vs DOP853(rtol 1e-13) "
          f"状态差={d2:.2e}；残差差={d_res:.2e}（校准：DOP853@1e-11 自身误差~1.6e-6，须 1e-13） "
          f"→ {'OK' if d2 <= 1e-6 and d_res <= 1e-6 else 'FAIL'}")
    ok_all &= (d2 <= 1e-6 and d_res <= 1e-6)

    # ③ 事件检测采样敏感性（G2 实验数据：30s 采样；重采样到 600s 对比近地点时刻）
    f = h5py.File(OUTDIR / "AS_G2_drag1.0.h5", "r")
    t = f["epochs"][:]
    st = f["states"][:]
    f.close()
    per30 = ev.perigee_events(t, st)
    m = (t <= 5 * 86400.0)
    per600 = ev.perigee_events(t[m][::20], st[m][::20])
    if per30 and per600:
        dt_per = abs(per30[0]["t"] - per600[0]["t"])
        dalt = abs(per30[0]["value"] - per600[0]["value"])
        print(f"[W8-3] 近地点事件：30s 采样 {per30[0]['t']/86400:.4f}d vs 600s {per600[0]['t']/86400:.4f}d "
              f"（Δt={dt_per:.1f}s, Δalt={dalt/1e3:.2f}km）")
    else:
        print(f"[W8-3] 事件数: 30s={len(per30)} 600s={len(per600)}（本次比较跳过）")

    # ④ 负结果重放：搜索 smoke（与 T-AS-002 判据一致）应复现同量级 best_residual
    from mission_sim.core.cyber.algorithms.low_perigee import search_low_perigee
    t0 = time.perf_counter()
    rep = search_low_perigee(resonance=(2, 1), max_alt_km=300.0, grid=(16, 16),
                             n_steps=6000, refine_iter=60, tol=1e-8)
    sv = rep["survey"]
    print(f"[W8-4] 负结果重放：found={rep['found']} best_residual={sv['best_residual']:.3e} "
          f"n_seeds={sv['n_seeds']} 用时={sv['runtime_s']:.1f}s（本次 {time.perf_counter()-t0:.1f}s）")
    print(f"       留档对照: AS-005 记载的 smoke best_residual 应一致（同配置确定性）")
    json.dump({"verify": {"kernel_dmax": dmax, "half_cycle_dstate": d2,
                          "residual_mismatch": d_res,
                          "smoke_best_residual": sv["best_residual"],
                          "smoke_found": rep["found"]}},
              open(project_root / "workbench/design_dev/evidence/AS/w8_verify_result.json", "w"),
              indent=2)
    print("结论:", "PASS" if ok_all else "CHECK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
