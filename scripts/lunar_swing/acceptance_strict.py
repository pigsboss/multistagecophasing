#!/usr/bin/env python3
"""
LUNAR-SWING 严格验收脚本 v2（判据无后门）
==========================================
LS-001 G5-⑤ / T-LS-008。全部判据必须 PASS 才算通过；不接受任何放宽或"取最佳结果"。

验收项：
  1. 对称 2:1 周期解收敛（tol=1e-8）
  2. 全状态闭合残差 ≤ 1e-8；半周期垂直截面残差 ≤ 1e-8
  3. Jacobi 相对漂移 ≤ 1e-10
  4. 初值扫描（网格）严格容差下 ≥3 个收敛初值
  5. 跨积分器复核（RK4 与 DOP853）：闭合残差 ≤1e-6 且互差 ≤1e-6

用法: python scripts/lunar_swing/acceptance_strict.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

CHAR_PERIOD = 4.342 * 86400.0
CLOSURE_TOL = 1e-8
JACOBI_TOL = 1e-10
CROSS_TOL = 1e-6
SCAN_MIN = 3


def main() -> int:
    print("=" * 70)
    print("LUNAR-SWING Strict Acceptance (v2)")
    print("=" * 70)

    crtbp = UniversalCRTBP.earth_moon_system()
    tg = LunarSwingTargeter(dynamics_model=crtbp, mu=crtbp.mu,
                            integrator_type="rk4", num_steps=40000)

    checks = {}

    # ---- 1-3：对称 2:1 严格解（内部求解容差收紧至 1e-10，为判据留裕度） ----
    res = tg.find_symmetric_periodic_orbit(
        resonance_ratio=(2, 1), x0_guess=0.85, vy0_guess=0.95,
        tol=1e-10, max_iter=50, char_period=CHAR_PERIOD, verbose=True)
    checks["对称 2:1 收敛"] = bool(res.get("success"))
    print(f"   [1] 收敛迭代: {res.get('iterations')}")
    checks["半周期截面残差≤1e-8"] = res.get("half_period_residual", 1.0) <= CLOSURE_TOL
    print(f"   [2] 半周期残差: {res.get('half_period_residual'):.3e}")
    checks["全状态闭合≤1e-8"] = res.get("full_state_residual", 1.0) <= CLOSURE_TOL
    print(f"   [3] 全状态闭合: {res.get('full_state_residual'):.3e}")
    checks["Jacobi 漂移≤1e-10"] = res.get("jacobi_drift", 1.0) <= JACOBI_TOL
    print(f"   [4] Jacobi 漂移: {res.get('jacobi_drift'):.3e}")

    # ---- 4：初值扫描（CR-LS-01：网格按实测族域定位；判据不变） ----
    found = tg.scan_resonant_orbits(
        resonance_ratio=(2, 1),
        x0_grid=np.linspace(0.78, 0.86, 5), vy0_grid=np.linspace(0.44, 0.60, 5),
        tol=1e-6, max_iter=20, num_steps=3000,
        polish_tol=1e-9, polish_num_steps=40000, char_period=CHAR_PERIOD,
        max_polish=6)
    checks[f"扫描收敛初值≥{SCAN_MIN}"] = len(found) >= SCAN_MIN
    print(f"   [5] 扫描收敛: {len(found)} 个初值")

    # ---- 5：跨积分器复核 ----
    if checks["对称 2:1 收敛"]:
        from scipy.integrate import solve_ivp
        x0 = np.asarray(res["state"])
        T_nd = res["period"] / CHAR_PERIOD
        dyn = tg._get_dynamics_func()
        xa, _ = tg._stm_calc.propagate_with_stm(
            dynamics=dyn, initial_state=x0, t0=0.0, tf=T_nd, method="rk4", num_steps=40000)
        sol = solve_ivp(lambda t, x: dyn(t, x), (0.0, T_nd), x0,
                        method="DOP853", rtol=1e-11, atol=1e-13)
        r_a = float(np.linalg.norm(xa - x0))
        r_b = float(np.linalg.norm(sol.y[:, -1] - x0))
        cross = float(np.linalg.norm(xa - sol.y[:, -1]))
        checks["跨积分器（RK4/DOP853）"] = (r_a <= CROSS_TOL and r_b <= CROSS_TOL and cross <= CROSS_TOL)
        print(f"   [6] 跨积分器: RK4={r_a:.3e}, DOP853={r_b:.3e}, 互差={cross:.3e}")
    else:
        checks["跨积分器（RK4/DOP853）"] = False

    # ---- 汇总 ----
    print("-" * 70)
    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"   {'✅' if passed else '❌'} {name}")
    print("=" * 70)
    print("Overall: " + ("🎉 PASSED" if ok else "❌ FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
