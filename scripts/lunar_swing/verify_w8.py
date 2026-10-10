#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNAR-SWING 独立验证补充（W8 证据）
====================================
① 步长敏感性：固定步长 RK4 单周期闭合残差随步数变化（预期 h⁴ 收敛）；
② 多积分器：RK4 / DOP853 / Radau 单周期闭合残差与互差（T-LS-007 校准表数据源）。

用法: python scripts/lunar_swing/verify_w8.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

CHAR_PERIOD = 4.342 * 86400.0
STATE_2TO1 = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
T_ND = 0.5 * 27.321661 * 86400.0 / CHAR_PERIOD


def main() -> int:
    crtbp = UniversalCRTBP.earth_moon_system()
    tg = LunarSwingTargeter(dynamics_model=crtbp, mu=crtbp.mu,
                            integrator_type="rk4", num_steps=40000)
    dyn = tg._get_dynamics_func()
    x0 = STATE_2TO1

    print("== ① 步长敏感性（RK4，单周期全状态闭合残差） ==")
    for n in [5000, 10000, 20000, 40000, 80000]:
        xa, _ = tg._stm_calc.propagate_with_stm(
            dynamics=dyn, initial_state=x0, t0=0.0, tf=T_ND, method="rk4", num_steps=n)
        print(f"   num_steps={n:6d}  closure={np.linalg.norm(xa - x0):.3e}")

    print("== ② 多积分器单周期闭合（x0 = 2:1 严格解） ==")
    from scipy.integrate import solve_ivp
    finals = {}
    xa, _ = tg._stm_calc.propagate_with_stm(
        dynamics=dyn, initial_state=x0, t0=0.0, tf=T_ND, method="rk4", num_steps=40000)
    finals["RK4-40000"] = xa
    for name in ["DOP853", "Radau"]:
        sol = solve_ivp(lambda t, x: dyn(t, x), (0.0, T_ND), x0,
                        method=name, rtol=1e-11, atol=1e-13)
        assert sol.success, f"{name} 积分失败"
        finals[name] = sol.y[:, -1]
    for name, xf in finals.items():
        print(f"   {name:10s} closure={np.linalg.norm(xf - x0):.3e}")
    names = list(finals)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(finals[names[i]] - finals[names[j]])
            print(f"   互差 {names[i]} vs {names[j]}: {d:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
