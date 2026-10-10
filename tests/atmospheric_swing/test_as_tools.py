# tests/atmospheric_swing/test_as_tools.py
# -*- coding: utf-8 -*-
"""
AS-003 T-AS-006/007/008：能量账本、敏感度工具、验收脚本。
红基线：先于实现落地。
"""
import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# T-AS-006：能量账本闭合性（校准）
# ---------------------------------------------------------------------------
def test_t_as_006_energy_ledger_closure():
    from scipy.integrate import solve_ivp

    from mission_sim.analysis.energy_ledger import ledger_closure

    mu = 3.986004418e14
    r0 = 7.0e6
    v0 = np.sqrt(mu / r0)
    state0 = np.array([r0, 0.0, 0.0, 0.0, v0 * 0.9, 0.0])
    a_t = 1e-4

    def rhs(t, y):
        r = np.linalg.norm(y[:3])
        v = y[3:6]
        a_g = -mu * y[:3] / r ** 3
        a_th = a_t * v / np.linalg.norm(v)
        return np.concatenate([v, a_g + a_th])

    t_end = 2.0 * np.pi * np.sqrt(r0 ** 3 / mu) * 0.8
    sol = solve_ivp(rhs, (0, t_end), state0, rtol=1e-12, atol=1e-9,
                    dense_output=True, method="DOP853")
    ts = np.linspace(0, t_end, 40001)
    xs = sol.sol(ts).T

    def parts_all(t, state):
        r = np.linalg.norm(state[:3])
        v = state[3:6]
        return {"central": -mu * state[:3] / r ** 3,
                "tangential": a_t * v / np.linalg.norm(v)}

    # ① 动能口径（含全部力）闭合（校准：实测 ~8.5e-9 @rtol 1e-12，门限 1e-7 = 12× 裕度）
    out_all = ledger_closure(ts, xs, parts_all, mu=mu)
    assert out_all["residual_kin_rel"] <= 1e-7, out_all

    # ② 机械能口径（账本 = 仅扰动力做功）：ΔE_mech = W_扰动
    def parts_pert(t, state):
        v = state[3:6]
        return {"tangential": a_t * v / np.linalg.norm(v)}

    out = ledger_closure(ts, xs, parts_pert, mu=mu)
    assert out["residual_rel"] <= 1e-7, out


# ---------------------------------------------------------------------------
# T-AS-007：敏感度工具（区间结论口径）
# ---------------------------------------------------------------------------
def test_t_as_007_sensitivity_sweep():
    from mission_sim.analysis.sensitivity import parameter_sweep

    def run_fn(p):
        return {"metric": 1.0 + 2.0 * p["density_scale"]}

    out = parameter_sweep(run_fn, parameter="density_scale",
                          values=[0.5, 0.75, 1.0, 1.25, 1.5])
    assert out["n_runs"] == 5
    curves = [r["metric"] for r in out["results"]]
    assert curves == sorted(curves)
    assert abs(out["interval"]["min"] - 2.0) < 1e-12
    assert abs(out["interval"]["max"] - 4.0) < 1e-12


# ---------------------------------------------------------------------------
# T-AS-008：AS 验收脚本存在且无容忍后门（静态）
# ---------------------------------------------------------------------------
def test_t_as_008_acceptance_script_no_backdoor():
    script = (pathlib.Path(__file__).resolve().parents[2]
              / "scripts" / "atmospheric_swing" / "acceptance_as.py")
    assert script.exists(), "AS 验收脚本不存在"
    text = script.read_text(encoding="utf-8")
    for banned in ["放宽", "临时方案", "Accepting due to", "取最佳", "fallback_pass"]:
        assert banned not in text, f"验收脚本残留容忍分支：{banned}"
