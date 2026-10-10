# tests/lunar_swing/test_ls_strict.py
# -*- coding: utf-8 -*-
"""
LS-003 T-LS-001~008：LUNAR-SWING 专题严格化测试。

红基线：本文件先于实现落地并跑红留档（O-9 / RE-LS-001 R-LS-4）。
判据标注"（校准）"者以实测标定（RE-LS-001 R-LS-2）。
"""

import pathlib

import h5py
import numpy as np
import pytest

CHAR_PERIOD = 4.342 * 86400.0  # 地月 CRTBP 特征周期（秒）


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------
def _crtbp():
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    return UniversalCRTBP.earth_moon_system()


def _targeter(crtbp=None, num_steps=40000, **kw):
    from mission_sim.core.cyber.algorithms.lunar_swing_targeter import LunarSwingTargeter
    crtbp = crtbp or _crtbp()
    return LunarSwingTargeter(dynamics_model=crtbp, mu=crtbp.mu,
                              integrator_type="rk4", num_steps=num_steps, **kw)


@pytest.fixture(scope="module")
def solution_2to1():
    crtbp = _crtbp()
    tg = _targeter(crtbp)
    res = tg.find_symmetric_periodic_orbit(
        resonance_ratio=(2, 1), x0_guess=0.85, vy0_guess=0.95,
        tol=1e-10, max_iter=50, char_period=CHAR_PERIOD)
    return tg, res


# ---------------------------------------------------------------------------
# T-LS-001：对称求解器 2:1 严格闭合 + 变分矩阵核对
# ---------------------------------------------------------------------------
def test_t_ls_001_symmetric_2to1_full_closure(solution_2to1):
    _, res = solution_2to1
    assert res.get("success"), f"2:1 未收敛：{res.get('error', res.get('half_period_residual'))}"
    assert res["half_period_residual"] <= 1e-8, f"半周期截面残差 {res['half_period_residual']:.2e}"
    assert res["full_state_residual"] <= 1e-8, f"全状态闭合残差 {res['full_state_residual']:.2e}"
    assert res["jacobi_drift"] <= 1e-10, f"Jacobi 漂移 {res['jacobi_drift']:.2e}"
    assert res["iterations"] <= 50


def test_t_ls_001_variational_matrix_check():
    crtbp = _crtbp()
    rng = np.random.default_rng(20261010)
    ok = 0
    while ok < 10:
        x = np.concatenate([rng.uniform(-0.8, 0.8, 3), rng.uniform(-1.0, 1.0, 3)])
        r1 = np.linalg.norm(x[:2] + [crtbp.mu, 0.0]) if False else np.linalg.norm(x[:3] - np.array([-crtbp.mu, 0, 0]))
        r2 = np.linalg.norm(x[:3] - np.array([1 - crtbp.mu, 0, 0]))
        if min(r1, r2) < 0.05:
            continue
        ok += 1
        A = crtbp.variational_matrix_nd(x)
        assert A.shape == (6, 6)
        h = 1e-7
        for j in range(3):
            dp = np.zeros(6); dp[j] = h
            col = (crtbp._crtbp_acceleration_nd(x + dp) - crtbp._crtbp_acceleration_nd(x - dp)) / (2 * h)
            assert np.allclose(A[3:6, j], col, rtol=1e-5, atol=2e-7), f"列 {j} 不匹配"
        assert np.allclose(A[0:3, 3:6], np.eye(3))
        assert np.allclose(A[3:6, 3:6], np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]]))


# ---------------------------------------------------------------------------
# T-LS-002：对称求解器 3:2（争取项；负结果按 RE-LS-001 R-LS-1 记录/变更单）
# ---------------------------------------------------------------------------
def test_t_ls_002_symmetric_3to2():
    tg = _targeter()
    ok = None
    for x0, vy0 in [(0.80, 0.95), (0.75, 1.00), (0.85, 1.05)]:
        res = tg.find_symmetric_periodic_orbit(
            resonance_ratio=(3, 2), x0_guess=x0, vy0_guess=vy0,
            tol=1e-8, max_iter=60, char_period=CHAR_PERIOD)
        if res.get("success"):
            ok = res
            break
    assert ok is not None, "3:2 未在任何候选初值下收敛（须按 R-LS-1 记录负结果并走变更单）"
    assert ok["full_state_residual"] <= 1e-8
    assert ok["jacobi_drift"] <= 1e-10


# ---------------------------------------------------------------------------
# T-LS-003：初值系统化扫描（≥3 收敛初值，无人工特调）
# ---------------------------------------------------------------------------
def test_t_ls_003_scan_multiple_solutions():
    tg = _targeter()
    # CR-LS-01（待 RE-LS-002 复核）：原网格 vy∈[0.85,1.05] 不含速度闭合族，
    # 首轮红；按实测族域（粗扫确认 vy≈0.42–0.60）重新系统扫描，判据不变。
    xs = np.linspace(0.78, 0.86, 5)
    vys = np.linspace(0.44, 0.60, 5)
    found = tg.scan_resonant_orbits(
        resonance_ratio=(2, 1), x0_grid=xs, vy0_grid=vys,
        tol=1e-6, max_iter=20, num_steps=3000,
        polish_tol=1e-9, polish_num_steps=40000, char_period=CHAR_PERIOD,
        max_polish=6)
    assert len(found) >= 3, f"扫描仅找到 {len(found)} 个收敛初值"
    for r in found:
        assert r["full_state_residual"] <= 1e-8


# ---------------------------------------------------------------------------
# T-LS-004：ThirdBodyGravity 单元
# ---------------------------------------------------------------------------
def test_t_ls_004_third_body_gravity():
    from mission_sim.core.physics.models.third_body import ThirdBodyGravity
    from mission_sim.core.physics.environment import CelestialEnvironment
    from mission_sim.core.physics.models.central_gravity import CentralGravity
    from mission_sim.core.spacetime.ids import CoordinateFrame

    mu3 = 4.9048695e12  # 月球 GM
    r3 = np.array([3.844e8, 0.0, 0.0]); v3 = np.array([0.0, 1.022e3, 0.0])
    model = ThirdBodyGravity(mu_third=mu3,
                             position_provider=lambda t: np.concatenate([r3 + v3 * t, v3]))
    state = np.array([7.0e6, 1.0e6, 0.0, 0.0, 7.5e3, 0.0])

    a = np.asarray(model.compute_accel(state, 0.0))
    d = r3 - state[:3]
    a_ref = mu3 * (d / np.linalg.norm(d) ** 3 - r3 / np.linalg.norm(r3) ** 3)
    assert np.allclose(a, a_ref, rtol=1e-12, atol=1e-16), "直接+间接项与解析式不符"

    model0 = ThirdBodyGravity(mu_third=0.0, position_provider=lambda t: np.concatenate([r3, v3]))
    assert np.allclose(np.asarray(model0.compute_accel(state, 0.0)), 0.0), "零质量极限非零"

    env = CelestialEnvironment(CoordinateFrame.J2000_ECI, verbose=False)
    env.register_force(CentralGravity())
    env.register_force(model)
    acc, _ = env.get_total_acceleration(state, CoordinateFrame.J2000_ECI)
    a_central = -3.986004418e14 * state[:3] / np.linalg.norm(state[:3]) ** 3
    assert np.allclose(acc, a_central + a, rtol=1e-12, atol=1e-16), "环境求和异常"


# ---------------------------------------------------------------------------
# T-LS-005：带谐重力场 J2–J4（势函数 FD 梯度 + J2 一致性）
# ---------------------------------------------------------------------------
def test_t_ls_005_zonal_geopotential():
    from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
    from mission_sim.core.physics.models.j2_gravity import J2Gravity

    MU = 3.986004418e14; RE = 6378137.0
    J2, J3, J4 = 1.08262668e-3, -2.53215306e-6, -1.61098761e-6

    m_full = HighOrderGeopotential(mu_earth=MU, j2=J2, j3=J3, j4=J4, r_earth=RE)
    m2only = HighOrderGeopotential(mu_earth=MU, j2=J2, j3=0.0, j4=0.0, r_earth=RE)
    m2ref = J2Gravity(mu_earth=MU)

    rng = np.random.default_rng(20261010)
    for _ in range(10):
        r = rng.normal(size=3); r = r / np.linalg.norm(r) * rng.uniform(6.6e6, 5.0e7)
        state = np.concatenate([r, rng.normal(0.0, 100.0, 3)])

        # J2 一致性：J3=J4=0 时与 J2Gravity 一致
        assert np.allclose(np.asarray(m2only.compute_accel(state, 0.0)),
                           np.asarray(m2ref.compute_accel(state, 0.0)),
                           rtol=1e-14, atol=1e-18), "J2 极限不一致"

        # 势函数 FD 梯度（独立校验闭式加速度）
        def V(rr):
            # 摄动势（不含中心项，与实现同口径，a = -∇V）：
            # V_pert = -(μ/r)[J2(Re/r)²P2 + J3(Re/r)³P3 + J4(Re/r)⁴P4]
            rn = np.linalg.norm(rr); u = rr[2] / rn
            P2 = 0.5 * (3 * u * u - 1)
            P3 = 0.5 * (5 * u ** 3 - 3 * u)
            P4 = (35 * u ** 4 - 30 * u ** 2 + 3) / 8.0
            return -(MU / rn) * (J2 * (RE / rn) ** 2 * P2
                                 + J3 * (RE / rn) ** 3 * P3
                                 + J4 * (RE / rn) ** 4 * P4)

        h = 10.0
        a_fd = np.zeros(3)
        for j in range(3):
            dp = np.zeros(3); dp[j] = h
            a_fd[j] = -(V(r + dp) - V(r - dp)) / (2 * h)
        a_impl = np.asarray(m_full.compute_accel(state, 0.0))
        rel = np.linalg.norm(a_impl - a_fd) / max(np.linalg.norm(a_fd), 1e-30)
        assert rel < 1e-5, f"J2–J4 与势梯度不符（rel={rel:.2e}）"


# ---------------------------------------------------------------------------
# T-LS-006：场景冒烟 + CRTBP→ECI 转换自检
# ---------------------------------------------------------------------------
def test_t_ls_006_scenario_smoke_and_conversion(tmp_path):
    from mission_sim.simulation.threebody.earth_moon import (
        EarthMoonResonanceSimulation, crtbp_state_to_eci)

    # 转换自检（正/逆映射闭环；CR-LS-03：计入质心→地心偏移 μ·L）
    L = 3.844e8
    omega = 2.0 * np.pi / (27.321661 * 86400.0)
    mu = 0.01215
    moon_r = np.array([3.844e8, 0.0, 0.0])
    moon_v = np.array([0.0, 1.022e3, 0.0])
    state_nd = np.array([0.8, 0.05, 0.0, 0.0, 0.9, 0.0])
    s_eci = crtbp_state_to_eci(state_nd, moon_r, moon_v, L, omega, mu)
    assert np.isclose(s_eci[0], L * (state_nd[0] + mu), rtol=0.0, atol=1e-6), "未计入质心→地心偏移"

    d_hat = moon_r / np.linalg.norm(moon_r)
    h_hat = np.cross(moon_r, moon_v); h_hat = h_hat / np.linalg.norm(h_hat)
    y_hat = np.cross(h_hat, d_hat)
    B = np.vstack([d_hat, y_hat, h_hat])
    r_nd_back = B @ (s_eci[:3] - mu * L * d_hat) / L
    omega_vec = omega * h_hat
    v_nd_back = B @ (s_eci[3:6] - np.cross(omega_vec, s_eci[:3])) / (L * omega)
    assert np.allclose(r_nd_back, state_nd[:3], atol=1e-12)
    assert np.allclose(v_nd_back, state_nd[3:], atol=1e-12)

    # 场景冒烟（确定性玩具月球，离线可复现）
    def moon_provider(t):
        th = omega * t
        r = L * np.array([np.cos(th), np.sin(th), 0.0])
        v = L * omega * np.array([-np.sin(th), np.cos(th), 0.0])
        return np.concatenate([r, v])

    init = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    sim = EarthMoonResonanceSimulation(dict(
        mission_name="LS_smoke", simulation_days=0.5, time_step=600.0,
        data_dir=str(tmp_path), initial_state=init.tolist(),
        moon_provider=moon_provider, enable_moon=True, enable_j2=True,
        enable_drag=False, verbose=False))
    assert sim.run()
    with h5py.File(sim.h5_file, "r") as f:
        for k in ["epochs", "states", "altitude_m", "moon_distance_m"]:
            assert k in f and f[k].shape[0] > 1, f"数据集缺失：{k}"
        r = np.linalg.norm(f["states"][:, :3], axis=1)
    assert np.all(np.isfinite(r)) and r.max() < 2.0e9


# ---------------------------------------------------------------------------
# T-LS-007：跨积分器一致性（校准）
# ---------------------------------------------------------------------------
def test_t_ls_007_cross_integrator_consistency(solution_2to1):
    from scipy.integrate import solve_ivp

    tg, res = solution_2to1
    assert res.get("success")
    x0 = np.asarray(res["state"])
    T_nd = res["period"] / CHAR_PERIOD

    dyn = tg._get_dynamics_func()
    xa, _ = tg._stm_calc.propagate_with_stm(
        dynamics=dyn, initial_state=x0, t0=0.0, tf=T_nd, method="rk4", num_steps=40000)

    sol = solve_ivp(lambda t, x: dyn(t, x), (0.0, T_nd), x0,
                    method="DOP853", rtol=1e-11, atol=1e-13)
    assert sol.success
    xb = sol.y[:, -1]

    r_a = float(np.linalg.norm(xa - x0))
    r_b = float(np.linalg.norm(xb - x0))
    assert r_a <= 1e-6, f"RK4 闭合残差 {r_a:.2e} 超校准门限"
    assert r_b <= 1e-6, f"DOP853 闭合残差 {r_b:.2e} 超校准门限"
    assert np.linalg.norm(xa - xb) <= 1e-6, "两积分器结果互差超限"


# ---------------------------------------------------------------------------
# T-LS-008：严格验收脚本 v2（无容忍后门；脚本运行见实验证据）
# ---------------------------------------------------------------------------
def test_t_ls_008_strict_acceptance_script_no_backdoor():
    script = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "lunar_swing" / "acceptance_strict.py"
    assert script.exists(), "验收脚本 v2 不存在"
    text = script.read_text(encoding="utf-8")
    for banned in ["放宽到 0.1", "临时方案", "Accepting due to", "based on shooting"]:
        assert banned not in text, f"验收脚本 v2 残留容忍分支：{banned}"


# ---------------------------------------------------------------------------
# T-LS-009：场景力模型历元同步（CR-LS-02；红→绿回归）
# ---------------------------------------------------------------------------
def test_t_ls_009_scenario_force_epoch_sync(tmp_path):
    from mission_sim.simulation.threebody.earth_moon import EarthMoonResonanceSimulation

    def moon_provider(t):
        # 快速线性移动的玩具月球：力对历元强敏感（防"冻结历元"缺陷回归）
        r = np.array([3.0e8 + 1.0e6 * t, 1.0e8, 0.0])
        return np.concatenate([r, np.array([1.0e6, 0.0, 0.0])])

    MU_E, MU_M = 3.986004418e14, 4.9048695e12
    init = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    sim = EarthMoonResonanceSimulation(dict(
        mission_name="LS_epoch", simulation_days=0.01, time_step=60.0,
        data_dir=str(tmp_path), initial_state=init.tolist(),
        moon_provider=moon_provider, enable_moon=True, enable_j2=False,
        enable_drag=False, verbose=False))
    accel = sim._build_accel()

    for t in [100.0, 1000.0]:
        a = np.asarray(accel(t, init), dtype=np.float64)
        r3 = moon_provider(t)[:3]
        d = r3 - init[:3]
        a_ref = (-MU_E * init[:3] / np.linalg.norm(init[:3]) ** 3
                 + MU_M * (d / np.linalg.norm(d) ** 3 - r3 / np.linalg.norm(r3) ** 3))
        assert np.allclose(a, a_ref, rtol=1e-9, atol=1e-12), f"t={t} 处时间相关力未按当前历元求值"
