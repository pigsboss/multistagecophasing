# tests/atmospheric_swing/test_as_solver.py
# -*- coding: utf-8 -*-
"""
AS-003 T-AS-001～005、009、010：求解策略、事件与加速内核。
红基线：本文件先于实现落地并跑红留档（AS-002 W1 / O-9 延续）。
判据标注"（校准）"者按实测标定（R-LS-2 口径）。
"""
import time

import numpy as np

CHAR_PERIOD = 4.342 * 86400.0
T_MOON = 27.321661 * 86400.0
T_HALF_2TO1 = 0.5 * T_MOON / CHAR_PERIOD
X_SOL_2TO1 = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])


# ---------------------------------------------------------------------------
# T-AS-001：连续法工具（合成族：解析可核）
# ---------------------------------------------------------------------------
def test_t_as_001_continuation_family():
    from mission_sim.core.cyber.algorithms.family_continuation import continue_family

    def residual(x, lam):
        return np.array([x[0] ** 2 - lam, x[1] - 2.0 * x[0]])

    def solve(x_guess, lam):
        x = np.asarray(x_guess, dtype=np.float64).copy()
        for _ in range(80):
            f = residual(x, lam)
            if np.linalg.norm(f) < 1e-13:
                break
            j = np.array([[2.0 * x[0], 0.0], [-2.0, 1.0]])
            x = x - np.linalg.solve(j, f)
        return x

    path = continue_family(residual, solve, x_initial=np.array([0.5, 1.0]),
                           final_value=4.0, param0=0.25, step=0.5,
                           tol=1e-10, max_steps=200)
    assert len(path) >= 4, f"延续路径过短：{len(path)}"
    lam_last, x_last = path[-1]
    assert abs(lam_last - 4.0) < 1e-9
    assert np.allclose(residual(x_last, lam_last), 0.0, atol=1e-9)
    lams = [p[0] for p in path]
    assert all(b >= a - 1e-12 for a, b in zip(lams[:-1], lams[1:])), lams


# ---------------------------------------------------------------------------
# T-AS-002：低近地点 2:1 严格解（负结果路径按 AS-001 G1 留档）
# ---------------------------------------------------------------------------
def test_t_as_002_low_perigee_2to1():
    from mission_sim.core.cyber.algorithms.low_perigee import search_low_perigee

    report = search_low_perigee(resonance=(2, 1), max_alt_km=300.0,
                                grid=(16, 16), n_steps=6000,
                                refine_iter=60, tol=1e-8)
    survey = report["survey"]
    assert survey["n_seeds"] >= 16 * 16, survey
    assert isinstance(survey["best_residual"], float)
    if report["found"]:
        sol = report["solutions"][0]
        assert sol["full_state_residual"] <= 1e-8, sol
        assert sol["perigee_alt_km"] <= 301.0, sol
        assert sol["jacobi_drift"] <= 1e-10, sol
    else:
        assert survey["refine_attempts"] >= survey["n_candidates"], survey


# ---------------------------------------------------------------------------
# T-AS-003：低近地点 3:2（争取项，不阻塞）
# ---------------------------------------------------------------------------
def test_t_as_003_low_perigee_3to2():
    from mission_sim.core.cyber.algorithms.low_perigee import search_low_perigee

    report = search_low_perigee(resonance=(3, 2), max_alt_km=300.0,
                                grid=(12, 12), n_steps=6000,
                                refine_iter=60, tol=1e-8)
    if report["found"]:
        sol = report["solutions"][0]
        assert sol["full_state_residual"] <= 1e-8, sol
        assert sol["perigee_alt_km"] <= 301.0, sol


# ---------------------------------------------------------------------------
# T-AS-004：分段/自适应打靶一致性（校准）
# ---------------------------------------------------------------------------
def test_t_as_004_variable_step_consistency():
    from mission_sim.utils.dynamics.compiled_kernels import (
        rk4_propagate, rk4_propagate_adaptive)

    mu = 0.012144731
    x_ref = rk4_propagate(X_SOL_2TO1, T_HALF_2TO1, 400000, mu)
    x_var, n_used = rk4_propagate_adaptive(X_SOL_2TO1, T_HALF_2TO1, mu)
    assert np.linalg.norm(np.asarray(x_var) - np.asarray(x_ref)) <= 1e-6
    assert n_used <= 100000, f"自适应步数过多：{n_used}"


# ---------------------------------------------------------------------------
# T-AS-005：事件检测（近地点 / 大气进出 / 月球近距）
# ---------------------------------------------------------------------------
def test_t_as_005_events():
    from mission_sim.utils.dynamics import events

    mu = 3.986004418e14
    a = 1.0e7
    e = 0.5
    n = np.sqrt(mu / a ** 3)
    t = np.linspace(0.0, 2 * np.pi / n, 20001)
    p = a * (1 - e * e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(n * t / 2))
    r = p / (1 + e * np.cos(nu))
    states = np.zeros((t.size, 6))
    states[:, 0] = r
    per = events.perigee_events(t, states)
    assert len(per) >= 2
    t_per = per[0]["t"]
    assert min(abs(t_per - k * 2 * np.pi / n) for k in range(2)) < 1e-1

    p_orb = 2 * np.pi / n
    t2 = np.linspace(0.0, 2.0 * p_orb, 40001)
    h2 = 100e3 + 60e3 * np.sin(2 * np.pi * t2 / p_orb)
    crossings = events.threshold_crossings(t2, h2, 130e3)
    assert len(crossings) == 4
    assert [c["direction"] for c in crossings] == ["up", "down", "up", "down"]

    d_moon = np.abs(1.0e8 - 1.0e4 * t)
    app = events.approach_events(t, d_moon, 5.0e6)
    assert len(app) >= 1 and app[0]["min_distance"] <= 5.0e6


# ---------------------------------------------------------------------------
# T-AS-009：加速内核一致性 / 筛选层 / 降级路径（CR-AS-01）
# ---------------------------------------------------------------------------
def test_t_as_009_compiled_kernel_identity():
    from mission_sim.utils.dynamics import compiled_kernels as ck
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

    c = UniversalCRTBP.earth_moon_system()
    mu = c.mu
    rng = np.random.default_rng(20261010)
    checked = 0
    while checked < 10:
        x = np.concatenate([rng.uniform(-0.8, 0.8, 3), rng.uniform(-1.0, 1.0, 3)])
        if (np.linalg.norm(x[:3] + [mu, 0, 0]) < 0.05
                or np.linalg.norm(x[:3] - [1 - mu, 0, 0]) < 0.05):
            continue
        checked += 1
        d = np.abs(np.asarray(ck.rhs_crtbp(x, mu))
                   - np.concatenate([x[3:], c._crtbp_acceleration_nd(x)])).max()
        assert d <= 1e-15, f"RHS 不一致 {d:.2e}"
        dj = np.abs(np.asarray(ck.jac_crtbp(x, mu)) - c.variational_matrix_nd(x)).max()
        assert dj <= 1e-15, f"Jac 不一致 {dj:.2e}"

    xa = ck.rk4_propagate(X_SOL_2TO1, T_HALF_2TO1, 40000, mu)
    xb = ck.rk4_propagate_batch(X_SOL_2TO1[None, :], T_HALF_2TO1, 40000, mu)[0]
    assert np.abs(np.asarray(xa) - np.asarray(xb)).max() == 0.0
    xs, phi = ck.rk4_propagate_aug(X_SOL_2TO1, T_HALF_2TO1, 20000, mu)
    x2 = ck.rk4_propagate(X_SOL_2TO1, T_HALF_2TO1, 20000, mu)
    assert np.abs(np.asarray(xs) - np.asarray(x2)).max() <= 1e-14
    assert np.asarray(phi).shape == (6, 6)


def test_t_as_009b_mlx_screening_and_degradation(monkeypatch):
    from mission_sim.utils.dynamics import mlx_screen

    assert isinstance(mlx_screen.screening_available(), bool)
    if mlx_screen.screening_available():
        seeds = (np.tile(X_SOL_2TO1, (64, 1))
                 + np.random.default_rng(3).normal(0, 1e-4, (64, 6)))
        res32 = mlx_screen.batch_half_period_residual_fp32(
            seeds, T_HALF_2TO1, 4000, 0.012144731)
        from mission_sim.utils.dynamics import compiled_kernels as ck
        res64 = ck.batch_half_period_residual(seeds, T_HALF_2TO1, 4000, 0.012144731)
        set32 = set(np.where(np.linalg.norm(res32, axis=1) < 1e-2)[0].tolist())
        set64 = set(np.where(np.linalg.norm(res64, axis=1) < 1e-2)[0].tolist())
        assert set64 <= set32, f"筛选层漏检：{sorted(set64 - set32)}"
    monkeypatch.setenv("MCPC_DISABLE_MLX", "1")
    assert mlx_screen.screening_available() is False


def test_t_as_010_acceleration_baseline(capsys):
    from mission_sim.utils.dynamics import compiled_kernels as ck
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

    c = UniversalCRTBP.earth_moon_system()
    n = 20000
    h = T_HALF_2TO1 / n

    def deriv_np(yy):
        xx = yy[:6]
        return np.concatenate(
            [np.concatenate([xx[3:], c._crtbp_acceleration_nd(xx)]),
             (c.variational_matrix_nd(xx) @ yy[6:].reshape(6, 6)).ravel()])

    t0 = time.perf_counter()
    y = np.zeros(42)
    y[:6] = X_SOL_2TO1
    y[6:] = np.eye(6).ravel()
    for _ in range(n):
        k1 = deriv_np(y)
        k2 = deriv_np(y + 0.5 * h * k1)
        k3 = deriv_np(y + 0.5 * h * k2)
        k4 = deriv_np(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    t_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    ck.rk4_propagate_aug(X_SOL_2TO1, T_HALF_2TO1, n, c.mu)
    t_nb = time.perf_counter() - t0
    speedup = t_np / max(t_nb, 1e-9)
    print(f"ACCEL_BASELINE numba_aug_speedup={speedup:.1f} (n={n})")
    assert speedup >= 3.0, f"加速比 {speedup:.1f}× 低于校准下限 3×"
