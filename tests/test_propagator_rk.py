"""
Tests for the Runge‑Kutta integrator family (RK45, DOP853, DP8(7))
located in mission_sim.utils.propagators.rk.

Uses a harmonic oscillator as a simple analytic test case,
plus a two‑body Kepler comparison against the batch Kepler solver.
"""

import numpy as np
import pytest
from numba import njit

from mission_sim.utils.propagators.rk import (
    integrate_rk45,
    integrate_dop853,
    integrate_dp8,
    integrate_dp8_batch,
)
from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch


# ---------------------------------------------------------------------------
# Oscillator helpers
# ---------------------------------------------------------------------------
@njit
def harmonic_f(t, y):
    """y = [x, v] ; d/dt [x, v] = [v, -x] (ω=1)"""
    return np.array([y[1], -y[0]])


def analytical_harmonic(t, x0=1.0, v0=0.0):
    """Return true [x, v] at time t."""
    x = x0 * np.cos(t) + v0 * np.sin(t)
    v = -x0 * np.sin(t) + v0 * np.cos(t)
    return np.array([x, v])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_output_shape():
    """Verify that the integrator returns correctly shaped arrays."""
    t0, t_end = 0.0, 1.0
    y0 = np.array([1.0, 0.0])
    t_arr, y_arr = integrate_rk45(harmonic_f, t0, y0, (t0, t_end))
    assert t_arr.ndim == 1
    assert y_arr.ndim == 2
    assert t_arr.shape[0] >= 2
    assert y_arr.shape[0] == t_arr.shape[0]
    assert y_arr.shape[1] == 2


def test_rk45_forward_harmonic():
    """RK45 propagates a harmonic oscillator to t=2π with good accuracy."""
    t0, tf = 0.0, 2.0 * np.pi
    y0 = np.array([1.0, 0.0])
    t_arr, y_arr = integrate_rk45(harmonic_f, t0, y0, (t0, tf))
    yf = y_arr[-1]
    true = analytical_harmonic(tf)
    err = np.abs(yf - true).max()
    assert err < 1e-6


def test_dop853_forward_harmonic():
    """DOP853 propagates a harmonic oscillator to t=2π with high accuracy."""
    t0, tf = 0.0, 2.0 * np.pi
    y0 = np.array([1.0, 0.0])
    t_arr, y_arr = integrate_dop853(harmonic_f, t0, y0, (t0, tf))
    yf = y_arr[-1]
    true = analytical_harmonic(tf)
    err = np.abs(yf - true).max()
    assert err < 1e-10


def test_dp8_forward_harmonic():
    """DP8(7) propagates a harmonic oscillator to t=2π with high accuracy."""
    t0, tf = 0.0, 2.0 * np.pi
    y0 = np.array([1.0, 0.0])
    t_arr, y_arr = integrate_dp8(harmonic_f, t0, y0, (t0, tf))
    yf = y_arr[-1]
    true = analytical_harmonic(tf)
    err = np.abs(yf - true).max()
    assert err < 1e-10


def test_rk45_backward():
    """Forward integration to 0.5 and backward to 0.0 recovers the initial state."""
    t0, t_mid, tf = 0.0, 0.5, 0.0
    y0 = np.array([1.0, 0.0])
    # forward
    _, y_mid_arr = integrate_rk45(harmonic_f, t0, y0, (t0, t_mid))
    y_mid = y_mid_arr[-1]
    # backward
    t_back, y_back_arr = integrate_rk45(harmonic_f, t_mid, y_mid, (t_mid, tf))
    y_back = y_back_arr[-1]
    err = np.abs(y_back - y0).max()
    assert err < 1e-9


def test_dp8_batch():
    """Batch integration of two oscillators with different initial conditions."""
    t0, tf = 0.0, np.pi
    y0_batch = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    t_batch, y_batch = integrate_dp8_batch(harmonic_f, t0, y0_batch, (t0, tf))
    assert len(t_batch) == 2
    assert len(y_batch) == 2
    for idx in range(2):
        t_arr = t_batch[idx]
        y_arr = y_batch[idx]
        assert t_arr.shape[0] >= 2
        assert y_arr.shape[1] == 2
        # final state close to analytic
        true = analytical_harmonic(tf, x0=y0_batch[idx, 0], v0=y0_batch[idx, 1])
        err = np.abs(y_arr[-1] - true).max()
        assert err < 1e-6


def test_dp8_kepler_precision():
    """
    Compare DP8 propagation of the two‑body problem with the analytic
    Kepler solver over one tenth of a period.
    """
    # Two‑body parameters (G = 1, m1=0.9, m2=0.1)
    m1, m2 = 0.9, 0.1
    mu_total = m1 + m2  # 1.0
    a = 1.0
    T = 2.0 * np.pi / np.sqrt(mu_total / a**3)  # period = 2π
    t_span = (0.0, 0.1 * T)

    # Centre of mass at rest at origin
    r_cm = np.zeros(3)
    v_cm = np.zeros(3)
    # relative state: body2 – body1
    r_rel = np.array([a, 0.0, 0.0])
    v_rel = np.array([0.0, np.sqrt(mu_total / a), 0.0])  # tangential
    total_mass = m1 + m2

    # initial individual states (inertial, same as used in analytical ephemeris test)
    r1 = r_cm - (m2 / total_mass) * r_rel
    v1 = v_cm - (m2 / total_mass) * v_rel
    r2 = r_cm + (m1 / total_mass) * r_rel
    v2 = v_cm + (m1 / total_mass) * v_rel
    Y0 = np.concatenate([r1, v1, r2, v2])  # 12‑component state

    # N‑body derivative for a pure two‑body system
    @njit
    def two_body_f(t, Y):
        r1 = Y[0:3]
        v1 = Y[3:6]
        r2 = Y[6:9]
        v2 = Y[9:12]
        dr = r2 - r1
        r = np.sqrt(np.sum(dr**2))
        acc = mu_total * dr / r**3
        dY = np.empty(12)
        dY[0:3] = v1
        dY[3:6] = (m2 / mu_total) * acc  # acc1 = mu2/M * a_rel
        dY[6:9] = v2
        dY[9:12] = -(m1 / mu_total) * acc  # acc2 = -mu1/M * a_rel
        return dY

    # Numerical integration
    t_arr, Y_arr = integrate_dp8(two_body_f, 0.0, Y0, t_span)
    Y_num = Y_arr[-1]

    # Analytic solution at final time using Kepler solver
    tf = t_span[1]
    # Mean anomaly at t=0 for circular orbit is 0
    M0 = 0.0
    n = np.sqrt(mu_total / a**3)
    Mf = M0 + n * tf
    # The Kepler solver expects arrays, so we wrap the scalar
    rel_state = kepler_elements_to_cartesian_batch(
        np.array([a]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([Mf]),
        mu_total,
    )[0]  # shape (6,)

    r_rel_ana = rel_state[:3]
    v_rel_ana = rel_state[3:]
    # CM remains at origin
    r1_ana = r_cm - (m2 / total_mass) * r_rel_ana
    v1_ana = v_cm - (m2 / total_mass) * v_rel_ana
    r2_ana = r_cm + (m1 / total_mass) * r_rel_ana
    v2_ana = v_cm + (m1 / total_mass) * v_rel_ana
    Y_ana = np.concatenate([r1_ana, v1_ana, r2_ana, v2_ana])

    err = np.abs(Y_num - Y_ana).max()
    assert err < 1e-6, f"DP8 Kepler error {err:.2e} exceeds 1e-6"
