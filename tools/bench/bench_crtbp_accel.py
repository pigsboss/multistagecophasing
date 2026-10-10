#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRTBP 求解内核加速基准（候选沉淀工具原型）
==========================================
对比：现网 numpy 逐步实现 vs Numba 编译内核（单轨迹 fp64 + 多核批量）。
用法: python3 tools/bench/bench_crtbp_accel.py
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
from numba import njit, prange, get_num_threads

from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP

CRTBP = UniversalCRTBP.earth_moon_system()
MU = CRTBP.mu
X0 = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
T_HALF = 1.5731034659749784  # 2:1 周期之半（TU）
N_RHS = 100_000
N_STEPS = 40_000
N_BATCH = 16


@njit(cache=True)
def rhs_nb(x, mu):
    xa, ya, za, vx, vy, vz = x
    r1sq = (xa + mu) ** 2 + ya ** 2 + za ** 2
    r2sq = (xa + mu - 1.0) ** 2 + ya ** 2 + za ** 2
    r1 = np.sqrt(r1sq + 1e-15)
    r2 = np.sqrt(r2sq + 1e-15)
    ax = 2 * vy + xa - (1 - mu) * (xa + mu) / r1 ** 3 - mu * (xa + mu - 1) / r2 ** 3
    ay = -2 * vx + ya - (1 - mu) * ya / r1 ** 3 - mu * ya / r2 ** 3
    az = -(1 - mu) * za / r1 ** 3 - mu * za / r2 ** 3
    return np.array([vx, vy, vz, ax, ay, az])


@njit(cache=True)
def jac_nb(x, mu):
    xa, ya, za = x[0], x[1], x[2]
    dx1 = xa + mu
    dx2 = xa + mu - 1.0
    r1sq = dx1 * dx1 + ya * ya + za * za + 1e-15
    r2sq = dx2 * dx2 + ya * ya + za * za + 1e-15
    r1_3 = r1sq ** 1.5
    r1_5 = r1sq ** 2.5
    r2_3 = r2sq ** 1.5
    r2_5 = r2sq ** 2.5
    m1, m2 = 1.0 - mu, mu
    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    Uxx = (1.0 - m1 / r1_3 + 3.0 * m1 * dx1 * dx1 / r1_5 - m2 / r2_3 + 3.0 * m2 * dx2 * dx2 / r2_5)
    Uyy = (1.0 - m1 / r1_3 + 3.0 * m1 * ya * ya / r1_5 - m2 / r2_3 + 3.0 * m2 * ya * ya / r2_5)
    Uzz = (-m1 / r1_3 + 3.0 * m1 * za * za / r1_5 - m2 / r2_3 + 3.0 * m2 * za * za / r2_5)
    Uxy = 3.0 * m1 * dx1 * ya / r1_5 + 3.0 * m2 * dx2 * ya / r2_5
    Uxz = 3.0 * m1 * dx1 * za / r1_5 + 3.0 * m2 * dx2 * za / r2_5
    Uyz = 3.0 * m1 * ya * za / r1_5 + 3.0 * m2 * ya * za / r2_5
    A[3, 0], A[3, 1], A[3, 2] = Uxx, Uxy, Uxz
    A[4, 0], A[4, 1], A[4, 2] = Uxy, Uyy, Uyz
    A[5, 0], A[5, 1], A[5, 2] = Uxz, Uyz, Uzz
    A[3, 4] = 2.0
    A[4, 3] = -2.0
    return A


@njit(cache=True)
def rk4_state_nb(x0, T, n, mu):
    h = T / n
    x = x0.copy()
    for _ in range(n):
        f1 = rhs_nb(x, mu)
        f2 = rhs_nb(x + 0.5 * h * f1, mu)
        f3 = rhs_nb(x + 0.5 * h * f2, mu)
        f4 = rhs_nb(x + h * f3, mu)
        x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return x


@njit(cache=True)
def aug_deriv(y, mu):
    x = y[:6]
    phi = y[6:].reshape(6, 6)
    d = np.empty(42)
    d[:6] = rhs_nb(x, mu)
    d[6:] = (jac_nb(x, mu) @ phi).reshape(36)
    return d


@njit(cache=True)
def rk4_aug_nb(x0, T, n, mu):
    h = T / n
    y = np.zeros(42)
    y[:6] = x0
    y[6:] = np.eye(6).reshape(36)
    for _ in range(n):
        k1 = aug_deriv(y, mu)
        k2 = aug_deriv(y + 0.5 * h * k1, mu)
        k3 = aug_deriv(y + 0.5 * h * k2, mu)
        k4 = aug_deriv(y + h * k3, mu)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


@njit(parallel=True, cache=True)
def rk4_batch_par(seeds, T, n, mu):
    m = seeds.shape[0]
    out = np.empty((m, 6))
    for i in prange(m):
        out[i] = rk4_state_nb(seeds[i], T, n, mu)
    return out


def aug_deriv_np(y, c):
    x = y[:6]
    phi = y[6:].reshape(6, 6)
    d = np.empty(42)
    d[:6] = np.concatenate([x[3:], c._crtbp_acceleration_nd(x)])
    d[6:] = (c.variational_matrix_nd(x) @ phi).ravel()
    return d


def rk4_aug_np(x0, T, n, c):
    h = T / n
    y = np.zeros(42)
    y[:6] = x0
    y[6:] = np.eye(6).ravel()
    for _ in range(n):
        k1 = aug_deriv_np(y, c)
        k2 = aug_deriv_np(y + 0.5 * h * k1, c)
        k3 = aug_deriv_np(y + 0.5 * h * k2, c)
        k4 = aug_deriv_np(y + h * k3, c)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def bench(fn, *args, repeat=1):
    fn(*args)  # 预热（含 numba 编译）
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    dt = (time.perf_counter() - t0) / repeat
    return dt, out


def main() -> int:
    mu = MU
    print(f"numba threads = {get_num_threads()}")

    # 等价性
    d_rhs = np.abs(rhs_nb(X0, mu) - np.concatenate([X0[3:], CRTBP._crtbp_acceleration_nd(X0)])).max()
    d_jac = np.abs(jac_nb(X0, mu) - CRTBP.variational_matrix_nd(X0)).max()
    xa_nb = rk4_state_nb(X0, T_HALF, 2000, mu)
    # numpy 状态版
    def rk4_state_np(x0, T, n, c):
        h = T / n
        x = x0.copy()
        for _ in range(n):
            f1 = np.concatenate([x[3:], c._crtbp_acceleration_nd(x)])
            f2 = np.concatenate([(x + 0.5 * h * f1)[3:], c._crtbp_acceleration_nd(x + 0.5 * h * f1)])
            f3 = np.concatenate([(x + 0.5 * h * f2)[3:], c._crtbp_acceleration_nd(x + 0.5 * h * f2)])
            f4 = np.concatenate([(x + h * f3)[3:], c._crtbp_acceleration_nd(x + h * f3)])
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x
    d_rk = np.abs(xa_nb - rk4_state_np(X0, T_HALF, 2000, CRTBP)).max()
    ya_nb = rk4_aug_nb(X0, T_HALF, 500, mu)
    ya_np = rk4_aug_np(X0, T_HALF, 500, CRTBP)
    d_aug = np.abs(ya_nb - ya_np).max()
    print(f"等价性: RHS {d_rhs:.2e} | Jac {d_jac:.2e} | RK4(2000步) {d_rk:.2e} | RK4-aug(500步) {d_aug:.2e}")

    # RHS 单次成本
    t_np, _ = bench(lambda: [np.concatenate([X0[3:], CRTBP._crtbp_acceleration_nd(X0)]) for _ in range(N_RHS)])
    t_nb, _ = bench(lambda: [rhs_nb(X0, mu) for _ in range(N_RHS)])
    print(f"RHS x{N_RHS}: numpy {t_np:.3f}s | numba {t_nb:.3f}s | 加速 {t_np/t_nb:.1f}x")

    # 单轨迹：state 40000 步
    t_np, _ = bench(rk4_state_np, X0, T_HALF, N_STEPS, CRTBP)
    t_nb, _ = bench(rk4_state_nb, X0, T_HALF, N_STEPS, mu)
    print(f"RK4-state {N_STEPS} 步: numpy {t_np:.3f}s | numba {t_nb:.3f}s | 加速 {t_np/t_nb:.1f}x")

    # 单轨迹：aug(42d) 40000 步
    t_np, _ = bench(rk4_aug_np, X0, T_HALF, N_STEPS, CRTBP)
    t_nb, _ = bench(rk4_aug_nb, X0, T_HALF, N_STEPS, mu)
    print(f"RK4-aug42 {N_STEPS} 步: numpy {t_np:.3f}s | numba {t_nb:.3f}s | 加速 {t_np/t_nb:.1f}x")

    # 批量（16 种子）并行
    rng = np.random.default_rng(42)
    seeds = np.tile(X0, (N_BATCH, 1)) + rng.normal(0, 1e-3, (N_BATCH, 6))
    t_ser, out_ser = bench(rk4_batch_par, seeds, T_HALF, N_STEPS, mu)  # numba parallel
    # 串行基线：单线程 numba × N
    import numba
    numba.set_num_threads(1)
    rk4_batch_par(seeds, T_HALF, 100, mu)
    t0 = time.perf_counter()
    for i in range(N_BATCH):
        rk4_state_nb(seeds[i], T_HALF, N_STEPS, mu)
    t_1th = time.perf_counter() - t0
    numba.set_num_threads(get_num_threads())
    d_chk = np.abs(out_ser[0] - rk4_state_nb(seeds[0], T_HALF, N_STEPS, mu)).max()
    print(f"批量 {N_BATCH} 种子 x{N_STEPS} 步: 单线程 {t_1th:.3f}s | 多核prange {t_ser:.3f}s | 加速 {t_1th/t_ser:.1f}x (一致性 {d_chk:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
