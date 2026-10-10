#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLX/Metal fp32 批量筛选层基准（AS-CR-01 证据）
===============================================
用法: python3 tools/bench/bench_mlx_crtbp.py
（mlx 为可选依赖；未安装时本脚本安全退出，对应"缺省降级"路径）
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    print("mlx 未安装：筛选层缺省降级（本脚本跳过）")
    sys.exit(0)

MU = np.float32(0.012144731)
T_HALF = 1.5731034659749784
X0 = np.array([0.8087346541, 0.0, 0.0, 0.0, 0.5158236131, 0.0], dtype=np.float32)


def rhs_mlx(x):
    xa, ya, za, vx, vy, vz = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    muf = mx.array(MU)
    r1 = mx.sqrt((xa + muf) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    r2 = mx.sqrt((xa + muf - 1.0) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    ax = 2 * vy + xa - (1 - muf) * (xa + muf) / r1 ** 3 - muf * (xa + muf - 1.0) / r2 ** 3
    ay = -2 * vx + ya - (1 - muf) * ya / r1 ** 3 - muf * ya / r2 ** 3
    az = -(1 - muf) * za / r1 ** 3 - muf * za / r2 ** 3
    return mx.stack([vx, vy, vz, ax, ay, az], axis=1)


def main() -> int:
    print("mlx", mx.__version__, "| default device:", mx.default_device())
    try:
        info = mx.device_info()
        print("device:", info.get("device_name"), "| memory(GB):",
              round(info.get("memory_size", 0) / 2 ** 30, 1))
    except Exception as e:  # noqa: BLE001
        print("device_info 不可用:", e)

    try:
        mx.eval(mx.sqrt(mx.array([1.0], dtype=mx.float64)))
        print("fp64 on GPU: 支持")
    except Exception as e:  # noqa: BLE001
        print(f"fp64 on GPU: 不支持（{type(e).__name__}: {str(e)[:60]}）")

    rng = np.random.default_rng(0)
    N = 1_000_000
    X = (rng.normal(size=(N, 6)) * 0.3).astype(np.float32)
    X[:, :3] += np.array([0.55, 0.0, 0.0], dtype=np.float32)
    x = mx.array(X)
    for _ in range(3):
        mx.eval(rhs_mlx(x))
    t0 = time.perf_counter()
    for _ in range(20):
        mx.eval(rhs_mlx(x))
    t = (time.perf_counter() - t0) / 20
    print(f"批量 RHS N={N}: {t * 1e3:.2f} ms/次 → {N / t / 1e6:.0f} M 状态/s")

    def run(K, steps, chunk):
        h = T_HALF / steps
        S = np.tile(X0, (K, 1)) + (rng.normal(size=(K, 6)) * 1e-3).astype(np.float32)

        @mx.compile
        def step(state):
            for _ in range(chunk):
                k1 = rhs_mlx(state)
                k2 = rhs_mlx(state + 0.5 * h * k1)
                k3 = rhs_mlx(state + 0.5 * h * k2)
                k4 = rhs_mlx(state + h * k3)
                state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return state

        st = mx.array(S)
        st = step(st)
        mx.eval(st)
        t0 = time.perf_counter()
        for _ in range(steps // chunk):
            st = step(st)
        mx.eval(st)
        dt = time.perf_counter() - t0
        print(f"批量 RK4: K={K} x {steps} 步 (chunk={chunk}): {dt:.3f}s → "
              f"{K * steps / dt / 1e6:.1f} M seed·步/s")

    run(65536, 2000, 292)
    run(262144, 2000, 292)

    from numba import njit
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    c = UniversalCRTBP.earth_moon_system()
    mu = c.mu

    @njit
    def rhs_nb(s, mu_):
        xa, ya, za, vx, vy, vz = s
        r1 = np.sqrt((xa + mu_) ** 2 + ya ** 2 + za ** 2 + 1e-15)
        r2 = np.sqrt((xa + mu_ - 1.0) ** 2 + ya ** 2 + za ** 2 + 1e-15)
        ax = 2 * vy + xa - (1 - mu_) * (xa + mu_) / r1 ** 3 - mu_ * (xa + mu_ - 1) / r2 ** 3
        ay = -2 * vx + ya - (1 - mu_) * ya / r1 ** 3 - mu_ * ya / r2 ** 3
        az = -(1 - mu_) * za / r1 ** 3 - mu_ * za / r2 ** 3
        return np.array([vx, vy, vz, ax, ay, az])

    @njit
    def rk4_nb(x0, T, n, mu_):
        h = T / n
        xx = x0.copy()
        for _ in range(n):
            f1 = rhs_nb(xx, mu_)
            f2 = rhs_nb(xx + 0.5 * h * f1, mu_)
            f3 = rhs_nb(xx + 0.5 * h * f2, mu_)
            f4 = rhs_nb(xx + h * f3, mu_)
            xx = xx + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return xx

    n = 2000
    h = T_HALF / n

    @mx.compile
    def step1(state):
        k1 = rhs_mlx(state)
        k2 = rhs_mlx(state + 0.5 * h * k1)
        k3 = rhs_mlx(state + 0.5 * h * k2)
        k4 = rhs_mlx(state + h * k3)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    st = mx.array(np.tile(X0, (1, 1)))
    st = step1(st)
    mx.eval(st)
    for _ in range(n - 1):
        st = step1(st)
    mx.eval(st)
    x32 = np.asarray(st)[0]
    x64 = rk4_nb(X0.astype(np.float64), T_HALF, n, mu)
    dr = float(np.abs(x32[:3] - x64[:3]).max())
    print(f"fp32 半周期 {n} 步 vs fp64: |Δr|max={dr:.2e} DU (≈{dr * 3.844e8 / 1e3:.2f} km)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
