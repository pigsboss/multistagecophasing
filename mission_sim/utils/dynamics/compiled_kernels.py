# mission_sim/utils/dynamics/compiled_kernels.py
# -*- coding: utf-8 -*-
"""
CRTBP 编译内核（numba fp64）与 Numpy 参考回退
==============================================
AS-CR-01 / G5⑦ 沉淀件（Apple Silicon 异构加速批处理求解内核）：

  · 与 numpy 参考实现逐位一致（RHS / 解析雅可比 / RK4 / 42 维增广）；
  · 单轨迹加速 ~25×（42 维）、多核批量加速 ~7×（12 核实测）；
  · 变步长自适应 RK4（近地点细分；供低近地点轨道使用）；
  · numba 不可用时自动回退 numpy 参考实现（API 不变）。

接口约定：全部无量纲（DU/TU），state = [x, y, z, vx, vy, vz]。
"""
import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def compiled_available() -> bool:
    """numba 编译内核是否可用。"""
    return _HAS_NUMBA


# ---------------------------------------------------------------------------
# Numpy 参考（回退路径；与编译版本同式）
# ---------------------------------------------------------------------------
def _rhs_np(x, mu):
    xa, ya, za, vx, vy, vz = (float(x[0]), float(x[1]), float(x[2]),
                              float(x[3]), float(x[4]), float(x[5]))
    r1 = np.sqrt((xa + mu) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    r2 = np.sqrt((xa + mu - 1.0) ** 2 + ya ** 2 + za ** 2 + 1e-15)
    ax = 2 * vy + xa - (1 - mu) * (xa + mu) / r1 ** 3 - mu * (xa + mu - 1) / r2 ** 3
    ay = -2 * vx + ya - (1 - mu) * ya / r1 ** 3 - mu * ya / r2 ** 3
    az = -(1 - mu) * za / r1 ** 3 - mu * za / r2 ** 3
    return np.array([vx, vy, vz, ax, ay, az])


def _jac_np(x, mu):
    xa, ya, za = float(x[0]), float(x[1]), float(x[2])
    dx1 = xa + mu
    dx2 = xa + mu - 1.0
    r1sq = dx1 * dx1 + ya * ya + za * za + 1e-15
    r2sq = dx2 * dx2 + ya * ya + za * za + 1e-15
    r1_3 = r1sq ** 1.5
    r1_5 = r1sq ** 2.5
    r2_3 = r2sq ** 1.5
    r2_5 = r2sq ** 2.5
    m1, m2 = 1.0 - mu, mu
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    uxx = (1.0 - m1 / r1_3 + 3.0 * m1 * dx1 * dx1 / r1_5
           - m2 / r2_3 + 3.0 * m2 * dx2 * dx2 / r2_5)
    uyy = (1.0 - m1 / r1_3 + 3.0 * m1 * ya * ya / r1_5
           - m2 / r2_3 + 3.0 * m2 * ya * ya / r2_5)
    uzz = (-m1 / r1_3 + 3.0 * m1 * za * za / r1_5
           - m2 / r2_3 + 3.0 * m2 * za * za / r2_5)
    uxy = 3.0 * m1 * dx1 * ya / r1_5 + 3.0 * m2 * dx2 * ya / r2_5
    uxz = 3.0 * m1 * dx1 * za / r1_5 + 3.0 * m2 * dx2 * za / r2_5
    uyz = 3.0 * m1 * ya * za / r1_5 + 3.0 * m2 * ya * za / r2_5
    a[3, 0], a[3, 1], a[3, 2] = uxx, uxy, uxz
    a[4, 0], a[4, 1], a[4, 2] = uxy, uyy, uyz
    a[5, 0], a[5, 1], a[5, 2] = uxz, uyz, uzz
    a[3, 4] = 2.0
    a[4, 3] = -2.0
    return a


def _rk4_state_np(x0, T, n, mu):
    h = T / n
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(n):
        f1 = _rhs_np(x, mu)
        f2 = _rhs_np(x + 0.5 * h * f1, mu)
        f3 = _rhs_np(x + 0.5 * h * f2, mu)
        f4 = _rhs_np(x + h * f3, mu)
        x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    return x


def _aug_deriv_np(y, mu):
    x = y[:6]
    phi = y[6:].reshape(6, 6)
    d = np.empty(42)
    d[:6] = _rhs_np(x, mu)
    d[6:] = (_jac_np(x, mu) @ phi).reshape(36)
    return d


def _rk4_aug_np(x0, T, n, mu):
    h = T / n
    y = np.zeros(42)
    y[:6] = x0
    y[6:] = np.eye(6).reshape(36)
    for _ in range(n):
        k1 = _aug_deriv_np(y, mu)
        k2 = _aug_deriv_np(y + 0.5 * h * k1, mu)
        k3 = _aug_deriv_np(y + 0.5 * h * k2, mu)
        k4 = _aug_deriv_np(y + h * k3, mu)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _rk4_adaptive_np(x0, T, mu, h_max, h_min, scale):
    t = 0.0
    x = np.asarray(x0, dtype=np.float64).copy()
    n = 0
    while t < T - 1e-15:
        r = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
        h = scale * r ** 1.5
        h = min(max(h, h_min), h_max)
        if t + h > T:
            h = T - t
        f1 = _rhs_np(x, mu)
        f2 = _rhs_np(x + 0.5 * h * f1, mu)
        f3 = _rhs_np(x + 0.5 * h * f2, mu)
        f4 = _rhs_np(x + h * f3, mu)
        x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        t += h
        n += 1
    return x, n


# ---------------------------------------------------------------------------
# numba 编译版本
# ---------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(cache=True)
    def _rhs_nb(x, mu):
        xa, ya, za, vx, vy, vz = x
        r1 = np.sqrt((xa + mu) ** 2 + ya ** 2 + za ** 2 + 1e-15)
        r2 = np.sqrt((xa + mu - 1.0) ** 2 + ya ** 2 + za ** 2 + 1e-15)
        ax = 2 * vy + xa - (1 - mu) * (xa + mu) / r1 ** 3 - mu * (xa + mu - 1) / r2 ** 3
        ay = -2 * vx + ya - (1 - mu) * ya / r1 ** 3 - mu * ya / r2 ** 3
        az = -(1 - mu) * za / r1 ** 3 - mu * za / r2 ** 3
        return np.array([vx, vy, vz, ax, ay, az])

    @njit(cache=True)
    def _jac_nb(x, mu):
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
        a = np.zeros((6, 6))
        a[0, 3] = a[1, 4] = a[2, 5] = 1.0
        uxx = (1.0 - m1 / r1_3 + 3.0 * m1 * dx1 * dx1 / r1_5
               - m2 / r2_3 + 3.0 * m2 * dx2 * dx2 / r2_5)
        uyy = (1.0 - m1 / r1_3 + 3.0 * m1 * ya * ya / r1_5
               - m2 / r2_3 + 3.0 * m2 * ya * ya / r2_5)
        uzz = (-m1 / r1_3 + 3.0 * m1 * za * za / r1_5
               - m2 / r2_3 + 3.0 * m2 * za * za / r2_5)
        uxy = 3.0 * m1 * dx1 * ya / r1_5 + 3.0 * m2 * dx2 * ya / r2_5
        uxz = 3.0 * m1 * dx1 * za / r1_5 + 3.0 * m2 * dx2 * za / r2_5
        uyz = 3.0 * m1 * ya * za / r1_5 + 3.0 * m2 * ya * za / r2_5
        a[3, 0], a[3, 1], a[3, 2] = uxx, uxy, uxz
        a[4, 0], a[4, 1], a[4, 2] = uxy, uyy, uyz
        a[5, 0], a[5, 1], a[5, 2] = uxz, uyz, uzz
        a[3, 4] = 2.0
        a[4, 3] = -2.0
        return a

    @njit(cache=True)
    def _rk4_state_nb(x0, T, n, mu):
        h = T / n
        x = x0.copy()
        for _ in range(n):
            f1 = _rhs_nb(x, mu)
            f2 = _rhs_nb(x + 0.5 * h * f1, mu)
            f3 = _rhs_nb(x + 0.5 * h * f2, mu)
            f4 = _rhs_nb(x + h * f3, mu)
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        return x

    @njit(cache=True)
    def _aug_deriv_nb(y, mu):
        x = y[:6]
        phi = y[6:].reshape(6, 6)
        d = np.empty(42)
        d[:6] = _rhs_nb(x, mu)
        d[6:] = (_jac_nb(x, mu) @ phi).reshape(36)
        return d

    @njit(cache=True)
    def _rk4_aug_nb(x0, T, n, mu):
        h = T / n
        y = np.zeros(42)
        y[:6] = x0
        y[6:] = np.eye(6).reshape(36)
        for _ in range(n):
            k1 = _aug_deriv_nb(y, mu)
            k2 = _aug_deriv_nb(y + 0.5 * h * k1, mu)
            k3 = _aug_deriv_nb(y + 0.5 * h * k2, mu)
            k4 = _aug_deriv_nb(y + h * k3, mu)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    @njit(parallel=True, cache=True)
    def _rk4_batch_nb(seeds, T, n, mu):
        m = seeds.shape[0]
        out = np.empty((m, 6))
        for i in prange(m):
            out[i] = _rk4_state_nb(seeds[i], T, n, mu)
        return out

    @njit(parallel=True, cache=True)
    def _residual_batch_nb(seeds, T, n, mu):
        m = seeds.shape[0]
        out = np.empty((m, 2))
        for i in prange(m):
            x = _rk4_state_nb(seeds[i], T, n, mu)
            out[i, 0] = x[1]
            out[i, 1] = x[3]
        return out

    @njit(parallel=True, cache=True)
    def _residual_extent_batch_nb(seeds, T, n, mu):
        """半周期残差 + 地心半径延展 [y, xdot, rmin, rmax, r_end]（筛选用）。"""
        m = seeds.shape[0]
        out = np.empty((m, 5))
        h = T / n
        for i in prange(m):
            x = seeds[i].copy()
            rmin = 1.0e9
            rmax = 0.0
            for _ in range(n):
                rg = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
                if rg < rmin:
                    rmin = rg
                if rg > rmax:
                    rmax = rg
                f1 = _rhs_nb(x, mu)
                f2 = _rhs_nb(x + 0.5 * h * f1, mu)
                f3 = _rhs_nb(x + 0.5 * h * f2, mu)
                f4 = _rhs_nb(x + h * f3, mu)
                x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
            rg = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
            if rg < rmin:
                rmin = rg
            if rg > rmax:
                rmax = rg
            out[i, 0] = x[1]
            out[i, 1] = x[3]
            out[i, 2] = rmin
            out[i, 3] = rmax
            out[i, 4] = rg
        return out

    @njit(cache=True)
    def _rk4_adaptive_nb(x0, T, mu, h_max, h_min, scale):
        t = 0.0
        x = x0.copy()
        n = 0
        while t < T - 1e-15:
            r = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
            h = scale * r ** 1.5
            if h > h_max:
                h = h_max
            if h < h_min:
                h = h_min
            if t + h > T:
                h = T - t
            f1 = _rhs_nb(x, mu)
            f2 = _rhs_nb(x + 0.5 * h * f1, mu)
            f3 = _rhs_nb(x + 0.5 * h * f2, mu)
            f4 = _rhs_nb(x + h * f3, mu)
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
            t += h
            n += 1
        return x, n


# ---------------------------------------------------------------------------
# 公共 API（numpy 输入输出；按可用性自动选择路径）
# ---------------------------------------------------------------------------
def rhs_crtbp(x, mu):
    x = np.asarray(x, dtype=np.float64)
    if _HAS_NUMBA:
        return np.asarray(_rhs_nb(x, float(mu)))
    return _rhs_np(x, float(mu))


def jac_crtbp(x, mu):
    x = np.asarray(x, dtype=np.float64)
    if _HAS_NUMBA:
        return np.asarray(_jac_nb(x, float(mu)))
    return _jac_np(x, float(mu))


def rk4_propagate(x0, T, n_steps, mu):
    x0 = np.asarray(x0, dtype=np.float64)
    if _HAS_NUMBA:
        return np.asarray(_rk4_state_nb(x0, float(T), int(n_steps), float(mu)))
    return _rk4_state_np(x0, float(T), int(n_steps), float(mu))


def rk4_propagate_aug(x0, T, n_steps, mu):
    """返回 (state, Phi)。"""
    x0 = np.asarray(x0, dtype=np.float64)
    if _HAS_NUMBA:
        y = np.asarray(_rk4_aug_nb(x0, float(T), int(n_steps), float(mu)))
    else:
        y = _rk4_aug_np(x0, float(T), int(n_steps), float(mu))
    return y[:6].copy(), y[6:].reshape(6, 6).copy()


def rk4_propagate_batch(seeds, T, n_steps, mu):
    seeds = np.asarray(seeds, dtype=np.float64)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    if _HAS_NUMBA:
        return np.asarray(_rk4_batch_nb(seeds, float(T), int(n_steps), float(mu)))
    return np.array([_rk4_state_np(s, float(T), int(n_steps), float(mu)) for s in seeds])


def batch_half_period_residual(seeds, T, n_steps, mu):
    """批量半周期截面残差 [y, ẋ]（对称判据）。"""
    seeds = np.asarray(seeds, dtype=np.float64)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    if _HAS_NUMBA:
        return np.asarray(_residual_batch_nb(seeds, float(T), int(n_steps), float(mu)))
    out = np.empty((seeds.shape[0], 2))
    for i, s in enumerate(seeds):
        x = _rk4_state_np(s, float(T), int(n_steps), float(mu))
        out[i] = [x[1], x[3]]
    return out


def batch_half_period_residual_extent(seeds, T, n_steps, mu):
    """批量半周期残差 + 地心半径延展 [y, ẋ, r_min, r_max, r_end]（DU）。"""
    seeds = np.asarray(seeds, dtype=np.float64)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    if _HAS_NUMBA:
        return np.asarray(_residual_extent_batch_nb(seeds, float(T), int(n_steps), float(mu)))
    out = np.empty((seeds.shape[0], 5))
    h = float(T) / int(n_steps)
    for i, s in enumerate(seeds):
        x = np.asarray(s, dtype=np.float64).copy()
        rmins, rmaxs = 1e9, 0.0
        for _ in range(int(n_steps)):
            rg = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
            rmins = min(rmins, rg)
            rmaxs = max(rmaxs, rg)
            f1 = _rhs_np(x, mu)
            f2 = _rhs_np(x + 0.5 * h * f1, mu)
            f3 = _rhs_np(x + 0.5 * h * f2, mu)
            f4 = _rhs_np(x + h * f3, mu)
            x = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        rg = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
        out[i] = [x[1], x[3], min(rmins, rg), max(rmaxs, rg), rg]
    return out


def rk4_propagate_adaptive(x0, T, mu, scale=2.0e-3, h_max=4.0e-3, h_min=4.0e-7):
    """变步长 RK4（h = clamp(scale·r_geo^1.5, h_min, h_max)）。

    返回 (state, n_used)。
    """
    x0 = np.asarray(x0, dtype=np.float64)
    if _HAS_NUMBA:
        x, n = _rk4_adaptive_nb(x0, float(T), float(mu), float(h_max),
                                float(h_min), float(scale))
        return np.asarray(x), int(n)
    return _rk4_adaptive_np(x0, float(T), float(mu), float(h_max),
                            float(h_min), float(scale))
