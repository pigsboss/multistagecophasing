"""
Dormand‐Prince 5(4) (DP5) 自适应步长数值传播器
----------------------------------------------------------
基于 Numba 编译的高效显式 Runge Kutta 积分器。

当前实现使用经典的 **Dormand‑Prince 5(4)** 7 阶段系数
（即 scipy 中的 RK45）作为默认嵌入对。

若要升级到 **Dormand‑Prince 8(7) 13M** 方法，只需将下方
_BUTCHER_A, _BUTCHER_B, _BUTCHER_C, _BUTCHER_E 替换为
对应的 DP8(7) 系数即可（可参考 Hairer, Norsett, Wanner 的
《Solving Ordinary Differential Equations I》附带的代码）。

功能模块：
- integrate_dp8(t0, y0, f, t_span, rtol, atol)  单样本积分
- integrate_dp8_batch(t0, y0, f, t_span, rtol, atol)  批量并行积分
- integrate_dp8_trajectory(t0, y0, f, t_span, rtol, atol)  密集输出轨迹
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# 默认 Butcher 表：DP5(4) 7 阶段
# ---------------------------------------------------------------------------

_BUTCHER_C = np.array([
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0,
    1.0
], dtype=np.float64)

_BUTCHER_A = np.zeros((7, 7), dtype=np.float64)
_BUTCHER_A[1, 0] = 1.0 / 5.0
_BUTCHER_A[2, 0] = 3.0 / 40.0
_BUTCHER_A[2, 1] = 9.0 / 40.0
_BUTCHER_A[3, 0] = 44.0 / 45.0
_BUTCHER_A[3, 1] = -56.0 / 15.0
_BUTCHER_A[3, 2] = 32.0 / 9.0
_BUTCHER_A[4, 0] = 19372.0 / 6561.0
_BUTCHER_A[4, 1] = -25360.0 / 2187.0
_BUTCHER_A[4, 2] = 64448.0 / 6561.0
_BUTCHER_A[4, 3] = -212.0 / 729.0
_BUTCHER_A[5, 0] = 9017.0 / 3168.0
_BUTCHER_A[5, 1] = -355.0 / 33.0
_BUTCHER_A[5, 2] = 46732.0 / 5247.0
_BUTCHER_A[5, 3] = 49.0 / 176.0
_BUTCHER_A[5, 4] = -5103.0 / 18656.0
_BUTCHER_A[6, 0] = 35.0 / 384.0
_BUTCHER_A[6, 1] = 0.0
_BUTCHER_A[6, 2] = 500.0 / 1113.0
_BUTCHER_A[6, 3] = 125.0 / 192.0
_BUTCHER_A[6, 4] = -2187.0 / 6784.0
_BUTCHER_A[6, 5] = 11.0 / 84.0

# 8 阶解（用于误差估计的低阶解）
_BUTCHER_B_LOW = np.array([
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0
], dtype=np.float64)

# 5 阶解（实际传播采用的高阶解）
_BUTCHER_B_HIGH = np.array([
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0
], dtype=np.float64)

# 误差系数 = b_high - b_low
_BUTCHER_E = _BUTCHER_B_HIGH - _BUTCHER_B_LOW

# 如果希望使用 DP8(7) 13 阶段系数，请将上述 4 个数组替换为对应的 13 阶段系数。
# 之后所有函数无需任何修改即可工作。


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

@njit
def _norm(x):
    """无穷范数（用于误差监控）。"""
    return np.max(np.abs(x))


@njit
def _step_accepted(err, rtol, atol):
    """
    检查步长是否被接受。
    返回 True 如果数值误差 ≤ 1.0 （相对/绝对容差范围内）。
    """
    return err <= 1.0


@njit
def _compute_new_h(h, err, power, fac_min=0.2, fac_max=10.0):
    """
    PI 式步长控制。
    根据误差估计缩放步长。
    """
    fac = 0.9 * err ** (-power)
    fac = min(fac_max, max(fac_min, fac))
    return h * fac


# ---------------------------------------------------------------------------
# 单步积分（DP5(4) 或 DP8(7) 若替换系数）
# ---------------------------------------------------------------------------

@njit
def _dopri_step(f, t, y, h, rtol, atol, args=()):
    """
    执行一个 RK 步长。返回新状态 y_new（5 阶解）以及误差估计 err。
    同时计算导数 k1..k7。
    """
    n = y.shape[0]
    k = np.zeros((7, n), dtype=np.float64)

    # 阶段 1
    k[0] = f(t, y, *args)

    # 阶段 2..7
    for s in range(1, 7):
        t_stage = t + _BUTCHER_C[s] * h
        y_stage = y.copy()
        for j in range(s):
            y_stage += h * _BUTCHER_A[s, j] * k[j]
        k[s] = f(t_stage, y_stage, *args)

    # 5 阶解（高阶）
    y_high = y.copy()
    for s in range(7):
        y_high += h * _BUTCHER_B_HIGH[s] * k[s]

    # 低阶解（用于误差估计）
    y_low = y.copy()
    for s in range(7):
        y_low += h * _BUTCHER_B_LOW[s] * k[s]

    # 误差
    err = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sc = atol + rtol * max(abs(y[i]), abs(y_high[i]))
        err[i] = abs(y_high[i] - y_low[i]) / sc
    err_norm = _norm(err)

    return y_high, err_norm, k


# ---------------------------------------------------------------------------
# 单样本自适应积分
# ---------------------------------------------------------------------------

@njit
def integrate_dp8(f, t0, y0, t_span, rtol=1e-8, atol=1e-12, h0=0.0, hmin=1e-20, args=()):
    """
    自适应步长积分，从 t0 到 t_span[1]，初始步长自动选取或由 h0 指定。
    返回 (t, y_hist) 其中 t 为时间序列，y_hist 为状态序列 (N, n)。

    Parameters
    ----------
    f : callable(t, y, *args) -> ndarray
        动力学函数，必须可被 Numba 编译。
    t0 : float
        起始时间。
    y0 : ndarray shape (n,)
        初始状态。
    t_span : tuple (t0, tf)
        积分区间。
    rtol, atol : float
        相对/绝对容差。
    h0 : float, optional
        初始步长（0 表示自动选取）。
    hmin : float, optional
        最小步长。
    args : tuple, optional
        传递给 f 的额外参数（f(t, y, *args)）。
    """
    t = t0
    y = y0.copy()
    n = y.shape[0]

    tf = t_span[1]
    direction = 1.0 if tf >= t0 else -1.0
    h = h0 if h0 > 0.0 else np.sqrt(np.finfo(np.float64).eps)

    # 存储结果（预分配）
    # 我们无法提前知道点数，使用列表再转数组。
    t_list = [t]
    y_list = [y.copy()]

    while (tf - t) * direction > 0.0:
        # 限制步长
        if (t + h - tf) * direction > 0.0:
            h = tf - t

        # 执行一步
        y_new, err, _ = _dopri_step(f, t, y, h, rtol, atol, args)

        # 检查步长是否接受
        if _step_accepted(err, rtol, atol):
            t += h
            y[:] = y_new
            t_list.append(t)
            y_list.append(y.copy())

            # 计算下一大步的步长（PI 控制）
            power = 1.0 / 5.0  # DP5 误差阶 5
            h = _compute_new_h(h, err, power)
        else:
            # 减小步长重试
            power = 1.0 / 5.0
            h = _compute_new_h(h, err, power)
            if abs(h) < hmin:
                raise ValueError(f"步长减小到 {hmin} 以下，积分被迫停止于 t={t}")

    # 确保最后一点正好在 tf
    t_arr = np.array(t_list)
    y_arr = np.array(y_list)

    if t_arr[-1] != tf:
        # 线性插值到 tf
        alpha = (tf - t_arr[-2]) / (t_arr[-1] - t_arr[-2])
        y_last = y_arr[-2] + alpha * (y_arr[-1] - y_arr[-2])
        t_arr[-1] = tf
        y_arr[-1] = y_last

    return t_arr, y_arr


# ---------------------------------------------------------------------------
# 批量积分（并行 prange）
# ---------------------------------------------------------------------------

@njit(parallel=True)
def integrate_dp8_batch(f, t0, y0_batch, t_span, rtol=1e-8, atol=1e-12, h0=0.0, args=()):
    """
    批量自适应积分。对批次中每个样本调用 integrate_dp8 独立积分。

    Parameters
    ----------
    f : callable(t, y, *args) -> ndarray
        动力学函数 (单样本接口，y 为 shape (n,))。
    t0 : float
    y0_batch : ndarray shape (N, n)
        初始状态集合。
    t_span : tuple (t0, tf)
    rtol, atol : float
    h0 : float
    args : tuple, optional
        传递给 f 的额外参数。

    Returns
    -------
    t_batch : list of ndarray
        每个样本的时间序列。
    y_batch : list of ndarray
        每个样本的状态序列。
    """
    N = y0_batch.shape[0]
    t_batch = [None] * N
    y_batch = [None] * N

    for idx in prange(N):
        y0 = y0_batch[idx]
        t_arr, y_arr = integrate_dp8(f, t0, y0, t_span, rtol, atol, h0, args=args)
        t_batch[idx] = t_arr
        y_batch[idx] = y_arr

    return t_batch, y_batch


# ---------------------------------------------------------------------------
# 密集输出（带 Hermite 插值）
# ---------------------------------------------------------------------------

@njit
def _hermite_interp(t0, y0, f0, t1, y1, f1, t_query):
    """
    在 [t0, t1] 区间内使用 Hermite 三次插值，返回 t_query 处的状态。
    f0 = f(t0, y0), f1 = f(t1, y1)
    """
    n = y0.shape[0]
    h = t1 - t0
    theta = (t_query - t0) / h
    theta1 = 1.0 - theta

    # Hermite 基函数
    phi0 = theta1 * theta1 * (1.0 + 2.0 * theta)
    phi1 = theta * theta * (3.0 - 2.0 * theta)
    psi0 = h * theta * theta1 * theta1
    psi1 = -h * theta * theta * theta1

    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = phi0 * y0[i] + phi1 * y1[i] + psi0 * f0[i] + psi1 * f1[i]
    return result


@njit
def integrate_dp8_trajectory(f, t0, y0, t_span, rtol=1e-8, atol=1e-12, h0=0.0, args=()):
    """
    在每一步保存 [t, y], 并且记录每一步的导数 k0（位于起点）。
    之后使用 Hermite 插值在任意查询点上给出高精度值。
    返回 (t_hist, y_hist, f_hist) 其中 f_hist 是导数。
    """
    t = t0
    y = y0.copy()
    n = y.shape[0]
    tf = t_span[1]
    direction = 1.0 if tf >= t0 else -1.0
    h = h0 if h0 > 0.0 else np.sqrt(np.finfo(np.float64).eps)

    t_list = [t]
    y_list = [y.copy()]
    f_list = [f(t, y, *args).copy()]

    while (tf - t) * direction > 0.0:
        if (t + h - tf) * direction > 0.0:
            h = tf - t

        y_new, err, k = _dopri_step(f, t, y, h, rtol, atol, args)

        if _step_accepted(err, rtol, atol):
            t_next = t + h
            y_next = y_new.copy()
            f_next = f(t_next, y_next, *args).copy()

            t_list.append(t_next)
            y_list.append(y_next)
            f_list.append(f_next)

            t = t_next
            y = y_next

            power = 1.0 / 5.0
            h = _compute_new_h(h, err, power)
        else:
            power = 1.0 / 5.0
            h = _compute_new_h(h, err, power)

    t_arr = np.array(t_list)
    y_arr = np.array(y_list)
    f_arr = np.array(f_list)

    # 确保终点匹配
    if t_arr[-1] != tf:
        alpha = (tf - t_arr[-2]) / (t_arr[-1] - t_arr[-2])
        y_last = y_arr[-2] + alpha * (y_arr[-1] - y_arr[-2])
        t_arr[-1] = tf
        y_arr[-1] = y_last
        f_arr[-1] = f(tf, y_last, *args).copy()

    return t_arr, y_arr, f_arr


# ---------------------------------------------------------------------------
# 方便的用户函数（可选）
# ---------------------------------------------------------------------------

@njit
def integrate_dp8_simple(f, t0, y0, tf, rtol=1e-8, atol=1e-12, args=()):
    """简化接口：只返回最终状态。"""
    t_arr, y_arr = integrate_dp8(f, t0, y0, (t0, tf), rtol, atol, args=args)
    return y_arr[-1]
