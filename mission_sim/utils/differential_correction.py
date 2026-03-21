# mission_sim/utils/differential_correction.py
"""
微分修正工具 (Differential Correction Utilities) - 增强版
新增基于状态转移矩阵的单参数和多参数修正函数，提高精度和收敛性。
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional, List


def compute_stm_numerical(
    dynamics: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    x0: np.ndarray,
    tf: float,
    eps: float = 1e-6,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = 'DOP853'
) -> np.ndarray:
    """
    数值计算状态转移矩阵 Φ(tf, t0)（差分法）。

    参数:
        dynamics: 动力学函数 f(t, x) -> dx/dt (n维)
        t0: 初始时间
        x0: 初始状态 (n维)
        tf: 最终时间
        eps: 扰动步长
        rtol, atol: 积分相对/绝对容差
        method: 积分器方法

    返回:
        Φ: 状态转移矩阵 (n x n)
    """
    n = len(x0)
    # 积分参考轨迹
    sol_ref = solve_ivp(dynamics, (t0, tf), x0, method=method, rtol=rtol, atol=atol)
    if not sol_ref.success:
        raise RuntimeError(f"Reference integration failed: {sol_ref.message}")
    xf_ref = sol_ref.y[:, -1]

    stm = np.zeros((n, n))
    for i in range(n):
        x0_pert = x0.copy()
        x0_pert[i] += eps
        sol_pert = solve_ivp(dynamics, (t0, tf), x0_pert, method=method, rtol=rtol, atol=atol)
        if not sol_pert.success:
            raise RuntimeError(f"Perturbed integration failed for index {i}: {sol_pert.message}")
        xf_pert = sol_pert.y[:, -1]
        stm[:, i] = (xf_pert - xf_ref) / eps
    return stm


def single_parameter_correction_with_stm(
    dynamics: Callable[[float, np.ndarray], np.ndarray],
    constraint: Callable[[np.ndarray], np.ndarray],
    t0: float,
    x0: np.ndarray,
    param_index: int,
    param_guess: float,
    tf: float,
    target: np.ndarray = None,
    max_iter: int = 20,
    tol: float = 1e-12,
    eps_perturb: float = 1e-6,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = 'DOP853'
) -> Tuple[float, np.ndarray, dict]:
    """
    基于状态转移矩阵的单参数微分修正。
    通过 STM 计算敏感性 ∂c/∂α = ∂c/∂xf * STM * e_param，更精确。

    参数：
        dynamics, constraint, t0, x0, param_index, param_guess, tf, target
        max_iter, tol, eps_perturb, rtol, atol, method
    返回：
        (param_opt, xf_opt, history)
    """
    if target is None:
        target = np.zeros_like(constraint(np.zeros(6)))

    history = {'alpha': [], 'c_norm': []}
    alpha = param_guess
    n = len(x0)
    e_param = np.zeros(n)
    e_param[param_index] = 1.0

    for it in range(max_iter):
        x0_curr = x0.copy()
        x0_curr[param_index] = alpha

        sol = solve_ivp(dynamics, (t0, tf), x0_curr, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed at iteration {it}: {sol.message}")
        xf = sol.y[:, -1]

        c = constraint(xf) - target
        c_norm = np.linalg.norm(c)
        history['alpha'].append(alpha)
        history['c_norm'].append(c_norm)

        if c_norm < tol:
            return alpha, xf, history

        stm = compute_stm_numerical(dynamics, t0, x0_curr, tf, eps=eps_perturb,
                                    rtol=rtol, atol=atol, method=method)

        # 约束雅可比 ∂c/∂xf
        m = len(c)
        j_c = np.zeros((m, n))
        eps_state = 1e-8
        for i in range(n):
            xf_pert = xf.copy()
            xf_pert[i] += eps_state
            c_pert = constraint(xf_pert) - target
            j_c[:, i] = (c_pert - c) / eps_state

        S = j_c @ (stm @ e_param)

        if np.linalg.norm(S) < 1e-12:
            print(f"Warning: Sensitivity near zero at iteration {it}, stopping.")
            break

        delta_alpha = -c / S
        alpha += delta_alpha[0]

    print(f"Single-parameter correction finished after {it+1} iterations, final residual = {c_norm:.2e}")
    return alpha, xf, history


def multi_parameter_correction(
    dynamics: Callable[[float, np.ndarray], np.ndarray],
    constraint: Callable[[np.ndarray], np.ndarray],
    t0: float,
    x0: np.ndarray,
    param_indices: List[int],
    param_guesses: List[float],
    tf: float,
    target: np.ndarray = None,
    max_iter: int = 20,
    tol: float = 1e-12,
    eps_perturb: float = 1e-6,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = 'DOP853'
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    多参数微分修正（通用）。
    同时修正多个初始状态参数，使约束满足目标值。

    参数:
        dynamics: 动力学函数
        constraint: 约束函数，输入终端状态，输出 m 维约束向量
        t0, x0, tf: 积分区间
        param_indices: 待修正参数在 x0 中的索引列表
        param_guesses: 参数初始猜测值列表（与 param_indices 对应）
        target: 期望的约束值（默认零向量）
        max_iter, tol: 迭代参数
        eps_perturb: 数值差分步长
        rtol, atol, method: 积分参数

    返回:
        (param_opt, xf_opt, history): 优化后的参数数组、终端状态、迭代历史
    """
    if target is None:
        # 通过一次积分猜测约束维数
        sol0 = solve_ivp(dynamics, (t0, tf), x0, method=method, rtol=rtol, atol=atol)
        if not sol0.success:
            raise RuntimeError(f"Initial integration failed: {sol0.message}")
        target = np.zeros_like(constraint(sol0.y[:, -1]))

    history = {'params': [], 'c_norm': []}
    alpha = np.array(param_guesses, dtype=float)
    n_params = len(param_indices)
    n = len(x0)

    for it in range(max_iter):
        # 构造当前初始状态
        x0_curr = x0.copy()
        for idx, val in zip(param_indices, alpha):
            x0_curr[idx] = val

        # 积分
        sol = solve_ivp(dynamics, (t0, tf), x0_curr, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed at iteration {it}: {sol.message}")
        xf = sol.y[:, -1]

        c = constraint(xf) - target
        c_norm = np.linalg.norm(c)
        history['params'].append(alpha.copy())
        history['c_norm'].append(c_norm)

        if c_norm < tol:
            return alpha, xf, history

        # 计算 STM
        stm = compute_stm_numerical(dynamics, t0, x0_curr, tf, eps=eps_perturb,
                                    rtol=rtol, atol=atol, method=method)

        # 计算约束雅可比 ∂c/∂xf
        m = len(c)
        j_c = np.zeros((m, n))
        eps_state = 1e-8
        for i in range(n):
            xf_pert = xf.copy()
            xf_pert[i] += eps_state
            c_pert = constraint(xf_pert) - target
            j_c[:, i] = (c_pert - c) / eps_state

        # 构造矩阵 E (n x n_params)，每列为 e_{param_indices[k]}
        E = np.zeros((n, n_params))
        for k, idx in enumerate(param_indices):
            E[idx, k] = 1.0

        # 敏感性矩阵 S = ∂c/∂α = j_c * STM * E
        S = j_c @ (stm @ E)

        # 解线性方程组 S * Δα = -c
        try:
            delta_alpha = np.linalg.lstsq(S, -c, rcond=None)[0]
        except np.linalg.LinAlgError:
            print(f"Warning: SVD failed at iteration {it}, using pseudo-inverse.")
            delta_alpha = np.linalg.pinv(S) @ (-c)

        alpha += delta_alpha

        if np.linalg.norm(delta_alpha) < 1e-12:
            break

    print(f"Multi-parameter correction finished after {it+1} iterations, final residual = {c_norm:.2e}")
    return alpha, xf, history


def compute_periodic_orbit(
    dynamics: Callable[[float, np.ndarray], np.ndarray],
    state0_guess: np.ndarray,
    param_index: int,
    constraint_func: Callable[[np.ndarray], float],
    tf_guess: float = None,
    max_iter: int = 20,
    tol: float = 1e-12,
    **kwargs
) -> Tuple[float, np.ndarray, dict]:
    """
    寻找周期轨道的微分修正专用函数（保留向后兼容）。
    """
    def constraint_wrapper(xf):
        return np.array([constraint_func(xf)])

    if tf_guess is None:
        raise ValueError("tf_guess must be provided for periodic orbit correction")

    return single_parameter_correction_with_stm(
        dynamics=dynamics,
        constraint=constraint_wrapper,
        t0=0.0,
        x0=state0_guess,
        param_index=param_index,
        param_guess=state0_guess[param_index],
        tf=tf_guess,
        target=np.array([0.0]),
        max_iter=max_iter,
        tol=tol,
        **kwargs
    )


def jacobi_constant(state_nd: np.ndarray, mu: float) -> float:
    """
    计算无量纲 CRTBP 中的雅可比常数。

    参数:
        state_nd: 无量纲状态 [x, y, z, vx, vy, vz]
        mu: 质量比

    返回:
        雅可比常数 C
    """
    x, y, z, vx, vy, vz = state_nd
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    U = (x**2 + y**2)/2 + (1-mu)/r1 + mu/r2
    v2 = vx**2 + vy**2 + vz**2
    return 2*U - v2