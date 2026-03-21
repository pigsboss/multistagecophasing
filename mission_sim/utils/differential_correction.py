# mission_sim/utils/differential_correction.py
"""
微分修正工具 (Differential Correction Utilities)
提供状态转移矩阵数值计算、单参数微分修正等通用函数，用于周期轨道生成（如 Halo 轨道）。

功能：
1. 数值计算状态转移矩阵 (STM)
2. 单参数微分修正（针对周期轨道约束）
3. 辅助函数：轨道积分、约束函数定义
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Optional


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
    数值计算状态转移矩阵 Φ(tf, t0)，通过差分法近似。
    对初始状态每个分量施加小扰动，分别积分，然后差分得到 STM。

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
    # 先积分参考轨迹得到最终状态
    sol_ref = solve_ivp(dynamics, (t0, tf), x0, method=method, rtol=rtol, atol=atol)
    if not sol_ref.success:
        raise RuntimeError(f"Reference trajectory integration failed: {sol_ref.message}")
    xf_ref = sol_ref.y[:, -1]

    # 初始化 STM 矩阵
    stm = np.zeros((n, n))

    # 对每个初始状态分量进行扰动
    for i in range(n):
        # 创建扰动初始状态
        x0_pert = x0.copy()
        x0_pert[i] += eps

        # 积分扰动轨迹
        sol_pert = solve_ivp(dynamics, (t0, tf), x0_pert, method=method, rtol=rtol, atol=atol)
        if not sol_pert.success:
            raise RuntimeError(f"Perturbed trajectory integration failed for index {i}: {sol_pert.message}")
        xf_pert = sol_pert.y[:, -1]

        # 差分得到 STM 的第 i 列
        stm[:, i] = (xf_pert - xf_ref) / eps

    return stm


def single_parameter_correction(
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
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    单参数微分修正，用于寻找使约束条件满足的参数值。
    例如，调整初始 vy 使轨道在 y=0 处闭合。

    算法：
        x0(α) = x0_nom + α * e_param
        对给定的 α，积分得到终端状态，计算约束偏差 Δc
        求解 Δc ≈ S * Δα，其中 S = ∂c/∂α
        更新 α := α - inv(S) * c，直到收敛。

    参数:
        dynamics: 动力学函数 f(t, x)
        constraint: 约束函数，输入终端状态，输出约束向量 (m维)
        t0: 初始时间
        x0: 名义初始状态 (n维)
        param_index: 待修正参数在 x0 中的索引 (0..n-1)
        param_guess: 参数初始猜测值
        tf: 积分最终时间
        target: 期望的约束值，默认为零向量
        max_iter: 最大迭代次数
        tol: 收敛容差
        eps_perturb: 参数扰动步长
        rtol, atol: 积分容差
        method: 积分方法

    返回:
        (param_opt, xf_opt, history): 优化后的参数值、终端状态、迭代历史
    """
    if target is None:
        target = np.zeros_like(constraint(np.zeros(6)))  # 假设输出维数

    # 记录迭代历史
    history = {'alpha': [], 'c_norm': []}

    alpha = param_guess
    x0_base = x0.copy()

    for it in range(max_iter):
        # 构造当前初始状态
        x0_curr = x0_base.copy()
        x0_curr[param_index] = alpha

        # 积分得到终端状态
        sol = solve_ivp(dynamics, (t0, tf), x0_curr, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed at iteration {it}: {sol.message}")
        xf = sol.y[:, -1]

        # 计算约束偏差
        c = constraint(xf) - target
        c_norm = np.linalg.norm(c)

        history['alpha'].append(alpha)
        history['c_norm'].append(c_norm)

        # 收敛检查
        if c_norm < tol:
            return alpha, xf, history

        # 数值计算敏感性矩阵 S = ∂c/∂α
        # 对参数进行小扰动
        alpha_pert = alpha + eps_perturb
        x0_pert = x0_base.copy()
        x0_pert[param_index] = alpha_pert
        sol_pert = solve_ivp(dynamics, (t0, tf), x0_pert, method=method, rtol=rtol, atol=atol)
        if not sol_pert.success:
            raise RuntimeError(f"Perturbed integration failed at iteration {it}: {sol_pert.message}")
        xf_pert = sol_pert.y[:, -1]

        c_pert = constraint(xf_pert) - target
        S = (c_pert - c) / eps_perturb  # m维

        # 牛顿步： Δα = - inv(S) * c
        # 对于单参数，S 是列向量，求伪逆
        # 如果 S 很小，则停止
        if np.linalg.norm(S) < 1e-12:
            print(f"Warning: Sensitivity matrix near zero at iteration {it}, stopping.")
            break

        delta_alpha = -c / S  # 标量除法
        alpha += delta_alpha[0]  # c 是标量时，直接取第一个

    print(f"Single-parameter correction finished after {it+1} iterations, final residual = {c_norm:.2e}")
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
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    寻找周期轨道的微分修正专用函数。
    约束通常定义为 y=0 穿越条件 (对于对称轨道) 或 x=0 穿越条件。

    参数:
        dynamics: 动力学函数
        state0_guess: 初始状态猜测 (n维)
        param_index: 待修正的初始状态索引
        constraint_func: 约束函数，输入终端状态，输出标量约束（如 y 坐标）
        tf_guess: 半周期时间猜测，如果为 None，则自动寻找第一个穿越点
        max_iter, tol: 迭代参数
        **kwargs: 传递给 solve_ivp 的参数

    返回:
        (alpha_opt, xf_opt, history): 优化后的参数值、终端状态、迭代历史
    """
    # 定义包装约束，返回 1 维数组
    def constraint_wrapper(xf):
        return np.array([constraint_func(xf)])

    if tf_guess is None:
        # 自动寻找第一个穿越点
        # 先积分一个足够长的时间，找到第一次穿越
        # 这里简化，假设用户提供 tf_guess，或者可以搜索
        raise ValueError("tf_guess must be provided for periodic orbit correction")

    # 调用单参数修正
    alpha_opt, xf_opt, history = single_parameter_correction(
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
    return alpha_opt, xf_opt, history


# 辅助函数：雅可比常数计算（用于验证）
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