# mission_sim/core/cyber/algorithms/family_continuation.py
# -*- coding: utf-8 -*-
"""
一维参数连续/同伦驱动（G5① 沉淀件）
=====================================
把"单点求解"升级为"族域理解"：以标量参数 λ 为延续变量，
predictor（线性外推）+ corrector（回调求解器）+ 自适应步长（成功放大/失败缩小）。

用途：共振族追踪、Lagrange/Halo 族研究、任意参数化根路径（合成问题亦可）。
"""
import numpy as np


def continue_family(residual_fn, solve_fn, x_initial, final_value, *,
                    param0=0.0, step=0.25, tol=1e-10, max_steps=500,
                    grow=1.5, shrink=0.5, max_step=None):
    """参数延续。

    Args:
        residual_fn: callable(x, lam) -> np.ndarray 残差向量
        solve_fn: callable(x_guess, lam) -> x_sol（收敛求解器，自行迭代到 ~tol）
        x_initial: param0 处的已知解
        final_value: 目标参数值（须 > param0）
        param0: 起始参数值
        step: 初始步长
        tol: 残差验收门限
        max_steps: 路径点数上限（防死循环）
        grow / shrink: 步长放大/缩小因子
        max_step: 步长上限（默认 8×初始步长）

    Returns:
        path: [(lam_0, x_0), ..., (final_value, x_final)]
    """
    x = np.asarray(x_initial, dtype=np.float64).copy()
    path = [(float(param0), x.copy())]
    lam = float(param0)
    step_cur = float(step)
    step_max = float(max_step) if max_step is not None else float(step) * 8.0
    prev_lam, prev_x = None, None
    guard = 0

    while lam < final_value - 1e-15:
        guard += 1
        if guard > max_steps:
            raise RuntimeError(f"continue_family 超出 max_steps={max_steps}")
        step_cur = min(step_cur, final_value - lam)
        lam_new = lam + step_cur
        if prev_lam is not None:
            rate = (x - prev_x) / (lam - prev_lam)
            x_guess = x + rate * (lam_new - lam)
        else:
            x_guess = x.copy()
        x_new = np.asarray(solve_fn(x_guess, lam_new), dtype=np.float64)
        if np.linalg.norm(residual_fn(x_new, lam_new)) <= max(tol, 1e-9):
            prev_lam, prev_x = lam, x
            lam, x = lam_new, x_new
            path.append((float(lam), x.copy()))
            step_cur = min(step_cur * grow, step_max)
        else:
            step_cur *= shrink
            if step_cur < 1e-12:
                raise RuntimeError("continue_family 步长收缩至下限，延续失败")
    return path
