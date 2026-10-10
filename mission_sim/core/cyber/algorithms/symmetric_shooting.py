# mission_sim/core/cyber/algorithms/symmetric_shooting.py
# -*- coding: utf-8 -*-
"""
对称轨道打靶求解器（AS W2 求解策略升级；编译内核加速）
=======================================================
垂直截面判据：设计变量 (x0, vy0)，残差 [y(T/2), ẋ(T/2)] → 0。
相对既有 `LunarSwingTargeter.find_symmetric_periodic_orbit` 的升级：
  · numba fp64 编译内核（默认路径，速度 ~25×，与参考逐位一致）；
  · 线搜索牛顿（回溯接受，抗非线性/近月强扰动）；
  · 停滞检测（离散误差下限附近自动收敛停止）；
  · 全周期核验 + Jacobi 漂移。
"""
import numpy as np

from mission_sim.utils.dynamics import compiled_kernels as ck

CHAR_PERIOD = 4.342 * 86400.0


def jacobi_constant(state, mu):
    """无量纲 CRTBP Jacobi 常数 C = 2Ω − v²。"""
    x, y, z = float(state[0]), float(state[1]), float(state[2])
    r1 = np.sqrt((x + mu) ** 2 + y * y + z * z)
    r2 = np.sqrt((x + mu - 1.0) ** 2 + y * y + z * z)
    omega2 = x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2
    v2 = float(np.sum(np.asarray(state)[3:6] ** 2))
    return omega2 - v2


class SymmetricShooter:
    """对称周期轨道打靶（线搜索牛顿 + 编译内核）。"""

    def __init__(self, mu, period_seconds, n_steps=40000, char_period=CHAR_PERIOD):
        self.mu = float(mu)
        self.period_seconds = float(period_seconds)
        self.t_half = 0.5 * self.period_seconds / char_period
        self.t_full = self.period_seconds / char_period
        self.n_steps = int(n_steps)

    # ------------------------------------------------------------------
    def _seed(self, x0, vy0):
        s = np.zeros(6)
        s[0] = float(x0)
        s[4] = float(vy0)
        return s

    def residual(self, x0, vy0, n_steps=None):
        n = int(n_steps or self.n_steps)
        xe = ck.rk4_propagate(self._seed(x0, vy0), self.t_half, n, self.mu)
        return np.array([xe[1], xe[3]])

    # ------------------------------------------------------------------
    def solve(self, x0_guess, vy0_guess, tol=1e-8, max_iter=80, polish_tol=1e-11):
        """线搜索牛顿求解。

        Returns: dict(success, x0, vy0, state, residual, iterations, history)
        """
        x = np.array([float(x0_guess), float(vy0_guess)])
        hist = []
        best_x, best_rn = x.copy(), float("inf")
        stall = 0
        for it in range(max_iter):
            seed = self._seed(x[0], x[1])
            xe, phi = ck.rk4_propagate_aug(seed, self.t_half, self.n_steps, self.mu)
            res = np.array([xe[1], xe[3]])
            rn = float(np.linalg.norm(res))
            hist.append(rn)
            if rn < best_rn:
                best_rn, best_x, stall = rn, x.copy(), 0
            else:
                stall += 1
            if best_rn <= polish_tol or stall >= 3:
                break
            j = np.array([[phi[1, 0], phi[1, 4]],
                          [phi[3, 0], phi[3, 4]]])
            ds = np.linalg.lstsq(j, -res, rcond=None)[0]
            accepted = False
            for tscale in (1.0, 0.5, 0.25, 0.1):
                x_try = x + tscale * ds
                res_t = self.residual(x_try[0], x_try[1])
                if np.linalg.norm(res_t) < rn:
                    x = x_try
                    accepted = True
                    break
            if not accepted:
                break
        return {
            "success": bool(best_rn <= tol),
            "x0": float(best_x[0]),
            "vy0": float(best_x[1]),
            "state": self._seed(best_x[0], best_x[1]),
            "residual": best_rn,
            "iterations": len(hist),
            "history": hist,
        }

    # ------------------------------------------------------------------
    def verify(self, x0, vy0, n_steps=None):
        """全周期闭合 + Jacobi 漂移核验。"""
        n = int(n_steps or self.n_steps)
        seed = self._seed(x0, vy0)
        xf = ck.rk4_propagate(seed, self.t_full, max(2 * n, n + 1), self.mu)
        full_res = float(np.linalg.norm(xf - seed))
        c0 = jacobi_constant(seed, self.mu)
        c1 = jacobi_constant(xf, self.mu)
        drift = abs(c1 - c0) / max(abs(c0), 1e-30)
        return {"full_state_residual": full_res, "jacobi_drift": float(drift),
                "state": seed, "x0": float(x0), "vy0": float(vy0)}
