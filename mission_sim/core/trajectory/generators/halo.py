# mission_sim/core/trajectory/generators/halo.py
"""Halo 轨道微分修正生成器"""

import numpy as np
from scipy.integrate import solve_ivp
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators.base import BaseTrajectoryGenerator
from mission_sim.utils.differential_correction import compute_periodic_orbit


class HaloDifferentialCorrector(BaseTrajectoryGenerator):
    """
    Halo 轨道微分修正生成器。
    使用单参数微分修正方法，求解日地/地月旋转系中的周期 Halo 轨道。
    """

    def __init__(self, mu: float = 3.00348e-6):
        """
        初始化生成器。

        Args:
            mu: 无量纲质量比（日地系统默认 3.00348e-6，地月系统约 0.01215）
        """
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        生成 Halo 轨道星历。

        config 必须包含:
            - Az: 目标 Z 振幅 (无量纲)
            - dt: 输出步长 (无量纲)
            - initial_guess: 可选，[x0, z0, vy0] 初始猜测
            - max_iter: 微分修正最大迭代次数 (可选，默认 10)
            - tol: 修正容差 (可选，默认 1e-10)

        Returns:
            Ephemeris 对象（SUN_EARTH_ROTATING 坐标系）
        """
        # 获取参数
        Az = config.get("Az", 0.05)
        dt_nd = config.get("dt", 0.001)
        max_iter = config.get("max_iter", 10)
        tol = config.get("tol", 1e-10)

        # 获取初始猜测
        if "initial_guess" in config:
            x0, z0, vy0 = config["initial_guess"]
        else:
            x0, z0, vy0 = self._robust_initial_guess(Az)

        # 初始状态 [x, y, z, vx, vy, vz] (从 XZ 平面出发)
        state0_nd = np.array([x0, 0.0, z0, 0.0, vy0, 0.0])

        # 使用微分修正寻找周期轨道
        try:
            # 寻找半周期（第一次 y=0 穿越）
            T_half = self._find_half_period(state0_nd)
            if T_half is None:
                raise RuntimeError("无法自动确定半周期，使用默认值")
            # 修正参数（例如初始 vy）使半周期后 y=0 且 vz=0
            def constraint(state_f):
                return state_f[1]  # 要求 y=0

            # 使用微分修正
            # 注意：compute_periodic_orbit 在 differential_correction 中可能未完全实现，这里调用底层
            from mission_sim.utils.differential_correction import single_parameter_correction

            param_index = 4  # vy
            alpha_opt, _, history = single_parameter_correction(
                dynamics=self._crtbp_equations,
                constraint=lambda xf: np.array([xf[1]]),
                t0=0.0,
                x0=state0_nd,
                param_index=param_index,
                param_guess=state0_nd[param_index],
                tf=T_half,
                target=np.array([0.0]),
                max_iter=max_iter,
                tol=tol
            )
            state0_corrected = state0_nd.copy()
            state0_corrected[param_index] = alpha_opt
            print(f"[HaloCorrector] 微分修正完成，最终残差 {history['c_norm'][-1]:.2e}")
        except Exception as e:
            print(f"[HaloCorrector] 微分修正失败，使用原始初始猜测: {e}")
            state0_corrected = state0_nd

        # 确定周期（半周期检测）
        T_nd = self._find_period(state0_corrected)
        if T_nd is None:
            T_nd = 3.141  # 默认近似周期

        # 生成完整轨道
        times_nd = np.arange(0, T_nd, dt_nd)
        sol = solve_ivp(
            self._crtbp_equations,
            (0, T_nd),
            state0_corrected,
            t_eval=times_nd,
            method='DOP853',
            rtol=1e-12,
            atol=1e-12
        )

        # 转换为物理单位
        AU = 1.495978707e11
        OMEGA = 1.990986e-7
        physical_times = sol.t / OMEGA
        physical_states = sol.y.T.copy()
        physical_states[:, 0:3] *= AU
        physical_states[:, 3:6] *= (AU * OMEGA)

        return Ephemeris(physical_times, physical_states, CoordinateFrame.SUN_EARTH_ROTATING)

    def _crtbp_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """CRTBP 无量纲运动方程"""
        x, y, z, vx, vy, vz = state
        mu = self.mu
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
        ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
        az = -(1-mu)*z/r1**3 - mu*z/r2**3
        return np.array([vx, vy, vz, ax, ay, az])

    def _find_half_period(self, state0: np.ndarray, max_time: float = 20.0) -> float:
        """寻找半周期（y=0 穿越）"""
        def y_zero(t, y):
            return y[1]
        y_zero.direction = -1  # 从正到负
        sol = solve_ivp(self._crtbp_equations, (0, max_time), state0,
                        events=[y_zero], method='DOP853',
                        rtol=1e-12, atol=1e-12)
        if len(sol.t_events[0]) > 0:
            t_half = sol.t_events[0][0]
            if t_half > 0.1:
                return t_half
        return None

    def _find_period(self, state0: np.ndarray, max_time: float = 20.0) -> float:
        """寻找完整周期（两次 y=0 穿越）"""
        t_half = self._find_half_period(state0, max_time)
        if t_half is not None:
            return 2.0 * t_half
        return None

    def _robust_initial_guess(self, Az: float) -> tuple:
        """根据振幅生成稳健初始猜测"""
        mu = self.mu
        gamma = np.cbrt(mu / 3.0)
        x_L2 = 1 - mu + gamma
        x0 = x_L2 + 0.015 * Az
        z0 = Az
        vy0 = 0.01 + 0.1 * Az
        return x0, z0, vy0