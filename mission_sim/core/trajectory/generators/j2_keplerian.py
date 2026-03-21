# mission_sim/core/trajectory/generators/j2_keplerian.py
"""带 J2 摄动的开普勒轨道生成器（数值积分）"""

import numpy as np
from scipy.integrate import solve_ivp
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators.base import BaseTrajectoryGenerator
from mission_sim.core.physics.models.j2_gravity import J2Gravity


class J2KeplerianGenerator(BaseTrajectoryGenerator):
    """
    带 J2 摄动的开普勒轨道生成器。
    通过数值积分二体 + J2 摄动，生成高精度参考星历。
    适用于 LEO/GEO 任务。
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        初始化生成器。

        Args:
            mu: 中心天体引力常数 (m³/s²)，默认地球。
        """
        self.mu = mu
        self.j2_model = J2Gravity(mu_earth=mu)  # 使用 J2 模型

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成 J2 摄动轨道。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数
            - dt: 输出步长 (s)
            - sim_time: 仿真时长 (s)
            - integrator: 积分器方法 (可选，默认 'DOP853')
            - rtol: 相对容差 (可选，默认 1e-12)

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("J2KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        integrator = config.get("integrator", 'DOP853')
        rtol = config.get("rtol", 1e-12)

        # 将轨道根数转换为笛卡尔状态
        state0 = self._elements_to_cartesian(elements)

        # 定义运动方程
        def dynamics(t, state):
            pos = state[:3]
            vel = state[3:6]
            r = np.linalg.norm(pos)
            # 中心引力
            acc_central = -self.mu * pos / r**3
            # J2 摄动
            acc_j2 = self.j2_model.compute_accel(state, t)
            return np.concatenate([vel, acc_central + acc_j2])

        # 积分
        times = np.arange(0, sim_time + dt, dt)
        sol = solve_ivp(
            dynamics,
            t_span=(0, sim_time),
            y0=state0,
            t_eval=times,
            method=integrator,
            rtol=rtol,
            atol=rtol
        )
        if not sol.success:
            raise RuntimeError(f"J2 轨道积分失败: {sol.message}")

        return Ephemeris(sol.t, sol.y.T, CoordinateFrame.J2000_ECI)

    def _elements_to_cartesian(self, elements):
        """将轨道根数转换为 J2000_ECI 笛卡尔状态（简化，仅圆轨道）"""
        a, e, i, Omega, omega, M0 = elements
        # 简化：仅支持圆轨道 e=0，且 i=0，Ω=0，ω=0
        # 实际应用需实现完整转换
        n = np.sqrt(self.mu / a**3)
        M = M0
        E = M  # 圆轨道近似
        nu = E
        r = a
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        vx = -np.sqrt(self.mu / a) * np.sin(nu)
        vy = np.sqrt(self.mu / a) * np.cos(nu)
        return np.array([x, y, 0.0, vx, vy, 0.0])