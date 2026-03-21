# mission_sim/core/trajectory/generators/keplerian.py
"""开普勒轨道生成器（二体解析解）"""

import numpy as np
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.trajectory.generators.base import BaseTrajectoryGenerator


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    开普勒轨道生成器（二体问题解析解）。
    使用经典开普勒公式生成参考星历，无摄动。
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        初始化生成器。

        Args:
            mu: 中心天体引力常数 (m³/s²)，默认地球。
        """
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成星历。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数
            - dt: 时间步长 (s)
            - sim_time: 仿真时长 (s)

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        times = np.arange(0, sim_time + dt, dt)

        a, e, i, Omega, omega, M0 = elements
        n = np.sqrt(self.mu / a**3)

        states = []
        for t in times:
            # 平近点角
            M = M0 + n * t
            # 解 Kepler 方程（简化：小 e 时可近似，这里使用牛顿迭代）
            E = self._kepler_solver(M, e)
            # 真近点角
            nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
            # 轨道面内坐标
            r = a * (1 - e * np.cos(E))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            vx_orb = -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu)
            vy_orb = np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu))

            # 坐标变换到 J2000_ECI (简化：仅处理圆轨道或忽略倾角)
            # 为简化，此处假设轨道在赤道平面内 (i=0, Omega=0, omega=0)
            # 实际应用中应包含完整的三维旋转。
            state_eci = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
            states.append(state_eci)

        return Ephemeris(times, np.array(states), CoordinateFrame.J2000_ECI)

    def _kepler_solver(self, M: float, e: float, tol: float = 1e-12) -> float:
        """解 Kepler 方程 M = E - e sin(E) (牛顿迭代)"""
        E = M if e < 0.8 else np.pi  # 初始猜测
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        return E