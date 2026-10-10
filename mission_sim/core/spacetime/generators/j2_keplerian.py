# mission_sim/core/spacetime/generators/j2_keplerian.py
"""
带 J2 摄动的开普勒轨道生成器（数值积分）

通过数值积分"二体 + J2 摄动"生成高精度参考星历，适用于 LEO/GEO 任务。

修订记录（WP-1.2，2026-10-10）：
- 删除重复定义的第二个 `J2KeplerianGenerator` 类（其 J2 公式符号写反，
  导致标称轨道与仿真动力学方向相反，见 M1 红基线/测试报告）；
- J2 公式统一取自 `physics/models/j2_gravity.py: J2Gravity`（单一来源，DD-001 D-01）；
- 并入 epoch 偏移与 j2_coefficient / earth_radius 配置项支持。
"""

import numpy as np
from scipy.integrate import solve_ivp

from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator
from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.utils.math_tools import elements_to_cartesian


class J2KeplerianGenerator(BaseTrajectoryGenerator):
    """
    带 J2 摄动的开普勒轨道生成器。

    通过数值积分"二体 + J2 摄动"生成高精度参考星历。
    在地心惯性系中计算航天器相对地球的运动（地球恒为原点），不需要外部星历。
    """

    def __init__(self, mu: float = 3.986004418e14, ephemeris=None, use_high_precision=False):
        """
        Args:
            mu: 中心天体引力常数 (m^3/s^2)，默认地球。
            ephemeris: 忽略此参数（本生成器不需要外部星历）。
            use_high_precision: 忽略此参数（本生成器不支持高精度星历模式）。
        """
        super().__init__(ephemeris=None, use_high_precision=False)
        self.mu = float(mu)

    def generate(self, config: dict) -> Ephemeris:
        """
        根据轨道根数生成 J2 摄动轨道。

        config 必须包含:
            - elements: [a, e, i, Omega, omega, M0] 轨道根数（SI/rad）
            - dt: 输出步长 (s)
            - sim_time: 仿真时长 (s)
        config 可选:
            - integrator: 积分器方法（默认 'DOP853'）
            - rtol: 积分相对容差（默认 1e-12，绝对容差同值）
            - epoch: 起始历元偏移 (s)
            - j2_coefficient / earth_radius: 覆盖默认 J2 参数

        Returns:
            Ephemeris 对象（J2000_ECI 坐标系）
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("J2KeplerianGenerator 必须在 config 中提供 6 个轨道根数 'elements'。")

        dt = float(config.get("dt", 1.0))
        sim_time = float(config.get("sim_time", 86400.0))
        integrator = config.get("integrator", "DOP853")
        rtol = float(config.get("rtol", 1e-12))
        epoch = float(config.get("epoch", 0.0))

        # J2 公式单一来源（DD-001 D-01）：统一使用 J2Gravity 模型
        j2_model = J2Gravity(
            mu_earth=self.mu,
            j2=config.get("j2_coefficient", None),
            r_earth=config.get("earth_radius", None),
        )

        a, e, i, Omega, omega, M0 = elements
        state0 = elements_to_cartesian(self.mu, a, e, i, Omega, omega, M0)

        def dynamics(t, state):
            pos = state[:3]
            vel = state[3:6]
            acc_central = -self.mu * pos / np.linalg.norm(pos) ** 3
            acc_j2 = j2_model.compute_accel(state, t)
            return np.concatenate([vel, acc_central + acc_j2])

        times = np.arange(0.0, sim_time + 0.5 * dt, dt)
        times = times[times <= sim_time + 1e-9]
        if times.size == 0 or times[-1] < sim_time - 1e-9:
            times = np.append(times, sim_time)
        sol = solve_ivp(
            dynamics,
            (0.0, sim_time),
            state0,
            t_eval=times,
            method=integrator,
            rtol=rtol,
            atol=rtol,
        )
        if not sol.success:
            raise RuntimeError(f"J2 轨道积分失败: {sol.message}")

        return Ephemeris(sol.t + epoch, sol.y.T, CoordinateFrame.J2000_ECI)


# Convenience function
def create_j2_keplerian_generator(**kwargs) -> J2KeplerianGenerator:
    """Factory function to create J2-perturbed orbit generator"""
    return J2KeplerianGenerator(**kwargs)
