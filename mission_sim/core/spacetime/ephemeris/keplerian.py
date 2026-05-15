"""
KeplerEphemeris – 直接求解开普勒方程的解析星历表

不需要外部文件或数值积分，对椭圆轨道提供精确的状态查询。
"""

import numpy as np
from numba import njit

from mission_sim.core.spacetime.ephemeris.base import Ephemeris
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch


class KeplerEphemeris(Ephemeris):
    """
    理想 Kepler 椭圆轨道星历表。
    所有轨道根数 (a, e, i, Ω, ω, M0) 在 epoch 确定后保持不变，
    get_state 直接求解 Kepler 方程，完全解析，无插值误差。
    """

    def __init__(
        self,
        a: float,
        e: float,
        i: float,
        Omega: float,
        omega: float,
        M0: float,
        epoch: float,
        mu: float,
        frame: CoordinateFrame = CoordinateFrame.J2000_ECI,
    ):
        """
        :param a: 半长轴 (m)
        :param e: 偏心率 (0 ≤ e < 1)
        :param i: 倾角 (rad)
        :param Omega: 升交点赤经 (rad)
        :param omega: 近地点辐角 (rad)
        :param M0: 历元平近点角 (rad)
        :param epoch: 历元时间 (TDB 秒，相对于 J2000)
        :param mu: 中心天体引力常数 (m³/s²)
        :param frame: 坐标系
        """
        # 调用基类构造，提供占位数据以满足契约
        dummy_times = np.array([epoch])
        dummy_states = np.zeros((1, 6))
        super().__init__(dummy_times, dummy_states, frame)

        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.omega = omega
        self.M0 = M0
        self.epoch = epoch
        self.mu = mu

        # 预计算平均运动
        self.n = np.sqrt(mu / a**3)

    def get_state(self, t: float) -> np.ndarray:
        """
        返回 t 时刻 (TDB 秒，相对 J2000) 的笛卡尔状态。

        重写基类方法，直接求解开普勒方程。
        """
        # 计算 t 时刻的平近点角
        M = self.M0 + self.n * (t - self.epoch)

        # 调用批量求解器（单个输入）
        state = kepler_elements_to_cartesian_batch(
            np.array([self.a]),
            np.array([self.e]),
            np.array([self.i]),
            np.array([self.Omega]),
            np.array([self.omega]),
            np.array([M]),
            self.mu,
        )
        return state[0].copy()  # shape (6,)
