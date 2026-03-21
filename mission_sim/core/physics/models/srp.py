# mission_sim/core/physics/models/srp.py
"""
太阳光压模型（Cannonball 模型）
适用于日地旋转系，假设太阳位于固定位置（日地质心系中负X方向）。
在 L1 级中，光压作为主要非保守力之一，用于深空任务（如 Halo 轨道）的轨道维持分析。
"""

import numpy as np
from mission_sim.core.physics.environment import IForceModel


class Cannonball_SRP(IForceModel):
    """
    球对称太阳光压模型（Cannonball 模型）
    继承自 IForceModel，提供基于面积质量比和反射系数的光压加速度计算。
    假设太阳位于固定位置（日地旋转系中），适用于日地 L1/L2 等深空任务。

    物理公式:
        a_srp = (1 + r) * P_solar * (AU² / r²) * (A/m) * û

    其中:
        r       : 反射系数 (0~1，完全吸收为0，完全反射为1)
        P_solar : 太阳辐射压力常数，在 1 AU 处约为 4.56e-6 N/m²
        AU      : 天文单位，1.495978707e11 m
        r       : 航天器到太阳的距离 (m)
        A/m     : 面积质量比 (m²/kg)
        û       : 从航天器指向太阳的单位方向向量

    坐标系假设: 日地旋转系，太阳位于 (-mu * AU, 0, 0)，地球位于 ((1-mu)*AU, 0, 0)
    """

    def __init__(self,
                 area_to_mass: float,
                 reflectivity: float = 1.0,
                 sun_position: np.ndarray = None,
                 AU: float = 1.495978707e11,
                 mu: float = 3.00348e-6,
                 P_solar: float = 4.56e-6):
        """
        初始化光压模型。

        Args:
            area_to_mass: 面积质量比 A/m (m²/kg)
            reflectivity: 表面反射系数 (0~1)，默认 1.0（完全反射）
            sun_position: 太阳在坐标系中的固定位置 (3 维向量)，默认为日地旋转系中的标准位置 [-mu*AU, 0, 0]
            AU: 天文单位 (m)，默认 1.495978707e11
            mu: 质量比（日地系统），默认 3.00348e-6
            P_solar: 1 AU 处的太阳辐射压力 (N/m²)，默认 4.56e-6
        """
        self.area_to_mass = float(area_to_mass)
        self.reflectivity = float(reflectivity)
        self.AU = float(AU)
        self.mu = float(mu)
        self.P_solar = float(P_solar)

        if sun_position is None:
            # 日地旋转系中太阳固定位置
            self.sun_position = np.array([-self.mu * self.AU, 0.0, 0.0], dtype=np.float64)
        else:
            self.sun_position = np.asarray(sun_position, dtype=np.float64)
            if self.sun_position.shape != (3,):
                raise ValueError(f"sun_position 必须是形状为 (3,) 的数组，当前形状: {self.sun_position.shape}")

        # 预计算常数因子，用于优化计算
        self._constant_factor = (1.0 + self.reflectivity) * self.P_solar * (self.AU ** 2) * self.area_to_mass

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算太阳光压加速度。

        Args:
            state: 航天器状态向量 [x, y, z, vx, vy, vz] (SI单位，日地旋转系)
            epoch: 当前仿真时间 (s)，未使用（模型与时间无关，但保留接口）

        Returns:
            np.ndarray: 加速度向量 [ax, ay, az] (m/s²)
        """
        # 提取位置
        pos_sc = state[:3]

        # 计算从航天器指向太阳的矢量
        r_vec = self.sun_position - pos_sc
        r_mag = np.linalg.norm(r_vec)

        # 防止除零
        if r_mag < 1.0:
            # 距离太近，返回零加速度（实际不会发生）
            return np.zeros(3, dtype=np.float64)

        # 单位方向向量
        dir_sun = r_vec / r_mag

        # 加速度大小: factor = (1+r)*P_solar*(AU²/r²)*(A/m)
        # 其中 r 是航天器到太阳的距离
        factor = self._constant_factor / (r_mag * r_mag)

        # 加速度向量
        acc = factor * dir_sun

        return acc.astype(np.float64)

    def __repr__(self) -> str:
        return (f"Cannonball_SRP(A/m={self.area_to_mass:.4f} m²/kg, "
                f"reflectivity={self.reflectivity:.2f}, "
                f"sun_pos=({self.sun_position[0]:.2e}, {self.sun_position[1]:.2e}, {self.sun_position[2]:.2e}) m)")