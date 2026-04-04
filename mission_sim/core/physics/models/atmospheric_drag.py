# mission_sim/core/physics/models/atmospheric_drag.py
"""
大气阻力模型 (Atmospheric Drag)
适用于地心惯性系 (J2000_ECI)，采用指数大气密度模型。
在 L1 级中，大气阻力是 LEO 任务的主导非保守力之一，用于高精度轨道维持分析。
"""

import numpy as np
from mission_sim.core.physics.environment import IForceModel


def _atmospheric_drag_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    area_to_mass: float,
    Cd: float,
    rho0: float,
    H: float,
    h0: float,
    R_earth: float
) -> np.ndarray:
    """
    计算大气阻力加速度的纯函数 (供 JIT 加速使用)。

    Args:
        pos: 航天器位置向量 [x, y, z] (m)
        vel: 航天器速度向量 [vx, vy, vz] (m/s)
        area_to_mass: 面积质量比 A/m (m²/kg)
        Cd: 阻力系数 (无量纲)
        rho0: 参考高度 h0 处的大气密度 (kg/m³)
        H: 标高 (m)
        h0: 参考高度 (m)
        R_earth: 地球半径 (m)

    Returns:
        加速度向量 [ax, ay, az] (m/s²)
    """
    # 地心距和高度
    r = np.linalg.norm(pos)
    h = r - R_earth

    # 高度低于参考高度时，使用参考密度（保守估计，实际更低）
    if h < h0:
        rho = rho0
    else:
        rho = rho0 * np.exp(-(h - h0) / H)

    # 速度大小
    v = np.linalg.norm(vel)

    # 避免除零
    if v < 1e-10:
        return np.zeros(3, dtype=np.float64)

    # 阻力加速度: a = -0.5 * Cd * (A/m) * rho * v * v_dir
    factor = -0.5 * Cd * area_to_mass * rho * v
    acc = factor * vel
    return acc.astype(np.float64)


class AtmosphericDrag(IForceModel):
    """
    大气阻力模型 (指数密度模型)
    继承自 IForceModel，计算大气阻力加速度（地心惯性系）。

    物理公式:
        a_drag = -0.5 * Cd * (A/m) * ρ * v * v̂

    其中:
        Cd : 阻力系数 (无量纲)
        A/m: 面积质量比 (m²/kg)
        ρ  : 大气密度 (kg/m³)，采用指数模型 ρ = ρ₀ * exp(-(h - h₀)/H)
        v  : 航天器相对于大气的速度大小 (m/s)，忽略大气自转，直接使用惯性速度
        v̂  : 速度方向单位向量

    坐标系假设: 地心惯性系 (J2000_ECI)
    """

    # 标准地球常数 (WGS84)
    R_EARTH = 6378137.0          # 地球半径 (m)
    RHO0 = 1.225                 # 海平面大气密度 (kg/m³)
    H = 8500.0                   # 标高 (m)，典型值
    H0 = 0.0                     # 参考高度 (m)

    def __init__(
        self,
        area_to_mass: float,
        Cd: float = 2.2,
        rho0: float = None,
        H: float = None,
        h0: float = None,
        R_earth: float = None
    ):
        """
        初始化大气阻力模型。

        Args:
            area_to_mass: 面积质量比 A/m (m²/kg)
            Cd: 阻力系数，默认 2.2 (航天器典型值)
            rho0: 参考高度 h0 处的大气密度 (kg/m³)，默认海平面密度 1.225
            H: 标高 (m)，默认 8500.0
            h0: 参考高度 (m)，默认 0.0 (海平面)
            R_earth: 地球半径 (m)，默认 WGS84 值 6378137.0
        """
        self.area_to_mass = float(area_to_mass)
        self.Cd = float(Cd)
        self.rho0 = float(rho0) if rho0 is not None else self.RHO0
        self.H = float(H) if H is not None else self.H
        self.h0 = float(h0) if h0 is not None else self.H0
        self.R_earth = float(R_earth) if R_earth is not None else self.R_EARTH

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算大气阻力加速度。

        Args:
            state: 航天器状态向量 [x, y, z, vx, vy, vz] (地心惯性系)
            epoch: 当前仿真时间 (s)，未使用（模型与时间无关）

        Returns:
            np.ndarray: 加速度向量 [ax, ay, az] (m/s²)
        """
        pos = state[:3]
        vel = state[3:6]

        # 调用纯函数计算加速度
        return _atmospheric_drag_accel(
            pos,
            vel,
            self.area_to_mass,
            self.Cd,
            self.rho0,
            self.H,
            self.h0,
            self.R_earth
        )

    def __repr__(self) -> str:
        return (f"AtmosphericDrag(A/m={self.area_to_mass:.4f} m²/kg, "
                f"Cd={self.Cd:.2f}, rho0={self.rho0:.3f} kg/m³, "
                f"H={self.H:.1f} m, R_earth={self.R_earth:.1f} m)")
