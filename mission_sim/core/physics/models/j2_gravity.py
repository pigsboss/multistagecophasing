# mission_sim/core/physics/models/j2_gravity.py
"""
地球 J2 摄动模型 (J2 Perturbation)
适用于地心惯性系 (J2000_ECI)，提供地球非球形引力中 J2 项导致的加速度。
在 L1 级中，J2 摄动是 LEO/GEO 任务的主导摄动之一，用于高精度轨道外推和标称轨道生成。
"""

import numpy as np
from mission_sim.core.physics.models.base import ForceModel

# 尝试导入 Numba 用于 JIT 加速，若不可用则回退到普通 Python
try:
    from numba import jit as njit
    _HAS_NUMBA = True
except ImportError:
    # 定义一个空装饰器，使代码在无 numba 时仍能运行
    def njit(func):
        return func
    _HAS_NUMBA = False


@njit
def _j2_accel(pos: np.ndarray, mu_earth: float, j2: float, r_earth: float) -> np.ndarray:
    """
    计算地球 J2 摄动加速度的纯函数 (可被 Numba JIT 编译)。

    物理公式:
        a_J2 = - (3/2) * μ_E * J2 * R_E² / r⁵ * [ x(5z²/r² - 1),
                                                  y(5z²/r² - 1),
                                                  z(5z²/r² - 3) ]

    Args:
        pos: 航天器位置向量 [x, y, z] (地心惯性系, m)
        mu_earth: 地球引力常数 (m³/s²)
        j2: 地球 J2 系数 (无量纲)
        r_earth: 地球赤道半径 (m)

    Returns:
        加速度向量 [ax, ay, az] (m/s²)
    """
    x, y, z = pos[0], pos[1], pos[2]

    # 地心距平方和五次方
    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    r5 = r2 * r2 * r

    # 避免除零
    if r5 < 1e-20:
        return np.zeros(3, dtype=np.float64)

    # 计算 z²/r²
    z2_r2 = (z * z) / r2

    # 预计算常数因子
    factor = -1.5 * mu_earth * j2 * (r_earth ** 2) / r5

    # 根据公式计算加速度分量
    ax = factor * x * (5.0 * z2_r2 - 1.0)
    ay = factor * y * (5.0 * z2_r2 - 1.0)
    az = factor * z * (5.0 * z2_r2 - 3.0)

    return np.array([ax, ay, az], dtype=np.float64)


class J2Gravity(IForceModel):
    """
    地球 J2 摄动模型
    继承自 IForceModel，计算地球 J2 项引起的加速度（地心惯性系）。

    坐标系假设: 地心惯性系 (J2000_ECI)
    """

    # 标准地球常数 (WGS84)
    MU_EARTH = 3.986004418e14      # m³/s²
    J2 = 1.08262668e-3             # 无量纲
    R_EARTH = 6378137.0            # m

    def __init__(self,
                 mu_earth: float = None,
                 j2: float = None,
                 r_earth: float = None):
        """
        初始化 J2 摄动模型。

        Args:
            mu_earth: 地球引力常数 (m³/s²)，默认使用 WGS84 值
            j2: 地球 J2 系数，默认使用 WGS84 值
            r_earth: 地球半径 (m)，默认使用 WGS84 值
        """
        self.mu_earth = mu_earth if mu_earth is not None else self.MU_EARTH
        self.j2 = j2 if j2 is not None else self.J2
        self.r_earth = r_earth if r_earth is not None else self.R_EARTH

        # 预计算常数因子（虽然纯函数中已计算，但保留属性以备其他用途）
        self._prefactor = 1.5 * self.mu_earth * self.j2 * (self.r_earth ** 2)

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        计算 J2 摄动加速度。

        Args:
            state: 航天器状态向量 [x, y, z, vx, vy, vz] (地心惯性系)
            epoch: 当前仿真时间 (s)，未使用（模型与时间无关）

        Returns:
            np.ndarray: 加速度向量 [ax, ay, az] (m/s²)
        """
        # 提取位置
        pos = state[:3]

        # 调用纯函数计算加速度
        return _j2_accel(pos, self.mu_earth, self.j2, self.r_earth)

    def __repr__(self) -> str:
        return (f"J2Gravity(mu_earth={self.mu_earth:.4e} m³/s², "
                f"j2={self.j2:.6e}, r_earth={self.r_earth:.3f} m)")
