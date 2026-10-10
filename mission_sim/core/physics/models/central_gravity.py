# mission_sim/core/physics/models/central_gravity.py
"""
MCPC Core Physics: 中心二体引力模型 (Central Two-Body Gravity)
---------------------------------------------------------------
a = -mu * r / |r|^3   （SI 单位，适用于惯性系）

按 DD-001 决策 D-01：中心引力作为独立 IForceModel 接入，
与 J2Gravity（摄动项）分离，各自保持单一职责。
"""

import numpy as np
from mission_sim.core.physics.environment import IForceModel

try:
    from numba import njit
except ImportError:  # pragma: no cover - 无 numba 环境回退
    def njit(func):
        return func


@njit
def _central_accel(pos: np.ndarray, mu: float, r_min: float) -> np.ndarray:
    """纯函数：a = -mu * r / |r|^3；r < r_min 时返回零向量（数值保护）。"""
    x, y, z = pos[0], pos[1], pos[2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    if r < r_min:
        return np.zeros(3, dtype=np.float64)
    factor = -mu / (r2 * r)
    return np.array([factor * x, factor * y, factor * z], dtype=np.float64)


class CentralGravity(IForceModel):
    """
    中心二体引力模型（惯性系）。

    坐标系假设: 惯性系（如 J2000_ECI）；与具体天体无关，由 mu 参数化。
    """

    MU_EARTH = 3.986004418e14  # m^3/s^2 (WGS84)

    def __init__(self, mu: float = None, r_min: float = 1.0):
        """
        Args:
            mu: 中心天体引力常数 (m^3/s^2)，默认 WGS84 地球值
            r_min: 数值保护半径 (m)，r < r_min 时输出零加速度
        """
        self.mu = float(mu if mu is not None else self.MU_EARTH)
        self.r_min = float(r_min)

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """a = -mu * r / |r|^3（state: [x,y,z,vx,vy,vz]，SI）。"""
        pos = np.asarray(state[:3], dtype=np.float64)
        return _central_accel(pos, self.mu, self.r_min)

    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """Batch compute for N states of shape (N, 6) -> (N, 3)。"""
        if state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
            raise ValueError(f"state_matrix must be of shape (N, 6), got {state_matrix.shape}")
        pos = state_matrix[:, :3]
        r = np.linalg.norm(pos, axis=1)
        out = np.zeros((pos.shape[0], 3), dtype=np.float64)
        valid = r >= self.r_min
        factor = -self.mu / (r[valid] ** 3)
        out[valid] = pos[valid] * factor[:, None]
        return out

    def __repr__(self) -> str:
        return f"CentralGravity(mu={self.mu:.6e} m^3/s^2)"
