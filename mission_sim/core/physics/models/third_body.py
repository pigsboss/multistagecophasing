# mission_sim/core/physics/models/third_body.py
"""
MCPC: 通用第三体引力力模型 (ThirdBodyGravity)
---------------------------------------------
a = μ3 · [ (r3 − r)/|r3 − r|³ − r3/|r3|³ ]   （含间接项，SI，惯性系）

第三体位置由「星历回调」提供：`position_provider(t) -> [x,y,z]`（或 6 维状态，取前 3）。
适用于任何多体场景（地月共振演化、日地系统、月球借力等）。

沉淀自 LUNAR-SWING 专题（LS-001 G5-①）。
"""

import numpy as np

from mission_sim.core.physics.environment import IForceModel


class ThirdBodyGravity(IForceModel):
    """第三体引力（含间接项；位置随时间的函数由调用方注入）。"""

    def __init__(self, mu_third: float, position_provider, r_min: float = 1.0,
                 name: str = "ThirdBody"):
        """
        Args:
            mu_third: 第三体引力常数 (m³/s²)
            position_provider: 可调用对象 t(s) → 第三体位置（至少 3 维，SI，惯性系）
            r_min: 数值保护距离 (m)
            name: 显示名（如 "Moon"）
        """
        self.mu = float(mu_third)
        self.position_provider = position_provider
        self.r_min = float(r_min)
        self.name = name

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        r3 = np.asarray(self.position_provider(epoch), dtype=np.float64)[:3]
        r = np.asarray(state[:3], dtype=np.float64)
        d = r3 - r
        dn = float(np.linalg.norm(d))
        r3n = float(np.linalg.norm(r3))
        if dn < self.r_min or r3n < self.r_min:
            return np.zeros(3, dtype=np.float64)
        return self.mu * (d / dn ** 3 - r3 / r3n ** 3)

    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """逐星循环回退（回调式星历不便向量化时的默认实现）。"""
        if state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
            raise ValueError(f"state_matrix must be of shape (N, 6), got {state_matrix.shape}")
        out = np.zeros((state_matrix.shape[0], 3), dtype=np.float64)
        for i in range(state_matrix.shape[0]):
            out[i] = self.compute_accel(state_matrix[i], epoch)
        return out

    def __repr__(self) -> str:
        return f"ThirdBodyGravity(name={self.name}, mu={self.mu:.4e} m³/s²)"
