# mission_sim/core/physics/payload/opd_link.py
"""
MCPC L3: OPD（光程差）简化链路
------------------------------
残余指向角 → OPD 的线性映射：opd = κ · ‖θ_res‖，κ = 2 × 力臂（简化几何）。
按 DD-001 D-06（不越界）：不做完整光学传播，仅保留等效一阶几何关系。

另提供 Allan 偏差（重叠式、τ 平均）计算，用于残余 OPD 的稳定度指标。
"""

import numpy as np


class OPDLink:
    """OPD 简化链路（κ = 2 × lever_arm_m）。"""

    def __init__(self, lever_arm_m: float = 2.5):
        self.lever_arm_m = float(lever_arm_m)

    @property
    def kappa(self) -> float:
        return 2.0 * self.lever_arm_m

    def opd(self, theta_residual_rad):
        """残余角 → OPD (m)。输入 (2,) 返回标量；输入 (N, 2) 返回 (N,)。"""
        th = np.asarray(theta_residual_rad, dtype=np.float64)
        if th.ndim == 1:
            return self.kappa * float(np.linalg.norm(th))
        return self.kappa * np.linalg.norm(th, axis=1)

    def __repr__(self) -> str:
        return f"OPDLink(lever={self.lever_arm_m} m, kappa={self.kappa} m/rad)"


def allan_deviation(x: np.ndarray, dt: float, taus) -> np.ndarray:
    """重叠 Allan 偏差：σ(τ)，τ = m·dt（对 OPD 残差时间序列做稳定度度量）。"""
    x = np.asarray(x, dtype=np.float64)
    out = []
    for tau in taus:
        m = max(1, int(round(float(tau) / float(dt))))
        if x.size < 2 * m:
            out.append(np.nan)
            continue
        n = x.size // m
        y = x[: n * m].reshape(n, m).mean(axis=1)
        d = np.diff(y)
        out.append(float(np.sqrt(0.5 * float(np.mean(d ** 2)))))
    return np.array(out, dtype=np.float64)
