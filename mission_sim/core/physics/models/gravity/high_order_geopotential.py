# mission_sim/core/physics/models/gravity/high_order_geopotential.py
"""
MCPC: 带谐地球重力场 J2–J4（HighOrderGeopotential）
---------------------------------------------------
由 J2 桩升级（LUNAR-SWING 专题沉淀件，LS-001 G5-③）。

势函数：
    V(r) = -(μ/r)·[1 - J2 (Re/r)² P2(u) - J3 (Re/r)³ P3(u) - J4 (Re/r)⁴ P4(u)],  u = z/r
加速度 a = −∇V（闭式实现；由势函数有限差分梯度独立校验，见 T-LS-005）。
J2 项与 `j2_gravity.J2Gravity` 逐位一致（同一公式）。
不适用的高阶/田谐项（>4 阶、C22/S22）留待后续版本。
"""

import numpy as np

from mission_sim.core.physics.environment import IForceModel


class HighOrderGeopotential(IForceModel):
    """带谐重力场（J2/J3/J4；惯性系，SI）。

    兼容旧桩接口（degree/order/coeff_file/set_max_degree，元数据），
    供 L1 存量测试使用；本版本物理实现上限为 J2–J4 带谐。
    """

    MU_EARTH = 3.986004418e14
    J2 = 1.08262668e-3
    J3 = -2.5321531e-6
    J4 = -1.61098761e-6
    R_EARTH = 6378137.0

    def __init__(self, mu_earth: float = None, j2: float = None, j3: float = None,
                 j4: float = None, r_earth: float = None,
                 degree: int = 10, order: int = 10, coeff_file=None):
        self.mu_earth = float(mu_earth if mu_earth is not None else self.MU_EARTH)
        self.j2 = float(j2 if j2 is not None else self.J2)
        self.j3 = float(j3 if j3 is not None else self.J3)
        self.j4 = float(j4 if j4 is not None else self.J4)
        self.r_earth = float(r_earth if r_earth is not None else self.R_EARTH)
        self.degree = int(degree)
        self.order = int(order)
        self.coeff_file = coeff_file

    def set_max_degree(self, degree: int) -> None:
        """兼容旧桩接口：记录阶次上限（物理实现仍为 J2–J4）。"""
        self.degree = int(degree)

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        x, y, z = float(state[0]), float(state[1]), float(state[2])
        mu, re = self.mu_earth, self.r_earth
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        if r < 1.0:
            return np.zeros(3, dtype=np.float64)
        u = z / r
        re_r = re / r
        r3 = r2 * r
        r5 = r2 * r2 * r

        # ---- J2（与 j2_gravity 同一公式）----
        f2 = -1.5 * mu * self.j2 * re * re / r5
        a2 = np.array([f2 * x * (5.0 * z * z / r2 - 1.0),
                       f2 * y * (5.0 * z * z / r2 - 1.0),
                       f2 * z * (5.0 * z * z / r2 - 3.0)])

        # ---- J3：a = -∇V3 ----
        c3 = mu * self.j3 * re ** 3
        f3 = -2.5 * c3 / (r5 * r2)          # -2.5·C3/r⁷
        u3 = 7.0 * u * u * u - 3.0 * u
        a3 = np.array([f3 * x * (u3 * r),
                       f3 * y * (u3 * r),
                       c3 * ((15.0 * u * u - 3.0) * (1.0 - u * u) / (2.0 * r5)
                             - 4.0 * z * (0.5 * (5.0 * u ** 3 - 3.0 * u)) / (r5 * r))])

        # ---- J4：a = -∇V4（V4 = -C4·P4/r⁵）----
        c4 = mu * self.j4 * re ** 4
        u4 = 21.0 * u ** 4 - 14.0 * u * u + 1.0
        p4 = (35.0 * u ** 4 - 30.0 * u * u + 3.0) / 8.0
        f4 = -(15.0 / 8.0) * c4 / (r5 * r2)          # -15·C4/(8r⁷)
        a4 = np.array([f4 * x * u4,
                       f4 * y * u4,
                       c4 * ((35.0 * u ** 3 - 15.0 * u) * (1.0 - u * u) / (2.0 * r5 * r)
                             - 5.0 * z * p4 / (r5 * r2))])

        return a2 + a3 + a4

    def __repr__(self) -> str:
        return (f"HighOrderGeopotential(J2={self.j2:.3e}, J3={self.j3:.3e}, "
                f"J4={self.j4:.3e}, Re={self.r_earth:.0f} m)")
