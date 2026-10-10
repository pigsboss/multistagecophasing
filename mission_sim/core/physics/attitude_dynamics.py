# mission_sim/core/physics/attitude_dynamics.py
"""
MCPC L4: 6-DOF 刚体姿轨耦合动力学（四元数）
--------------------------------------------
按 DD-001 R-L4-01 / D-09：航天器姿态动力学新建于物理域（不复活旧 attitude 包职责）。

约定（M4 修正：以数值守恒验证为准）：
- 四元数 q = [w, x, y, z]（单位四元数），表示 **body→inertial** 旋转（R(q) = C_in←b）：
  v_inertial = R(q) · v_body；q̇ = ½ q ⊗ [0, ω]（ω 为体坐标系角速度）；
- 动力学：轨道 r̈ = −μr/|r|³（可叠加控制/外力加速度）；姿态 I·ω̇ = τ − ω×(I·ω)；
- 无控自由演化守恒量：惯性系角动量 L = R·(I·ω)、机械能（轨道 + 旋转动能）。
"""

import numpy as np


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """单位四元数 → 旋转矩阵（body→inertial，即 v_inertial = R·v_body）。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """q̇ = ½ q ⊗ [0, ω]（体坐标系角速度）。"""
    w, x, y, z = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -(x * wx + y * wy + z * wz),
        w * wx + y * wz - z * wy,
        w * wy - x * wz + z * wx,
        w * wz + x * wy - y * wx,
    ])


class RigidBody6DOF:
    """6-DOF 刚体（质量 + 对角惯量），RK4 离散步进。状态 13 维：r(3)+v(3)+q(4)+ω(3)。"""

    def __init__(self, mass_kg: float, inertia_diag_kgm2, mu: float = 3.986004418e14,
                 r_min: float = 1.0):
        self.mass = float(mass_kg)
        self.I = np.diag(np.asarray(inertia_diag_kgm2, dtype=np.float64))
        self.I_inv = np.diag(1.0 / np.asarray(inertia_diag_kgm2, dtype=np.float64))
        self.mu = float(mu)
        self.r_min = float(r_min)
        # 默认初始状态（需通过 set_state 显式设定）
        self._state = np.zeros(13, dtype=np.float64)
        self._state[6] = 1.0  # 单位四元数

    # ------------------------------------------------------------------
    def set_state(self, r, v, quaternion, angular_velocity) -> None:
        q = np.asarray(quaternion, dtype=np.float64)
        q = q / np.linalg.norm(q)
        self._state = np.concatenate([np.asarray(r, float), np.asarray(v, float), q,
                                      np.asarray(angular_velocity, float)])

    @property
    def position(self) -> np.ndarray:
        return self._state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._state[3:6].copy()

    @property
    def quaternion(self) -> np.ndarray:
        return self._state[6:10].copy()

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._state[10:13].copy()

    # ------------------------------------------------------------------
    def angular_momentum_inertial(self) -> np.ndarray:
        """惯性系角动量 L = R·(I·ω)（R = C_in←b，见 quaternion_to_matrix）。"""
        R = quaternion_to_matrix(self.quaternion)
        return R @ (self.I @ self.angular_velocity)

    def mechanical_energy(self) -> float:
        """机械能 = 轨道动能 + 势能 + 旋转动能。"""
        v = self.velocity
        r = float(np.linalg.norm(self.position))
        w = self.angular_velocity
        ke_orb = 0.5 * self.mass * float(np.dot(v, v))
        pe = -self.mu * self.mass / max(r, self.r_min)
        ke_rot = 0.5 * float(w @ (self.I @ w))
        return ke_orb + pe + ke_rot

    # ------------------------------------------------------------------
    def _deriv(self, state: np.ndarray, torque: np.ndarray) -> np.ndarray:
        r = state[0:3]
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]
        r_norm = float(np.linalg.norm(r))
        a = -self.mu * r / max(r_norm, self.r_min) ** 3
        dw = self.I_inv @ (torque - np.cross(w, self.I @ w))
        dq = quaternion_derivative(q, w)
        return np.concatenate([v, a, dq, dw])

    def step(self, torque: np.ndarray, dt: float) -> None:
        """RK4 步进 dt；步末归一化四元数。"""
        torque = np.asarray(torque, dtype=np.float64)
        y = self._state
        k1 = self._deriv(y, torque)
        k2 = self._deriv(y + 0.5 * dt * k1, torque)
        k3 = self._deriv(y + 0.5 * dt * k2, torque)
        k4 = self._deriv(y + dt * k3, torque)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[6:10] = y[6:10] / np.linalg.norm(y[6:10])
        self._state = y

    def __repr__(self) -> str:
        return (f"RigidBody6DOF(m={self.mass} kg, I_diag={np.diag(self.I)}, "
                f"|r|={np.linalg.norm(self.position):.3e} m)")
