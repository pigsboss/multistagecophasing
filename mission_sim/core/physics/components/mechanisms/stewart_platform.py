# mission_sim/core/physics/components/mechanisms/stewart_platform.py
"""
MCPC L3: Stewart 平台（6-6 构型）运动学映射
--------------------------------------------
- 逆运动学：位姿 → 6 支杆长度（解析）；
- 正运动学：6 支杆长度 → 位姿（牛顿迭代 + 数值雅可比）。

按 DD-001 D-06（不越界）：本模块仅实现运动学映射；动力学由二阶传递函数表示
（在装配层 L3 闭环原型中使用）。
位姿定义：pose = [x, y, z, rx, ry, rz]，平移 (m) + 旋转向量 (rad)，相对零位的小量程。
"""

import numpy as np


def rotation_vector_to_matrix(rv: np.ndarray) -> np.ndarray:
    """罗德里格斯公式：旋转向量 (rad) → 旋转矩阵。"""
    rv = np.asarray(rv, dtype=np.float64)
    theta = float(np.linalg.norm(rv))
    K = np.array([[0.0, -rv[2], rv[1]],
                  [rv[2], 0.0, -rv[0]],
                  [-rv[1], rv[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K + 0.5 * (K @ K)
    K = K / theta
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class StewartPlatform:
    """6-6 Stewart 平台运动学（上下各 6 个铰点，顶/底错开 30° 布置）。"""

    def __init__(self, base_radius_m: float = 0.20, top_radius_m: float = 0.15,
                 nominal_height_m: float = 0.20,
                 base_pair_offset_deg: float = 8.0, top_pair_offset_deg: float = 14.0):
        self.base_radius = float(base_radius_m)
        self.top_radius = float(top_radius_m)
        self.height0 = float(nominal_height_m)

        # 经典 6-6 构型：三个"角"上成对布置，顶/底错开 60°。
        # 注意（M3 调试记录）：
        # ① 均布 60° 的"全对称"构型雅可比秩亏（6 DOF 退化到 3 维）；
        # ② 顶/底对间距必须不同（α_b ≠ α_t）——同名间距时锚点构型互为"同伦像"，
        #    中心位形奇异（雅可比秩亏）。
        off_b = np.deg2rad(float(base_pair_offset_deg))
        off_t = np.deg2rad(float(top_pair_offset_deg))
        corners = np.deg2rad(np.array([0.0, 120.0, 240.0]))
        base_angles = (corners[:, None] + np.array([-off_b, off_b])[None, :]).ravel()
        top_angles = (corners[:, None] + np.deg2rad(60.0) + np.array([-off_t, off_t])[None, :]).ravel()
        self.base_anchors = np.array([
            [self.base_radius * np.cos(a), self.base_radius * np.sin(a), 0.0]
            for a in base_angles
        ], dtype=np.float64)
        self.top_anchors = np.array([
            [self.top_radius * np.cos(a), self.top_radius * np.sin(a), 0.0]
            for a in top_angles
        ], dtype=np.float64)
        self._top_origin_nominal = np.array([0.0, 0.0, self.height0], dtype=np.float64)

    # ------------------------------------------------------------------
    def inverse_kinematics(self, pose: np.ndarray) -> np.ndarray:
        """位姿 (6,) → 支杆长度 (6,)。"""
        pose = np.asarray(pose, dtype=np.float64)
        R = rotation_vector_to_matrix(pose[3:6])
        top_abs = self._top_origin_nominal + pose[0:3] + (R @ self.top_anchors.T).T
        return np.linalg.norm(top_abs - self.base_anchors, axis=1)

    def _jacobian(self, pose: np.ndarray, h: float = 1e-7) -> np.ndarray:
        J = np.zeros((6, 6), dtype=np.float64)
        for j in range(6):
            dp = np.zeros(6)
            dp[j] = h
            fp = self.inverse_kinematics(pose + dp)
            fm = self.inverse_kinematics(pose - dp)
            J[:, j] = (fp - fm) / (2.0 * h)
        return J

    def forward_kinematics(self, lengths: np.ndarray, guess: np.ndarray = None,
                           tol: float = 1e-13, max_iter: int = 200) -> np.ndarray:
        """支杆长度 (6,) → 位姿 (6,)（牛顿迭代）。"""
        lengths = np.asarray(lengths, dtype=np.float64)
        pose = np.zeros(6) if guess is None else np.array(guess, dtype=np.float64)
        for _ in range(max_iter):
            f = self.inverse_kinematics(pose) - lengths
            if float(np.max(np.abs(f))) < tol:
                break
            J = self._jacobian(pose)
            dpose, *_ = np.linalg.lstsq(J, -f, rcond=None)
            pose = pose + dpose
        return pose

    def __repr__(self) -> str:
        return (f"StewartPlatform(base_r={self.base_radius} m, top_r={self.top_radius} m, "
                f"height={self.height0} m)")
