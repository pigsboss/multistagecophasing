# mission_sim/core/physics/components/mechanisms/fast_steering_mirror.py
"""
MCPC L3: 快摆镜（FSM）组件
--------------------------
2 轴快摆镜的简化传递函数模型（二阶动力学 + 行程限幅），按 DD-001 D-06 不引入
全局非线性多体耦合；机构级细节（压电迟滞等）留待 L4。
"""

import numpy as np


class FastSteeringMirror:
    """2 轴快摆镜：二阶动力学（wn, zeta）+ 行程限幅，RK4 离散步进。"""

    def __init__(self, wn_hz: float = 500.0, zeta: float = 0.7, stroke_rad: float = 1e-3):
        self.wn = 2.0 * np.pi * float(wn_hz)
        self.zeta = float(zeta)
        self.stroke_rad = float(stroke_rad)
        self._theta = np.zeros(2, dtype=np.float64)
        self._omega = np.zeros(2, dtype=np.float64)

    @property
    def theta(self) -> np.ndarray:
        """当前镜面偏转角 (2,)（rad）。"""
        return self._theta.copy()

    def reset(self) -> None:
        self._theta[:] = 0.0
        self._omega[:] = 0.0

    def step(self, commanded_rad: np.ndarray, dt: float) -> np.ndarray:
        """按指令角步进 dt，返回更新后的偏转角。"""
        cmd = np.clip(np.asarray(commanded_rad, dtype=np.float64),
                      -self.stroke_rad, self.stroke_rad)
        wn, z = self.wn, self.zeta

        def deriv(state: np.ndarray) -> np.ndarray:
            th, om = state[:2], state[2:]
            return np.concatenate([om, wn * wn * (cmd - th) - 2.0 * z * wn * om])

        y = np.concatenate([self._theta, self._omega])
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._theta = np.clip(y[:2], -self.stroke_rad, self.stroke_rad)
        self._omega = y[2:]
        return self._theta.copy()

    def __repr__(self) -> str:
        return (f"FastSteeringMirror(wn={self.wn / (2 * np.pi):.0f} Hz, "
                f"zeta={self.zeta}, stroke={self.stroke_rad * 1e3:.1f} mrad)")
