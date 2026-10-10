# mission_sim/core/cyber/payload_ctrl/hierarchical_controller.py
"""
MCPC L3: 分层共相控制（粗调 Stewart + 精调 FSM）
------------------------------------------------
按 DD-001 R-L3-05：平台粗调（低带宽、承担慢漂移与大行程）+ 载荷精调（高带宽、抑制快抖动）。
输出为两轴期望修正角（rad）：
    u_stewart → Stewart 平台；u_fsm → 快摆镜。

实现：精调环为 PI + 滤波微分（lead）；粗调环为低增益纯积分（cutoff 远低于精调带宽）。
"""

import numpy as np


class HierarchicalPhasingController:
    """分层共相控制器（2 轴解耦，参数对各轴相同）。"""

    def __init__(self, dt: float,
                 kp_fsm: float = 2.5, ki_fsm: float = 800.0, kd_fsm: float = 2.0e-4,
                 kp_stewart: float = 0.0, ki_stewart: float = 20.0,
                 derivative_lp_hz: float = 2000.0, integral_limit_rad: float = 1e-2):
        self.dt = float(dt)
        self.kp_fsm = float(kp_fsm)
        self.ki_fsm = float(ki_fsm)
        self.kd_fsm = float(kd_fsm)
        self.kp_stewart = float(kp_stewart)
        self.ki_stewart = float(ki_stewart)
        self._w_d = 2.0 * np.pi * float(derivative_lp_hz)
        self._ilim = float(integral_limit_rad)
        self.reset()

    def reset(self) -> None:
        self._ie_f = np.zeros(2, dtype=np.float64)
        self._ie_s = np.zeros(2, dtype=np.float64)
        self._e_prev = np.zeros(2, dtype=np.float64)
        self._e_dot_f = np.zeros(2, dtype=np.float64)

    def update(self, error_rad: np.ndarray):
        """输入当前 LOS 误差角 (2,)（rad），返回 (u_stewart, u_fsm)。"""
        e = np.asarray(error_rad, dtype=np.float64)

        # 滤波微分（一阶低通，抑制噪声放大）
        raw = (e - self._e_prev) / self.dt
        a = self._w_d * self.dt / (1.0 + self._w_d * self.dt)
        self._e_dot_f = (1.0 - a) * self._e_dot_f + a * raw
        self._e_prev = e.copy()

        # 积分（带限幅）
        self._ie_f = np.clip(self._ie_f + e * self.dt, -self._ilim, self._ilim)
        self._ie_s = np.clip(self._ie_s + e * self.dt, -self._ilim, self._ilim)

        # 精调环（FSM）：PI + lead
        u_fsm = self.kp_fsm * e + self.ki_fsm * self._ie_f + self.kd_fsm * self._e_dot_f
        # 粗调环（Stewart）：低增益积分（慢漂移/大行程）
        u_stewart = self.kp_stewart * e + self.ki_stewart * self._ie_s
        return u_stewart, u_fsm

    def __repr__(self) -> str:
        return (f"HierarchicalPhasingController(dt={self.dt:.1e}, kp_f={self.kp_fsm}, "
                f"ki_f={self.ki_fsm}, kd_f={self.kd_fsm}, ki_s={self.ki_stewart})")
