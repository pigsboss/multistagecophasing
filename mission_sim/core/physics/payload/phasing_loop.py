# mission_sim/core/physics/payload/phasing_loop.py
"""
MCPC L3: 共相闭环原型装配器（PhasingLoop）
------------------------------------------
把 L3 各件装配为可运行的简化闭环：
    平台抖动（平台-载荷双层模型）
      → Stewart 平台（粗调，二阶传递函数 2 Hz）
      → 快摆镜 FSM（精调，二阶传递函数 500 Hz）
      → OPD 简化链路（κ·‖θ_res‖）
      ↺ 分层控制（粗调 + 精调，payload_ctrl）

简化假设（DD-001 D-06 不越界）：
- 机构为二阶传递函数模型，不做全局非线性多体耦合；
- OPD 为角残差的线性映射 κ·‖θ‖；
- 解耦假设由 PlatformPayloadModel.validate_decoupling() 校验。
"""

import numpy as np

from mission_sim.core.physics.components.mechanisms.stewart_platform import StewartPlatform
from mission_sim.core.physics.components.mechanisms.fast_steering_mirror import FastSteeringMirror
from mission_sim.core.cyber.payload_ctrl.hierarchical_controller import HierarchicalPhasingController
from .platform_payload import PlatformPayloadModel
from .opd_link import OPDLink


class PhasingLoop:
    """L3 共相闭环原型装配器（默认参数即验收基线配置）。"""

    def __init__(self, lever_arm_m: float = 2.5, dt: float = 1e-4, seed: int = 20261010,
                 platform_mass_kg: float = 1000.0, payload_mass_kg: float = 50.0,
                 stewart_wn_hz: float = 2.0, stewart_zeta: float = 0.9,
                 fsm_wn_hz: float = 500.0, fsm_zeta: float = 0.7,
                 disturbance_scale: float = 1.0,
                 controller: HierarchicalPhasingController = None):
        self.dt = float(dt)
        self.disturbance_scale = float(disturbance_scale)
        self.platform = PlatformPayloadModel(
            platform_mass_kg=platform_mass_kg,
            payload_mass_kg=payload_mass_kg,
            seed=seed,
        )
        self.platform.validate_decoupling()
        self.opd_link = OPDLink(lever_arm_m=lever_arm_m)
        self.stewart = StewartPlatform()  # 几何参考（运动学校验用）
        self.fsm = FastSteeringMirror(wn_hz=fsm_wn_hz, zeta=fsm_zeta)
        self._wn_s = 2.0 * np.pi * float(stewart_wn_hz)
        self._zeta_s = float(stewart_zeta)
        self._stroke_s = 2e-3  # Stewart 行程（rad，简化）
        self.controller = controller if controller is not None else HierarchicalPhasingController(dt=self.dt)
        self.reset()

    def reset(self) -> None:
        self._th_s = np.zeros(2, dtype=np.float64)
        self._om_s = np.zeros(2, dtype=np.float64)
        self.fsm.reset()
        self.controller.reset()

    def _stewart_step(self, commanded_rad: np.ndarray) -> np.ndarray:
        """Stewart 平台（简化二轴）二阶动力学步进。"""
        cmd = np.clip(np.asarray(commanded_rad, dtype=np.float64), -self._stroke_s, self._stroke_s)
        wn, z = self._wn_s, self._zeta_s

        def deriv(state):
            th, om = state[:2], state[2:]
            return np.concatenate([om, wn * wn * (cmd - th) - 2.0 * z * wn * om])

        y = np.concatenate([self._th_s, self._om_s])
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * self.dt * k1)
        k3 = deriv(y + 0.5 * self.dt * k2)
        k4 = deriv(y + self.dt * k3)
        y = y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._th_s, self._om_s = y[:2], y[2:]
        return self._th_s.copy()

    # ------------------------------------------------------------------
    def run(self, duration_s: float = 4.0) -> dict:
        """抖动抑制闭环运行；同时给出开环对照（同一扰动、无控制）。"""
        n = int(round(float(duration_s) / self.dt))
        t = np.arange(n) * self.dt
        jitter = self.disturbance_scale * self.platform.jitter_series(t)
        self.reset()

        opd_cl = np.zeros(n, dtype=np.float64)
        opd_ol = np.zeros(n, dtype=np.float64)
        th_s_hist = np.zeros((n, 2), dtype=np.float64)
        th_f_hist = np.zeros((n, 2), dtype=np.float64)

        for k in range(n):
            err = jitter[k] - (self._th_s + self.fsm.theta)
            u_s, u_f = self.controller.update(err)
            self._stewart_step(u_s)
            self.fsm.step(u_f, self.dt)

            th_res = jitter[k] - (self._th_s + self.fsm.theta)
            opd_cl[k] = self.opd_link.opd(th_res)
            opd_ol[k] = self.opd_link.opd(jitter[k])
            th_s_hist[k] = self._th_s
            th_f_hist[k] = self.fsm.theta

        return {
            "t": t, "dt": self.dt,
            "opd_residual": opd_cl, "opd_open_loop": opd_ol,
            "theta_platform": jitter, "theta_stewart": th_s_hist, "theta_fsm": th_f_hist,
        }

    def step_response(self, step_rad: float = 1e-5, duration_s: float = 1.0) -> dict:
        """指向指令阶跃响应（轴 0 阶跃，其余为 0）。"""
        n = int(round(float(duration_s) / self.dt))
        t = np.arange(n) * self.dt
        self.reset()

        ref = np.array([float(step_rad), 0.0])
        corr = np.zeros(n, dtype=np.float64)
        for k in range(n):
            err = ref - (self._th_s + self.fsm.theta)
            u_s, u_f = self.controller.update(err)
            self._stewart_step(u_s)
            self.fsm.step(u_f, self.dt)
            corr[k] = float((self._th_s + self.fsm.theta)[0])

        return {"t": t, "ref_rad": float(step_rad), "correction": corr}
