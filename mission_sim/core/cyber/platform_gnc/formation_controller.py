"""
MCPC Cyber Domain: Formation Controller (L2) — v2（接通修订）
------------------------------------------------------------
三段式编队控制状态机：
    - GENERATION: 初始获取（大误差收敛）
    - KEEPING: 稳态保持（LQR + STM 时延预测）
    - RECONFIGURATION: 外部任务指令触发的构型重构（RE-001 I-1 闭合）

消费（可能延迟的）ISL 网络帧，输出 LVLH 加速度指令（Telecommand）。

v2 修订（WP-2.1，对应 DD-001 v1.1 R-L2-01/02/08）：
- 支持目标相对状态（target_state），状态机与 LQR 均基于“相对目标的误差”；
- 新增 command_reconfiguration() 外部触发（KEEPING→RECONFIGURATION）；
- 测量处理：球坐标→笛卡尔后，用连续测量有限差分估计 LVLH 相对速度
  （原实现将速度置零，属占位；现按旋转系相对速度口径近似，DD-001 R-L2-02）。
"""

import numpy as np
from typing import List, Optional, Tuple
from enum import Enum, auto

from mission_sim.core.cyber.ids import ISLNetworkFrame, FormationMode
from mission_sim.core.spacetime.ids import Telecommand, CoordinateFrame
from mission_sim.core.cyber.models.relative_dynamics import RelativeDynamics
from mission_sim.utils.math_tools import get_lqr_gain


class FormationController:
    """
    L2 Formation Controller with three-stage state machine and LQR control.

    输出语义：Telecommand.force_vector 为 **LVLH 加速度指令**（设计矩阵 B=I 口径），
    由仿真编排层换算为推力（×质量）并送入执行器。
    """

    def __init__(
        self,
        deputy_id: str,
        chief_id: str,
        dynamics: RelativeDynamics,
        # Control gains (can be precomputed)
        K_lqr: np.ndarray = None,
        # Thresholds for state machine
        generation_threshold_pos: float = 100.0,   # m
        generation_threshold_vel: float = 0.5,     # m/s
        keeping_threshold_pos: float = 1.0,        # m
        keeping_threshold_vel: float = 0.01,       # m/s
        # LQR weights (used if K_lqr not provided)
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        self.deputy_id = deputy_id
        self.chief_id = chief_id
        self.dynamics = dynamics

        # Default LQR weights (tunable)
        if Q is None:
            Q = np.diag([1.0, 1.0, 1.0, 1e4, 1e4, 1e4])
        if R is None:
            R = np.diag([1e6, 1e6, 1e6])

        if K_lqr is None:
            # Compute continuous-time LQR gain for CW dynamics
            A = dynamics._continuous_matrix() if hasattr(dynamics, '_continuous_matrix') else None
            if A is None:
                raise ValueError("Dynamics model must provide continuous matrix for LQR design when K_lqr is not provided")
            B = np.zeros((6, 3))
            B[3:6, 0:3] = np.eye(3)
            self.K = get_lqr_gain(A, B, Q, R)
        else:
            self.K = K_lqr

        # State machine thresholds
        self.gen_thresh_pos = generation_threshold_pos
        self.gen_thresh_vel = generation_threshold_vel
        self.keep_thresh_pos = keeping_threshold_pos
        self.keep_thresh_vel = keeping_threshold_vel

        # Internal state
        self.mode = FormationMode.GENERATION
        self.last_estimated_state = np.zeros(6)  # 相对主星状态 [dx,dy,dz,dvx,dvy,dvz]（LVLH）
        self.control_force = np.zeros(3)

        # v2 新增状态
        self.target_state = np.zeros(6)          # LVLH 目标相对状态
        self.update_calls = 0                    # 接通性计数（T-L2-001）
        self._prev_meas_pos: Optional[np.ndarray] = None
        self._prev_meas_time: Optional[float] = None

    # ------------------------------------------------------------------
    # v2：外部重构指令（三态机第三态触发）
    # ------------------------------------------------------------------
    def command_reconfiguration(self, new_target_state: np.ndarray) -> None:
        """外部任务指令：进入 RECONFIGURATION 并设定新目标（RE-001 I-1）。"""
        self.target_state = np.asarray(new_target_state, dtype=np.float64)
        self.mode = FormationMode.RECONFIGURATION

    def update(
        self,
        current_time: float,
        frames: List[ISLNetworkFrame],
        dt: float,
        target_state: Optional[np.ndarray] = None,
    ) -> Telecommand:
        """
        Update controller: process received frames, estimate current relative state,
        determine mode, compute control acceleration command.

        Args:
            current_time: Current simulation time (s)
            frames: List of received network frames (may be delayed or stale)
            dt: Time step for prediction (s)
            target_state: LVLH 目标相对状态 (6,)；若提供则更新内部目标（外部任务序列）

        Returns:
            Telecommand: 加速度指令（LVLH 系）
        """
        self.update_calls += 1
        if target_state is not None:
            self.target_state = np.asarray(target_state, dtype=np.float64)

        # 1. Extract valid measurements (not stale, destined for this deputy)
        valid_meas = []
        for frame in frames:
            if frame.dest_id != self.deputy_id:
                continue
            if frame.is_stale(current_time, max_delay=10.0):  # Configurable
                continue
            valid_meas.append(frame)

        # 2. If no valid measurement, use last state and predict
        if not valid_meas:
            stm = self.dynamics.compute_discrete_stm(dt)
            self.last_estimated_state = self.dynamics.predict_state(self.last_estimated_state, stm)
        else:
            # Use most recent measurement (by payload timestamp)
            latest = max(valid_meas, key=lambda f: f.payload.phys_timestamp)
            meas = latest.payload
            t_meas = meas.phys_timestamp

            if hasattr(meas, 'range_m'):
                # 球坐标 → 笛卡尔（LVLH）
                r = meas.range_m
                az = meas.azimuth_rad
                el = meas.elevation_rad
                pos = np.array([
                    r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az),
                    r * np.sin(el),
                ])
            else:
                pos = np.asarray(meas[:3], dtype=np.float64) if hasattr(meas, '__len__') else np.zeros(3)

            # v2：连续测量有限差分估计速度（LVLH 内差分 ≈ 旋转系相对速度）
            v_est = np.zeros(3)
            if self._prev_meas_pos is not None and (t_meas - self._prev_meas_time) > 1e-6:
                v_est = (pos - self._prev_meas_pos) / (t_meas - self._prev_meas_time)
            self._prev_meas_pos = pos.copy()
            self._prev_meas_time = t_meas

            measured_state = np.concatenate([pos, v_est])

            # Simple complementary filter: update with measurement (weighted)
            alpha = 0.7  # Tuning parameter
            self.last_estimated_state = alpha * measured_state + (1 - alpha) * self.last_estimated_state

            # Predict forward to current time (if measurement is older)
            age = current_time - t_meas
            if age > 1e-6:
                stm = self.dynamics.compute_discrete_stm(age)
                self.last_estimated_state = self.dynamics.predict_state(self.last_estimated_state, stm)

        # 3. State machine mode transition（基于“相对目标的误差”）
        error_state = self.last_estimated_state - self.target_state
        pos_err_norm = np.linalg.norm(error_state[0:3])
        vel_err_norm = np.linalg.norm(error_state[3:6])

        if self.mode == FormationMode.GENERATION:
            if pos_err_norm < self.gen_thresh_pos and vel_err_norm < self.gen_thresh_vel:
                self.mode = FormationMode.KEEPING
        elif self.mode == FormationMode.KEEPING:
            # 等待外部 command_reconfiguration() 触发（不做自动迁移）
            pass
        elif self.mode == FormationMode.RECONFIGURATION:
            if pos_err_norm < self.keep_thresh_pos and vel_err_norm < self.keep_thresh_vel:
                self.mode = FormationMode.KEEPING

        # 4. Compute control acceleration command（u = -K e，LVLH）
        u = -self.K @ error_state
        self.control_force = u

        # 5. Create telecommand（LVLH 加速度指令）
        cmd = Telecommand(
            force_vector=self.control_force,
            frame=CoordinateFrame.LVLH,
            duration_s=dt,
            actuator_id=f"{self.deputy_id}_thruster"
        )
        return cmd

    def get_mode(self) -> FormationMode:
        """Return current control mode."""
        return self.mode

    def reset(self):
        """Reset controller state."""
        self.mode = FormationMode.GENERATION
        self.last_estimated_state = np.zeros(6)
        self.control_force = np.zeros(3)
        self.target_state = np.zeros(6)
        self._prev_meas_pos = None
        self._prev_meas_time = None
