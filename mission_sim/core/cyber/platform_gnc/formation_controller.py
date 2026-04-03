"""
MCPC Cyber Domain: Formation Controller (L2)
--------------------------------------------
Implements three-stage formation control:
    - GENERATION: Initial acquisition (reduce large errors)
    - KEEPING: Steady-state maintenance (LQR with STM prediction)
    - RECONFIGURATION: Trajectory tracking for configuration change

Consumes delayed network frames and outputs thrust commands in LVLH frame.
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
        """
        Initialize formation controller.

        Args:
            deputy_id: ID of this deputy spacecraft
            chief_id: ID of the chief/reference spacecraft
            dynamics: Relative dynamics model (e.g., CWDynamics)
            K_lqr: Precomputed LQR gain matrix (3x6). If None, compute from Q,R.
            generation_threshold_pos: Position error below which GENERATION considered converged
            generation_threshold_vel: Velocity error below which GENERATION considered converged
            keeping_threshold_pos: Position error threshold for KEEPING mode
            keeping_threshold_vel: Velocity error threshold for KEEPING mode
            Q: State weighting matrix (6x6) for LQR (if K_lqr not provided)
            R: Control weighting matrix (3x3) for LQR (if K_lqr not provided)
        """
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
            # (linearized around zero relative state)
            A = dynamics._continuous_matrix() if hasattr(dynamics, '_continuous_matrix') else None
            if A is None:
                raise ValueError("Dynamics model must provide continuous matrix for LQR design")
            B = np.zeros((6, 3))
            B[3:6, 0:3] = np.eye(3)  # Control affects acceleration directly
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
        self.last_estimated_state = np.zeros(6)  # Relative state [dx,dy,dz,dvx,dvy,dvz]
        self.control_force = np.zeros(3)

    def update(
        self,
        current_time: float,
        frames: List[ISLNetworkFrame],
        dt: float
    ) -> Telecommand:
        """
        Update controller: process received frames, estimate current relative state,
        determine mode, compute control force.

        Args:
            current_time: Current simulation time (s)
            frames: List of received network frames (may be delayed or stale)
            dt: Time step for prediction (s)

        Returns:
            Telecommand: Force command in LVLH frame
        """
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
            # Predict forward using STM
            stm = self.dynamics.compute_discrete_stm(dt)
            self.last_estimated_state = self.dynamics.predict_state(self.last_estimated_state, stm)
        else:
            # Use most recent measurement (by timestamp)
            latest = max(valid_meas, key=lambda f: f.payload.phys_timestamp)
            # Extract relative position from measurement payload
            # Assumes payload is MicrowaveISLMeasurement (range, azimuth, elevation)
            # Convert to Cartesian in LVLH frame
            meas = latest.payload
            # Simplified: assume measurement directly provides Cartesian? For now, placeholder.
            # In real implementation, convert spherical to Cartesian.
            # Placeholder: assume measurement contains relative position vector
            # For L2, we'll assume the frame already provides relative position in LVLH.
            # Here we directly use the relative state from the measurement.
            # For simplicity, we treat the measurement as relative position (m) and zero velocity.
            # A more complete implementation would use a relative navigation filter.
            # For Sprint 3, we assume the measurement gives relative position directly.
            # TODO: integrate with relative navigation filter.
            if hasattr(meas, 'range_m'):
                # Spherical to Cartesian conversion
                r = meas.range_m
                az = meas.azimuth_rad
                el = meas.elevation_rad
                dx = r * np.cos(el) * np.cos(az)
                dy = r * np.cos(el) * np.sin(az)
                dz = r * np.sin(el)
                measured_state = np.array([dx, dy, dz, 0.0, 0.0, 0.0])
            else:
                # Assume direct Cartesian vector (for testing)
                measured_state = np.array(meas) if hasattr(meas, '__len__') else np.zeros(6)

            # Simple complementary filter: update with measurement (weighted)
            alpha = 0.7  # Tuning parameter
            self.last_estimated_state = alpha * measured_state + (1 - alpha) * self.last_estimated_state

            # Predict forward to current time (if measurement is older)
            age = current_time - meas.phys_timestamp
            if age > 1e-6:
                stm = self.dynamics.compute_discrete_stm(age)
                self.last_estimated_state = self.dynamics.predict_state(self.last_estimated_state, stm)

        # 3. State machine mode transition
        pos_err_norm = np.linalg.norm(self.last_estimated_state[0:3])
        vel_err_norm = np.linalg.norm(self.last_estimated_state[3:6])

        if self.mode == FormationMode.GENERATION:
            if pos_err_norm < self.gen_thresh_pos and vel_err_norm < self.gen_thresh_vel:
                self.mode = FormationMode.KEEPING
        elif self.mode == FormationMode.KEEPING:
            # Stay in KEEPING unless commanded to reconfigure (external trigger)
            # For now, no auto-transition from KEEPING to RECONFIGURATION
            pass
        elif self.mode == FormationMode.RECONFIGURATION:
            # Check if reconfiguration target achieved (target would be set externally)
            # For simplicity, if error small, go back to KEEPING
            if pos_err_norm < self.keep_thresh_pos and vel_err_norm < self.keep_thresh_vel:
                self.mode = FormationMode.KEEPING

        # 4. Compute control force using LQR
        # For GENERATION mode, use same LQR but with higher gains? Use same for now.
        # For RECONFIGURATION, could use different gains, but same for simplicity.
        u = -self.K @ self.last_estimated_state  # u = -K * x (force in LVLH)
        self.control_force = u

        # 5. Create telecommand
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