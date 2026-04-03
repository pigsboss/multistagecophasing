"""
MCPC Cyber Domain: Clohessy-Wiltshire (CW) Relative Dynamics
-------------------------------------------------------------
Implements the linearized relative motion equations for circular reference orbits.
Provides discrete-time state transition matrix for formation keeping.
"""

import numpy as np
from mission_sim.core.cyber.models.relative_dynamics import RelativeDynamics


class CWDynamics(RelativeDynamics):
    """
    Clohessy-Wiltshire (CW) / Hill's equations for circular orbits.
    Assumes:
        - Chief in circular orbit (constant angular velocity n)
        - Small relative distances compared to orbit radius
        - No perturbations (J2, drag, etc.)
        - LVLH frame: X=radial, Y=along-track, Z=cross-track
    """

    def __init__(self, n: float):
        """
        Initialize CW dynamics.

        Args:
            n: Mean orbital angular velocity of the chief (rad/s) = sqrt(mu / r^3)
        """
        self.n = n
        # Precompute common matrix components (time-invariant)
        self._A = self._continuous_matrix()

    def _continuous_matrix(self) -> np.ndarray:
        """Continuous-time state matrix A (6x6) for CW equations."""
        n = self.n
        A = np.zeros((6, 6))
        # Position to velocity
        A[0:3, 3:6] = np.eye(3)
        # Acceleration components
        A[3, 0] = 3 * n**2
        A[3, 4] = 2 * n
        A[4, 1] = 0.0
        A[4, 3] = -2 * n
        A[5, 2] = -n**2
        return A

    def compute_discrete_stm(self, dt: float) -> np.ndarray:
        """
        Compute discrete-time STM Φ(dt) = exp(A * dt) using analytical formula.

        The CW STM has a closed-form expression (see Vallado or Curtis).
        For small dt (< 0.1 orbit period), matrix exponential is also accurate.
        Here we use analytical formulas for exactness.
        """
        n = self.n
        s = np.sin(n * dt)
        c = np.cos(n * dt)

        # Position-to-position (3x3)
        phi_rr = np.array([
            [4 - 3*c, 0, 0],
            [6*(s - n*dt), 1, 0],
            [0, 0, c]
        ])

        # Position-to-velocity (3x3)
        phi_rv = np.array([
            [s / n, 2*(1 - c)/n, 0],
            [2*(c - 1)/n, (4*s - 3*n*dt)/n, 0],
            [0, 0, s / n]
        ])

        # Velocity-to-position (3x3)
        phi_vr = np.array([
            [3*n*s, 0, 0],
            [6*n*(c - 1), 0, 0],
            [0, 0, -n*s]
        ])

        # Velocity-to-velocity (3x3)
        phi_vv = np.array([
            [c, 2*s, 0],
            [-2*s, 4*c - 3, 0],
            [0, 0, c]
        ])

        # Assemble 6x6 STM
        stm = np.zeros((6, 6))
        stm[0:3, 0:3] = phi_rr
        stm[0:3, 3:6] = phi_rv
        stm[3:6, 0:3] = phi_vr
        stm[3:6, 3:6] = phi_vv

        return stm.astype(np.float64)

    def predict_state(self, current_state: np.ndarray, stm: np.ndarray) -> np.ndarray:
        """Apply STM to current state."""
        return stm @ current_state

    def __repr__(self) -> str:
        return f"CWDynamics(n={self.n:.6e} rad/s)"