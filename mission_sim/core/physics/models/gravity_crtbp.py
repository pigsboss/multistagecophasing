# mission_sim/core/physics/models/gravity_crtbp.py
"""
MCPC Core Physics Models: Dual-Engine Vectorized CRTBP
------------------------------------------------------
Implements the Circular Restricted Three-Body Problem (CRTBP) in SI units.
Optimized with Numba JIT for L1 single-satellite precision and 
NumPy vectorization for L2 multi-satellite formation performance.
"""

import numpy as np
from mission_sim.core.physics.models.base import ForceModel

# --- Numba Optimization Guard ---
try:
    from numba import jit as njit
    _HAS_NUMBA = True
except ImportError:
    def njit(func): return func
    _HAS_NUMBA = False

@njit
def _crtbp_accel_numba(pos, vel, gm1, gm2, omega, x1, x2):
    """[L1-OPTIMIZED] Numba-accelerated single state calculation."""
    dx1 = pos[0] - x1
    dx2 = pos[0] - x2
    r1_3 = (dx1**2 + pos[1]**2 + pos[2]**2)**1.5
    r2_3 = (dx2**2 + pos[1]**2 + pos[2]**2)**1.5
    
    # Gravitational components
    ax = -gm1 * dx1 / r1_3 - gm2 * dx2 / r2_3
    ay = -gm1 * pos[1] / r1_3 - gm2 * pos[1] / r2_3
    az = -gm1 * pos[2] / r1_3 - gm2 * pos[2] / r2_3
    
    # Fictitious forces (Centrifugal & Coriolis)
    ax += omega**2 * pos[0] + 2.0 * omega * vel[1]
    ay += omega**2 * pos[1] - 2.0 * omega * vel[0]
    
    return np.array([ax, ay, az], dtype=np.float64)

class GravityCRTBP(ForceModel):
    """
    [MCPC UNIVERSAL] CRTBP Gravity Model.
    Supports Sun-Earth system by default using SI units.
    """

    def __init__(self):
        # Physical Constants (SI Units: m, s, kg)
        self.GM_SUN = 1.32712440018e20
        self.GM_EARTH = 3.986004418e14
        self.AU = 1.495978707e11
        self.OMEGA = 1.990986e-7  # Mean angular velocity of Earth (rad/s)

        # Pre-computed System Parameters
        self.mu = self.GM_EARTH / (self.GM_SUN + self.GM_EARTH)
        self._x1 = -self.mu * self.AU
        self._x2 = (1.0 - self.mu) * self.AU
        self._omega_sq = self.OMEGA**2

    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """[L1 LEGACY] Compute acceleration for a SINGLE spacecraft using Numba."""
        return _crtbp_accel_numba(
            state[0:3], state[3:6], 
            self.GM_SUN, self.GM_EARTH, self.OMEGA, 
            self._x1, self._x2
        )

    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """
        [L2-SPECIFIC / PARALLELIZATION] 
        Batch compute accelerations for N spacecraft using NumPy broadcasting.
        O(1) Python overhead, O(N) C-backend performance.
        """
        # 1. Extract position and velocity matrix views
        x, y, z = state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2]
        vx, vy = state_matrix[:, 3], state_matrix[:, 4]

        # 2. Relative distances to primaries
        dx1, dx2 = x - self._x1, x - self._x2
        r1_3 = (dx1**2 + y**2 + z**2 + 1e-15)**1.5
        r2_3 = (dx2**2 + y**2 + z**2 + 1e-15)**1.5

        # 3. Summing all force components (Gravity + Centrifugal + Coriolis)
        ax = (-self.GM_SUN * dx1 / r1_3 - self.GM_EARTH * dx2 / r2_3 + 
              self._omega_sq * x + 2.0 * self.OMEGA * vy)
        ay = (-self.GM_SUN * y / r1_3 - self.GM_EARTH * y / r2_3 + 
              self._omega_sq * y - 2.0 * self.OMEGA * vx)
        az = (-self.GM_SUN * z / r1_3 - self.GM_EARTH * z / r2_3)

        # 4. Assemble result matrix (N, 3)
        return np.column_stack((ax, ay, az))

    def __repr__(self) -> str:
        return f"GravityCRTBP(mu={self.mu:.2e}, OMEGA={self.OMEGA:.2e})"
