"""
Universal Circular Restricted Three-Body Problem (CRTBP) Model

This module provides a generic CRTBP implementation that supports any two-body system
(Earth-Moon, Sun-Earth, etc.) and follows the IForceModel interface.

Implements both scalar and vectorized computations for optimal performance.
"""

import numpy as np
from typing import Dict, Any
from mission_sim.core.physics.environment import IForceModel
from mission_sim.utils.math_tools import inertial_to_rotating, rotating_to_inertial


class UniversalCRTBP(IForceModel):
    """
    Universal Circular Restricted Three-Body Problem (CRTBP) Dynamics Model
    
    Implements the CRTBP dynamics for any two-body system (Earth-Moon, Sun-Earth, etc.)
    in the rotating frame centered at the barycenter.
    
    Coordinate System (rotating frame):
        - Origin: Barycenter of the two primary bodies
        - X-axis: Points from barycenter to the smaller primary (e.g., from Earth to Moon)
        - Y-axis: In the orbital plane, 90° ahead of X in the direction of motion
        - Z-axis: Completes right-handed system (perpendicular to orbital plane)
    
    Physical Parameters:
        - m1: Mass of larger primary (kg)
        - m2: Mass of smaller primary (kg)
        - L: Distance between primaries (m)
        - ω: System angular velocity (rad/s)
        - μ: Mass ratio = m2/(m1 + m2)
    """
    
    # Gravitational constant (m³ kg⁻¹ s⁻²)
    G = 6.67430e-11
    
    def __init__(self, primary_mass: float, secondary_mass: float, 
                 distance: float, system_name: str = 'custom',
                 use_numba: bool = False):
        """
        Initialize CRTBP system with physical parameters.
        
        Args:
            primary_mass: Mass of larger primary body (kg)
            secondary_mass: Mass of smaller primary body (kg)
            distance: Distance between primaries (m)
            system_name: System identifier (e.g., 'earth_moon', 'sun_earth')
            use_numba: Enable Numba acceleration for single state computations
        """
        self._primary_mass = float(primary_mass)
        self._secondary_mass = float(secondary_mass)
        self._distance = float(distance)
        self._system_name = system_name
        self._use_numba = use_numba
        
        # Compute derived parameters
        total_mass = self._primary_mass + self._secondary_mass
        self._mu = self._secondary_mass / total_mass  # Mass ratio
        
        # Angular velocity: ω = sqrt(G*(m1+m2)/L³)
        self._omega = np.sqrt(self.G * total_mass / self._distance**3)
        
        # Characteristic length and time scales
        self._L = self._distance  # Characteristic length (m)
        self._T = 1.0 / self._omega if self._omega != 0 else 1.0  # Characteristic time (s)
        
        # Positions of primaries in rotating frame (normalized by L)
        self._primary_pos = np.array([-self._mu, 0.0, 0.0])    # Larger primary (e.g., Earth)
        self._secondary_pos = np.array([1.0 - self._mu, 0.0, 0.0])  # Smaller primary (e.g., Moon)
        
        # Compute GM values for compatibility with legacy code
        self._gm1 = self.G * self._primary_mass
        self._gm2 = self.G * self._secondary_mass
        
        # Numba acceleration setup
        if use_numba:
            try:
                from numba import jit as njit
                self._accel_numba = njit(self._crtbp_acceleration_nd)
            except ImportError:
                self._use_numba = False
                print("警告: Numba 不可用，将使用纯 NumPy 实现")
    
    @classmethod
    def earth_moon_system(cls, use_numba: bool = False) -> 'UniversalCRTBP':
        """
        Create Earth-Moon system CRTBP instance.
        
        Returns:
            UniversalCRTBP instance configured for Earth-Moon system
        """
        earth_mass = 5.972e24  # kg
        moon_mass = 7.342e22   # kg
        distance = 3.844e8     # m (semi-major axis)
        
        return cls(earth_mass, moon_mass, distance, 'earth_moon', use_numba)
    
    @classmethod
    def sun_earth_system(cls, use_numba: bool = False) -> 'UniversalCRTBP':
        """
        Create Sun-Earth system CRTBP instance.
        
        Returns:
            UniversalCRTBP instance configured for Sun-Earth system
        """
        sun_mass = 1.989e30    # kg
        earth_mass = 5.972e24  # kg
        distance = 1.496e11    # m (1 AU)
        
        return cls(sun_mass, earth_mass, distance, 'sun_earth', use_numba)
    
    @property
    def mu(self) -> float:
        """Mass ratio μ = m2/(m1+m2)"""
        return self._mu
    
    @property
    def distance(self) -> float:
        """Distance between primary bodies (m)"""
        return self._distance
    
    @property
    def omega(self) -> float:
        """System angular velocity (rad/s)"""
        return self._omega
    
    @property
    def system_name(self) -> str:
        """System identifier"""
        return self._system_name
    
    @property
    def primary_mass(self) -> float:
        """Mass of larger primary body (kg)"""
        return self._primary_mass
    
    @property
    def secondary_mass(self) -> float:
        """Mass of smaller primary body (kg)"""
        return self._secondary_mass
    
    @property
    def gm1(self) -> float:
        """GM of larger primary (m³/s²)"""
        return self._gm1
    
    @property
    def gm2(self) -> float:
        """GM of smaller primary (m³/s²)"""
        return self._gm2
    
    def _to_nd(self, state_physical: np.ndarray) -> np.ndarray:
        """
        Convert physical state to dimensionless state.
        
        Args:
            state_physical: Physical state [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            Dimensionless state [x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd]
        """
        state_nd = np.zeros_like(state_physical)
        
        # Position: divide by characteristic length L
        state_nd[0:3] = state_physical[0:3] / self._L
        
        # Velocity: divide by characteristic velocity L*ω
        state_nd[3:6] = state_physical[3:6] / (self._L * self._omega)
        
        return state_nd
    
    def _to_physical(self, state_nd: np.ndarray) -> np.ndarray:
        """
        Convert dimensionless state to physical state.
        
        Args:
            state_nd: Dimensionless state [x_nd, y_nd, z_nd, vx_nd, vy_nd, vz_nd]
            
        Returns:
            Physical state [x, y, z, vx, vy, vz] (m, m/s)
        """
        state_physical = np.zeros_like(state_nd)
        
        # Position: multiply by characteristic length L
        state_physical[0:3] = state_nd[0:3] * self._L
        
        # Velocity: multiply by characteristic velocity L*ω
        state_physical[3:6] = state_nd[3:6] * self._L * self._omega
        
        return state_physical
    
    def _crtbp_acceleration_nd(self, state_nd: np.ndarray) -> np.ndarray:
        """
        Compute CRTBP acceleration in dimensionless units.
        
        Args:
            state_nd: Dimensionless state [x, y, z, vx, vy, vz]
            
        Returns:
            Dimensionless acceleration [ax, ay, az]
        """
        x, y, z, vx, vy, _ = state_nd
        mu = self._mu
        
        # Distances to primaries (squared)
        r1_sq = (x + mu)**2 + y**2 + z**2
        r2_sq = (x + mu - 1)**2 + y**2 + z**2
        
        # Avoid division by zero (add small epsilon)
        eps = 1e-15
        r1 = np.sqrt(r1_sq + eps)
        r2 = np.sqrt(r2_sq + eps)
        
        # Compute accelerations
        # CRTBP equations in rotating frame:
        # ax = 2*vy + x - (1-μ)*(x+μ)/r1³ - μ*(x+μ-1)/r2³
        # ay = -2*vx + y - (1-μ)*y/r1³ - μ*y/r2³
        # az = -(1-μ)*z/r1³ - μ*z/r2³
        
        ax = 2*vy + x - (1 - mu)*(x + mu)/(r1**3) - mu*(x + mu - 1)/(r2**3)
        ay = -2*vx + y - (1 - mu)*y/(r1**3) - mu*y/(r2**3)
        az = - (1 - mu)*z/(r1**3) - mu*z/(r2**3)
        
        return np.array([ax, ay, az])
    
    def _crtbp_acceleration_physical(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Direct computation of CRTBP acceleration in physical units.
        This method is equivalent to the original GravityCRTBP approach.
        
        Args:
            pos: Position vector [x, y, z] (m) in rotating frame
            vel: Velocity vector [vx, vy, vz] (m/s) in rotating frame
            
        Returns:
            Acceleration [ax, ay, az] (m/s²)
        """
        # Positions of primaries in rotating frame (physical units)
        x1 = -self._mu * self._L
        x2 = (1.0 - self._mu) * self._L
        
        # Relative distances to primaries
        dx1 = pos[0] - x1
        dx2 = pos[0] - x2
        r1_3 = (dx1**2 + pos[1]**2 + pos[2]**2)**1.5
        r2_3 = (dx2**2 + pos[1]**2 + pos[2]**2)**1.5
        
        # Gravitational components
        ax = -self._gm1 * dx1 / r1_3 - self._gm2 * dx2 / r2_3
        ay = -self._gm1 * pos[1] / r1_3 - self._gm2 * pos[1] / r2_3
        az = -self._gm1 * pos[2] / r1_3 - self._gm2 * pos[2] / r2_3
        
        # Fictitious forces (Centrifugal & Coriolis)
        ax += self._omega**2 * pos[0] + 2.0 * self._omega * vel[1]
        ay += self._omega**2 * pos[1] - 2.0 * self._omega * vel[0]
        
        return np.array([ax, ay, az], dtype=np.float64)
    
    def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
        """
        [L1 LEGACY & FALLBACK]
        Compute CRTBP acceleration in physical units for a single spacecraft.
        
        Args:
            state: Physical state [x, y, z, vx, vy, vz] (m, m/s) in rotating frame
            epoch: Current time (seconds) - used for consistency with IForceModel
            
        Returns:
            Acceleration [ax, ay, az] (m/s²)
        """
        if self._use_numba:
            # Use physical units with Numba acceleration
            return self._crtbp_acceleration_physical(state[0:3], state[3:6])
        else:
            # Use dimensionless units (standard approach)
            state_nd = self._to_nd(state)
            accel_nd = self._crtbp_acceleration_nd(state_nd)
            accel_physical = accel_nd * (self._L * self._omega**2)
            return accel_physical
    
    def compute_vectorized_acc(self, state_matrix: np.ndarray, epoch: float) -> np.ndarray:
        """
        [L2-SPECIFIC / PARALLELIZATION]
        Batch compute CRTBP accelerations for multiple spacecraft using vectorization.
        
        Args:
            state_matrix: 2D array of shape (N, 6) representing N spacecraft states
            epoch: Current time (seconds)
            
        Returns:
            2D array of shape (N, 3) representing accelerations for each spacecraft
        """
        if state_matrix.ndim != 2 or state_matrix.shape[1] != 6:
            raise ValueError(f"state_matrix must be of shape (N, 6), got {state_matrix.shape}")
        
        N = state_matrix.shape[0]
        
        # Convert all states to dimensionless at once
        # Extract positions and velocities
        pos = state_matrix[:, 0:3]  # (N, 3)
        vel = state_matrix[:, 3:6]  # (N, 3)
        
        # Convert to dimensionless
        pos_nd = pos / self._L
        vel_nd = vel / (self._L * self._omega)
        
        # Extract components
        x = pos_nd[:, 0]  # (N,)
        y = pos_nd[:, 1]
        z = pos_nd[:, 2]
        vx = vel_nd[:, 0]
        vy = vel_nd[:, 1]
        
        mu = self._mu
        
        # Compute distances to primaries for all spacecraft
        # r1^2 = (x + μ)^2 + y^2 + z^2
        # r2^2 = (x + μ - 1)^2 + y^2 + z^2
        r1_sq = (x + mu)**2 + y**2 + z**2
        r2_sq = (x + mu - 1)**2 + y**2 + z**2
        
        # Avoid division by zero
        eps = 1e-15
        r1 = np.sqrt(r1_sq + eps)  # (N,)
        r2 = np.sqrt(r2_sq + eps)
        
        # Compute accelerations in dimensionless units
        ax_nd = 2*vy + x - (1 - mu)*(x + mu)/(r1**3) - mu*(x + mu - 1)/(r2**3)
        ay_nd = -2*vx + y - (1 - mu)*y/(r1**3) - mu*y/(r2**3)
        az_nd = - (1 - mu)*z/(r1**3) - mu*z/(r2**3)
        
        # Stack into (N, 3) matrix
        accel_nd = np.column_stack((ax_nd, ay_nd, az_nd))
        
        # Convert to physical units
        accel_physical = accel_nd * (self._L * self._omega**2)
        
        return accel_physical
    
    def jacobi_constant(self, state: np.ndarray) -> float:
        """
        Compute Jacobi constant (dimensionless).
        
        The Jacobi constant is an integral of motion in the CRTBP:
        C = 2Ω - (vx² + vy² + vz²)
        where Ω = (1-μ)/r1 + μ/r2 + 0.5*(x² + y²)
        
        Args:
            state: Physical state [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            Jacobi constant C (dimensionless)
        """
        # Convert to dimensionless
        state_nd = self._to_nd(state)
        x, y, z, vx, vy, vz = state_nd
        mu = self._mu
        
        # Distances to primaries
        r1_sq = (x + mu)**2 + y**2 + z**2
        r2_sq = (x + mu - 1)**2 + y**2 + z**2
        
        eps = 1e-15
        r1 = np.sqrt(r1_sq + eps)
        r2 = np.sqrt(r2_sq + eps)
        
        # Effective potential
        omega = (1 - mu)/r1 + mu/r2 + 0.5*(x**2 + y**2)
        
        # Velocity squared
        v_sq = vx**2 + vy**2 + vz**2
        
        # Jacobi constant
        C = 2*omega - v_sq
        
        return C
    
    def to_rotating_frame(self, state_inertial: np.ndarray, epoch: float) -> np.ndarray:
        """
        Convert state from inertial frame (J2000) to rotating frame.
        
        Args:
            state_inertial: State in inertial frame [x, y, z, vx, vy, vz] (m, m/s)
            epoch: Current time (seconds since J2000)
            
        Returns:
            State in rotating frame [x, y, z, vx, vy, vz] (m, m/s)
        """
        # Use math_tools conversion with system angular velocity
        state_rotating = inertial_to_rotating(state_inertial, epoch, self._omega)
        return state_rotating
    
    def to_inertial_frame(self, state_rotating: np.ndarray, epoch: float) -> np.ndarray:
        """
        Convert state from rotating frame to inertial frame (J2000).
        
        Args:
            state_rotating: State in rotating frame [x, y, z, vx, vy, vz] (m, m/s)
            epoch: Current time (seconds since J2000)
            
        Returns:
            State in inertial frame [x, y, z, vx, vy, vz] (m, m/s)
        """
        # Use math_tools conversion with system angular velocity
        state_inertial = rotating_to_inertial(state_rotating, epoch, self._omega)
        return state_inertial
    
    def get_primary_positions_nd(self) -> (np.ndarray, np.ndarray):
        """
        Get positions of primary bodies in dimensionless coordinates.
        
        Returns:
            Tuple of (primary_position, secondary_position) in rotating frame
        """
        return self._primary_pos.copy(), self._secondary_pos.copy()
    
    def get_effective_potential_nd(self, state_nd: np.ndarray) -> float:
        """Compute effective potential Ω at given dimensionless state."""
        x, y, z, _, _, _ = state_nd
        mu = self._mu
        
        # Distances to primaries
        r1_sq = (x + mu)**2 + y**2 + z**2
        r2_sq = (x + mu - 1)**2 + y**2 + z**2
        
        eps = 1e-15
        r1 = np.sqrt(r1_sq + eps)
        r2 = np.sqrt(r2_sq + eps)
        
        # Effective potential
        omega = (1 - mu)/r1 + mu/r2 + 0.5*(x**2 + y**2)
        
        return omega
    
    def get_lagrange_points_nd(self) -> Dict[str, np.ndarray]:
        """
        Compute positions of Lagrange points in dimensionless coordinates.
        
        Returns:
            Dictionary with keys 'L1' through 'L5' and their positions [x, y, z]
        """
        mu = self._mu
        
        # Approximate positions (for mu < 0.5)
        # L1, L2, L3 are collinear points
        gamma1 = ((mu/3)**(1/3))  # L1 approximate
        gamma2 = ((mu/3)**(1/3))  # L2 approximate
        gamma3 = 1 + (7/12)*mu    # L3 approximate
        
        L1 = np.array([1 - mu - gamma1, 0, 0])
        L2 = np.array([1 - mu + gamma2, 0, 0])
        L3 = np.array([-mu - gamma3, 0, 0])
        
        # L4 and L5 are triangular points
        L4 = np.array([0.5 - mu, np.sqrt(3)/2, 0])
        L5 = np.array([0.5 - mu, -np.sqrt(3)/2, 0])
        
        return {
            'L1': L1,
            'L2': L2,
            'L3': L3,
            'L4': L4,
            'L5': L5
        }
    
    def get_system_parameters(self) -> Dict[str, Any]:
        """
        Get all system parameters.
        
        Returns:
            Dictionary containing all system parameters
        """
        return {
            'system_name': self._system_name,
            'primary_mass': self._primary_mass,
            'secondary_mass': self._secondary_mass,
            'distance': self._distance,
            'mu': self._mu,
            'omega': self._omega,
            'L': self._L,
            'T': self._T,
            'primary_position_nd': self._primary_pos.tolist(),
            'secondary_position_nd': self._secondary_pos.tolist()
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"UniversalCRTBP(system='{self._system_name}', "
                f"μ={self._mu:.6f}, ω={self._omega:.3e} rad/s, "
                f"L={self._distance:.3e} m)")
    
    def __str__(self) -> str:
        """Human-readable string"""
        params = self.get_system_parameters()
        return (f"Universal CRTBP System: {params['system_name']}\n"
                f"  Mass ratio (μ): {params['mu']:.6f}\n"
                f"  Distance: {params['distance']:.3e} m\n"
                f"  Angular velocity: {params['omega']:.3e} rad/s\n"
                f"  Primary mass: {params['primary_mass']:.3e} kg\n"
                f"  Secondary mass: {params['secondary_mass']:.3e} kg")


# 注意：特定系统类已移动到独立文件中
# SunEarthCRTBP 类已移动到 sun_earth_crtbp.py
# EarthMoonCRTBP 类已移动到 earth_moon_crtbp.py
# 工厂函数已移动到 __init__.py 中

# 导出通用 CRTBP 类
__all__ = ['UniversalCRTBP']
