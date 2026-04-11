# mission_sim/utils/math_tools.py
"""
MCPC Core Math Utility Library
------------------------------
Provides high-precision coordinate frame transformations, matrix operations, 
and quaternion processing across domains.
Specifically targets L2 multi-satellite formations, providing rigorous 
bidirectional transformations between the absolute frame and the relative (LVLH) frame.

LVLH Frame Definition (Unified):
    - X axis (Radial): along the chief position vector (from central body to chief)
    - Y axis (Along-track): along the chief velocity direction (completes right-handed system)
    - Z axis (Cross-track): along the chief angular momentum vector (r × v)
"""
import numpy as np
import scipy.linalg
from typing import Tuple

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Calculate the unit vector of a given vector.
    Returns a zero vector if the norm is extremely small to prevent division by zero.
    
    Args:
        v: Input vector (numpy array).
        
    Returns:
        Unit vector or zero vector.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def get_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the optimal feedback gain matrix K for a continuous-time Linear Quadratic Regulator (LQR).
    Solves the Continuous-time Algebraic Riccati Equation (CARE):
        A^T P + P A - P B R^-1 B^T P + Q = 0

    Args:
        A: State matrix (n x n)
        B: Control input matrix (n x m)
        Q: State weighting matrix (n x n), semi-positive definite
        R: Control weighting matrix (m x m), positive definite

    Returns:
        K: Optimal feedback gain matrix (m x n), such that u(t) = -K * e(t)
    """
    try:
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    except scipy.linalg.LinAlgError as e:
        raise RuntimeError(f"LQR Riccati equation solving failed. Check if A, B are stabilizable! Error: {e}")

    # K = R^-1 B^T P
    K = np.linalg.inv(R) @ B.T @ P
    return K


def elements_to_cartesian(
    mu: float,
    a: float,
    e: float,
    i: float,
    Omega: float,
    omega: float,
    M: float
) -> np.ndarray:
    """
    Convert classical orbital elements to Cartesian state vector (J2000_ECI or similar inertial frame).

    The conversion follows the standard procedure:
        1. Solve Kepler's equation for eccentric anomaly E.
        2. Compute true anomaly nu.
        3. Compute position and velocity in the orbital plane.
        4. Rotate to the inertial frame using the rotation sequence: Omega (Z), i (X), omega (Z).

    Args:
        mu: Gravitational parameter of the central body (m³/s²)
        a: Semi-major axis (m) - must be positive
        e: Eccentricity (0 <= e < 1 for elliptical orbits)
        i: Inclination (rad)
        Omega: Right ascension of ascending node (rad)
        omega: Argument of periapsis (rad)
        M: Mean anomaly (rad)

    Returns:
        Cartesian state vector [x, y, z, vx, vy, vz] (m, m/s)

    Raises:
        ValueError: If orbital elements are invalid
    """
    # Validate input parameters according to MCPC coding standards
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a:.6e} m")
    
    if e < 0 or e >= 1:
        raise ValueError(f"Eccentricity must be 0 <= e < 1 for elliptical orbits, got e={e:.6f}")
    
    # Additional check for the square root term that caused warnings
    if abs(e - 1.0) < 1e-12:
        raise ValueError(f"Parabolic orbits (e=1) are not supported, got e={e:.6f}")
    
    # For stability, also check that 1 - e² is not too small to avoid division by zero
    if abs(1 - e**2) < 1e-12:
        raise ValueError(f"Nearly parabolic orbit (1 - e² ≈ 0) may cause numerical issues, got e={e:.6f}")
    
    # 1. Solve Kepler's equation: M = E - e sin(E) using Newton-Raphson
    # Wrap mean anomaly to [0, 2π) to handle values outside this range
    M_wrapped = M % (2 * np.pi)
    E = M_wrapped
    for _ in range(10):
        f = E - e * np.sin(E) - M_wrapped
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        if abs(delta) < 1e-12:
            break

    # 2. True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

    # 3. Position and velocity in orbital plane
    r = a * (1 - e * np.cos(E))
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    p = a * (1 - e ** 2)
    # Additional safety check for p to avoid sqrt of negative or zero
    if p <= 0:
        raise ValueError(f"Parameter p = a*(1-e²) must be positive, got p={p:.6e} (a={a:.6e}, e={e:.6f})")
    
    vx_orb = -np.sqrt(mu / p) * np.sin(nu)
    vy_orb = np.sqrt(mu / p) * (e + np.cos(nu))

    # 4. Rotation matrix from orbital plane to inertial frame
    # Sequence: rotate by Omega around Z, then i around X, then omega around Z
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    R = np.array([
        [cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i,
         -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i,
         sin_Omega * sin_i],
        [sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i,
         -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i,
         -cos_Omega * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i]
    ])

    pos = R @ [x_orb, y_orb, 0.0]
    vel = R @ [vx_orb, vy_orb, 0.0]

    return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=np.float64)


def inertial_to_rotating(
    state_inertial: np.ndarray,
    t: float,
    omega: float
) -> np.ndarray:
    """
    Convert a state vector from the inertial frame (J2000_SSB or similar)
    to the rotating frame that rotates about the Z-axis with constant angular velocity omega.

    Args:
        state_inertial: State vector in inertial frame [x, y, z, vx, vy, vz] (m, m/s)
        t: Time (s)
        omega: Angular velocity of rotating frame (rad/s)

    Returns:
        State vector in rotating frame [x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot] (m, m/s)
    """
    x, y, z, vx, vy, vz = state_inertial

    cos_theta = np.cos(omega * t)
    sin_theta = np.sin(omega * t)

    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta
    z_rot = z

    vx_rot = vx * cos_theta + vy * sin_theta + omega * y_rot
    vy_rot = -vx * sin_theta + vy * cos_theta - omega * x_rot
    vz_rot = vz

    return np.array([x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot], dtype=np.float64)


def rotating_to_inertial(
    state_rotating: np.ndarray,
    t: float,
    omega: float
) -> np.ndarray:
    """
    Convert a state vector from the rotating frame back to the inertial frame.

    Args:
        state_rotating: State vector in rotating frame [x, y, z, vx, vy, vz] (m, m/s)
        t: Time (s)
        omega: Angular velocity of rotating frame (rad/s)

    Returns:
        State vector in inertial frame [x_inertial, y_inertial, z_inertial, vx_inertial, vy_inertial, vz_inertial] (m, m/s)
    """
    x_rot, y_rot, z_rot, vx_rot, vy_rot, vz_rot = state_rotating

    cos_theta = np.cos(omega * t)
    sin_theta = np.sin(omega * t)

    x = x_rot * cos_theta - y_rot * sin_theta
    y = x_rot * sin_theta + y_rot * cos_theta
    z = z_rot

    vx = vx_rot * cos_theta - vy_rot * sin_theta - omega * y
    vy = vx_rot * sin_theta + vy_rot * cos_theta + omega * x
    vz = vz_rot

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)


# =====================================================================
# Earth-Moon System Specific Functions
# =====================================================================

def inertial_to_earth_moon_rotating(state_inertial: np.ndarray, t: float) -> np.ndarray:
    """
    Convert state from J2000_ECI to Earth-Moon rotating frame.
    
    Earth-Moon rotating frame definition:
        - Origin: Earth-Moon barycenter
        - X-axis: Points from barycenter to Earth
        - Z-axis: Perpendicular to Earth-Moon orbital plane
        - Y-axis: Completes right-handed system
        - Angular velocity: ω = 2.6617e-6 rad/s
    
    Args:
        state_inertial: State vector in J2000_ECI [x, y, z, vx, vy, vz] (m, m/s)
        t: Time since J2000 epoch (s)
        
    Returns:
        State vector in Earth-Moon rotating frame
    """
    omega_moon = 2.6617e-6  # Lunar orbital angular velocity (rad/s)
    
    # Note: This assumes state_inertial is already relative to Earth-Moon barycenter
    # For precise conversion, need to subtract barycenter position/velocity
    return inertial_to_rotating(state_inertial, t, omega_moon)


def earth_moon_rotating_to_inertial(state_rotating: np.ndarray, t: float) -> np.ndarray:
    """
    Convert state from Earth-Moon rotating frame to J2000_ECI.
    
    Args:
        state_rotating: State vector in Earth-Moon rotating frame [x, y, z, vx, vy, vz] (m, m/s)
        t: Time since J2000 epoch (s)
        
    Returns:
        State vector in J2000_ECI
    """
    omega_moon = 2.6617e-6  # Lunar orbital angular velocity (rad/s)
    return rotating_to_inertial(state_rotating, t, omega_moon)


def get_earth_moon_system_parameters() -> dict:
    """
    Get Earth-Moon system physical parameters.
    
    Returns:
        Dictionary containing system parameters
    """
    return {
        'mu': 0.01215,  # Mass ratio: m_moon/(m_earth + m_moon)
        'distance': 3.844e8,  # Earth-Moon distance (m)
        'omega': 2.6617e-6,  # Angular velocity (rad/s)
        'period': 27.321661 * 24 * 3600,  # Sidereal period (s)
        'earth_mass': 5.972e24,  # kg
        'moon_mass': 7.342e22,  # kg
        'earth_mu': 3.986004418e14,  # m³/s²
        'moon_mu': 4.9048695e12  # m³/s²
    }


# =====================================================================
# L2 Level Core: High-precision Bidirectional Transformations 
# Between Absolute Frame and Relative Frame (LVLH)
# =====================================================================

def compute_lvlh_dcm(r_chief_abs: np.ndarray, v_chief_abs: np.ndarray) -> np.ndarray:
    """
    Compute the Direction Cosine Matrix (DCM) from the Absolute frame 
    to the LVLH frame (X: radial, Y: along-track, Z: cross-track).

    Args:
        r_chief_abs: Chief's position vector in the absolute frame (3x1).
        v_chief_abs: Chief's velocity vector in the absolute frame (3x1).
        
    Returns:
        dcm_abs_to_lvlh: 3x3 rotation matrix.
    """
    r = np.array(r_chief_abs, dtype=np.float64)
    v = np.array(v_chief_abs, dtype=np.float64)
    
    # Radial (X axis)
    x_hat = normalize_vector(r)
    
    # Angular momentum vector
    h = np.cross(r, v)
    # Cross-track (Z axis)
    z_hat = normalize_vector(h)
    
    # Along-track (Y axis) completes right-handed system: Y = Z × X
    y_hat = np.cross(z_hat, x_hat)
    
    # DCM rows are the basis vectors in the absolute frame
    dcm_abs_to_lvlh = np.vstack((x_hat, y_hat, z_hat))
    
    return dcm_abs_to_lvlh


def absolute_to_lvlh(r_chief_abs: np.ndarray, v_chief_abs: np.ndarray, 
                     r_deputy_abs: np.ndarray, v_deputy_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the absolute state of a deputy spacecraft into the relative state 
    within the Chief's LVLH coordinate frame.
    
    LVLH definition: X = radial (from central body to chief), 
                     Y = along-track (velocity direction),
                     Z = cross-track (angular momentum direction).
    
    Args:
        r_chief_abs: Chief absolute position (3,)
        v_chief_abs: Chief absolute velocity (3,)
        r_deputy_abs: Deputy absolute position (3,)
        v_deputy_abs: Deputy absolute velocity (3,)
    
    Returns:
        rho_lvlh: Relative position in LVLH (3,)
        rho_dot_lvlh: Relative velocity in LVLH (3,)
    """
    r_c = np.array(r_chief_abs, dtype=np.float64)
    v_c = np.array(v_chief_abs, dtype=np.float64)
    r_d = np.array(r_deputy_abs, dtype=np.float64)
    v_d = np.array(v_deputy_abs, dtype=np.float64)
    
    # Relative vectors in absolute frame
    delta_r_abs = r_d - r_c
    delta_v_abs = v_d - v_c
    
    # Compute DCM
    dcm = compute_lvlh_dcm(r_c, v_c)
    rho_lvlh = dcm @ delta_r_abs
    
    # Angular velocity of LVLH frame
    h = np.cross(r_c, v_c)
    r_norm_sq = np.dot(r_c, r_c)
    omega = h / r_norm_sq if r_norm_sq > 1e-12 else np.zeros(3)
    
    # Relative velocity in LVLH (subtract transport term)
    v_rel_abs = delta_v_abs - np.cross(omega, delta_r_abs)
    rho_dot_lvlh = dcm @ v_rel_abs
    
    return rho_lvlh, rho_dot_lvlh


def lvlh_to_absolute(r_chief_abs: np.ndarray, v_chief_abs: np.ndarray, 
                     rho_lvlh: np.ndarray, rho_dot_lvlh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transformation: convert LVLH relative state back to absolute deputy state.

    Args:
        r_chief_abs: Chief absolute position (3,)
        v_chief_abs: Chief absolute velocity (3,)
        rho_lvlh: Relative position in LVLH (3,)
        rho_dot_lvlh: Relative velocity in LVLH (3,)
    
    Returns:
        r_deputy_abs: Deputy absolute position (3,)
        v_deputy_abs: Deputy absolute velocity (3,)
    """
    r_c = np.array(r_chief_abs, dtype=np.float64)
    v_c = np.array(v_chief_abs, dtype=np.float64)
    rho = np.array(rho_lvlh, dtype=np.float64)
    rho_dot = np.array(rho_dot_lvlh, dtype=np.float64)
    
    dcm = compute_lvlh_dcm(r_c, v_c)
    dcm_inv = dcm.T  # Orthogonal matrix inverse = transpose
    
    delta_r_abs = dcm_inv @ rho
    r_deputy_abs = r_c + delta_r_abs
    
    # Angular velocity
    h = np.cross(r_c, v_c)
    r_norm_sq = np.dot(r_c, r_c)
    omega = h / r_norm_sq if r_norm_sq > 1e-12 else np.zeros(3)
    
    v_rel_abs = dcm_inv @ rho_dot
    delta_v_abs = v_rel_abs + np.cross(omega, delta_r_abs)
    v_deputy_abs = v_c + delta_v_abs
    
    return r_deputy_abs, v_deputy_abs


# =====================================================================
# 通用数学工具：开普勒方程求解器（跨域共享）
# =====================================================================

def solve_kepler_equation_batch(
    M_array: np.ndarray, 
    e: float, 
    max_iter: int = 10, 
    tol: float = 1e-12
) -> np.ndarray:
    """
    Batch solve Kepler's equation: M = E - e * sin(E)
    
    Uses vectorized Newton-Raphson iteration for efficient computation.
    
    Args:
        M_array: Mean anomaly array (rad), shape (N,)
        e: Eccentricity (0 ≤ e < 1)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        np.ndarray: Eccentric anomaly array (rad), shape (N,)
        
    Raises:
        ValueError: If parameters are invalid
    """
    if e < 0 or e >= 1:
        raise ValueError(
            f"Eccentricity must be 0 ≤ e < 1, got e={e:.6f}"
        )
    
    # Wrap mean anomaly to [0, 2π)
    M_wrapped = M_array % (2 * np.pi)
    
    # Initial guess: for small eccentricity, E ≈ M
    E = M_wrapped.copy()
    
    # Newton-Raphson iteration: E_{i+1} = E_i - (E_i - e*sin(E_i) - M) / (1 - e*cos(E_i))
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M_wrapped
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E -= delta
        
        # Check convergence
        if np.max(np.abs(delta)) < tol:
            break
    
    return E


def solve_kepler_equation_scalar(
    M: float, 
    e: float, 
    max_iter: int = 10, 
    tol: float = 1e-12
) -> float:
    """
    Scalar version to solve Kepler's equation: M = E - e * sin(E)
    
    Args:
        M: Mean anomaly (rad)
        e: Eccentricity (0 ≤ e < 1)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        float: Eccentric anomaly (rad)
    """
    # Wrap call to batch version (for consistency)
    M_array = np.array([M])
    E_array = solve_kepler_equation_batch(M_array, e, max_iter, tol)
    return E_array[0]


def orbital_elements_to_cartesian_batch(
    a: float,
    e: float,
    i: float,
    Omega: float,
    omega: float,
    M_array: np.ndarray,
    mu: float
) -> np.ndarray:
    """
    Batch convert orbital elements to Cartesian state vectors (vectorized)
    
    Convert a set of mean anomalies to corresponding state vectors,
    suitable for generating orbital time series.
    
    Args:
        a: Semi-major axis (m), scalar
        e: Eccentricity, scalar
        i: Inclination (rad), scalar
        Omega: Right ascension of ascending node (rad), scalar
        omega: Argument of periapsis (rad), scalar
        M_array: Mean anomaly array (rad), shape (N,)
        mu: Gravitational parameter (m³/s²), scalar
        
    Returns:
        np.ndarray: Cartesian state vector array, shape (N, 6)
        
    Raises:
        ValueError: If orbital parameters are invalid
    """
    # Parameter validation
    if a <= 0:
        raise ValueError(f"Semi-major axis must be positive, got a={a:.6e} m")
    
    if e < 0 or e >= 1:
        raise ValueError(
            f"Eccentricity must be 0 ≤ e < 1 for elliptical orbits, got e={e:.6f}"
        )
    
    # Check p = a*(1-e²) to avoid division by zero or negative
    p = a * (1 - e**2)
    if p <= 0:
        raise ValueError(
            f"Parameter p = a*(1-e²) must be positive, got p={p:.6e} "
            f"(a={a:.6e}, e={e:.6f})"
        )
    
    # Solve Kepler's equation (batch)
    E_array = solve_kepler_equation_batch(M_array, e)
    
    # Calculate true anomaly (vectorized)
    sqrt_one_plus_e = np.sqrt(1 + e)
    sqrt_one_minus_e = np.sqrt(1 - e)
    nu_array = 2 * np.arctan2(
        sqrt_one_plus_e * np.sin(E_array / 2),
        sqrt_one_minus_e * np.cos(E_array / 2)
    )
    
    # Calculate orbital plane positions and velocities (vectorized)
    r_array = a * (1 - e * np.cos(E_array))
    x_orb_array = r_array * np.cos(nu_array)
    y_orb_array = r_array * np.sin(nu_array)
    
    sqrt_mu_over_p = np.sqrt(mu / p)
    vx_orb_array = -sqrt_mu_over_p * np.sin(nu_array)
    vy_orb_array = sqrt_mu_over_p * (e + np.cos(nu_array))
    
    # Build rotation matrix (3-1-3 sequence: Ω, i, ω)
    cos_Omega = np.cos(Omega)
    sin_Omega = np.sin(Omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    
    R = np.array([
        [cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i,
         -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i,
         sin_Omega * sin_i],
        [sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i,
         -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i,
         -cos_Omega * sin_i],
        [sin_omega * sin_i,
         cos_omega * sin_i,
         cos_i]
    ])
    
    # Batch rotation to inertial frame
    # Build orbital plane state matrix (N, 6)
    N = len(M_array)
    states_orbital = np.zeros((N, 6))
    states_orbital[:, 0] = x_orb_array
    states_orbital[:, 1] = y_orb_array
    # z_orb remains 0
    states_orbital[:, 3] = vx_orb_array
    states_orbital[:, 4] = vy_orb_array
    # vz_orb remains 0
    
    # Apply rotation matrix to position and velocity
    pos_inertial = states_orbital[:, 0:3] @ R.T
    vel_inertial = states_orbital[:, 3:6] @ R.T
    
    # Combine results
    states_inertial = np.concatenate([pos_inertial, vel_inertial], axis=1)
    
    return states_inertial


# Backward compatibility: Update existing scalar function to use new function
def elements_to_cartesian(
    mu: float,
    a: float,
    e: float,
    i: float,
    Omega: float,
    omega: float,
    M: float
) -> np.ndarray:
    """
    Convert orbital elements to Cartesian state vector (scalar version, backward compatible)
    
    Note: This function has been updated to call the new vectorized implementation
          for consistency. New code should use orbital_elements_to_cartesian_batch
          directly for better performance.
    """
    # Wrap call to batch version
    M_array = np.array([M])
    states = orbital_elements_to_cartesian_batch(a, e, i, Omega, omega, M_array, mu)
    return states[0]
