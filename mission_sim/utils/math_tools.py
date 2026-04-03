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
        a: Semi-major axis (m)
        e: Eccentricity (0 <= e < 1)
        i: Inclination (rad)
        Omega: Right ascension of ascending node (rad)
        omega: Argument of periapsis (rad)
        M: Mean anomaly (rad)

    Returns:
        Cartesian state vector [x, y, z, vx, vy, vz] (m, m/s)
    """
    # 1. Solve Kepler's equation: M = E - e sin(E) using Newton-Raphson
    E = M
    for _ in range(10):
        f = E - e * np.sin(E) - M
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