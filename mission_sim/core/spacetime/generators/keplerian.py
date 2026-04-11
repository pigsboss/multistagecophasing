# mission_sim/core/trajectory/generators/keplerian.py
"""
开普勒轨道生成器

基于二体问题假设，根据轨道根数生成开普勒轨道。
支持椭圆轨道（0 ≤ e < 1），提供高精度数值积分。

**接口规范：**
1. 调用者负责确保输入参数的有效性
2. 所有计算都向量化，支持批量处理
3. 为GPU移植优化，避免Python循环

**输入要求：**
- a > 0 (半长轴必须为正)
- 0 ≤ e < 1 (椭圆轨道)
- i, Ω, ω, M0 在合理范围内（弧度制）
- dt > 0
- sim_time > 0
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator


@dataclass
class KeplerianConfig:
    """开普勒轨道配置"""
    elements: List[float]  # [a, e, i, Ω, ω, M0]
    epoch: float = 0.0
    dt: float = 60.0
    sim_time: float = 3600.0
    mu: float = 3.986004418e14  # 地球引力常数 (m³/s²)


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    开普勒轨道生成器（向量化版本）
    
    根据经典轨道根数生成二体问题轨道。
    所有计算都向量化，支持批量处理，为GPU移植优化。
    """
    
    def __init__(self, ephemeris=None, use_high_precision=False):
        """初始化生成器
        
        Note:
            不进行任何验证，保持最小初始化。
            所有验证由调用者负责。
        """
        super().__init__(ephemeris, use_high_precision)
        self.mu = 3.986004418e14  # 默认地球引力常数
        
    def generate(self, config: Dict[str, Any]) -> Ephemeris:
        """
        Generate Keplerian orbit based on orbital elements (vectorized version).
        
        Args:
            config: Configuration dictionary, must contain:
                - elements: Orbital elements list [a, e, i, Ω, ω, M0]
                - dt: Output time step (s)
                - sim_time: Simulation duration (s)
                Optional:
                - epoch: Epoch time (s)
                - mu: Gravitational parameter (m³/s²)
                
        Returns:
            Ephemeris: Generated orbit ephemeris
            
        Raises:
            ValueError: If required parameters are missing or invalid
            
        Note:
            - Uses vectorized batch computation, no Python loops
            - Output state array shape is (N, 6), N = number of time points
        """
        # Validate required parameters
        required_keys = ['elements', 'dt', 'sim_time']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required parameter: '{key}'")
        
        # Extract parameters
        elements = config['elements']
        dt = config['dt']
        sim_time = config['sim_time']
        epoch = config.get('epoch', 0.0)
        mu = config.get('mu', self.mu)
        
        # Validate orbital elements
        if len(elements) != 6:
            raise ValueError(
                f"Orbital elements must have exactly 6 values, got {len(elements)}"
            )
        
        # Validate element values
        a, e, i, Omega, omega, M0 = elements
        
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got a={a:.6e} m")
        
        if e < 0 or e >= 1:
            raise ValueError(
                f"Eccentricity must be 0 ≤ e < 1 for elliptical orbits, got e={e:.6f}"
            )
        
        # Validate time parameters
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got dt={dt:.6e} s")
        
        if sim_time < 0:
            raise ValueError(f"Simulation time must be non-negative, got sim_time={sim_time:.6e} s")
        
        # Generate time sequence (vector)
        num_points = int(sim_time / dt) + 1
        times = np.linspace(0, sim_time, num_points) + epoch
        
        # Calculate mean anomaly array (vectorized)
        n = np.sqrt(mu / a**3)
        M_array = M0 + n * times
        
        # Batch convert to Cartesian coordinates (vectorized)
        states_array = self.elements_to_cartesian_batch(a, e, i, Omega, omega, M_array, mu)
        
        return Ephemeris(times, states_array, CoordinateFrame.J2000_ECI)
    
    def elements_to_cartesian_batch(self, a, e, i, Omega, omega, M_array, mu):
        """
        Batch convert orbital elements to Cartesian state vectors (vectorized version).
        
        Args:
            a: Semi-major axis (m), scalar
            e: Eccentricity, scalar
            i: Inclination (rad), scalar
            Omega: Right ascension of ascending node (rad), scalar
            omega: Argument of perigee (rad), scalar
            M_array: Mean anomaly array (rad), shape (N,)
            mu: Gravitational parameter (m³/s²), scalar
            
        Returns:
            np.ndarray: Cartesian state vector array, shape (N, 6)
            
        Raises:
            ValueError: If orbital parameters are invalid
            
        Note:
            - Uses vectorized operations, no Python loops
            - Designed for GPU portability
        """
        # Validate parameters (similar to scalar version in math_tools.py)
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got a={a:.6e} m")
        
        if e < 0 or e >= 1:
            raise ValueError(f"Eccentricity must be 0 ≤ e < 1 for elliptical orbits, got e={e:.6f}")
        
        # Additional check to avoid numerical issues
        if abs(1 - e**2) < 1e-12:
            raise ValueError(
                f"Nearly parabolic orbit (1 - e² ≈ 0) may cause numerical issues, got e={e:.6f}"
            )
        
        # Calculate mean motion
        n = np.sqrt(mu / a**3)
        
        # Vectorized solution of Kepler's equation
        M_wrapped = M_array % (2 * np.pi)
        
        # Initial guess: use M as initial value for E (good for small e)
        E = M_wrapped.copy()
        
        # Iteratively solve Kepler's equation: E = M + e * sin(E)
        for _ in range(10):
            delta = (E - e * np.sin(E) - M_wrapped) / (1 - e * np.cos(E))
            E -= delta
        
        # Calculate true anomaly
        sqrt_one_plus_e = np.sqrt(1 + e)
        sqrt_one_minus_e = np.sqrt(1 - e)
        nu = 2 * np.arctan2(sqrt_one_plus_e * np.sin(E/2), sqrt_one_minus_e * np.cos(E/2))
        
        # Orbital plane coordinates
        r = a * (1 - e * np.cos(E))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Orbital plane velocity
        p = a * (1 - e**2)  # Semi-latus rectum
        if p <= 0:
            raise ValueError(
                f"Parameter p = a*(1-e²) must be positive, got p={p:.6e} (a={a:.6e}, e={e:.6f})"
            )
        
        sqrt_mu_over_p = np.sqrt(mu / p)
        vx_orb = -sqrt_mu_over_p * np.sin(nu)
        vy_orb = sqrt_mu_over_p * (e + np.cos(nu))
        
        # Rotation matrix (same for all points)
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
        
        # Vectorized rotation (batch processing)
        states_orbital = np.stack([x_orb, y_orb, np.zeros_like(x_orb), 
                                   vx_orb, vy_orb, np.zeros_like(vx_orb)], axis=1)
        
        # Apply rotation matrix
        pos_inertial = states_orbital[:, 0:3] @ R.T
        vel_inertial = states_orbital[:, 3:6] @ R.T
        
        # Combine results
        states_inertial = np.concatenate([pos_inertial, vel_inertial], axis=1)
        
        return states_inertial
    
    def elements_to_cartesian_scalar(self, a, e, i, Omega, omega, M, mu):
        """
        标量版本的轨道根数转换（用于向后兼容）。
        
        Note:
            内部调用向量化版本，效率较低。
            新代码应使用向量化接口。
        """
        M_array = np.array([M])
        states = self.elements_to_cartesian_batch(a, e, i, Omega, omega, M_array, mu)
        return states[0]


# 便捷函数
def create_keplerian_generator(**kwargs) -> KeplerianGenerator:
    """创建开普勒轨道生成器的工厂函数"""
    return KeplerianGenerator(**kwargs)
