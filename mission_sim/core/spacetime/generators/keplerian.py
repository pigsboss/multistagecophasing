# mission_sim/core/trajectory/generators/keplerian.py
"""Keplerian orbit generator (two-body analytical solution)"""

import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.generators.base import BaseTrajectoryGenerator
from mission_sim.utils.math_tools import elements_to_cartesian


class KeplerianGenerator(BaseTrajectoryGenerator):
    """
    Keplerian orbit generator (two-body analytical solution).
    Generates reference ephemeris using classical Kepler formulas, no perturbations.
    
    This is a simplified model for L1-level baseline calibration and ideal nominal
    orbit generation. It does not integrate with high-precision ephemeris.
    """

    def __init__(self, mu: float = 3.986004418e14):
        """
        Initialize the generator.

        Args:
            mu: Gravitational parameter of central body (m³/s²), default Earth.
        """
        # Note: KeplerianGenerator is a simplified model and does not use
        # high-precision ephemeris. The ephemeris and use_high_precision parameters
        # from BaseTrajectoryGenerator are intentionally ignored.
        self.mu = mu

    def generate(self, config: dict) -> Ephemeris:
        """
        Generate ephemeris from orbital elements.

        config must contain:
            - elements: [a, e, i, Omega, omega, M0] orbital elements
            - dt: time step (s)
            - sim_time: simulation duration (s)

        Returns:
            Ephemeris object (J2000_ECI frame)
        """
        elements = config.get("elements")
        if elements is None or len(elements) != 6:
            raise ValueError("KeplerianGenerator requires 6 orbital elements 'elements' in config.")

        dt = config.get("dt", 1.0)
        sim_time = config.get("sim_time", 86400.0)
        times = np.arange(0, sim_time + dt, dt)

        a, e, i, Omega, omega, M0 = elements
        n = np.sqrt(self.mu / a**3)

        states = []
        for t in times:
            # Mean anomaly
            M = M0 + n * t
            # Solve Kepler's equation M = E - e sin(E) (Newton iteration)
            E = self._kepler_solver(M, e)
            # True anomaly
            nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
            # In-plane coordinates
            r = a * (1 - e * np.cos(E))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            vx_orb = -np.sqrt(self.mu / (a * (1 - e**2))) * np.sin(nu)
            vy_orb = np.sqrt(self.mu / (a * (1 - e**2))) * (e + np.cos(nu))

            # Use full six-element conversion function
            state_eci = elements_to_cartesian(self.mu, a, e, i, Omega, omega, M)
            states.append(state_eci)

        return Ephemeris(times, np.array(states), CoordinateFrame.J2000_ECI)

    def _kepler_solver(self, M: float, e: float, tol: float = 1e-12) -> float:
        """Solve Kepler's equation M = E - e sin(E) (Newton iteration)"""
        E = M if e < 0.8 else np.pi  # Initial guess
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < tol:
                break
        return E
"""
开普勒轨道生成器

基于二体问题假设，根据轨道根数生成开普勒轨道。
支持椭圆轨道（0 ≤ e < 1），提供高精度数值积分。
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
    开普勒轨道生成器
    
    根据经典轨道根数生成二体问题轨道。
    支持椭圆轨道（0 ≤ e < 1），使用高精度数值积分。
    """
    
    def __init__(self, ephemeris=None, use_high_precision=False):
        """初始化生成器"""
        super().__init__(ephemeris, use_high_precision)
        self.mu = 3.986004418e14  # 默认地球引力常数
        
    def generate(self, config: Dict[str, Any]) -> Ephemeris:
        """
        根据轨道根数生成开普勒轨道。
        
        Args:
            config: 配置字典，必须包含：
                - elements: 轨道根数列表 [a, e, i, Ω, ω, M0]
                - dt: 输出步长 (s)
                - sim_time: 仿真时长 (s)
                可选：
                - epoch: 历元时间 (s)
                - mu: 引力常数 (m³/s²)
                
        Returns:
            Ephemeris: 生成的轨道星历
        """
        # 验证配置参数
        required_keys = ['elements', 'dt', 'sim_time']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        elements = config['elements']
        if len(elements) != 6:
            raise ValueError(f"Elements must have 6 values, got {len(elements)}")
        
        # 提取参数
        a, e, i, Omega, omega, M0 = elements
        dt = config['dt']
        sim_time = config['sim_time']
        epoch = config.get('epoch', 0.0)
        mu = config.get('mu', self.mu)
        
        # 验证轨道根数
        self._validate_elements(a, e, i, Omega, omega, M0, mu)
        
        # 生成时间序列
        num_points = int(sim_time / dt) + 1
        times = np.linspace(0, sim_time, num_points) + epoch
        
        # 生成轨道
        states = []
        for t in times:
            # 计算当前时刻的平近点角
            n = np.sqrt(mu / a**3)  # 平运动
            M = M0 + n * t
            
            # 转换为笛卡尔坐标
            state = self.elements_to_cartesian(a, e, i, Omega, omega, M, mu)
            states.append(state)
        
        states_array = np.array(states)
        
        return Ephemeris(times, states_array, CoordinateFrame.J2000_ECI)
    
    def elements_to_cartesian(self, a, e, i, Omega, omega, M, mu):
        """
        将轨道根数转换为笛卡尔状态向量。
        
        Args:
            a: 半长轴 (m)
            e: 偏心率
            i: 倾角 (rad)
            Omega: 升交点赤经 (rad)
            omega: 近地点幅角 (rad)
            M: 平近点角 (rad)
            mu: 引力常数 (m³/s²)
            
        Returns:
            np.ndarray: 笛卡尔状态向量 [x, y, z, vx, vy, vz]
        """
        # ========== 严格的参数验证 ==========
        # 1. 验证半长轴
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got a={a}")
        
        # 2. 验证偏心率（只支持椭圆轨道 e < 1）
        if e < 0:
            raise ValueError(f"Eccentricity must be non-negative, got e={e}")
        if e >= 1:
            # 根据MCPC编码标准，使用英文错误信息
            raise ValueError(f"Eccentricity must be < 1 for elliptical orbits, got e={e}")
        
        # 3. 验证轨道参数 p = a * (1 - e^2) 为正
        p = a * (1 - e**2)
        if p <= 0:
            raise ValueError(
                f"Orbital parameter p = a*(1-e^2) must be positive, "
                f"got p={p} (a={a}, e={e})"
            )
        
        # 4. 验证引力常数
        if mu <= 0:
            raise ValueError(f"Gravitational parameter must be positive, got mu={mu}")
        
        # 5. 验证角度参数在合理范围内
        if not (-2*np.pi <= i <= 2*np.pi):
            raise ValueError(f"Inclination must be in radians, got i={i} rad")
        if not (-2*np.pi <= Omega <= 2*np.pi):
            raise ValueError(f"RAAN must be in radians, got Omega={Omega} rad")
        if not (-2*np.pi <= omega <= 2*np.pi):
            raise ValueError(f"Argument of perigee must be in radians, got omega={omega} rad")
        if not (-2*np.pi <= M <= 2*np.pi):
            raise ValueError(f"Mean anomaly must be in radians, got M={M} rad")
        
        # ========== 原有计算逻辑 ==========
        # 1. 解开普勒方程求偏近点角 E
        E = M
        for _ in range(10):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            delta = f / f_prime
            E -= delta
            if abs(delta) < 1e-12:
                break
        
        # 2. 计算真近点角
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # 3. 计算轨道平面内的位置和速度
        r = a * (1 - e * np.cos(E))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        vx_orb = -np.sqrt(mu / (a * (1 - e**2))) * np.sin(nu)
        vy_orb = np.sqrt(mu / (a * (1 - e**2))) * (e + np.cos(nu))
        
        # 4. 旋转到惯性系
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        
        # 旋转矩阵
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
        
        return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
    
    def _validate_elements(self, a, e, i, Omega, omega, M0, mu):
        """验证轨道根数"""
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got a={a}")
        if e < 0 or e >= 1:
            raise ValueError(f"Eccentricity must be 0 ≤ e < 1, got e={e}")
        if mu <= 0:
            raise ValueError(f"Gravitational parameter must be positive, got mu={mu}")
        
        # 验证角度在合理范围内
        for name, value in [("i", i), ("Omega", Omega), ("omega", omega), ("M0", M0)]:
            if not (-2*np.pi <= value <= 2*np.pi):
                raise ValueError(f"{name} must be in radians between -2π and 2π, got {value}")


# 便捷函数
def create_keplerian_generator(**kwargs) -> KeplerianGenerator:
    """创建开普勒轨道生成器的工厂函数"""
    return KeplerianGenerator(**kwargs)
