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
        根据轨道根数生成开普勒轨道（向量化版本）。
        
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
            
        Note:
            - 调用者负责验证输入参数有效性
            - 使用向量化批量计算，无Python循环
            - 输出状态数组形状为 (N, 6)，N=时间点数量
        """
        # 提取参数（无验证）
        elements = config['elements']
        a, e, i, Omega, omega, M0 = elements
        dt = config['dt']
        sim_time = config['sim_time']
        epoch = config.get('epoch', 0.0)
        mu = config.get('mu', self.mu)
        
        # 生成时间序列（向量）
        num_points = int(sim_time / dt) + 1
        times = np.linspace(0, sim_time, num_points) + epoch
        
        # 计算平近点角数组（向量化）
        n = np.sqrt(mu / a**3)
        M_array = M0 + n * times
        
        # 批量转换为笛卡尔坐标（向量化）
        states_array = self.elements_to_cartesian_batch(a, e, i, Omega, omega, M_array, mu)
        
        return Ephemeris(times, states_array, CoordinateFrame.J2000_ECI)
    
    def elements_to_cartesian_batch(self, a, e, i, Omega, omega, M_array, mu):
        """
        将轨道根数批量转换为笛卡尔状态向量（向量化版本）。
        
        Args:
            a: 半长轴 (m)，标量
            e: 偏心率，标量
            i: 倾角 (rad)，标量
            Omega: 升交点赤经 (rad)，标量
            omega: 近地点幅角 (rad)，标量
            M_array: 平近点角数组 (rad)，形状为 (N,)
            mu: 引力常数 (m³/s²)，标量
            
        Returns:
            np.ndarray: 笛卡尔状态向量数组，形状为 (N, 6)
            
        Note:
            - 使用向量化运算，避免Python循环
            - 设计为可移植到GPU
            - 不进行输入验证，调用者负责
        """
        # GPU兼容性提示：以下函数设计为可移植到GPU
        # 使用纯NumPy操作，无Python循环，无递归
        # 未来可替换为ROCm/OpenCL内核
        
        # 计算平运动
        n = np.sqrt(mu / a**3)
        
        # 向量化解开普勒方程（使用定点迭代）
        M_wrapped = M_array % (2 * np.pi)
        
        # 初始猜测：使用M作为E的初始值（对小偏心率有效）
        E = M_wrapped.copy()
        
        # 迭代求解开普勒方程 E = M + e * sin(E)
        # 使用向量化迭代（适合批量计算）
        for _ in range(10):
            delta = (E - e * np.sin(E) - M_wrapped) / (1 - e * np.cos(E))
            E -= delta
        
        # 计算真近点角
        sqrt_one_plus_e = np.sqrt(1 + e)
        sqrt_one_minus_e = np.sqrt(1 - e)
        nu = 2 * np.arctan2(sqrt_one_plus_e * np.sin(E/2), sqrt_one_minus_e * np.cos(E/2))
        
        # 轨道平面坐标
        r = a * (1 - e * np.cos(E))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # 轨道平面速度
        p = a * (1 - e**2)  # 半通径
        sqrt_mu_over_p = np.sqrt(mu / p)
        vx_orb = -sqrt_mu_over_p * np.sin(nu)
        vy_orb = sqrt_mu_over_p * (e + np.cos(nu))
        
        # 旋转矩阵（对所有点相同）
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
        
        # 向量化旋转（批量处理）
        states_orbital = np.stack([x_orb, y_orb, np.zeros_like(x_orb), 
                                   vx_orb, vy_orb, np.zeros_like(vx_orb)], axis=1)
        
        # 应用旋转矩阵
        pos_inertial = states_orbital[:, 0:3] @ R.T
        vel_inertial = states_orbital[:, 3:6] @ R.T
        
        # 组合结果
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
