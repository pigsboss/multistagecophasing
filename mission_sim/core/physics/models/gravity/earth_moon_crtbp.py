"""
Earth-Moon Circular Restricted Three-Body Problem (CRTBP) Model

地月系统 CRTBP 专用实现。

继承自 UniversalCRTBP，使用地月系统精确参数初始化。
适用于月球轨道附近和地月拉格朗日点的任务。
"""

import numpy as np
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class EarthMoonCRTBP(UniversalCRTBP):
    """
    地月系统 CRTBP 专用实现。
    
    使用精确的地月系统参数：
    - GM_EARTH = 3.986004418e14 m³/s²
    - GM_MOON = 4.9048695e12 m³/s²
    - L_EM = 3.844e8 m (地月平均距离)
    
    坐标系: 地月旋转系，原点在地月质心
    """
    
    # 精确常数
    GM_EARTH = 3.986004418e14    # m³/s²
    GM_MOON = 4.9048695e12       # m³/s²
    L_EM = 3.844e8              # m (地月平均距离)
    
    def __init__(self, use_numba: bool = False):
        """
        初始化地月系统 CRTBP 模型。
        
        Args:
            use_numba: 是否启用 Numba 加速单状态计算
        """
        # 计算质量
        earth_mass = self.GM_EARTH / self.G
        moon_mass = self.GM_MOON / self.G
        
        # 调用父类初始化
        super().__init__(
            primary_mass=earth_mass,
            secondary_mass=moon_mass,
            distance=self.L_EM,
            system_name='earth_moon',
            use_numba=use_numba
        )
        
        # 保留原常量名
        self._earth_gm = self.GM_EARTH
        self._moon_gm = self.GM_MOON
        
        # 计算系统中天体的精确位置 (物理单位)
        self._earth_position_physical = np.array([-self._mu * self._L, 0.0, 0.0])
        self._moon_position_physical = np.array([(1.0 - self._mu) * self._L, 0.0, 0.0])
    
    @property
    def earth_gm(self) -> float:
        """地球的 GM (m³/s²)"""
        return self._earth_gm
    
    @property
    def moon_gm(self) -> float:
        """月球的 GM (m³/s²)"""
        return self._moon_gm
    
    @property
    def earth_position(self) -> np.ndarray:
        """地球在旋转系中的位置 (m)"""
        return self._earth_position_physical.copy()
    
    @property
    def moon_position(self) -> np.ndarray:
        """月球在旋转系中的位置 (m)"""
        return self._moon_position_physical.copy()
    
    def get_lagrange_points_physical(self) -> dict:
        """
        获取地月系统拉格朗日点的物理位置 (m)。
        
        Returns:
            包含 L1-L5 位置的字典，单位为米
        """
        lagrange_nd = self.get_lagrange_points_nd()
        
        lagrange_physical = {}
        for key, pos_nd in lagrange_nd.items():
            # 将无量纲位置转换为物理位置
            pos_physical = pos_nd * self._L
            lagrange_physical[key] = pos_physical
        
        return lagrange_physical
    
    def get_l1_distance_from_moon(self) -> float:
        """
        获取地月 L1 点距离月球的近似距离。
        
        Returns:
            L1 点距离月球的距离 (m)
        """
        # 近似公式: d ≈ (μ/3)^(1/3) * R
        gamma = (self._mu / 3.0) ** (1.0/3.0)
        l1_distance_from_moon = gamma * self._L
        return l1_distance_from_moon
    
    def get_l2_distance_from_moon(self) -> float:
        """
        获取地月 L2 点距离月球的近似距离。
        
        Returns:
            L2 点距离月球的距离 (m)
        """
        # 近似公式: d ≈ (μ/3)^(1/3) * R
        gamma = (self._mu / 3.0) ** (1.0/3.0)
        l2_distance_from_moon = gamma * self._L
        return l2_distance_from_moon
    
    def get_distance_earth_to_moon(self) -> float:
        """
        获取地球到月球的物理距离。
        
        Returns:
            地球到月球的距离 (m)
        """
        return self._L
    
    def compute_earth_centered_state(self, state_barycenter: np.ndarray) -> np.ndarray:
        """
        将质心系状态转换为地球中心系状态。
        
        Args:
            state_barycenter: 质心系状态 [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            地球中心系状态 [x, y, z, vx, vy, vz] (m, m/s)
        """
        state_earth = state_barycenter.copy()
        state_earth[0] -= self._earth_position_physical[0]  # x 方向平移
        return state_earth
    
    def compute_moon_centered_state(self, state_barycenter: np.ndarray) -> np.ndarray:
        """
        将质心系状态转换为月球中心系状态。
        
        Args:
            state_barycenter: 质心系状态 [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            月球中心系状态 [x, y, z, vx, vy, vz] (m, m/s)
        """
        state_moon = state_barycenter.copy()
        state_moon[0] -= self._moon_position_physical[0]  # x 方向平移
        return state_moon
    
    def compute_accel_moon_centered(self, state_moon_centered: np.ndarray) -> np.ndarray:
        """
        计算以月球为中心的加速度（用于月球轨道任务）。
        
        Args:
            state_moon_centered: 以月球为中心的状态 [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            以月球为中心的加速度 [ax, ay, az] (m/s²)
        """
        # 将月球中心状态转换到质心系
        state_barycenter = state_moon_centered.copy()
        state_barycenter[0] += self._moon_position_physical[0]
        
        # 计算在质心系的加速度
        accel_barycenter = self.compute_accel(state_barycenter, epoch=0.0)
        
        # 加速度在月球系和质心系中相同（纯平移）
        return accel_barycenter
    
    def get_system_info(self) -> dict:
        """
        获取地月系统详细信息。
        
        Returns:
            包含系统参数的字典
        """
        base_params = self.get_system_parameters()
        
        # 添加地月系统特定信息
        system_info = {
            **base_params,
            'earth_gm': self._earth_gm,
            'moon_gm': self._moon_gm,
            'earth_moon_distance': self._L,
            'earth_position_physical': self._earth_position_physical.tolist(),
            'moon_position_physical': self._moon_position_physical.tolist(),
            'l1_distance_from_moon_m': self.get_l1_distance_from_moon(),
            'l2_distance_from_moon_m': self.get_l2_distance_from_moon(),
            'earth_to_moon_distance_m': self.get_distance_earth_to_moon()
        }
        
        return system_info
    
    def __repr__(self) -> str:
        return f"EarthMoonCRTBP(μ={self.mu:.6f}, ω={self.omega:.2e} rad/s, L={self.distance:.2e} m)"
    
    def __str__(self) -> str:
        info = self.get_system_info()
        return (f"Earth-Moon CRTBP System:\n"
                f"  Mass ratio (μ): {info['mu']:.6e}\n"
                f"  Distance: {info['distance']:.3e} m\n"
                f"  Angular velocity: {info['omega']:.3e} rad/s\n"
                f"  Earth GM: {info['earth_gm']:.3e} m³/s²\n"
                f"  Moon GM: {info['moon_gm']:.3e} m³/s²\n"
                f"  L1 from Moon: {info['l1_distance_from_moon_m']:.3e} m "
                f"({info['l1_distance_from_moon_m']/self._L:.6f} 地月距离)")
