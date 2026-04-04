"""
Sun-Earth Circular Restricted Three-Body Problem (CRTBP) Model

日地系统 CRTBP 专用实现。

继承自 UniversalCRTBP，使用日地系统精确参数初始化。
此实现替代了原有的 GravityCRTBP 类。
"""

import numpy as np
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


class SunEarthCRTBP(UniversalCRTBP):
    """
    日地系统 CRTBP 专用实现。
    
    使用精确的日地系统参数：
    - GM_SUN = 1.32712440018e20 m³/s²
    - GM_EARTH = 3.986004418e14 m³/s²
    - AU = 1.495978707e11 m (天文单位)
    
    坐标系: 日地旋转系，原点在日地质心
    """
    
    # 精确常数 (与原有 GravityCRTBP 保持一致)
    GM_SUN = 1.32712440018e20    # m³/s²
    GM_EARTH = 3.986004418e14    # m³/s²
    AU = 1.495978707e11          # m (天文单位)
    
    # 地-太阳平均角速度 (rad/s) - 来自 GravityCRTBP
    MEAN_ANGULAR_VELOCITY = 1.990986e-7  # rad/s
    
    def __init__(self, use_numba: bool = False):
        """
        初始化日地系统 CRTBP 模型。
        
        Args:
            use_numba: 是否启用 Numba 加速单状态计算
        """
        # 计算质量 (通过 GM = G * M)
        sun_mass = self.GM_SUN / self.G
        earth_mass = self.GM_EARTH / self.G
        
        # 调用父类初始化
        super().__init__(
            primary_mass=sun_mass,
            secondary_mass=earth_mass,
            distance=self.AU,
            system_name='sun_earth',
            use_numba=use_numba
        )
        
        # 保留原 GravityCRTBP 的常量名以便兼容
        self._sun_gm = self.GM_SUN
        self._earth_gm = self.GM_EARTH
        self._au = self.AU
        
        # 计算系统中天体的精确位置 (物理单位)
        self._sun_position_physical = np.array([-self._mu * self._L, 0.0, 0.0])
        self._earth_position_physical = np.array([(1.0 - self._mu) * self._L, 0.0, 0.0])
    
    @property
    def sun_gm(self) -> float:
        """太阳的 GM (m³/s²)"""
        return self._sun_gm
    
    @property
    def earth_gm(self) -> float:
        """地球的 GM (m³/s²)"""
        return self._earth_gm
    
    @property
    def au(self) -> float:
        """天文单位 (m)"""
        return self._au
    
    @property
    def sun_position(self) -> np.ndarray:
        """太阳在旋转系中的位置 (m)"""
        return self._sun_position_physical.copy()
    
    @property
    def earth_position(self) -> np.ndarray:
        """地球在旋转系中的位置 (m)"""
        return self._earth_position_physical.copy()
    
    def get_lagrange_points_physical(self) -> dict:
        """
        获取日地系统拉格朗日点的物理位置 (m)。
        
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
    
    def get_l1_distance_from_earth(self) -> float:
        """
        获取日地 L1 点距离地球的近似距离。
        
        Returns:
            L1 点距离地球的距离 (m)
        """
        # 近似公式: d ≈ (μ/3)^(1/3) * R
        gamma = (self._mu / 3.0) ** (1.0/3.0)
        l1_distance_from_earth = gamma * self._L
        return l1_distance_from_earth
    
    def get_l2_distance_from_earth(self) -> float:
        """
        获取日地 L2 点距离地球的近似距离。
        
        Returns:
            L2 点距离地球的距离 (m)
        """
        # 近似公式: d ≈ (μ/3)^(1/3) * R
        gamma = (self._mu / 3.0) ** (1.0/3.0)
        l2_distance_from_earth = gamma * self._L
        return l2_distance_from_earth
    
    def compute_accel_earth_centered(self, state: np.ndarray) -> np.ndarray:
        """
        计算以地球为中心的加速度（用于验证）。
        
        注意：此方法仅用于测试验证，实际仿真请使用标准的 compute_accel 方法。
        
        Args:
            state: 以地球为中心的状态 [x, y, z, vx, vy, vz] (m, m/s)
            
        Returns:
            以地球为中心的加速度 [ax, ay, az] (m/s²)
        """
        # 将状态转换到质心旋转系
        state_barycenter = state.copy()
        state_barycenter[0] += self._earth_position_physical[0]  # x 方向平移
        
        # 计算在质心系的加速度
        accel_barycenter = self.compute_accel(state_barycenter, epoch=0.0)
        
        # 加速度在地球系和质心系中相同（纯平移）
        return accel_barycenter
    
    def get_system_info(self) -> dict:
        """
        获取日地系统详细信息。
        
        Returns:
            包含系统参数的字典
        """
        base_params = self.get_system_parameters()
        
        # 添加日地系统特定信息
        system_info = {
            **base_params,
            'sun_gm': self._sun_gm,
            'earth_gm': self._earth_gm,
            'au': self._au,
            'sun_position_physical': self._sun_position_physical.tolist(),
            'earth_position_physical': self._earth_position_physical.tolist(),
            'l1_distance_from_earth_m': self.get_l1_distance_from_earth(),
            'l2_distance_from_earth_m': self.get_l2_distance_from_earth(),
            'mean_angular_velocity': self.MEAN_ANGULAR_VELOCITY
        }
        
        return system_info
    
    def __repr__(self) -> str:
        return f"SunEarthCRTBP(μ={self.mu:.6f}, ω={self.omega:.2e} rad/s, L={self.distance:.2e} m)"
    
    def __str__(self) -> str:
        info = self.get_system_info()
        return (f"Sun-Earth CRTBP System:\n"
                f"  Mass ratio (μ): {info['mu']:.6e}\n"
                f"  Distance (AU): {info['distance']:.3e} m ({info['distance']/self.AU:.6f} AU)\n"
                f"  Angular velocity: {info['omega']:.3e} rad/s\n"
                f"  Sun GM: {info['sun_gm']:.3e} m³/s²\n"
                f"  Earth GM: {info['earth_gm']:.3e} m³/s²\n"
                f"  L1 from Earth: {info['l1_distance_from_earth_m']:.3e} m "
                f"({info['l1_distance_from_earth_m']/self.AU:.6f} AU)")
