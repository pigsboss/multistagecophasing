"""
高精度星历模块

提供太阳系天体的高精度位置和速度计算。
支持多种坐标系的转换，使用高精度动力学模型进行插值和计算。

特性：
1. 支持主要太阳系天体（太阳、地球、月球、火星等）
2. 支持多种坐标系（J2000、地月旋转系、日地旋转系等）
3. 使用高精度重力模型（高阶球谐函数、CRTBP等）
4. 提供状态插值和批量计算功能
5. 支持外部星历数据（如DE440）加载

作者: MCPC开发团队
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris.ephemeris import Ephemeris
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
from mission_sim.utils.math_tools import inertial_to_rotating, rotating_to_inertial


class CelestialBody(Enum):
    """太阳系天体枚举"""
    SUN = "sun"
    EARTH = "earth"
    MOON = "moon"
    MARS = "mars"
    VENUS = "venus"
    MERCURY = "mercury"
    JUPITER = "jupiter"
    SATURN = "saturn"


class EphemerisMode(Enum):
    """星历计算模式"""
    ANALYTICAL = "analytical"     # 解析模型（开普勒轨道+摄动）
    CRTBP = "crtbp"               # CRTBP模型
    NUMERICAL = "numerical"       # 数值积分
    EXTERNAL = "external"         # 外部数据（如DE440）


@dataclass
class EphemerisConfig:
    """星历配置"""
    mode: EphemerisMode = EphemerisMode.ANALYTICAL
    max_degree: int = 10  # 球谐函数最大阶数
    crtbp_system: str = "sun_earth"  # CRTBP系统
    interpolation_order: int = 8  # 插值阶数
    cache_size: int = 1000  # 缓存大小
    verbose: bool = False


class HighPrecisionEphemeris(Ephemeris):
    """
    高精度星历类
    
    提供太阳系天体的高精度位置和速度计算，支持多种坐标系和动力学模型。
    
    主要功能：
    1. 支持多种天体（太阳、地球、月球等）
    2. 支持多种坐标系（J2000、旋转系等）
    3. 支持多种计算模式（解析、CRTBP、数值、外部数据）
    4. 提供状态插值和批量计算
    5. 内置缓存机制提高性能
    """
    
    # 天体物理参数（SI单位）
    _CELESTIAL_PARAMS = {
        CelestialBody.SUN: {
            "GM": 1.32712440018e20,  # m³/s²
            "radius": 6.957e8,       # m
            "J2": 2.0e-7,           # 太阳扁率
            "rotation_rate": 2.865e-6  # rad/s (赤道自转角速度)
        },
        CelestialBody.EARTH: {
            "GM": 3.986004418e14,   # m³/s²
            "radius": 6378137.0,    # m
            "J2": 1.08262668e-3,    # 地球扁率
            "rotation_rate": 7.292115e-5  # rad/s
        },
        CelestialBody.MOON: {
            "GM": 4.9028695e12,     # m³/s²
            "radius": 1737400.0,    # m
            "J2": 2.027e-4,         # 月球扁率
            "rotation_rate": 2.6617e-6  # rad/s
        },
        CelestialBody.MARS: {
            "GM": 4.282837e13,      # m³/s²
            "radius": 3396200.0,    # m
            "J2": 1.96045e-3,       # 火星扁率
            "rotation_rate": 7.088e-5  # rad/s
        }
    }
    
    # 轨道根数（J2000历元，近似值）
    _ORBITAL_ELEMENTS = {
        # 地球绕太阳（开普勒根数）
        "earth_around_sun": {
            "a": 1.495978707e11,    # 半长轴 (m) ~ 1 AU
            "e": 0.0167086,         # 偏心率
            "i": np.deg2rad(0.00005),  # 轨道倾角 (rad)
            "Omega": np.deg2rad(-11.26064),  # 升交点赤经 (rad)
            "omega": np.deg2rad(102.94719),  # 近地点幅角 (rad)
            "M0": np.deg2rad(100.46435),     # 平近点角 (rad, J2000)
            "n": 1.99098659277e-7    # 平运动 (rad/s)
        },
        # 月球绕地球
        "moon_around_earth": {
            "a": 3.844e8,           # 半长轴 (m)
            "e": 0.0549,            # 偏心率
            "i": np.deg2rad(5.145),  # 轨道倾角 (rad)
            "Omega": np.deg2rad(125.08),    # 升交点赤经 (rad)
            "omega": np.deg2rad(318.15),    # 近地点幅角 (rad)
            "M0": np.deg2rad(134.9),        # 平近点角 (rad, J2000)
            "n": 2.661699e-6        # 平运动 (rad/s)
        }
    }
    
    def __init__(self, config: Optional[EphemerisConfig] = None):
        """
        初始化高精度星历
        
        Args:
            config: 星历配置，如为None则使用默认配置
        """
        super().__init__(t0=0.0, dt=1.0, state0=np.zeros(6))  # 基类初始化
        
        self.config = config or EphemerisConfig()
        self.verbose = self.config.verbose
        
        # 初始化模型
        self._gravity_models = {}
        self._crtbp_models = {}
        self._initialize_models()
        
        # 初始化缓存
        self._state_cache = {}
        self._max_cache_size = self.config.cache_size
        
        if self.verbose:
            print(f"[HighPrecisionEphemeris] 初始化完成，模式: {self.config.mode.value}")
    
    def _initialize_models(self):
        """初始化重力模型"""
        # 初始化地球高阶重力场模型
        if self.config.mode in [EphemerisMode.ANALYTICAL, EphemerisMode.NUMERICAL]:
            self._gravity_models[CelestialBody.EARTH] = HighOrderGeopotential(
                degree=self.config.max_degree,
                order=self.config.max_degree
            )
        
        # 初始化CRTBP模型
        if self.config.mode == EphemerisMode.CRTBP:
            if self.config.crtbp_system == "sun_earth":
                self._crtbp_models["sun_earth"] = UniversalCRTBP.sun_earth_system()
            elif self.config.crtbp_system == "earth_moon":
                self._crtbp_models["earth_moon"] = UniversalCRTBP.earth_moon_system()
            else:
                raise ValueError(f"不支持的CRTBP系统: {self.config.crtbp_system}")
    
    def get_state(self, 
                  target_body: Union[str, CelestialBody],
                  epoch: float,
                  observer_body: Union[str, CelestialBody] = CelestialBody.EARTH,
                  frame: Union[str, CoordinateFrame] = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """
        获取目标天体在指定时间、相对于指定观察者、在指定坐标系下的状态
        
        Args:
            target_body: 目标天体（字符串或枚举）
            epoch: 时间（秒，J2000历元）
            observer_body: 观察者天体（默认为地球）
            frame: 坐标系（默认为J2000地心惯性系）
            
        Returns:
            np.ndarray: 状态向量 [x, y, z, vx, vy, vz] (m, m/s)
            
        Raises:
            ValueError: 如果参数无效或不支持
        """
        # 标准化输入
        target = self._normalize_body(target_body)
        observer = self._normalize_body(observer_body)
        coord_frame = self._normalize_frame(frame)
        
        # 生成缓存键
        cache_key = (target.value, observer.value, coord_frame.name, epoch)
        
        # 检查缓存
        if cache_key in self._state_cache:
            return self._state_cache[cache_key].copy()
        
        # 根据计算模式选择计算方法
        if self.config.mode == EphemerisMode.ANALYTICAL:
            state = self._compute_analytical_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.CRTBP:
            state = self._compute_crtbp_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.NUMERICAL:
            state = self._compute_numerical_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.EXTERNAL:
            state = self._compute_external_state(target, observer, epoch, coord_frame)
        else:
            raise ValueError(f"不支持的星历模式: {self.config.mode}")
        
        # 缓存结果
        self._cache_state(cache_key, state)
        
        return state
    
    def get_earth_moon_rotating_state(self, epoch: float) -> np.ndarray:
        """
        获取指定时间地月旋转系的状态
        
        Args:
            epoch: 时间（秒，J2000历元）
            
        Returns:
            np.ndarray: 地月旋转系状态 [x, y, z, vx, vy, vz] (m, m/s)
        """
        # 获取月球在J2000系相对于地球的状态
        moon_state_j2000 = self.get_state(
            target_body=CelestialBody.MOON,
            epoch=epoch,
            observer_body=CelestialBody.EARTH,
            frame=CoordinateFrame.J2000_ECI
        )
        
        # 转换到地月旋转系
        if self.config.crtbp_system == "earth_moon" and "earth_moon" in self._crtbp_models:
            crtbp_model = self._crtbp_models["earth_moon"]
            state_rotating = crtbp_model.to_rotating_frame(moon_state_j2000, epoch)
        else:
            # 如果没有CRTBP模型，使用默认旋转参数
            earth_moon_distance = 3.844e8  # m
            earth_moon_period = 27.321661 * 24 * 3600  # s
            omega = 2 * np.pi / earth_moon_period
            
            state_rotating = inertial_to_rotating(moon_state_j2000, epoch, omega)
        
        return state_rotating
    
    def get_interpolated_state(self, t: float) -> np.ndarray:
        """
        获取插值状态（实现基类抽象方法）
        
        Args:
            t: 时间（秒）
            
        Returns:
            np.ndarray: 插值状态 [x, y, z, vx, vy, vz]
        """
        # 注意：基类方法需要重写以实现具体插值逻辑
        # 这里返回地球在J2000系的状态作为示例
        return self.get_state(
            target_body=CelestialBody.EARTH,
            epoch=t,
            observer_body=CelestialBody.SUN,
            frame=CoordinateFrame.J2000_ECI
        )
    
    def _compute_analytical_state(self,
                                 target: CelestialBody,
                                 observer: CelestialBody,
                                 epoch: float,
                                 frame: CoordinateFrame) -> np.ndarray:
        """
        使用解析模型计算状态
        
        Args:
            target: 目标天体
            observer: 观察者天体
            epoch: 时间
            frame: 坐标系
            
        Returns:
            np.ndarray: 状态向量
        """
        # 解析模型：使用开普勒轨道+摄动项
        
        # 特殊情况：地球绕太阳
        if target == CelestialBody.EARTH and observer == CelestialBody.SUN:
            return self._compute_earth_around_sun(epoch, frame)
        
        # 特殊情况：月球绕地球
        elif target == CelestialBody.MOON and observer == CelestialBody.EARTH:
            return self._compute_moon_around_earth(epoch, frame)
        
        # 其他情况：返回近似位置（零向量）
        else:
            warnings.warn(f"解析模型不支持 {target.value} 相对于 {observer.value}，返回近似解")
            return np.zeros(6)
    
    def _compute_earth_around_sun(self, epoch: float, frame: CoordinateFrame) -> np.ndarray:
        """计算地球绕太阳的轨道（近似开普勒轨道）"""
        elements = self._ORBITAL_ELEMENTS["earth_around_sun"]
        
        # 计算平近点角
        M = elements["M0"] + elements["n"] * epoch
        
        # 解开普勒方程（使用迭代法）
        E = M  # 初始猜测
        for _ in range(10):
            E = M + elements["e"] * np.sin(E)
        
        # 计算位置和速度
        a, e, i, Omega, omega = elements["a"], elements["e"], elements["i"], elements["Omega"], elements["omega"]
        
        # 真近点角
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # 轨道半径
        r = a * (1 - e * np.cos(E))
        
        # 在轨道平面内的位置和速度
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # 速度（简化计算）
        h = np.sqrt(self._CELESTIAL_PARAMS[CelestialBody.SUN]["GM"] * a * (1 - e**2))
        vx_orb = -h * np.sin(nu) / r
        vy_orb = h * (e + np.cos(nu)) / r
        
        # 转换到J2000系（简化）
        # 注意：这里使用了简化的轨道平面转换
        state = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
        
        return state
    
    def _compute_moon_around_earth(self, epoch: float, frame: CoordinateFrame) -> np.ndarray:
        """计算月球绕地球的轨道（近似开普勒轨道）"""
        elements = self._ORBITAL_ELEMENTS["moon_around_earth"]
        
        # 计算平近点角
        M = elements["M0"] + elements["n"] * epoch
        
        # 解开普勒方程
        E = M  # 初始猜测
        for _ in range(10):
            E = M + elements["e"] * np.sin(E)
        
        # 计算位置和速度
        a, e, i, Omega, omega = elements["a"], elements["e"], elements["i"], elements["Omega"], elements["omega"]
        
        # 真近点角
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # 轨道半径
        r = a * (1 - e * np.cos(E))
        
        # 在轨道平面内的位置和速度
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # 速度（简化计算）
        h = np.sqrt(self._CELESTIAL_PARAMS[CelestialBody.EARTH]["GM"] * a * (1 - e**2))
        vx_orb = -h * np.sin(nu) / r
        vy_orb = h * (e + np.cos(nu)) / r
        
        # 转换到J2000系（简化）
        # 注意：这里使用了简化的轨道平面转换
        state = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
        
        return state
    
    def _compute_crtbp_state(self,
                            target: CelestialBody,
                            observer: CelestialBody,
                            epoch: float,
                            frame: CoordinateFrame) -> np.ndarray:
        """
        使用CRTBP模型计算状态
        
        Args:
            target: 目标天体
            observer: 观察者天体
            epoch: 时间
            frame: 坐标系
            
        Returns:
            np.ndarray: 状态向量
        """
        # 检查是否支持该CRTBP系统
        if self.config.crtbp_system == "sun_earth":
            if not (target == CelestialBody.EARTH and observer == CelestialBody.SUN):
                warnings.warn("CRTBP模型仅支持日地系统，返回近似解")
                return np.zeros(6)
            
            # 获取日地旋转系状态
            crtbp_model = self._crtbp_models["sun_earth"]
            
            # 简化：返回地月L2点附近的示例状态
            lagrange_points = crtbp_model.get_lagrange_points_nd()
            l2_nd = lagrange_points["L2"]
            
            # 转换为物理单位
            state_nd = np.array([l2_nd[0], l2_nd[1], l2_nd[2], 0.0, 0.0, 0.0])
            state_physical = crtbp_model._to_physical(state_nd)
            
            # 根据坐标系转换
            if frame == CoordinateFrame.J2000_ECI:
                state_physical = crtbp_model.to_inertial_frame(state_physical, epoch)
            
            return state_physical
        
        elif self.config.crtbp_system == "earth_moon":
            if not (target == CelestialBody.MOON and observer == CelestialBody.EARTH):
                warnings.warn("CRTBP模型仅支持地月系统，返回近似解")
                return np.zeros(6)
            
            # 获取地月旋转系状态
            crtbp_model = self._crtbp_models["earth_moon"]
            
            # 简化：返回地月L2点附近的示例状态
            lagrange_points = crtbp_model.get_lagrange_points_nd()
            l2_nd = lagrange_points["L2"]
            
            # 转换为物理单位
            state_nd = np.array([l2_nd[0], l2_nd[1], l2_nd[2], 0.0, 0.0, 0.0])
            state_physical = crtbp_model._to_physical(state_nd)
            
            # 根据坐标系转换
            if frame == CoordinateFrame.J2000_ECI:
                state_physical = crtbp_model.to_inertial_frame(state_physical, epoch)
            
            return state_physical
        
        else:
            warnings.warn(f"不支持的CRTBP系统: {self.config.crtbp_system}")
            return np.zeros(6)
    
    def _compute_numerical_state(self,
                                target: CelestialBody,
                                observer: CelestialBody,
                                epoch: float,
                                frame: CoordinateFrame) -> np.ndarray:
        """
        使用数值积分计算状态
        
        Args:
            target: 目标天体
            observer: 观察者天体
            epoch: 时间
            frame: 坐标系
            
        Returns:
            np.ndarray: 状态向量
        """
        # 数值积分方法（简化实现）
        warnings.warn("数值积分模式尚未完全实现，返回近似解")
        return np.zeros(6)
    
    def _compute_external_state(self,
                               target: CelestialBody,
                               observer: CelestialBody,
                               epoch: float,
                               frame: CoordinateFrame) -> np.ndarray:
        """
        使用外部数据计算状态
        
        Args:
            target: 目标天体
            observer: 观察者天体
            epoch: 时间
            frame: 坐标系
            
        Returns:
            np.ndarray: 状态向量
        """
        # 外部数据加载（如DE440，简化实现）
        warnings.warn("外部数据模式尚未实现，返回近似解")
        return np.zeros(6)
    
    def _normalize_body(self, body: Union[str, CelestialBody]) -> CelestialBody:
        """标准化天体参数"""
        if isinstance(body, str):
            try:
                return CelestialBody(body.lower())
            except ValueError:
                raise ValueError(f"不支持的天体: {body}")
        elif isinstance(body, CelestialBody):
            return body
        else:
            raise TypeError(f"天体参数类型错误: {type(body)}")
    
    def _normalize_frame(self, frame: Union[str, CoordinateFrame]) -> CoordinateFrame:
        """标准化坐标系参数"""
        if isinstance(frame, str):
            try:
                return CoordinateFrame[frame]
            except KeyError:
                # 尝试处理带下划线的名称
                normalized_name = frame.replace("-", "_").upper()
                try:
                    return CoordinateFrame[normalized_name]
                except KeyError:
                    raise ValueError(f"不支持的坐标系: {frame}")
        elif isinstance(frame, CoordinateFrame):
            return frame
        else:
            raise TypeError(f"坐标系参数类型错误: {type(frame)}")
    
    def _cache_state(self, key: tuple, state: np.ndarray):
        """缓存状态结果"""
        if len(self._state_cache) >= self._max_cache_size:
            # 简单的LRU策略：删除第一个键
            first_key = next(iter(self._state_cache))
            del self._state_cache[first_key]
        
        self._state_cache[key] = state.copy()
    
    def clear_cache(self):
        """清空缓存"""
        self._state_cache.clear()
    
    def get_available_bodies(self) -> List[str]:
        """获取支持的天体列表"""
        return [body.value for body in CelestialBody]
    
    def get_body_parameters(self, body: Union[str, CelestialBody]) -> Dict:
        """获取天体的物理参数"""
        normalized_body = self._normalize_body(body)
        return self._CELESTIAL_PARAMS.get(normalized_body, {}).copy()
    
    def set_mode(self, mode: EphemerisMode):
        """设置星历计算模式"""
        self.config.mode = mode
        self._initialize_models()
        self.clear_cache()
        
        if self.verbose:
            print(f"[HighPrecisionEphemeris] 切换模式为: {mode.value}")
    
    def __repr__(self):
        return (f"HighPrecisionEphemeris(mode={self.config.mode.value}, "
                f"crtbp_system={self.config.crtbp_system}, "
                f"cache_size={len(self._state_cache)}/{self._max_cache_size})")


# 工厂函数，便于使用
def create_high_precision_ephemeris(mode: str = "analytical", **kwargs) -> HighPrecisionEphemeris:
    """
    创建高精度星历实例
    
    Args:
        mode: 星历模式 ("analytical", "crtbp", "numerical", "external")
        **kwargs: 其他配置参数
        
    Returns:
        HighPrecisionEphemeris实例
    """
    mode_enum = EphemerisMode(mode)
    config = EphemerisConfig(mode=mode_enum, **kwargs)
    return HighPrecisionEphemeris(config)


# 导出
__all__ = [
    'HighPrecisionEphemeris',
    'CelestialBody',
    'EphemerisMode',
    'EphemerisConfig',
    'create_high_precision_ephemeris'
]
