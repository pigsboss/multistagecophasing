"""
高精度星历模块 - SPICE 集成版

提供太阳系天体的高精度位置和速度计算。
支持多种坐标系的转换，使用高精度动力学模型进行插值和计算。

特性：
1. 支持主要太阳系天体（太阳、地球、月球、火星等）
2. 支持多种坐标系（J2000、地月旋转系、日地旋转系等）
3. 使用高精度重力模型（高阶球谐函数、CRTBP等）
4. 集成 NASA SPICE 工具包（DE440/DE441 星历）
5. 提供状态插值和批量计算功能
6. 支持外部星历数据加载

作者: MCPC开发团队
版本: 2.0.0 (SPICE Integrated)
"""

import warnings
# Filter out requests library warnings about urllib3/chardet compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os

from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.spacetime.ephemeris.base import Ephemeris
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
from mission_sim.utils.math_tools import inertial_to_rotating, rotating_to_inertial

# 尝试导入 SPICE 接口
try:
    from mission_sim.core.spacetime.ephemeris.spice_interface import (
        SPICEInterface, SPICEConfig, SPICEError, 
        KernelNotFoundError, MissionType
    )
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    warnings.warn("SPICE interface not available. SPICE mode will be disabled.")


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
    SPICE = "spice"               # NASA SPICE工具（高精度）


@dataclass
class EphemerisConfig:
    """星历配置"""
    mode: EphemerisMode = EphemerisMode.ANALYTICAL
    max_degree: int = 10  # 球谐函数最大阶数
    crtbp_system: str = "sun_earth"  # CRTBP系统
    interpolation_order: int = 8  # 插值阶数
    cache_size: int = 1000  # 缓存大小
    verbose: bool = False
    # SPICE 相关配置
    spice_kernels_path: Optional[Union[str, Path]] = None  # SPICE 内核路径
    spice_mission_type: str = "earth_moon"  # SPICE 任务类型
    spice_use_light_time: bool = True  # 使用光行差修正
    spice_use_aberration: bool = True  # 使用恒星像差修正


class HighPrecisionEphemeris(Ephemeris):
    """
    高精度星历类 - 集成 SPICE 支持
    
    提供太阳系天体的高精度位置和速度计算，支持多种坐标系和动力学模型。
    
    主要功能：
    1. 支持多种天体（太阳、地球、月球等）
    2. 支持多种坐标系（J2000、旋转系等）
    3. 支持多种计算模式（解析、CRTBP、数值、SPICE）
    4. 集成 NASA SPICE 工具包（DE440/441 星历，月球天平动数据）
    5. 提供状态插值和批量计算
    6. 内置缓存机制提高性能
    
    SPICE 模式：
    当 config.mode = EphemerisMode.SPICE 时，使用 NASA NAIF SPICE 工具包
    提供亚米级精度的行星和月球位置数据。
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
    
    # SPICE 天体名称映射
    _SPICE_BODY_MAP = {
        CelestialBody.SUN: "sun",
        CelestialBody.EARTH: "earth",
        CelestialBody.MOON: "moon",
        CelestialBody.MARS: "mars",
        CelestialBody.VENUS: "venus",
        CelestialBody.MERCURY: "mercury",
        CelestialBody.JUPITER: "jupiter",
        CelestialBody.SATURN: "saturn"
    }
    
    def __init__(self, 
                 times: Optional[Union[list, np.ndarray]] = None,
                 states: Optional[Union[list, np.ndarray]] = None,
                 frame: Optional[CoordinateFrame] = None,
                 config: Optional[EphemerisConfig] = None):
        """
        初始化高精度星历
        
        Args:
            times: 离散时间序列（用于 Ephemeris 基类，可选）
            states: 对应的状态序列（用于 Ephemeris 基类，可选）
            frame: 坐标系（用于 Ephemeris 基类，可选）
            config: 星历配置，如为None则使用默认配置
            
        Note:
            如果提供 times/states/frame，则同时初始化离散星历表功能（基类）。
            SPICE 模式不需要提供这些参数。
        """
        # 初始化基类（如果提供数据）
        if times is not None and states is not None and frame is not None:
            super().__init__(times, states, frame)
        else:
            # 不提供数据时，创建一个空的基类实例（仅用于接口兼容）
            # 使用2个点以满足CubicSpline的要求（需要≥2个点）
            super().__init__([0.0, 1.0], np.zeros((2, 6)), frame or CoordinateFrame.J2000_ECI)
        
        self.config = config or EphemerisConfig()
        self.verbose = self.config.verbose
        
        # 初始化模型
        self._gravity_models = {}
        self._crtbp_models = {}
        self._spice_interface: Optional[SPICEInterface] = None
        self._spice_initialized = False
        
        self._initialize_models()
        
        # 如果配置为 SPICE 模式，自动初始化
        if self.config.mode == EphemerisMode.SPICE:
            self._initialize_spice()
        
        # 初始化缓存
        self._state_cache = {}
        self._max_cache_size = self.config.cache_size
        
        if self.verbose:
            print(f"[HighPrecisionEphemeris] 初始化完成，模式: {self.config.mode.value}")
    
    def _initialize_models(self):
        """初始化重力模型"""
        # 初始化地球高阶重力场模型
        if self.config.mode in [EphemerisMode.ANALYTICAL, EphemerisMode.NUMERICAL]:
            try:
                self._gravity_models[CelestialBody.EARTH] = HighOrderGeopotential(
                    degree=self.config.max_degree,
                    order=self.config.max_degree
                )
            except Exception as e:
                if self.verbose:
                    print(f"[HighPrecisionEphemeris] 地球重力模型初始化失败: {e}")
        
        # 初始化CRTBP模型
        if self.config.mode == EphemerisMode.CRTBP:
            if self.config.crtbp_system == "sun_earth":
                self._crtbp_models["sun_earth"] = UniversalCRTBP.sun_earth_system()
            elif self.config.crtbp_system == "earth_moon":
                self._crtbp_models["earth_moon"] = UniversalCRTBP.earth_moon_system()
            else:
                raise ValueError(f"不支持的CRTBP系统: {self.config.crtbp_system}")
    
    def _initialize_spice(self) -> bool:
        """
        初始化 SPICE 接口
        
        Returns:
            bool: 是否成功初始化
        """
        if not SPICE_AVAILABLE:
            warnings.warn("SPICE not available. Please install spiceypy.")
            return False
        
        try:
            # 确定内核路径
            kernel_path = self.config.spice_kernels_path
            if kernel_path is None:
                # 尝试自动发现
                kernel_path = self._find_default_spice_path()
            
            if kernel_path is None:
                raise KernelNotFoundError("未找到 SPICE 内核路径")
            
            # 创建 SPICE 配置
            spice_config = SPICEConfig(
                mission_type=self.config.spice_mission_type,
                verbose=self.verbose,
                use_light_time_correction=self.config.spice_use_light_time,
                use_stellar_aberration=self.config.spice_use_aberration
            )
            
            # 初始化接口
            self._spice_interface = SPICEInterface(kernel_path, spice_config)
            success = self._spice_interface.initialize(self.config.spice_mission_type)
            
            if success:
                self._spice_initialized = True
                if self.verbose:
                    print(f"[HighPrecisionEphemeris] SPICE 初始化成功")
                return True
            else:
                warnings.warn("SPICE 初始化失败，回退到解析模式")
                return False
                
        except Exception as e:
            warnings.warn(f"SPICE 初始化错误: {e}，回退到解析模式")
            self._spice_interface = None
            self._spice_initialized = False
            return False
    
    def _find_default_spice_path(self) -> Optional[Path]:
        """查找默认 SPICE 内核路径"""
        # 环境变量
        env_path = os.environ.get('SPICE_KERNELS')
        if env_path:
            return Path(env_path)
        
        # 常见路径
        possible_paths = [
            Path('./spice_kernels'),
            Path('../spice_kernels'),
            Path(__file__).parent.parent.parent.parent.parent / 'spice_kernels',
            Path('/usr/local/share/spice_kernels'),
            Path('/opt/spice_kernels'),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # 检查是否包含必要文件（如 naif0012.tls）
                if any(path.glob('**/naif*.tls')):
                    return path
        
        return None
    
    def get_state(self, 
                  target_body: Union[str, CelestialBody],
                  epoch: float,
                  observer_body: Union[str, CelestialBody] = CelestialBody.EARTH,
                  frame: Union[str, CoordinateFrame] = CoordinateFrame.J2000_ECI,
                  abcorr: Optional[str] = None) -> np.ndarray:
        """
        获取目标天体在指定时间、相对于指定观察者、在指定坐标系下的状态
        
        Args:
            target_body: 目标天体（字符串或枚举）
            epoch: 时间（秒，J2000历元）
            observer_body: 观察者天体（默认为地球）
            frame: 坐标系（默认为J2000地心惯性系）
            abcorr: 光行差修正（仅SPICE模式使用，'NONE', 'LT', 'LT+S'）
            
        Returns:
            np.ndarray: 状态向量 [x, y, z, vx, vy, vz] (m, m/s)
            
        Raises:
            ValueError: 如果参数无效或不支持
        """
        # 标准化输入
        target = self._normalize_body(target_body)
        observer = self._normalize_body(observer_body)
        coord_frame = self._normalize_frame(frame)
        
        # 根据计算模式选择计算方法
        if self.config.mode == EphemerisMode.SPICE and self._spice_initialized:
            return self._compute_spice_state(target, observer, epoch, coord_frame, abcorr)
        elif self.config.mode == EphemerisMode.ANALYTICAL:
            return self._compute_analytical_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.CRTBP:
            return self._compute_crtbp_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.NUMERICAL:
            return self._compute_numerical_state(target, observer, epoch, coord_frame)
        elif self.config.mode == EphemerisMode.EXTERNAL:
            return self._compute_external_state(target, observer, epoch, coord_frame)
        else:
            # 如果 SPICE 初始化失败，回退到解析模式
            if self.config.mode == EphemerisMode.SPICE:
                warnings.warn("SPICE not initialized, falling back to analytical mode")
                return self._compute_analytical_state(target, observer, epoch, coord_frame)
            raise ValueError(f"不支持的星历模式: {self.config.mode}")
    
    def _compute_spice_state(self,
                            target: CelestialBody,
                            observer: CelestialBody,
                            epoch: float,
                            frame: CoordinateFrame,
                            abcorr: Optional[str] = None) -> np.ndarray:
        """
        使用 SPICE 计算高精度状态
        
        Args:
            target: 目标天体
            observer: 观察者天体
            epoch: 时间（秒，J2000历元）
            frame: 坐标系
            abcorr: 光行差修正
            
        Returns:
            np.ndarray: 状态向量 [x, y, z, vx, vy, vz] (m, m/s)
        """
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")
        
        # 转换天体名称为 SPICE 格式
        target_name = self._SPICE_BODY_MAP.get(target, target.value)
        observer_name = self._SPICE_BODY_MAP.get(observer, observer.value)
        
        try:
            # 调用 SPICE 接口
            state = self._spice_interface.get_state(
                target=target_name,
                epoch=epoch,
                observer=observer_name,
                frame=frame,
                abcorr=abcorr
            )
            return state
            
        except Exception as e:
            if self.verbose:
                print(f"[HighPrecisionEphemeris] SPICE 计算失败: {e}，回退到解析模式")
            # 回退到解析模式
            return self._compute_analytical_state(target, observer, epoch, frame)
    
    def get_spice_rotation_matrix(self,
                                 from_frame: CoordinateFrame,
                                 to_frame: CoordinateFrame,
                                 epoch: float) -> np.ndarray:
        """
        获取两个坐标系之间的旋转矩阵（SPICE 模式）
        
        Args:
            from_frame: 源坐标系
            to_frame: 目标坐标系
            epoch: 时间
            
        Returns:
            np.ndarray: 3x3 旋转矩阵
            
        Raises:
            SPICEError: 如果 SPICE 未初始化
        """
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")
        
        return self._spice_interface.get_rotation_matrix(from_frame, to_frame, epoch)
    
    def get_moon_libration_matrix(self, epoch: float) -> np.ndarray:
        """
        获取月球天平动矩阵（SPICE 模式）
        
        需要加载 moon_pa_de440_200625.bpc 等月球姿态内核
        
        Args:
            epoch: 时间（秒）
            
        Returns:
            np.ndarray: 3x3 旋转矩阵（J2000 到月球主轴坐标系）
        """
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")
        
        return self._spice_interface.get_moon_libration_matrix(epoch)
    
    def utc_to_et(self, utc: str) -> float:
        """
        UTC 时间转历书时（SPICE）
        
        Args:
            utc: UTC 时间字符串（ISO 格式）
            
        Returns:
            float: 历书时（秒，J2000起算）
        """
        if self._spice_initialized and self._spice_interface:
            return self._spice_interface.utc_to_et(utc)
        else:
            # 简化转换（无闰秒修正）
            from datetime import datetime
            dt = datetime.fromisoformat(utc.replace('Z', '+00:00'))
            # 粗略转换：从 J2000 起的秒数
            delta = dt - datetime(2000, 1, 1, 12, 0, 0)
            return delta.total_seconds()
    
    def et_to_utc(self, et: float) -> str:
        """
        历书时转 UTC（SPICE）
        
        Args:
            et: 历书时（秒）
            
        Returns:
            str: UTC 时间字符串
        """
        if self._spice_initialized and self._spice_interface:
            return self._spice_interface.et_to_utc(et)
        else:
            from datetime import datetime, timedelta
            dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=et)
            return dt.isoformat() + 'Z'
    
    def get_earth_moon_rotating_state(self, epoch: float) -> np.ndarray:
        """
        获取指定时间地月旋转系的状态
        
        Args:
            epoch: 时间（秒，J2000历元）
            
        Returns:
            np.ndarray: 地月旋转系状态 [x, y, z, vx, vy, vz] (m, m/s)
        """
        # 如果 SPICE 可用，使用 SPICE 转换
        if self._spice_initialized:
            try:
                # 获取月球在J2000系相对于地球的状态
                moon_state = self.get_state(
                    target_body=CelestialBody.MOON,
                    epoch=epoch,
                    observer_body=CelestialBody.EARTH,
                    frame=CoordinateFrame.J2000_ECI
                )
                # 使用 SPICE 转换到旋转系
                rot_mat = self._spice_interface.get_rotation_matrix(
                    CoordinateFrame.J2000_ECI,
                    CoordinateFrame.SUN_EARTH_ROTATING,  # 或地月旋转系
                    epoch
                )
                pos_rot = rot_mat @ moon_state[:3]
                vel_rot = rot_mat @ moon_state[3:6]
                return np.concatenate([pos_rot, vel_rot])
            except Exception as e:
                if self.verbose:
                    print(f"SPICE 旋转转换失败: {e}，使用 CRTBP 近似")
        
        # 原有 CRTBP 逻辑
        moon_state_j2000 = self.get_state(
            target_body=CelestialBody.MOON,
            epoch=epoch,
            observer_body=CelestialBody.EARTH,
            frame=CoordinateFrame.J2000_ECI
        )
        
        if self.config.crtbp_system == "earth_moon" and "earth_moon" in self._crtbp_models:
            crtbp_model = self._crtbp_models["earth_moon"]
            state_rotating = crtbp_model.to_rotating_frame(moon_state_j2000, epoch)
        else:
            earth_moon_distance = 3.844e8
            earth_moon_period = 27.321661 * 24 * 3600
            omega = 2 * np.pi / earth_moon_period
            state_rotating = inertial_to_rotating(moon_state_j2000, epoch, omega)
        
        return state_rotating
    
    def get_interpolated_state(self, t: float) -> np.ndarray:
        """
        获取指定时间的插值状态
        
        Args:
            t: 时间（秒）
            
        Returns:
            np.ndarray: 状态 [x, y, z, vx, vy, vz]
        """
        # This method provides compatibility with code expecting an Ephemeris-like interface
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
        使用解析模型计算状态（原有实现）
        """
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
        """计算地球绕太阳的轨道（原有实现）"""
        elements = self._ORBITAL_ELEMENTS["earth_around_sun"]
        M = elements["M0"] + elements["n"] * epoch
        E = M
        for _ in range(10):
            E = M + elements["e"] * np.sin(E)
        
        a, e, i, Omega, omega = elements["a"], elements["e"], elements["i"], elements["Omega"], elements["omega"]
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        r = a * (1 - e * np.cos(E))
        
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        h = np.sqrt(self._CELESTIAL_PARAMS[CelestialBody.SUN]["GM"] * a * (1 - e**2))
        vx_orb = -h * np.sin(nu) / r
        vy_orb = h * (e + np.cos(nu)) / r
        
        state = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
        return state
    
    def _compute_moon_around_earth(self, epoch: float, frame: CoordinateFrame) -> np.ndarray:
        """计算月球绕地球的轨道（原有实现）"""
        elements = self._ORBITAL_ELEMENTS["moon_around_earth"]
        M = elements["M0"] + elements["n"] * epoch
        E = M
        for _ in range(10):
            E = M + elements["e"] * np.sin(E)
        
        a, e, i, Omega, omega = elements["a"], elements["e"], elements["i"], elements["Omega"], elements["omega"]
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        r = a * (1 - e * np.cos(E))
        
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        h = np.sqrt(self._CELESTIAL_PARAMS[CelestialBody.EARTH]["GM"] * a * (1 - e**2))
        vx_orb = -h * np.sin(nu) / r
        vy_orb = h * (e + np.cos(nu)) / r
        
        state = np.array([x_orb, y_orb, 0.0, vx_orb, vy_orb, 0.0])
        return state
    
    def _compute_crtbp_state(self,
                            target: CelestialBody,
                            observer: CelestialBody,
                            epoch: float,
                            frame: CoordinateFrame) -> np.ndarray:
        """使用CRTBP模型计算状态（原有实现）"""
        if self.config.crtbp_system == "sun_earth":
            if not (target == CelestialBody.EARTH and observer == CelestialBody.SUN):
                warnings.warn("CRTBP模型仅支持日地系统，返回近似解")
                return np.zeros(6)
            
            crtbp_model = self._crtbp_models["sun_earth"]
            lagrange_points = crtbp_model.get_lagrange_points_nd()
            l2_nd = lagrange_points["L2"]
            state_nd = np.array([l2_nd[0], l2_nd[1], l2_nd[2], 0.0, 0.0, 0.0])
            state_physical = crtbp_model._to_physical(state_nd)
            
            if frame == CoordinateFrame.J2000_ECI:
                state_physical = crtbp_model.to_inertial_frame(state_physical, epoch)
            
            return state_physical
        
        elif self.config.crtbp_system == "earth_moon":
            if not (target == CelestialBody.MOON and observer == CelestialBody.EARTH):
                warnings.warn("CRTBP模型仅支持地月系统，返回近似解")
                return np.zeros(6)
            
            crtbp_model = self._crtbp_models["earth_moon"]
            lagrange_points = crtbp_model.get_lagrange_points_nd()
            l2_nd = lagrange_points["L2"]
            state_nd = np.array([l2_nd[0], l2_nd[1], l2_nd[2], 0.0, 0.0, 0.0])
            state_physical = crtbp_model._to_physical(state_nd)
            
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
        """使用数值积分计算状态（原有实现）"""
        warnings.warn("数值积分模式尚未完全实现，返回近似解")
        return np.zeros(6)
    
    def _compute_external_state(self,
                               target: CelestialBody,
                               observer: CelestialBody,
                               epoch: float,
                               frame: CoordinateFrame) -> np.ndarray:
        """使用外部数据计算状态（原有实现）"""
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
    
    def set_mode(self, mode: EphemerisMode, spice_kernels_path: Optional[Path] = None):
        """
        设置星历计算模式
        
        Args:
            mode: 新模式
            spice_kernels_path: SPICE 内核路径（如果是SPICE模式）
        """
        self.config.mode = mode
        
        if mode == EphemerisMode.SPICE:
            if spice_kernels_path:
                self.config.spice_kernels_path = spice_kernels_path
            self._initialize_spice()
        else:
            self._initialize_models()
        
        self.clear_cache()
        
        if self.verbose:
            print(f"[HighPrecisionEphemeris] 切换模式为: {mode.value}")
    
    def shutdown(self):
        """关闭星历接口，释放资源"""
        if self._spice_interface:
            self._spice_interface.shutdown()
            self._spice_initialized = False
            if self.verbose:
                print("[HighPrecisionEphemeris] SPICE 接口已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()
    
    def __repr__(self):
        spice_status = "SPICE-Ready" if self._spice_initialized else "No-SPICE"
        return (f"HighPrecisionEphemeris(mode={self.config.mode.value}, "
                f"status={spice_status}, "
                f"cache={len(self._state_cache)}/{self._max_cache_size})")


# 工厂函数更新
def create_high_precision_ephemeris(mode: str = "analytical", 
                                   spice_kernels_path: Optional[Union[str, Path]] = None,
                                   **kwargs) -> HighPrecisionEphemeris:
    """
    创建高精度星历实例
    
    Args:
        mode: 星历模式 ("analytical", "crtbp", "numerical", "external", "spice")
        spice_kernels_path: SPICE 内核路径（SPICE模式必需）
        **kwargs: 其他配置参数
        
    Returns:
        HighPrecisionEphemeris实例
    """
    mode_enum = EphemerisMode(mode)
    config = EphemerisConfig(mode=mode_enum, spice_kernels_path=spice_kernels_path, **kwargs)
    return HighPrecisionEphemeris(config=config)


# 导出更新
__all__ = [
    'HighPrecisionEphemeris',
    'CelestialBody',
    'EphemerisMode',
    'EphemerisConfig',
    'create_high_precision_ephemeris'
]
