"""
SPICE Interface Module - High-Precision Ephemeris for MCPC

基于 NASA NAIF SPICE 工具包 (spiceypy) 的高精度星历计算接口。
提供内核管理、时间转换、几何状态计算和坐标系转换功能。

Dependencies:
    - spiceypy: SPICE 工具包 Python 接口
    - numpy: 数值计算

Usage:
    >>> from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
    >>> spice = SPICEInterface('./spice_kernels')
    >>> spice.initialize(mission_type='earth_moon')
    >>> state = spice.get_state('moon', epoch=0.0, observer='earth', frame='J2000')
"""

import warnings
# Filter out requests library warnings about urllib3/chardet compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import os

# 尝试导入 spiceypy，若失败则标记为不可用
try:
    import spiceypy as spice
    from spiceypy.utils.exceptions import SpiceyError
    SPICE_AVAILABLE = True
except ImportError:
    spice = None
    SpiceyError = None
    SPICE_AVAILABLE = False
    warnings.warn("spiceypy not installed. SPICE functionality will be disabled.")

from mission_sim.core.spacetime.ids import CoordinateFrame


class SPICEError(Exception):
    """SPICE 操作相关错误"""
    pass


class KernelNotFoundError(SPICEError):
    """内核文件未找到错误"""
    pass


class KernelLoadError(SPICEError):
    """内核加载失败错误"""
    pass


@dataclass
class SPICEConfig:
    """SPICE 配置参数"""
    mission_type: str = "earth_moon"  # earth_moon, sun_earth, earth_only
    auto_discover: bool = True        # 自动发现内核文件
    verbose: bool = False             # 详细输出
    use_light_time_correction: bool = True  # 使用光行差修正
    use_stellar_aberration: bool = True     # 使用恒星像差修正
    max_cache_size: int = 1000        # 状态缓存大小


class MissionType(Enum):
    """任务类型枚举，决定加载哪些内核"""
    EARTH_MOON = "earth_moon"      # 地月系任务（Halo轨道等）
    SUN_EARTH = "sun_earth"        # 日地系任务（L1/L2点）
    EARTH_ONLY = "earth_only"      # 仅地球任务（LEO/GEO）
    LUNAR_ONLY = "lunar_only"      # 仅月球任务（月球轨道）
    INTERPLANETARY = "interplanetary"  # 行星际任务


class SPICEKernelManager:
    """
    SPICE 内核管理器
    
    负责内核文件的自动发现、加载和卸载。
    支持内核优先级管理和冲突检测。
    """
    
    # 内核类型到文件模式的映射
    KERNEL_PATTERNS = {
        'lsk': {
            'patterns': ['naif00*.tls', 'latest_leapseconds.tls'],
            'required': True,
            'description': 'Leapseconds kernel (time conversion)'
        },
        'pck': {
            'patterns': ['pck*.tpc', 'moon_pa*.bpc', 'earth*.bpc', 'earth*.tf'],
            'required': False,
            'description': 'Planetary constants (orientation/shape)'
        },
        'spk_planets': {
            'patterns': ['de440.bsp', 'de441.bsp', 'de442.bsp', 'de430.bsp'],
            'required': True,
            'description': 'Planetary ephemeris'
        },
        'spk_moon': {
            'patterns': ['moon_*.tf', 'moon_pa_*.bpc'],
            'required': False,
            'description': 'Lunar orientation and reference frames'
        },
        'fk': {
            'patterns': ['moon_*.tf', 'earth_*.tf'],
            'required': False,
            'description': 'Frame definitions'
        },
        'ik': {
            'patterns': ['*_ik.tf'],
            'required': False,
            'description': 'Instrument kernels'
        }
    }
    
    def __init__(self, kernel_root: Union[str, Path], config: Optional[SPICEConfig] = None):
        """
        初始化内核管理器
        
        Args:
            kernel_root: SPICE 内核根目录路径
            config: 配置对象
        """
        if not SPICE_AVAILABLE:
            raise SPICEError("spiceypy not available. Cannot initialize kernel manager.")
            
        self.kernel_root = Path(kernel_root).resolve()
        self.config = config or SPICEConfig()
        self.loaded_kernels: List[Path] = []
        self._kernel_handles: Dict[str, int] = {}  # 内核句柄（用于卸载特定内核）
        self._is_initialized = False
        
        if not self.kernel_root.exists():
            raise KernelNotFoundError(f"Kernel root directory not found: {self.kernel_root}")
    
    def initialize(self, mission_type: Optional[str] = None) -> None:
        """
        初始化并加载内核
        
        Args:
            mission_type: 任务类型，覆盖配置中的设置
        """
        if self._is_initialized:
            if self.config.verbose:
                print("[SPICE] Already initialized, skipping.")
            return
            
        mtype = mission_type or self.config.mission_type
        if self.config.verbose:
            print(f"[SPICE] Initializing for mission type: {mtype}")
        
        try:
            # 清空现有内核（如果有）
            self.unload_all()
            
            # 1. 加载必需的 LSK（闰秒内核）
            self._load_kernel_type('lsk')
            
            # 2. 根据任务类型加载其他内核
            if mtype in ['earth_moon', 'earth_only', 'interplanetary']:
                self._load_kernel_type('spk_planets')
                self._load_kernel_type('pck')
                
            if mtype in ['earth_moon', 'lunar_only']:
                self._load_kernel_type('spk_moon')
                self._load_kernel_type('fk')
            
            if mtype == 'sun_earth':
                self._load_kernel_type('spk_planets')  # 包含太阳和地球数据
                self._load_kernel_type('pck')
            
            self._is_initialized = True
            
            if self.config.verbose:
                print(f"[SPICE] Successfully loaded {len(self.loaded_kernels)} kernels")
                
        except Exception as e:
            # 发生错误时清理
            self.unload_all()
            raise KernelLoadError(f"Failed to initialize SPICE kernels: {e}")
    
    def _load_kernel_type(self, ktype: str) -> None:
        """
        加载特定类型的内核
        
        Args:
            ktype: 内核类型键（如 'lsk', 'spk_planets'）
        """
        if ktype not in self.KERNEL_PATTERNS:
            warnings.warn(f"Unknown kernel type: {ktype}")
            return
            
        config = self.KERNEL_PATTERNS[ktype]
        found = False
        
        # 按优先级顺序尝试加载
        for pattern in config['patterns']:
            # 在根目录和子目录中搜索
            paths = list(self.kernel_root.rglob(pattern))
            # 也检查直接子目录
            if not paths:
                for subdir in ['lsk', 'pck', 'spk', 'fk', 'ik', 'spk/planets', 'fk/satellites']:
                    candidate = self.kernel_root / subdir / pattern
                    if candidate.exists():
                        paths.append(candidate)
            
            if paths:
                # 按文件名排序，通常最新的版本排在前面（如 de440 > de430）
                paths.sort(reverse=True)
                selected = paths[0]
                
                try:
                    self._load_single_kernel(selected, ktype)
                    found = True
                    if self.config.verbose:
                        print(f"[SPICE] Loaded {ktype}: {selected.name}")
                    break
                except Exception as e:
                    warnings.warn(f"Failed to load kernel {selected}: {e}")
                    continue
        
        if not found and config['required']:
            raise KernelNotFoundError(
                f"Required kernel '{ktype}' not found. "
                f"Searched patterns: {config['patterns']}"
            )
    
    def _load_single_kernel(self, path: Path, ktype: str) -> None:
        """加载单个内核文件"""
        if not path.exists():
            raise KernelNotFoundError(f"Kernel file not found: {path}")
        
        # 使用 spiceypy 加载
        spice.furnsh(str(path))
        self.loaded_kernels.append(path)
        self._kernel_handles[ktype] = len(self.loaded_kernels) - 1
    
    def unload_all(self) -> None:
        """卸载所有内核"""
        if SPICE_AVAILABLE:
            spice.kclear()
        self.loaded_kernels.clear()
        self._kernel_handles.clear()
        self._is_initialized = False
        
        if self.config.verbose:
            print("[SPICE] All kernels unloaded")
    
    def get_loaded_kernels(self) -> List[Path]:
        """获取已加载内核列表"""
        return self.loaded_kernels.copy()
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._is_initialized


class SPICECalculator:
    """
    SPICE 计算核心类
    
    提供基于 SPICE 的几何计算功能，包括：
    - 天体状态查询（位置/速度）
    - 姿态/旋转矩阵查询
    - 坐标系转换
    - 时间转换
    """
    
    # NAIF ID 映射表
    NAIF_IDS = {
        'sun': 10,
        'mercury': 199,
        'venus': 299,
        'earth': 399,
        'moon': 301,
        'mars': 499,
        'mars_barycenter': 4,
        'jupiter': 599,
        'saturn': 699,
        'l2_earth_moon': 302,  # 地月 L2 点的虚拟 ID（需自定义）
    }
    
    # 坐标系名称映射：MCPC -> SPICE
    FRAME_MAP = {
        CoordinateFrame.J2000_ECI: 'J2000',
        CoordinateFrame.SUN_EARTH_ROTATING: 'SUN_EARTH_L2',  # 需要定义
        CoordinateFrame.LVLH: 'LVLH',  # 局部垂直局部水平（动态定义）
        CoordinateFrame.VVLH: 'VVLH',
    }
    
    def __init__(self, kernel_manager: SPICEKernelManager):
        """
        初始化计算器
        
        Args:
            kernel_manager: 已初始化的内核管理器
        """
        if not SPICE_AVAILABLE:
            raise SPICEError("spiceypy not available.")
            
        self.km = kernel_manager
        if not self.km.is_initialized():
            raise SPICEError("Kernel manager not initialized. Call initialize() first.")
        
        self.config = self.km.config
    
    def get_state(self,
                  target: Union[str, int],
                  epoch: float,
                  observer: Union[str, int] = 'earth',
                  frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI,
                  abcorr: Optional[str] = None) -> np.ndarray:
        """
        获取目标天体相对于观察者的状态（位置/速度）
        
        Args:
            target: 目标天体名称或 NAIF ID
            epoch: 历书时（秒，J2000 起算，即 TDB/ET）
            observer: 观察者天体名称或 NAIF ID
            frame: 参考坐标系（MCPC CoordinateFrame 或 SPICE 框架名）
            abcorr: 光行差修正模式：
                   'NONE' - 无修正
                   'LT' - 光行时间修正
                   'LT+S' - 光行时间+恒星像差
                   None - 使用配置默认值
        
        Returns:
            np.ndarray: 6 维状态向量 [x, y, z, vx, vy, vz] (m, m/s)
        
        Raises:
            SPICEError: 计算失败时抛出
        """
        # 确定光行差修正模式
        if abcorr is None:
            abcorr = self._get_default_abcorr()
        
        # 转换目标/观察者为 NAIF ID
        target_id = self._to_naif_id(target)
        observer_id = self._to_naif_id(observer)
        
        # 转换坐标系
        spice_frame = self._to_spice_frame(frame)
        
        try:
            # 调用 SPICE spkezr 函数
            # 返回状态（km, km/s）和光行时间（s）
            state_km, lt = spice.spkezr(
                target_id,
                epoch,
                spice_frame,
                abcorr,
                observer_id
            )
            
            # 转换为米制（MCPC 标准单位）
            state_m = np.array(state_km) * 1000.0
            
            return state_m
            
        except SpiceyError as e:
            raise SPICEError(f"SPICE calculation failed for {target} wrt {observer} at ET={epoch}: {e}")
    
    def get_geometric_state(self,
                           target: Union[str, int],
                           epoch: float,
                           observer: Union[str, int] = 'earth',
                           frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """
        获取几何状态（无相对论修正，最快）
        
        等价于 abcorr='NONE' 的 get_state()
        """
        return self.get_state(target, epoch, observer, frame, abcorr='NONE')
    
    def get_light_time_corrected_state(self,
                                      target: Union[str, int],
                                      epoch: float,
                                      observer: Union[str, int] = 'earth',
                                      frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> Tuple[np.ndarray, float]:
        """
        获取光行时间修正后的状态
        
        Returns:
            Tuple[状态向量, 光行时间秒]
        """
        state_m = self.get_state(target, epoch, observer, frame, abcorr='LT')
        
        # 计算光行时间（用于信息输出）
        target_id = self._to_naif_id(target)
        observer_id = self._to_naif_id(observer)
        spice_frame = self._to_spice_frame(frame)
        
        _, lt = spice.spkezr(target_id, epoch, spice_frame, 'LT', observer_id)
        
        return state_m, lt
    
    def get_rotation_matrix(self,
                           from_frame: Union[CoordinateFrame, str],
                           to_frame: Union[CoordinateFrame, str],
                           epoch: float) -> np.ndarray:
        """
        获取两个坐标系之间的旋转矩阵
        
        Args:
            from_frame: 源坐标系
            to_frame: 目标坐标系
            epoch: 时间（历书时秒）
        
        Returns:
            np.ndarray: 3x3 旋转矩阵
        """
        from_str = self._to_spice_frame(from_frame)
        to_str = self._to_spice_frame(to_frame)
        
        try:
            rot_mat = spice.pxform(from_str, to_str, epoch)
            return np.array(rot_mat)
        except SpiceyError as e:
            raise SPICEError(f"Failed to get rotation matrix from {from_str} to {to_str}: {e}")
    
    def transform_state(self,
                       state: np.ndarray,
                       from_frame: Union[CoordinateFrame, str],
                       to_frame: Union[CoordinateFrame, str],
                       epoch: float) -> np.ndarray:
        """
        转换状态向量到不同坐标系
        
        Args:
            state: 6 维状态向量 [pos(3), vel(3)]
            from_frame: 源坐标系
            to_frame: 目标坐标系
            epoch: 时间
        
        Returns:
            np.ndarray: 转换后的状态向量
        """
        # 获取旋转矩阵
        rot_mat = self.get_rotation_matrix(from_frame, to_frame, epoch)
        
        # 转换位置和速度
        new_pos = rot_mat @ state[:3]
        new_vel = rot_mat @ state[3:6]
        
        return np.concatenate([new_pos, new_vel])
    
    def get_moon_libration_matrix(self, epoch: float) -> np.ndarray:
        """
        获取月球天平动矩阵（J2000 -> MOON_PA）
        
        需要加载 moon_pa_de440_200625.bpc 等月球姿态内核
        
        Args:
            epoch: 历书时秒
        
        Returns:
            np.ndarray: 3x3 旋转矩阵（从 J2000 到月球主轴坐标系）
        """
        try:
            # MOON_PA 是 Principal Axis 坐标系（物理主轴）
            rot_mat = spice.pxform('J2000', 'MOON_PA', epoch)
            return np.array(rot_mat)
        except SpiceyError as e:
            # 如果 MOON_PA 未定义，尝试 IAU_MOON
            try:
                rot_mat = spice.pxform('J2000', 'IAU_MOON', epoch)
                warnings.warn("MOON_PA not available, using IAU_MOON instead")
                return np.array(rot_mat)
            except SpiceyError:
                raise SPICEError(f"Failed to get moon libration matrix: {e}")
    
    def utc_to_et(self, utc: str) -> float:
        """
        UTC 时间字符串转历书时（秒）
        
        Args:
            utc: UTC 时间字符串（ISO 格式，如 '2026-04-10T12:00:00'）
        
        Returns:
            float: 历书时（秒，J2000 起算）
        """
        try:
            return spice.utc2et(utc)
        except SpiceyError as e:
            raise SPICEError(f"Failed to convert UTC '{utc}' to ET: {e}")
    
    def et_to_utc(self, et: float, format: str = 'ISOC') -> str:
        """
        历书时转 UTC 字符串
        
        Args:
            et: 历书时秒
            format: 输出格式 ('ISOC' ISO日历, 'JULIAN', 等)
        
        Returns:
            str: UTC 时间字符串
        """
        try:
            return spice.et2utc(et, format, 6)  # 6位小数精度
        except SpiceyError as e:
            raise SPICEError(f"Failed to convert ET {et} to UTC: {e}")
    
    def get_lagrange_point_state(self,
                                point: str,
                                epoch: float,
                                frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """
        获取拉格朗日点状态（基于 CRTBP 近似）
        
        注意：这是近似计算，基于瞬时 CRTBP 参数
        
        Args:
            point: 'L1', 'L2', 'L3', 'L4', 'L5'
            epoch: 时间
            frame: 坐标系
        
        Returns:
            np.ndarray: 状态向量
        """
        # 获取地月质心和距离
        earth_state = self.get_state('earth', epoch, 'sun', frame, 'NONE')
        moon_state = self.get_state('moon', epoch, 'earth', frame, 'NONE')
        
        # 计算瞬时距离向量
        r_vec = moon_state[:3] - earth_state[:3]
        r = np.linalg.norm(r_vec)
        
        # 质量比（太阳-地球或地球-月球）
        # 这里简化处理，假设是地月系
        mu = 0.012150585609624  # 地月系统质量比
        
        # 拉格朗日点距离（简化公式）
        if point == 'L1':
            # L1 在地球和月球之间
            alpha = 1 - (mu/3)**(1/3)
            pos = earth_state[:3] + alpha * r_vec
        elif point == 'L2':
            # L2 在月球外侧
            alpha = 1 + (mu/3)**(1/3)
            pos = earth_state[:3] + alpha * r_vec
        elif point == 'L3':
            # L3 在地球另一侧
            pos = earth_state[:3] - (1 - 5*mu/12) * r_vec
        else:
            raise NotImplementedError(f"Lagrange point {point} calculation not implemented")
        
        # 速度（近似为与月球相同速度，实际应计算旋转系中的平衡）
        vel = moon_state[3:6]
        
        return np.concatenate([pos, vel])
    
    def _to_naif_id(self, body: Union[str, int]) -> int:
        """转换天体名称为 NAIF ID"""
        if isinstance(body, int):
            return body
        if isinstance(body, str):
            if body.lower() in self.NAIF_IDS:
                return self.NAIF_IDS[body.lower()]
            # 尝试直接解析为整数
            try:
                return int(body)
            except ValueError:
                pass
        raise SPICEError(f"Unknown body identifier: {body}")
    
    def _to_spice_frame(self, frame: Union[CoordinateFrame, str]) -> str:
        """转换 MCPC CoordinateFrame 到 SPICE 框架名"""
        if isinstance(frame, CoordinateFrame):
            if frame in self.FRAME_MAP:
                return self.FRAME_MAP[frame]
            else:
                # 默认使用枚举名
                return frame.name
        elif isinstance(frame, str):
            return frame
        else:
            raise SPICEError(f"Invalid frame type: {type(frame)}")
    
    def _get_default_abcorr(self) -> str:
        """获取默认光行差修正模式"""
        if self.config.use_light_time_correction and self.config.use_stellar_aberration:
            return 'LT+S'
        elif self.config.use_light_time_correction:
            return 'LT'
        else:
            return 'NONE'


class SPICEInterface:
    """
    SPICE 高层接口类
    
    整合内核管理和计算功能，提供简单易用的 API。
    这是 MCPC 与 SPICE 交互的主要入口点。
    """
    
    def __init__(self, 
                 kernel_root: Optional[Union[str, Path]] = None,
                 config: Optional[SPICEConfig] = None):
        """
        初始化 SPICE 接口
        
        Args:
            kernel_root: SPICE 内核根目录，若为 None 则尝试环境变量或默认位置
            config: 配置对象
        """
        if not SPICE_AVAILABLE:
            raise SPICEError(
                "spiceypy is not installed. "
                "Please install it with: pip install spiceypy"
            )
        
        self.config = config or SPICEConfig()
        
        # 确定内核根目录
        if kernel_root is None:
            kernel_root = self._find_default_kernel_path()
        
        self.kernel_root = Path(kernel_root) if kernel_root else None
        
        if self.kernel_root and not self.kernel_root.exists():
            raise KernelNotFoundError(f"Kernel directory not found: {self.kernel_root}")
        
        self._km: Optional[SPICEKernelManager] = None
        self._calc: Optional[SPICECalculator] = None
    
    def initialize(self, mission_type: Optional[str] = None) -> bool:
        """
        初始化 SPICE 接口
        
        Args:
            mission_type: 任务类型，如 'earth_moon', 'sun_earth'
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            if self.kernel_root is None:
                raise KernelNotFoundError("No kernel root specified")
            
            self._km = SPICEKernelManager(self.kernel_root, self.config)
            self._km.initialize(mission_type or self.config.mission_type)
            self._calc = SPICECalculator(self._km)
            
            return True
            
        except Exception as e:
            warnings.warn(f"SPICE initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查 SPICE 是否可用且已初始化"""
        return SPICE_AVAILABLE and self._km is not None and self._km.is_initialized()
    
    def get_state(self,
                  target: Union[str, int],
                  epoch: float,
                  observer: Union[str, int] = 'earth',
                  frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI,
                  abcorr: Optional[str] = None) -> np.ndarray:
        """
        获取天体状态（委托给 SPICECalculator）
        
        See SPICECalculator.get_state() for parameters.
        """
        if not self.is_available():
            raise SPICEError("SPICE not initialized. Call initialize() first.")
        return self._calc.get_state(target, epoch, observer, frame, abcorr)
    
    def get_geometric_state(self,
                           target: Union[str, int],
                           epoch: float,
                           observer: Union[str, int] = 'earth',
                           frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """获取几何状态"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.get_geometric_state(target, epoch, observer, frame)
    
    def get_light_time_corrected_state(self,
                                      target: Union[str, int],
                                      epoch: float,
                                      observer: Union[str, int] = 'earth',
                                      frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> Tuple[np.ndarray, float]:
        """获取光行时间修正后的状态"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.get_light_time_corrected_state(target, epoch, observer, frame)
    
    def get_rotation_matrix(self,
                           from_frame: Union[CoordinateFrame, str],
                           to_frame: Union[CoordinateFrame, str],
                           epoch: float) -> np.ndarray:
        """获取旋转矩阵"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.get_rotation_matrix(from_frame, to_frame, epoch)
    
    def transform_state(self,
                       state: np.ndarray,
                       from_frame: Union[CoordinateFrame, str],
                       to_frame: Union[CoordinateFrame, str],
                       epoch: float) -> np.ndarray:
        """转换状态"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.transform_state(state, from_frame, to_frame, epoch)
    
    def utc_to_et(self, utc: str) -> float:
        """UTC 转历书时"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.utc_to_et(utc)
    
    def et_to_utc(self, et: float, format: str = 'ISOC') -> str:
        """历书时转 UTC"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.et_to_utc(et, format)
    
    def get_moon_libration_matrix(self, epoch: float) -> np.ndarray:
        """获取月球天平动矩阵"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.get_moon_libration_matrix(epoch)
    
    def get_lagrange_point_state(self,
                                point: str,
                                epoch: float,
                                frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """获取拉格朗日点状态"""
        if not self.is_available():
            raise SPICEError("SPICE not initialized.")
        return self._calc.get_lagrange_point_state(point, epoch, frame)
    
    def shutdown(self):
        """关闭接口，卸载所有内核"""
        if self._km:
            self._km.unload_all()
            self._km = None
            self._calc = None
    
    def _find_default_kernel_path(self) -> Optional[Path]:
        """尝试查找默认内核路径"""
        # 1. 环境变量
        env_path = os.environ.get('SPICE_KERNELS')
        if env_path:
            return Path(env_path)
        
        # 2. 项目默认位置
        default_paths = [
            Path('./spice_kernels'),
            Path('../spice_kernels'),
            Path(__file__).parent.parent.parent.parent.parent / 'spice_kernels'
        ]
        
        for path in default_paths:
            if path.exists():
                return path
        
        return None
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()


# 便捷函数：快速获取状态
def get_spice_state(kernel_root: Union[str, Path],
                   target: str,
                   epoch: float,
                   observer: str = 'earth',
                   frame: Union[CoordinateFrame, str] = CoordinateFrame.J2000_ECI) -> np.ndarray:
    """
    一次性获取 SPICE 状态的便捷函数
    
    适用于简单的单次查询，自动管理初始化和清理。
    
    Example:
        >>> state = get_spice_state('./spice_kernels', 'moon', 0.0)
    """
    spice_if = SPICEInterface(kernel_root)
    try:
        spice_if.initialize()
        return spice_if.get_state(target, epoch, observer, frame)
    finally:
        spice_if.shutdown()


# 导出
__all__ = [
    'SPICEInterface',
    'SPICEKernelManager',
    'SPICECalculator',
    'SPICEConfig',
    'SPICEError',
    'KernelNotFoundError',
    'KernelLoadError',
    'MissionType',
    'get_spice_state'
]
