# -*- coding: utf-8 -*-
"""
高精度星历模块 — 纯 SPICE 实现 + 最小分析回退

提供太阳系天体的高精度位置和速度计算，集成 NASA SPICE 工具包。
新增 ANALYTICAL 模式用于测试环境。

特性：
1. 支持主要太阳系天体（太阳、地球、月球、火星等）
2. 支持多种坐标系（J2000、地月旋转系、日地旋转系等）
3. 集成 NASA SPICE 工具包（DE440/DE441 星历）
4. 提供状态插值和批量计算功能
5. 支持外部星历数据加载

作者: MCPC开发团队
版本: 3.1.0
"""

import os
import numpy as np
import warnings
import math
from typing import Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

from mission_sim.core.spacetime.ids import CoordinateFrame

# 尝试导入 SPICE 接口
try:
    from mission_sim.core.spacetime.ephemeris.spice_interface import (
        SPICEInterface,
        SPICEConfig,
        SPICEError,
        KernelNotFoundError,
    )
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    SPICEInterface = None
    warnings.warn("SPICE interface not available. All operations will fail.")

# 分析回退需要的导入
from mission_sim.core.spacetime.ephemeris.jpl_ssb_keplerian_elements import get_elements_short
from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch


# 黄道→赤道旋转常量 (J2000)
_OBLIQUITY_J2000 = math.radians(23.4392911111)
_R_ECL2EQ = np.array([
    [1.0, 0.0,                        0.0                      ],
    [0.0, math.cos(_OBLIQUITY_J2000), -math.sin(_OBLIQUITY_J2000)],
    [0.0, math.sin(_OBLIQUITY_J2000),  math.cos(_OBLIQUITY_J2000)]
])


class CelestialBody(Enum):
    """太阳系天体枚举"""
    SUN = "sun"
    EARTH = "earth"
    EARTH_BARYCENTER = "earth_barycenter"
    MOON = "moon"
    MARS = "mars"
    VENUS = "venus"
    MERCURY = "mercury"
    JUPITER = "jupiter"
    SATURN = "saturn"
    URANUS = "uranus"
    NEPTUNE = "neptune"
    SSB = "ssb"


class EphemerisMode(Enum):
    """星历计算模式"""
    SPICE = "spice"
    ANALYTICAL = "analytical"


class EphemerisConfig:
    """星历配置"""
    def __init__(self,
                 mode: EphemerisMode = EphemerisMode.SPICE,
                 spice_kernels_path: Optional[Union[str, Path]] = None,
                 spice_mission_type: str = "earth_moon",
                 spice_use_light_time: bool = True,
                 spice_use_aberration: bool = True,
                 verbose: bool = False,
                 **kwargs):
        self.mode = mode
        self.spice_kernels_path = spice_kernels_path
        self.spice_mission_type = spice_mission_type
        self.spice_use_light_time = spice_use_light_time
        self.spice_use_aberration = spice_use_aberration
        self.verbose = verbose


class HighPrecisionEphemeris:
    """
    高精度星历类 — 纯 SPICE 实现 + 最小分析回退
    
    提供太阳系天体的高精度位置和速度计算，支持多种坐标系。
    """
    
    # SPICE 天体名称映射
    _SPICE_BODY_MAP = {
        CelestialBody.SUN: "sun",
        CelestialBody.EARTH: "earth",
        CelestialBody.MOON: "moon",
        CelestialBody.MARS: "mars",
        CelestialBody.VENUS: "venus",
        CelestialBody.MERCURY: "mercury",
        CelestialBody.JUPITER: "jupiter",
        CelestialBody.SATURN: "saturn",
        CelestialBody.URANUS: "uranus",
        CelestialBody.NEPTUNE: "neptune",
        CelestialBody.SSB: "0",
    }

    def __init__(self, config: Optional[EphemerisConfig] = None):
        """
        初始化高精度星历
        
        Args:
            config: 星历配置，如为None则使用默认配置
        """
        self.config = config or EphemerisConfig()
        self.verbose = self.config.verbose
        self._spice_interface: Optional[SPICEInterface] = None
        self._spice_initialized = False

        if self.config.mode == EphemerisMode.ANALYTICAL:
            if self.verbose:
                print("[HighPrecisionEphemeris] Using analytical mode (limited)")
            return

        # 尝试 SPICE 初始化，失败则回退到分析模式
        try:
            self._initialize_spice()
        except Exception as e:
            if self.verbose:
                print(f"[HighPrecisionEphemeris] SPICE initialization failed, "
                      f"falling back to analytical mode: {e}")
            self.config.mode = EphemerisMode.ANALYTICAL
            self._spice_interface = None
            self._spice_initialized = False

        if self.verbose:
            print(f"[HighPrecisionEphemeris] Initialization complete (mode={self.config.mode.value})")

    def _initialize_spice(self) -> bool:
        """初始化 SPICE 接口"""
        if not SPICE_AVAILABLE:
            raise RuntimeError("SPICE (spiceypy) not installed. Cannot initialize.")
        
        try:
            # 确定内核路径
            kernel_path = self.config.spice_kernels_path
            if kernel_path is None:
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
                    print("[HighPrecisionEphemeris] SPICE initialized successfully")
            else:
                self._spice_interface = None
                self._spice_initialized = False
                raise RuntimeError("SPICE initialization returned failure")
            return success

        except Exception as e:
            self._spice_interface = None
            self._spice_initialized = False
            raise RuntimeError(f"SPICE initialization error: {e}") from e

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
                if any(path.glob('**/naif*.tls')):
                    return path
        return None

    def set_mode(self, mode: EphemerisMode,
                 spice_kernels_path: Optional[Union[str, Path]] = None):
        """切换模式（主要用于测试）"""
        if mode == self.config.mode:
            return
        if mode == EphemerisMode.SPICE:
            if spice_kernels_path is not None:
                self.config.spice_kernels_path = spice_kernels_path
            self.config.mode = EphemerisMode.SPICE
            self._initialize_spice()
        elif mode == EphemerisMode.ANALYTICAL:
            self.shutdown()
            self.config.mode = EphemerisMode.ANALYTICAL

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
        """
        # 分析模式使用简单的开普勒回退
        if self.config.mode == EphemerisMode.ANALYTICAL:
            return self._compute_analytical_state(target_body, observer_body, epoch, frame)
        
        if not self._spice_initialized:
            raise SPICEError("SPICE not initialized. Cannot compute state.")
        
        target = self._normalize_body(target_body)
        observer = self._normalize_body(observer_body)
        coord_frame = self._normalize_frame(frame)

        return self._compute_spice_state(target, observer, epoch, coord_frame, abcorr)

    def _compute_analytical_state(self,
                                  target_body: Union[str, CelestialBody],
                                  observer_body: Union[str, CelestialBody],
                                  epoch: float,
                                  frame: Union[str, CoordinateFrame]) -> np.ndarray:
        """
        最小分析回退：支持 Sun, Earth (EM Bary), Moon 相对于任意已支持观察者。
        使用短周期 Kepler 根数计算 SSB 状态，然后通过向量差转换到任意观察者。
        """
        target = self._normalize_body(target_body)
        observer = self._normalize_body(observer_body)

        # 计算 SSB 状态的辅助函数（避免递归）
        def _ssb_state(body: CelestialBody) -> np.ndarray:
            if body == CelestialBody.SUN:
                return np.zeros(6)
            elif body in (CelestialBody.EARTH, CelestialBody.EARTH_BARYCENTER):
                t_cy = epoch / (36525.0 * 86400.0)
                el = get_elements_short("EM Bary", t_cy)
                a, e, inc, Omega, omega, M = (
                    el["a"], el["e"], el["i"], el["Omega"], el["omega"], el["M"]
                )
                mu_sun = 1.32712440041279419e20
                state_ecl = kepler_elements_to_cartesian_batch(
                    np.array([a]), np.array([e]), np.array([inc]),
                    np.array([Omega]), np.array([omega]), np.array([M]), mu_sun
                )[0]
                return np.concatenate([
                    _R_ECL2EQ @ state_ecl[:3],
                    _R_ECL2EQ @ state_ecl[3:6]
                ])
            elif body == CelestialBody.MOON:
                # Moon relative to SSB: EM Bary + fixed offset
                em_state = _ssb_state(CelestialBody.EARTH_BARYCENTER)
                r_moon_rel = np.array([384400e3, 0.0, 0.0])
                return em_state + np.concatenate([r_moon_rel, np.zeros(3)])
            else:
                raise NotImplementedError(
                    f"Analytical mode does not support body {body}"
                )

        # 计算相对于非 SSB 观察者的状态
        if observer != CelestialBody.SSB:
            return _ssb_state(target) - _ssb_state(observer)
        else:
            return _ssb_state(target)

    def _compute_spice_state(self,
                            target: CelestialBody,
                            observer: CelestialBody,
                            epoch: float,
                            frame: CoordinateFrame,
                            abcorr: Optional[str] = None) -> np.ndarray:
        """使用 SPICE 计算高精度状态"""
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")

        target_name = target.value
        observer_name = self._SPICE_BODY_MAP.get(observer, observer.value)

        try:
            state = self._spice_interface.get_state(
                target=target_name,
                epoch=epoch,
                observer=observer_name,
                frame=frame,
                abcorr=abcorr
            )
            return state
        except Exception as e:
            raise SPICEError(
                f"SPICE calculation failed for target={target_name}, "
                f"observer={observer_name}: {e}"
            ) from e

    def get_spice_rotation_matrix(self,
                                 from_frame: CoordinateFrame,
                                 to_frame: CoordinateFrame,
                                 epoch: float) -> np.ndarray:
        """获取两个坐标系之间的旋转矩阵（SPICE 模式）"""
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")
        return self._spice_interface.get_rotation_matrix(from_frame, to_frame, epoch)

    def get_moon_libration_matrix(self, epoch: float) -> np.ndarray:
        """获取月球天平动矩阵（SPICE 模式）"""
        if not self._spice_initialized or self._spice_interface is None:
            raise SPICEError("SPICE not initialized")
        return self._spice_interface.get_moon_libration_matrix(epoch)

    def utc_to_et(self, utc: str) -> float:
        """UTC 时间转历书时（SPICE）"""
        if self._spice_initialized and self._spice_interface:
            return self._spice_interface.utc_to_et(utc)
        else:
            from datetime import datetime
            dt = datetime.fromisoformat(utc.replace('Z', '+00:00'))
            delta = dt - datetime(2000, 1, 1, 12, 0, 0)
            return delta.total_seconds()

    def et_to_utc(self, et: float) -> str:
        """历书时转 UTC（SPICE）"""
        if self._spice_initialized and self._spice_interface:
            return self._spice_interface.et_to_utc(et)
        else:
            from datetime import datetime, timedelta
            dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=et)
            return dt.isoformat() + 'Z'

    def get_earth_moon_rotating_state(self, epoch: float) -> np.ndarray:
        """
        获取指定时间地月旋转系的状态（仅 SPICE 模式）
        """
        if self.config.mode == EphemerisMode.ANALYTICAL:
            raise NotImplementedError("Rotating state not available in analytical mode")

        if self._spice_initialized:
            try:
                moon_state = self.get_state(
                    target_body=CelestialBody.MOON,
                    epoch=epoch,
                    observer_body=CelestialBody.EARTH,
                    frame=CoordinateFrame.J2000_ECI
                )
                rot_mat = self._spice_interface.get_rotation_matrix(
                    CoordinateFrame.J2000_ECI,
                    CoordinateFrame.SUN_EARTH_ROTATING,
                    epoch
                )
                pos_rot = rot_mat @ moon_state[:3]
                vel_rot = rot_mat @ moon_state[3:6]
                return np.concatenate([pos_rot, vel_rot])
            except Exception as e:
                if self.verbose:
                    print(f"SPICE rotation failed: {e}")

        # 回退：构建瞬时旋转矩阵
        if not self._spice_initialized or self._spice_interface is None:
            # 使用分析模式的地月状态
            moon_rel = self.get_state(CelestialBody.MOON, epoch,
                                      CelestialBody.EARTH, CoordinateFrame.J2000_ECI)
        else:
            moon_rel = self._spice_interface.get_state(
                'moon', epoch, 'earth', CoordinateFrame.J2000_ECI, 'NONE'
            )
        r = moon_rel[:3]
        v = moon_rel[3:6]
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-15:
            return np.zeros(6)
        x = r / r_norm
        z = np.cross(r, v)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-15:
            z = np.array([0.0, 0.0, 1.0])
        else:
            z /= z_norm
        y = np.cross(z, x)
        rot = np.array([x, y, z]).T  # 从 J2000 到旋转系的转换矩阵
        return np.concatenate([rot.T @ r, rot.T @ v])

    def _normalize_body(self, body: Union[str, CelestialBody]) -> CelestialBody:
        """标准化天体参数"""
        if isinstance(body, str):
            try:
                return CelestialBody(body.lower())
            except ValueError:
                raise ValueError(f"Unsupported celestial body: {body}")
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
                    raise ValueError(f"Unsupported coordinate frame: {frame}")
        elif isinstance(frame, CoordinateFrame):
            return frame
        else:
            raise TypeError(f"坐标系参数类型错误: {type(frame)}")

    def shutdown(self):
        """关闭星历接口，释放资源"""
        if self._spice_interface:
            self._spice_interface.shutdown()
            self._spice_initialized = False
            if self.verbose:
                print("[HighPrecisionEphemeris] SPICE interface closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __repr__(self):
        if self.config.mode == EphemerisMode.ANALYTICAL:
            return "HighPrecisionEphemeris(mode=ANALYTICAL)"
        status = "SPICE-Ready" if self._spice_initialized else "SPICE-Failed"
        return f"HighPrecisionEphemeris(status={status})"


# 工厂函数（保留兼容）
def create_high_precision_ephemeris(mode: str = "spice", 
                                   spice_kernels_path: Optional[Union[str, Path]] = None,
                                   **kwargs) -> HighPrecisionEphemeris:
    """
    创建高精度星历实例（纯 SPICE）
    """
    config = EphemerisConfig(
        mode=EphemerisMode.SPICE,
        spice_kernels_path=spice_kernels_path,
        **kwargs
    )
    return HighPrecisionEphemeris(config=config)


__all__ = [
    'HighPrecisionEphemeris',
    'CelestialBody',
    'EphemerisMode',
    'EphemerisConfig',
    'create_high_precision_ephemeris'
]
