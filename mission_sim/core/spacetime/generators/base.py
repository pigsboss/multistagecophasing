# mission_sim/core/trajectory/generators/base.py
"""标称轨道生成器抽象基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
from mission_sim.core.spacetime.ephemeris import Ephemeris
from mission_sim.core.spacetime.ids import CoordinateFrame, CelestialBody


class BaseTrajectoryGenerator(ABC):
    """
    标称轨道生成器抽象基类。
    所有具体生成器必须实现 generate 方法，返回 Ephemeris 对象。
    
    支持高精度星历集成，可从星历获取天体精确位置作为参考。
    可根据需要选择使用简化模型或高精度星历。
    """

    def __init__(self, ephemeris: Optional[Any] = None, use_high_precision: bool = False):
        """
        初始化生成器。
        
        Args:
            ephemeris: 可选的高精度星历实例（如 HighPrecisionEphemeris 或 SPICEInterface）
            use_high_precision: 是否使用高精度星历（如果可用）
        """
        self.ephemeris = ephemeris
        self.use_high_precision = use_high_precision and (ephemeris is not None)
        self._config = {}

    @abstractmethod
    def generate(self, config: Dict[str, Any]) -> Ephemeris:
        """
        生成标称轨道星历。

        Args:
            config: 生成器特定配置字典

        Returns:
            Ephemeris: 标称轨道星历表
        """
        pass

    def _get_celestial_state(self, body: Union[str, CelestialBody], epoch: float, 
                            observer: Optional[Union[str, CelestialBody]] = None,
                            frame: CoordinateFrame = CoordinateFrame.J2000_ECI) -> np.ndarray:
        """
        从高精度星历获取天体状态。
        
        Args:
            body: 目标天体名称或枚举
            epoch: 历元时间（秒，J2000起算）
            observer: 观察者天体（默认为None，使用系统默认）
            frame: 坐标系
            
        Returns:
            np.ndarray: 天体状态向量 [x, y, z, vx, vy, vz]
            
        Raises:
            ValueError: 如果未提供星历实例或高精度模式未启用
        """
        if not self.use_high_precision or self.ephemeris is None:
            raise ValueError(f"高精度星历模式未启用或未提供星历实例")
        
        try:
            # 尝试从 HighPrecisionEphemeris 获取状态
            if hasattr(self.ephemeris, 'get_state'):
                if observer is None:
                    # 根据天体类型选择默认观察者
                    if isinstance(body, str):
                        body_str = body.lower()
                    else:
                        body_str = body.value if isinstance(body, CelestialBody) else str(body)
                    
                    # 默认观察者选择
                    if body_str in ['moon', 'lunar']:
                        observer = CelestialBody.EARTH
                    elif body_str in ['earth']:
                        observer = CelestialBody.SUN
                    else:
                        observer = CelestialBody.EARTH
                
                return self.ephemeris.get_state(body, epoch, observer, frame)
            
            # 尝试从 SPICEInterface 获取状态
            elif hasattr(self.ephemeris, 'get_state'):
                # SPICEInterface 有类似的接口
                target_name = body.value if isinstance(body, CelestialBody) else body
                observer_name = observer.value if isinstance(observer, CelestialBody) else observer
                return self.ephemeris.get_state(target_name, epoch, observer_name, frame)
            
            else:
                raise AttributeError("星历实例不支持 get_state 方法")
                
        except Exception as e:
            raise RuntimeError(f"获取天体 '{body}' 状态失败: {e}")

    def _get_moon_libration_matrix(self, epoch: float) -> Optional[np.ndarray]:
        """
        获取月球天平动矩阵（如果可用）。
        
        Args:
            epoch: 历元时间（秒）
            
        Returns:
            Optional[np.ndarray]: 3x3 旋转矩阵，或 None（如果不支持）
        """
        if not self.use_high_precision or self.ephemeris is None:
            return None
            
        try:
            # 检查是否支持月球天平动矩阵
            if hasattr(self.ephemeris, 'get_moon_libration_matrix'):
                return self.ephemeris.get_moon_libration_matrix(epoch)
            return None
        except Exception:
            return None

    def _validate_config(self, config: Dict[str, Any], required_keys: list) -> None:
        """
        验证配置字典是否包含必需的键。
        
        Args:
            config: 配置字典
            required_keys: 必需键列表
            
        Raises:
            ValueError: 如果缺少必需的键
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"配置缺少必需的键: {missing_keys}")

    def _log_generation_info(self, orbit_type: str, config: Dict[str, Any]) -> None:
        """记录轨道生成信息（可被子类重写）"""
        precision_mode = "高精度" if self.use_high_precision else "简化"
        print(f"[轨道生成器] 生成 {orbit_type} 轨道 ({precision_mode}模式)")
        print(f"[轨道生成器] 使用星历: {'是' if self.ephemeris else '否'}")

    def get_precision_info(self) -> Dict[str, Any]:
        """
        获取生成器的精度信息。
        
        Returns:
            Dict[str, Any]: 包含精度信息的字典
        """
        return {
            "use_high_precision": self.use_high_precision,
            "has_ephemeris": self.ephemeris is not None,
            "ephemeris_type": type(self.ephemeris).__name__ if self.ephemeris else None,
        }
