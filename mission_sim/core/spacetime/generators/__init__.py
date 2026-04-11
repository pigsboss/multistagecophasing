# mission_sim/core/trajectory/generators/__init__.py
"""
时空域标称轨道生成器工厂。
支持高精度星历集成，可生成基于精确天体位置的参考轨道。
"""

from .base import BaseTrajectoryGenerator
from .keplerian import KeplerianGenerator
from .j2_keplerian import J2KeplerianGenerator
from .halo import HaloDifferentialCorrector

__all__ = [
    "BaseTrajectoryGenerator",
    "KeplerianGenerator",
    "J2KeplerianGenerator",
    "HaloDifferentialCorrector",
    "create_generator",
    "create_generator_with_ephemeris",
    "create_high_precision_generator",
]

def create_generator(orbit_type: str, **kwargs) -> BaseTrajectoryGenerator:
    """工厂函数：根据轨道类型创建对应的生成器实例"""
    orbit_type = orbit_type.lower()
    if orbit_type == "keplerian":
        return KeplerianGenerator(**kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(**kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(**kwargs)
    else:
        raise ValueError(f"未知的轨道类型: {orbit_type}")


def create_generator_with_ephemeris(orbit_type: str, ephemeris, **kwargs) -> BaseTrajectoryGenerator:
    """
    工厂函数：创建带有星历的生成器实例。
    
    Args:
        orbit_type: 轨道类型
        ephemeris: 星历实例（HighPrecisionEphemeris 或 SPICEInterface）
        **kwargs: 生成器特定参数
        
    Returns:
        BaseTrajectoryGenerator: 生成器实例
    """
    orbit_type = orbit_type.lower()
    if orbit_type == "keplerian":
        return KeplerianGenerator(ephemeris=ephemeris, **kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(ephemeris=ephemeris, **kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    else:
        raise ValueError(f"未知的轨道类型: {orbit_type}")


def create_high_precision_generator(orbit_type: str, ephemeris, **kwargs) -> BaseTrajectoryGenerator:
    """
    工厂函数：创建高精度星历生成器实例。
    
    Args:
        orbit_type: 轨道类型
        ephemeris: 高精度星历实例（必须支持 get_state 方法）
        **kwargs: 生成器特定参数
        
    Returns:
        BaseTrajectoryGenerator: 生成器实例（启用高精度模式）
    """
    orbit_type = orbit_type.lower()
    
    # 确保传递 use_high_precision=True
    kwargs['use_high_precision'] = True
    
    if orbit_type == "keplerian":
        return KeplerianGenerator(ephemeris=ephemeris, **kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(ephemeris=ephemeris, **kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    else:
        raise ValueError(f"未知的轨道类型: {orbit_type}")
