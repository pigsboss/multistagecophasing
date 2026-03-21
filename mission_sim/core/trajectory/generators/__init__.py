# mission_sim/core/trajectory/generators/__init__.py
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