# mission_sim/core/trajectory/generators/__init__.py
"""标称轨道生成器工厂及导出"""

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
    """
    工厂函数：根据轨道类型创建对应的生成器实例。

    Args:
        orbit_type: 轨道类型，可选 "keplerian", "j2_keplerian", "halo"
        **kwargs: 传递给生成器构造函数的参数（如 mu, 等）

    Returns:
        BaseTrajectoryGenerator: 生成器实例

    Raises:
        ValueError: 未知的 orbit_type
    """
    orbit_type = orbit_type.lower()
    if orbit_type == "keplerian":
        return KeplerianGenerator(**kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(**kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(**kwargs)
    else:
        raise ValueError(f"未知的轨道类型: {orbit_type}。支持的类型: keplerian, j2_keplerian, halo")