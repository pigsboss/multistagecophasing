# mission_sim/core/trajectory/generators/__init__.py
"""
时空域标称轨道生成器工厂。
支持高精度星历集成，可生成基于精确天体位置的参考轨道。
"""

from .base import BaseTrajectoryGenerator
from .keplerian import KeplerianGenerator
from .j2_keplerian import J2KeplerianGenerator
from .halo import HaloDifferentialCorrector
from .crtbp import CRTBPOrbitGenerator, CRTBPOrbitType, SymmetryType, CRTBPOrbitConfig, create_crtbp_generator, generate_family

__all__ = [
    "BaseTrajectoryGenerator",
    "KeplerianGenerator",
    "J2KeplerianGenerator",
    "HaloDifferentialCorrector",
    "CRTBPOrbitGenerator",
    "CRTBPOrbitType",
    "SymmetryType",
    "CRTBPOrbitConfig",
    "create_crtbp_generator",
    "generate_family",
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
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        # 对于 CRTBP 轨道类型，使用新的 CRTBPOrbitGenerator
        # 需要将字符串转换为 CRTBPOrbitType 枚举
        if orbit_type == "crtbp":
            # 默认使用 HALO
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        # 从 kwargs 中提取系统类型，默认为 sun_earth
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            **kwargs
        )
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
    
    # 开普勒轨道生成器不需要高精度星历
    if orbit_type in ["keplerian", "j2_keplerian"]:
        import warnings
        warnings.warn(
            f"{orbit_type} 生成器不需要高精度星历，将忽略 ephemeris 参数",
            UserWarning
        )
        # 移除 ephemeris 参数，避免传递给生成器
        if 'ephemeris' in kwargs:
            del kwargs['ephemeris']
    
    if orbit_type == "keplerian":
        return KeplerianGenerator(**kwargs)
    elif orbit_type == "j2_keplerian":
        return J2KeplerianGenerator(**kwargs)
    elif orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        # 对于 CRTBP 轨道类型，使用新的 CRTBPOrbitGenerator
        if orbit_type == "crtbp":
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            ephemeris=ephemeris,
            use_high_precision=True,
            **kwargs
        )
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
    
    # 开普勒轨道不支持高精度模式
    if orbit_type in ["keplerian", "j2_keplerian"]:
        raise ValueError(
            f"{orbit_type} 生成器不支持高精度模式。"
            "请使用 create_generator() 或考虑使用数值积分生成器。"
        )
    
    # 确保传递 use_high_precision=True
    kwargs['use_high_precision'] = True
    
    if orbit_type == "halo":
        return HaloDifferentialCorrector(ephemeris=ephemeris, **kwargs)
    elif orbit_type in ["crtbp", "halo", "dro", "lyapunov", "vertical", "resonant", "lissajous", "leader_follower"]:
        if orbit_type == "crtbp":
            orbit_type_enum = CRTBPOrbitType.HALO
        else:
            orbit_type_enum = CRTBPOrbitType[orbit_type.upper()]
        
        system_type = kwargs.pop("system_type", "sun_earth")
        return CRTBPOrbitGenerator(
            system_type=system_type,
            orbit_type=orbit_type_enum,
            ephemeris=ephemeris,
            use_high_precision=True,
            **kwargs
        )
    else:
        raise ValueError(f"未知的轨道类型: {orbit_type}")
