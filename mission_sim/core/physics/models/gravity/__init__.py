"""
重力模型模块 - 统一导出接口

导出通用 CRTBP 类和特定系统类。
提供向后兼容的别名。
"""

from .universal_crtbp import UniversalCRTBP
from .sun_earth_crtbp import SunEarthCRTBP
from .earth_moon_crtbp import EarthMoonCRTBP

# 向后兼容：导入适配器类
try:
    from ..gravity_crtbp import GravityCRTBP as LegacyGravityCRTBP
    GravityCRTBP = LegacyGravityCRTBP
except ImportError:
    # 如果找不到适配器类，则回退到 SunEarthCRTBP（可能会引发警告）
    import warnings
    warnings.warn(
        "gravity_crtbp.py not found, falling back to SunEarthCRTBP for GravityCRTBP",
        ImportWarning,
        stacklevel=2
    )
    GravityCRTBP = SunEarthCRTBP

__all__ = [
    'UniversalCRTBP',
    'SunEarthCRTBP', 
    'EarthMoonCRTBP',
    'GravityCRTBP',  # 向后兼容适配器类
]
