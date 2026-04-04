"""
重力模型模块 - 统一导出接口

导出通用 CRTBP 类和特定系统类。
提供向后兼容的别名。
"""

from .universal_crtbp import UniversalCRTBP
from .sun_earth_crtbp import SunEarthCRTBP
from .earth_moon_crtbp import EarthMoonCRTBP

# 向后兼容：GravityCRTBP 指向 SunEarthCRTBP
GravityCRTBP = SunEarthCRTBP

__all__ = [
    'UniversalCRTBP',
    'SunEarthCRTBP', 
    'EarthMoonCRTBP',
    'GravityCRTBP',  # 向后兼容别名
]
