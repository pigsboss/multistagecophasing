"""
重力模型模块 - 统一导出接口

导出通用 CRTBP 类和特定系统类。
提供向后兼容的别名。
"""

from .universal_crtbp import UniversalCRTBP
from .sun_earth_crtbp import SunEarthCRTBP
from .earth_moon_crtbp import EarthMoonCRTBP

# 向后兼容：将 GravityCRTBP 直接指向 SunEarthCRTBP
# 注意：SunEarthCRTBP 已经提供了完整的接口兼容性
GravityCRTBP = SunEarthCRTBP

__all__ = [
    'UniversalCRTBP',
    'SunEarthCRTBP', 
    'EarthMoonCRTBP',
    'GravityCRTBP',  # 向后兼容别名
]
