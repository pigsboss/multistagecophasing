# mission_sim/core/types.py
"""
MCPC 框架全局类型与契约定义 (Level 1 - Level 5)
职责：提供全系统统一的枚举与数据结构，确保物理域与信息域之间的坐标系与数据格式绝对一致。
"""

from enum import Enum
from dataclasses import dataclass
import numpy as np

class CoordinateFrame(Enum):
    """
    空间坐标系定义字典 (全域跨界支持)
    涵盖从近地 LEO/GEO 到深空平动点、尾随轨道的所需计算基准。
    """
    # === 惯性系 (Inertial Frames) ===
    J2000_ECI = "J2000_Earth_Centered_Inertial"      # 地心惯性系 (LEO/GEO 绝对动力学计算基准)
    J2000_SSB = "J2000_Solar_System_Barycenter"      # 太阳系质心惯性系 (Deep Space/Trailing 轨道基准)
    
    # === 旋转系 (Rotating Frames) ===
    WGS84_ECEF = "WGS84_Earth_Centered_Earth_Fixed"  # 地心地固系 (LEO/GEO 测控站位置与可视弧段计算)
    SUN_EARTH_ROTATING = "Sun_Earth_Rotating"        # 日地旋转系 (日地 L1/L2 Halo 轨道基准)
    EARTH_MOON_ROTATING = "Earth_Moon_Rotating"      # 地月旋转系 (NRHO/地月转移轨道基准)
    
    # === 相对与本体坐标系 (Relative & Body Frames) ===
    LVLH = "Local_Vertical_Local_Horizontal"         # 局部轨道直角坐标系 (L2级编队协同与相对测量基准)
    BODY_FRAME = "Spacecraft_Body_Frame"             # 航天器本体坐标系 (L3级以上姿态与敏感器测量基准)


@dataclass
class Telecommand:
    """
    全局遥控指令契约 (Telecommand Data Contract)
    约束：所有跨模块（尤其是从地面站到 GNC，或编队网络间）的控制目标必须封装为此类。
    """
    cmd_type: str                  # 指令类型，例如: "ORBIT_MAINTENANCE", "FORMATION_KEEPING"
    target_state: np.ndarray       # 目标状态向量 (通常为 6x1 的位置速度向量)
    frame: CoordinateFrame         # 目标状态所处的坐标系 (强校验字段，防止坐标系混用)
    execution_epoch: float = 0.0   # 指令生效的历元时间 (预留字段：支持绝对时间的程控指令注入)
    
    def __post_init__(self):
        """数据格式的底层防呆校验"""
        if not isinstance(self.target_state, np.ndarray):
            self.target_state = np.array(self.target_state, dtype=np.float64)
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError(f"指令的坐标系字段必须是 CoordinateFrame 枚举类型，当前为: {type(self.frame)}")


class CelestialBody(Enum):
    """太阳系天体枚举"""
    SUN = "sun"
    EARTH = "earth"
    MOON = "moon"
    MARS = "mars"
    VENUS = "venus"
    MERCURY = "mercury"
    JUPITER = "jupiter"
    SATURN = "saturn"
    URANUS = "uranus"
    NEPTUNE = "neptune"
