# mission_sim/core/types.py
from enum import Enum
from dataclasses import dataclass
import numpy as np

class CoordinateFrame(Enum):
    J2000_ECI = "J2000_ECI"
    J2000_SSB = "J2000_SSB"
    SUN_EARTH_ROTATING = "SUN_EARTH_ROTATING"
    EARTH_MOON_ROTATING = "EARTH_MOON_ROTATING"
    BODY_FRAME = "BODY_FRAME"
    # L2 级新增：相对运动坐标系
    LVLH = "LVLH"

@dataclass
class Telecommand:
    """
    全局指令契约
    所有跨模块的控制目标必须封装为此类，强制携带坐标系信息。
    """
    cmd_type: str
    target_state: np.ndarray
    frame: CoordinateFrame
