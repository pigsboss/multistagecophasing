"""
MCPC Core Law: Spacetime Domain Interface Definition Specification (IDS)
------------------------------------------------------------------------
This file defines universally accepted coordinate frames, command formats, 
and the Level 2 (L2) multi-satellite FormationState container. 
All cross-domain interactions must be based on the structures defined here.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Tuple, Optional


class CoordinateFrame(Enum):
    """Global unified coordinate frame contracts."""
    J2000_ECI = auto()           # J2000 Earth-Centered Inertial / Heliocentric Inertial
    SUN_EARTH_ROTATING = auto()  # Sun-Earth Rotating Frame (CRTBP)
    EARTH_MOON_ROTATING = auto() # Earth-Moon Rotating Frame (CRTBP)
    LVLH = auto()                # Local Vertical Local Horizontal (Relative Frame)
    VVLH = auto()                # Velocity-Velocity Local Horizontal
    
    # 高精度星历支持的坐标系
    ICRF = auto()                # International Celestial Reference Frame
    ITRF93 = auto()              # International Terrestrial Reference Frame 1993
    MOON_PA = auto()             # Moon Principal Axes Frame (from SPICE)
    IAU_EARTH = auto()           # IAU Earth body-fixed frame
    IAU_MOON = auto()            # IAU Moon body-fixed frame
    SUN_EARTH_BARYCENTER = auto()  # Sun-Earth barycenter frame
    ECLIPJ2000 = auto()          # Ecliptic frame at J2000
    IAU_SUN = auto()             # IAU Sun body-fixed frame
    
    # 特殊坐标系
    MARS_CENTERED_INERTIAL = auto()  # Mars-centered inertial
    MARS_FIXED = auto()              # Mars body-fixed frame


class CelestialBody(Enum):
    """标准天体枚举，用于高精度星历查询"""
    SUN = "sun"
    EARTH = "earth"
    MOON = "moon"
    MERCURY = "mercury"
    VENUS = "venus"
    MARS = "mars"
    JUPITER = "jupiter"
    SATURN = "saturn"
    URANUS = "uranus"
    NEPTUNE = "neptune"
    
    # 特殊天体
    EARTH_MOON_BARYCENTER = "earth_moon_barycenter"
    SUN_EARTH_BARYCENTER = "sun_earth_barycenter"
    SOLAR_SYSTEM_BARYCENTER = "solar_system_barycenter"
    
    # 拉格朗日点（虚拟天体）
    EARTH_MOON_L1 = "earth_moon_l1"
    EARTH_MOON_L2 = "earth_moon_l2"
    EARTH_MOON_L3 = "earth_moon_l3"
    EARTH_MOON_L4 = "earth_moon_l4"
    EARTH_MOON_L5 = "earth_moon_l5"
    SUN_EARTH_L1 = "sun_earth_l1"
    SUN_EARTH_L2 = "sun_earth_l2"
    SUN_EARTH_L3 = "sun_earth_l3"
    SUN_EARTH_L4 = "sun_earth_l4"
    SUN_EARTH_L5 = "sun_earth_l5"


@dataclass
class EphemerisConfig:
    """高精度星历配置"""
    mode: str = "analytical"          # 模式: analytical, crtbp, numerical, spice
    spice_kernels_path: Optional[str] = None  # SPICE内核路径
    mission_type: str = "earth_moon"  # 任务类型
    use_light_time: bool = True       # 使用光行差修正
    use_aberration: bool = True       # 使用恒星像差修正
    max_degree: int = 10              # 球谐函数最大阶数
    verbose: bool = False             # 详细输出


@dataclass
class HighPrecisionOrbitConfig:
    """高精度轨道生成配置"""
    # 基础配置
    orbit_type: str                    # 轨道类型: keplerian, j2_keplerian, halo, resonant
    duration: float                    # 轨道持续时间 (秒)
    step_size: float                   # 输出步长 (秒)
    
    # 坐标系配置
    reference_frame: CoordinateFrame = CoordinateFrame.J2000_ECI
    central_body: CelestialBody = CelestialBody.EARTH
    
    # 精度配置
    use_high_precision: bool = False   # 是否使用高精度星历
    ephemeris_config: Optional[EphemerisConfig] = None  # 星历配置
    
    # 轨道参数（根据轨道类型变化）
    orbit_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Telecommand:
    """
    Commands issued by the Cyber brain to the Physics actuators.
    Strictly constrained by coordinate frame and duration.
    Follows Law 3: Impulse Equivalence Principle.
    """
    force_vector: np.ndarray     # 3x1 Desired force vector (N)
    frame: CoordinateFrame       # Frame in which the force is defined (typically LVLH)
    duration_s: float            # Expected duration of the force (s)
    actuator_id: str             # Target actuator ID (e.g., "THR_MAIN_1")

@dataclass
class Telemetry:
    """
    [MCPC UNIVERSAL]
    Standard telemetry report from the Physics domain to the Cyber domain.
    Used for single-satellite absolute state reporting (legacy L1 & universal base).
    """
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    frame: CoordinateFrame

@dataclass
class FormationState:
    """
    L2 Core Data Bus: Multi-satellite Formation State Container.
    Carries the absolute state of the Chief and the relative states of N Deputies.
    """
    timestamp: float
    chief_position: np.ndarray       # Absolute position of Chief (3x1)
    chief_velocity: np.ndarray       # Absolute velocity of Chief (3x1)
    chief_frame: CoordinateFrame     # Chief's absolute frame (J2000 or ROTATING)
    
    # Explicitly maintain Deputy IDs to prevent index-mismatch bugs in multi-star topologies
    deputy_ids: List[str] = field(default_factory=list)
    deputy_relative_positions: List[np.ndarray] = field(default_factory=list)
    deputy_relative_velocities: List[np.ndarray] = field(default_factory=list)
    deputy_frame: CoordinateFrame = CoordinateFrame.LVLH  # Default frame for relative states

    def get_num_deputies(self) -> int:
        """Returns the number of deputies currently in the formation."""
        return len(self.deputy_ids)

    def add_deputy_state(self, deputy_id: str, rel_pos: np.ndarray, rel_vel: np.ndarray) -> None:
        """Mounts the relative state of a deputy spacecraft."""
        self.deputy_ids.append(deputy_id)
        self.deputy_relative_positions.append(np.array(rel_pos, dtype=np.float64))
        self.deputy_relative_velocities.append(np.array(rel_vel, dtype=np.float64))

    def get_deputy_index(self, deputy_id: str) -> int:
        """Safely retrieves the data index for a deputy based on its ID."""
        try:
            return self.deputy_ids.index(deputy_id)
        except ValueError:
            raise KeyError(f"Deputy ID '{deputy_id}' not found in current formation.")

    # --- Serialization methods for HDF5 logging and configuration support ---

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the FormationState to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "chief_position": self.chief_position.tolist(),
            "chief_velocity": self.chief_velocity.tolist(),
            "chief_frame": self.chief_frame.name,
            "deputy_ids": self.deputy_ids,
            "deputy_relative_positions": [p.tolist() for p in self.deputy_relative_positions],
            "deputy_relative_velocities": [v.tolist() for v in self.deputy_relative_velocities],
            "deputy_frame": self.deputy_frame.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormationState':
        """Deserializes a dictionary back into a FormationState instance."""
        return cls(
            timestamp=data["timestamp"],
            chief_position=np.array(data["chief_position"]),
            chief_velocity=np.array(data["chief_velocity"]),
            chief_frame=CoordinateFrame[data["chief_frame"]],
            deputy_ids=data.get("deputy_ids", []),
            deputy_relative_positions=[np.array(p) for p in data.get("deputy_relative_positions", [])],
            deputy_relative_velocities=[np.array(v) for v in data.get("deputy_relative_velocities", [])],
            deputy_frame=CoordinateFrame[data.get("deputy_frame", "LVLH")]
        )
