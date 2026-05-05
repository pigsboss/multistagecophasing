# -*- coding: utf-8 -*-
"""
MCPC Unified Scene Graph for 3D Visualization.

Provides an abstract scene description independent of any rendering engine.
Supports nonlinear scale mapping for multi-scale views (e.g. Sun-Earth-Moon).
All runtime messages strictly in English per MCPC internationalization policy.
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from mission_sim.core.spacetime.ids import CoordinateFrame


# ---------------------------------------------------------------------------
# Scale Functions (non‑hardcoded, swappable at runtime)
# ---------------------------------------------------------------------------

class ScaleFunction(ABC):
    """Abstract base for any monotonically increasing distance mapping f: d -> d'.

    Must satisfy:
      - f(0) = 0
      - f strictly monotone
    """

    @abstractmethod
    def map_distance(self, d: float) -> float:
        """Map a single radial distance."""
        ...

    def map_vector(self, vec: np.ndarray) -> np.ndarray:
        """Apply radial scaling to a vector (preserves direction)."""
        r = np.linalg.norm(vec)
        if r == 0.0:
            return vec.copy()
        return vec * (self.map_distance(r) / r)


class LinearScale(ScaleFunction):
    """Identity mapping (no compression)."""

    def map_distance(self, d: float) -> float:
        return d


class LogScale(ScaleFunction):
    """Logarithmic compression after a linear threshold.

    For d ≤ linear_threshold: f(d) = d
    For d > linear_threshold: f(d) = d_th + compression * ln(1 + (d-d_th)/compression)
    """

    def __init__(self, linear_threshold: float = 3.8e8, compression: float = 5e8):
        self.linear_threshold = linear_threshold
        self.compression = compression

    def map_distance(self, d: float) -> float:
        abs_d = abs(d)
        if abs_d <= self.linear_threshold:
            return d
        sign = np.sign(d)
        beyond = abs_d - self.linear_threshold
        return sign * (self.linear_threshold +
                       self.compression * np.log1p(beyond / self.compression))


class PiecewiseLinearScale(ScaleFunction):
    """Piecewise linear mapping defined by a sorted list of (real, display) knots."""

    def __init__(self, knots: List[Tuple[float, float]]):
        # knots must be sorted by real distance
        self.knots = sorted(knots, key=lambda x: x[0])
        if len(self.knots) < 2:
            raise ValueError("PiecewiseLinearScale needs at least two knots")

    def map_distance(self, d: float) -> float:
        if d <= self.knots[0][0]:
            return d  # extrapolate identity to left
        if d >= self.knots[-1][0]:
            # linear extrapolation using last segment
            (x0, y0), (x1, y1) = self.knots[-2], self.knots[-1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 1.0
            return y1 + slope * (d - x1)
        # binary search
        lo, hi = 0, len(self.knots) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.knots[mid][0] <= d:
                lo = mid
            else:
                hi = mid
        (x0, y0), (x1, y1) = self.knots[lo], self.knots[hi]
        t = (d - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# Scene Node Hierarchy
# ---------------------------------------------------------------------------

@dataclass
class Transform:
    """Local transform in parent's coordinate system (without nonlinear scaling)."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))    # anisotropic factor


class SceneNode(ABC):
    """Abstract base for all objects in the scene graph."""

    def __init__(self, name: str = ""):
        self.name = name
        self.children: List[SceneNode] = []
        self.parent: Optional[SceneNode] = None
        self.transform = Transform()

    def add_child(self, child: SceneNode) -> SceneNode:
        child.parent = self
        self.children.append(child)
        return child


class Group(SceneNode):
    """A container that applies a common transform to its children."""
    pass


class ScaledGroup(Group):
    """A group that applies a ScaleFunction to the radial distance of its children's positions
    relative to its own origin. This enables non-linear space compression.
    """

    def __init__(self, name: str = "", scale_function: Optional[ScaleFunction] = None):
        super().__init__(name)
        self.scale_function = scale_function or LinearScale()


class Ellipsoid(SceneNode):
    """A celestial body or spacecraft represented as a triaxial ellipsoid.

    radii: (a, b, c) along local X, Y, Z axes.
    """
    def __init__(self, name: str = "", radii: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__(name)
        self.radii = np.array(radii, dtype=float)


class Arrow(SceneNode):
    """A directed arrow (e.g. velocity vector)."""

    def __init__(self, name: str = "", direction: np.ndarray = np.zeros(3),
                 length: float = 1.0, color: str = "red"):
        super().__init__(name)
        self.direction = np.array(direction, dtype=float)
        self.length = length
        self.color = color


class Trajectory(SceneNode):
    """A polyline (orbit trace)."""
    def __init__(self, name: str = "", points: np.ndarray = np.empty((0, 3))):
        super().__init__(name)
        self.points = np.array(points, dtype=float)


class Camera(SceneNode):
    """Viewpoint definition."""
    def __init__(self, name: str = "Camera",
                 target: np.ndarray = np.zeros(3),
                 up: np.ndarray = np.array([0.0, 0.0, 1.0]),
                 fov: float = 45.0):
        super().__init__(name)
        self.target = np.array(target, dtype=float)
        self.up = np.array(up, dtype=float)
        self.fov = fov


class Background(SceneNode):
    """Skybox or distant star field, unaffected by nonlinear scaling."""
    def __init__(self, name: str = "Background"):
        super().__init__(name)


class Scene:
    """Root of the scene graph, holding the entire visual state."""
    def __init__(self, name: str = "MCPC Scene"):
        self.name = name
        self.root = Group("Root")
        self.camera: Optional[Camera] = None
        self.background: Optional[Background] = None
        self.time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialize the scene graph to a dictionary."""
        def _serialize(node: SceneNode) -> Dict[str, Any]:
            entry: Dict[str, Any] = {
                "type": type(node).__name__,
                "name": node.name,
                "transform": {
                    "position": node.transform.position.tolist(),
                    "rotation": node.transform.rotation.tolist(),
                    "scale": node.transform.scale.tolist()
                },
                "children": [_serialize(c) for c in node.children]
            }
            if isinstance(node, Ellipsoid):
                entry["radii"] = node.radii.tolist()
            elif isinstance(node, Arrow):
                entry.update({
                    "direction": node.direction.tolist(),
                    "length": node.length,
                    "color": node.color
                })
            elif isinstance(node, Trajectory):
                entry["points"] = node.points.tolist()
            elif isinstance(node, ScaledGroup):
                entry["scale_function"] = node.scale_function.__class__.__name__
            return entry

        data = {
            "name": self.name,
            "time": self.time,
            "root": _serialize(self.root),
            "camera": _serialize(self.camera) if self.camera else None,
            "background": _serialize(self.background) if self.background else None
        }
        return data


# ---------------------------------------------------------------------------
# Scene Builder – populates a Scene from MCPC / SPICE data
# ---------------------------------------------------------------------------

class SceneBuilder:
    """Creates a Scene using high‑precision ephemeris (SPICE) and scale mapping."""

    def __init__(self, scale_function: Optional[ScaleFunction] = None):
        self.scale_function = scale_function or LogScale(linear_threshold=4e8, compression=5e8)

    def build_solar_system_demo(self, epoch: float,
                                ephemeris_handler) -> Scene:
        """
        Populate a scene with Sun, Earth, Moon using the provided ephemeris handler.

        ephemeris_handler must be an instance of HighPrecisionEphemeris (SPICE mode).
        """
        scene = Scene("Sun-Earth-Moon")
        scene.time = epoch

        # --- Sun (root) ---
        sun_node = Ellipsoid("Sun", radii=(6.957e8, 6.957e8, 6.957e8))
        scene.root.add_child(sun_node)

        # --- Earth heliocentric state ---
        earth_state = ephemeris_handler.get_state(
            "earth", epoch,
            observer_body="sun",
            frame=CoordinateFrame.J2000_ECI
        )
        earth_pos = earth_state[:3]

        # WGS84 ellipsoid
        earth_radii = (6378137.0, 6378137.0, 6356752.3)

        # Earth rotation matrix (J2000 → IAU_EARTH)
        earth_rot = np.eye(3)
        try:
            earth_rot = ephemeris_handler.get_spice_rotation_matrix(
                CoordinateFrame.J2000_ECI,
                CoordinateFrame.IAU_EARTH,
                epoch
            )
        except Exception:
            pass  # keep identity if SPICE does not provide it

        # --- Scaled group for Sun‑Earth distance compression ---
        scaled_se = ScaledGroup("Sun-Earth Scale", scale_function=self.scale_function)
        scene.root.add_child(scaled_se)

        earth_group = Group("Earth")
        scaled_se.add_child(earth_group)
        earth_group.transform.position = earth_pos
        earth_group.transform.rotation = earth_rot

        earth_node = Ellipsoid("Earth", radii=earth_radii)
        earth_group.add_child(earth_node)

        # --- Moon relative to Earth (no extra scaling) ---
        moon_state = ephemeris_handler.get_state(
            "moon", epoch,
            observer_body="earth",
            frame=CoordinateFrame.J2000_ECI
        )
        moon_pos = moon_state[:3]

        # Approximate Moon ellipsoid
        moon_radii = (1737400.0, 1737400.0, 1735972.0)

        try:
            moon_rot = ephemeris_handler.get_moon_libration_matrix(epoch)
        except Exception:
            moon_rot = np.eye(3)

        moon_group = Group("Moon")
        earth_group.add_child(moon_group)
        moon_group.transform.position = moon_pos
        moon_group.transform.rotation = moon_rot

        moon_node = Ellipsoid("Moon", radii=moon_radii)
        moon_group.add_child(moon_node)

        # Compute display position of Earth (after nonlinear scaling)
        earth_display_pos = self.scale_function.map_vector(earth_pos)

        # --- Camera ---
        camera = Camera("Main Camera")
        # Place camera outside the Sun, roughly one‑third of the way to Earth’s display position
        cam_dir = earth_display_pos / np.linalg.norm(earth_display_pos)
        camera.transform.position = cam_dir * np.linalg.norm(earth_display_pos) * 0.33
        camera.target = earth_display_pos  # look at the displayed Earth
        scene.camera = camera
        scene.root.add_child(camera)

        # --- Background ---
        bg = Background()
        scene.background = bg
        scene.root.add_child(bg)

        return scene
