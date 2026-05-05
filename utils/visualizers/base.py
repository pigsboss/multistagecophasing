# -*- coding: utf-8 -*-
"""
MCPC Unified Scene Graph for 3D Visualization.

Provides an abstract scene description independent of any rendering engine.
Supports nonlinear scale mapping for multi-scale views (e.g. Sun-Earth-Moon).
All runtime messages strictly in English per MCPC internationalization policy.
"""

from __future__ import annotations
import numpy as np
import warnings
import math
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
    color: colour name for rendering (e.g. 'yellow', 'blue', 'gray').
    """
    def __init__(self, name: str = "", radii: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 color: str = "white"):
        super().__init__(name)
        self.radii = np.array(radii, dtype=float)
        self.color = color


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
                entry["color"] = node.color
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

    # Predefined planetary data (radii in meters, colour names)
    _PLANET_DATA = {
        "mercury": {"radii": (2439.4e3, 2439.4e3, 2439.4e3), "color": "grey"},
        "venus":   {"radii": (6052e3,   6052e3,   6052e3),   "color": "orange"},
        "earth":   {"radii": (6378.137e3, 6378.137e3, 6356.752e3), "color": "blue"},
        "mars":    {"radii": (3396.2e3, 3396.2e3, 3376.2e3), "color": "red"},
        "jupiter": {"radii": (71492e3,  71492e3,  66854e3),  "color": "brown"},
        "saturn":  {"radii": (60268e3,  60268e3,  54364e3),  "color": "gold"},
        "uranus":  {"radii": (25559e3,  25559e3,  24973e3),  "color": "lightblue"},
        "neptune": {"radii": (24764e3,  24764e3,  24341e3),  "color": "darkblue"},
    }

    def __init__(self, scale_function: Optional[ScaleFunction] = None):
        self.scale_function = scale_function or LogScale(linear_threshold=4e8, compression=5e8)
        # Cache the computed camera height across calls to avoid frame‑to‑frame jitter
        self._cached_camera_height: Optional[float] = None

    @staticmethod
    def _to_ecliptic(real_pos: np.ndarray) -> Tuple[float, float, float]:
        """Convert J2000 equatorial position to ecliptic longitude (deg), latitude (deg), distance (m)."""
        # Rotation matrix: equatorial -> ecliptic (rotation about X by obliquity)
        obliquity = math.radians(23.4392911)   # J2000 mean obliquity
        R = np.array([
            [1, 0, 0],
            [0, math.cos(obliquity), math.sin(obliquity)],
            [0, -math.sin(obliquity), math.cos(obliquity)]
        ])
        ecl = R @ real_pos
        x, y, z = ecl
        distance = math.hypot(z, math.hypot(x, y))
        lng = math.degrees(math.atan2(y, x)) % 360.0
        lat = math.degrees(math.asin(z / distance))
        return lng, lat, distance

    def build_solar_system(self, epoch: float,
                           ephemeris_handler) -> Scene:
        """
        Build a scene containing the Sun and all eight planets.

        Parameters:
            epoch: Ephemeris time (seconds past J2000).
            ephemeris_handler: An instance of HighPrecisionEphemeris (SPICE mode).
        """
        import math
        AU = 149597870700.0        # metres
        R_EARTH = 6378137.0        # metres

        scene = Scene("Solar-System")
        scene.time = epoch

        # Camera parameters (will be set later, but we compute them now for scaling)
        fov_deg = 60.0
        fov_rad = math.radians(fov_deg)
        # assume a 720p vertical resolution for conservative pixel sizing
        target_vertical_px = 720

        # Desired minimum pixel radii
        sun_pixel_radius = 50      # sun appears ~100px diameter
        planet_pixel_radius = 15   # planets appear ~30px diameter

        # --- Sun at origin ---
        sun_radii = (6.957e8, 6.957e8, 6.957e8)
        sun_node = Ellipsoid("Sun", radii=sun_radii, color="yellow")

        planet_data = {}  # name -> (real_pos, display_pos, info_dict)

        # Diagnostic: which scale function is active?
        print(f"[DIAG] ScaleFunction: {self.scale_function.__class__.__name__}")
        if hasattr(self.scale_function, 'linear_threshold'):
            print(f"[DIAG]   linear_threshold = {self.scale_function.linear_threshold:.3e}")
        if hasattr(self.scale_function, 'compression'):
            print(f"[DIAG]   compression       = {self.scale_function.compression:.3e}")
        print(f"[DIAG]   example (1e12 m) -> {self.scale_function.map_distance(1e12):.3e} m")

        # First pass: collect display positions to determine camera height
        for name, data in self._PLANET_DATA.items():
            try:
                state = ephemeris_handler.get_state(
                    name, epoch,
                    observer_body="sun",
                    frame=CoordinateFrame.J2000_ECI
                )
                real_pos = state[:3]

                # Check for suspiciously small position (likely analytic model failure)
                if np.linalg.norm(real_pos) < 1e8:
                    fallback_ok = False
                    if hasattr(ephemeris_handler, '_spice_if') and ephemeris_handler._spice_if is not None:
                        try:
                            state = ephemeris_handler._spice_if.get_state(
                                name, epoch, observer="sun",
                                frame=CoordinateFrame.J2000_ECI
                            )
                            real_pos = state[:3]
                            fallback_ok = True
                        except Exception:
                            pass
                    if not fallback_ok:
                        warnings.warn(f"Position data invalid for {name}, skipping.")
                        continue

                # Apply scale mapping
                display_pos = self.scale_function.map_vector(real_pos)

                # Ecliptic coordinates
                ecl_lng, ecl_lat, ecl_dist = self._to_ecliptic(real_pos)
                dist_au = np.linalg.norm(real_pos) / AU
                display_dist_au = np.linalg.norm(display_pos) / AU
                print(f"[DIAG] {name.capitalize():8s} "
                      f"ecl: l={ecl_lng:7.2f}° b={ecl_lat:+7.2f}° "
                      f"real={real_pos!r} "
                      f"display={display_pos!r} "
                      f"dist_au={dist_au:.4f} display_au={display_dist_au:.4f}")

                planet_data[name] = (real_pos, display_pos, data)
            except Exception as exc:
                warnings.warn(f"Failed to get state for {name}, skipping: {exc}")
                continue

        if not planet_data:
            warnings.warn("No valid planetary ephemeris data; scene will only contain the Sun.")

        # Compute camera placement (cache after first call)
        if planet_data:
            max_display_dist = max(np.linalg.norm(d[1]) for d in planet_data.values())
        else:
            max_display_dist = 0.0
        if self._cached_camera_height is None:
            self._cached_camera_height = max_display_dist * 3.0 if max_display_dist > 0 else 1.0
        camera_height = self._cached_camera_height
        camera_pos = np.array([0.0, 0.0, camera_height])

        # Distance from camera to Sun (in compressed space)
        sun_dist = camera_height   # Sun is at origin
        # World size per pixel at that distance
        sun_world_per_px = (2.0 * sun_dist * math.tan(fov_rad / 2.0)) / target_vertical_px
        sun_min_world_radius = sun_pixel_radius * sun_world_per_px
        sun_scale = sun_min_world_radius / sun_radii[0]   # all radii equal
        sun_node.transform.scale = np.array([sun_scale] * 3)
        scene.root.add_child(sun_node)

        # Second pass: create planet nodes directly under root (bypass Group wrapper)
        # DIAGNOSTIC: if this makes all 8 planets appear, vedo ignores Group.transform
        for name, (real_pos, display_pos, data) in planet_data.items():
            dist = np.linalg.norm(display_pos - camera_pos)
            world_per_px = (2.0 * dist * math.tan(fov_rad / 2.0)) / target_vertical_px
            min_world_radius = planet_pixel_radius * world_per_px

            real_radius = data["radii"][0]   # use equatorial radius as representative
            scale_factor = max(min_world_radius / real_radius, 1.0)  # never shrink below natural size

            ellip = Ellipsoid(name.capitalize(), radii=data["radii"], color=data["color"])
            # Position directly on the leaf node, not on an intermediate Group
            ellip.transform.position = display_pos
            ellip.transform.scale = np.array([scale_factor] * 3)
            scene.root.add_child(ellip)

        # --- Camera ---
        camera = Camera("Top-Down Camera")
        camera.transform.position = camera_pos
        camera.target = np.zeros(3)
        camera.up = np.array([0.0, 1.0, 0.0])
        camera.fov = fov_deg
        scene.camera = camera
        # DO NOT attach camera to root – it is metadata, not geometry

        # --- Background ---
        bg = Background()
        scene.background = bg
        # DO NOT attach background to root – it is metadata, not geometry

        # DIAGNOSTIC: print world positions so we can verify without relying on renderer
        sun_radius_R = sun_radii[0] / R_EARTH
        sun_world_radius_R = sun_radii[0] * sun_node.transform.scale[0] / R_EARTH
        print(f"[DIAG] Sun at {sun_node.transform.position}, "
              f"scale={sun_node.transform.scale[0]:.3e}, "
              f"R={sun_radius_R:.1f} Re, world_R={sun_world_radius_R:.3f} Re")
        for child in scene.root.children:
            if isinstance(child, Ellipsoid) and child.name != "Sun":
                pos = child.transform.position
                dist_to_cam = np.linalg.norm(pos - camera_pos)
                world_per_px = (2.0 * dist_to_cam * math.tan(fov_rad / 2.0)) / target_vertical_px
                w_radius = child.radii[0] * child.transform.scale[0]
                w_radius_R = w_radius / R_EARTH
                print(f"[DIAG] {child.name:8s} pos={pos!r} "
                      f"dist_cam={dist_to_cam:.3e} m "
                      f"world_radius={w_radius:.3e} m ({w_radius_R:.3f} Re) "
                      f"scale={child.transform.scale[0]:.3e}")
        print(f"[DIAG] Camera target={camera.target} pos={camera.transform.position}")

        return scene

    def build_solar_system_demo(self, epoch: float,
                                ephemeris_handler) -> Scene:
        """
        Populate a scene with Sun, Earth, Moon using the provided ephemeris handler.

        ephemeris_handler must be an instance of HighPrecisionEphemeris (SPICE mode).
        """
        import math
        LD = 3.844e8               # metres
        R_EARTH = 6378137.0        # metres

        scene = Scene("Sun-Earth-Moon")
        scene.time = epoch

        # --- Sun (root) ---
        sun_node = Ellipsoid("Sun", radii=(6.957e8, 6.957e8, 6.957e8),
                             color="yellow")
        sun_node.transform.scale = np.array([0.05, 0.05, 0.05])   # shrink for debug
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

        # Compute display position of Earth (after nonlinear scaling)
        earth_display_pos = self.scale_function.map_vector(earth_pos)

        # Earth group placed at display position under root (no scaling)
        earth_group = Group("Earth")
        scene.root.add_child(earth_group)
        earth_group.transform.position = earth_display_pos  # use display coordinates
        earth_group.transform.rotation = earth_rot

        earth_node = Ellipsoid("Earth", radii=earth_radii, color="blue")
        earth_node.transform.scale = np.array([10.0, 10.0, 10.0])
        earth_group.add_child(earth_node)

        # --- Moon: compute heliocentric display position = Earth_display + Moon_rel ---
        moon_state = ephemeris_handler.get_state(
            "moon", epoch,
            observer_body="earth",
            frame=CoordinateFrame.J2000_ECI
        )
        moon_rel_pos = moon_state[:3]
        moon_display_pos = earth_display_pos + moon_rel_pos   # add without scaling

        moon_radii = (1737400.0, 1737400.0, 1735972.0)
        try:
            moon_rot = ephemeris_handler.get_moon_libration_matrix(epoch)
        except Exception:
            moon_rot = np.eye(3)

        moon_group = Group("Moon")
        scene.root.add_child(moon_group)          # child of root, not of Earth
        moon_group.transform.position = moon_display_pos
        moon_group.transform.rotation = moon_rot

        moon_node = Ellipsoid("Moon", radii=moon_radii, color="gray")
        moon_node.transform.scale = np.array([10.0, 10.0, 10.0])
        moon_group.add_child(moon_node)

        # === Diagnostic for Earth‑Moon scene ===
        earth_real_dist = np.linalg.norm(earth_pos)
        earth_display_dist = np.linalg.norm(earth_display_pos)
        moon_real_pos_hc = earth_pos + moon_rel_pos
        moon_real_dist = np.linalg.norm(moon_real_pos_hc)
        moon_display_dist = np.linalg.norm(moon_display_pos)
        print(f"[DIAG DEMO] Sun        R={sun_radii[0]/R_EARTH:.1f} Re, "
              f"world_R={sun_radii[0]*sun_node.transform.scale[0]/R_EARTH:.3f} Re")
        print(f"[DIAG DEMO] Earth      real_dist={earth_real_dist/LD:.4f} LD, "
              f"display_dist={earth_display_dist/LD:.4f} LD, "
              f"R={earth_radii[0]/R_EARTH:.1f} Re, "
              f"world_R={earth_radii[0]*earth_node.transform.scale[0]/R_EARTH:.3f} Re")
        print(f"[DIAG DEMO] Moon       real_dist={moon_real_dist/LD:.4f} LD, "
              f"display_dist={moon_display_dist/LD:.4f} LD, "
              f"R={moon_radii[0]/R_EARTH:.1f} Re, "
              f"world_R={moon_radii[0]*moon_node.transform.scale[0]/R_EARTH:.3f} Re")

        # Camera – look at Earth's display position (Moon is now offset visibly)
        camera = Camera("Main Camera")
        sun_center = np.zeros(3)
        mid_point = (sun_center + earth_display_pos) / 2.0
        cam_pos = mid_point + np.array([0.0, 0.0, 1e9])
        camera.transform.position = cam_pos
        camera.target = earth_display_pos
        camera.up = np.array([0.0, 0.0, 1.0])
        scene.camera = camera
        scene.root.add_child(camera)

        # --- Background ---
        bg = Background()
        scene.background = bg
        scene.root.add_child(bg)

        return scene
