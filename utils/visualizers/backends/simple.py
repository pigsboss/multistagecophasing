# -*- coding: utf-8 -*-
"""
Simple Backend – prints the scene tree and optionally opens a 3D view via vedo.
Supports both interactive single‑frame and batch rendering to images.
"""

from ..base import (
    Scene, SceneNode, Group, ScaledGroup,
    Ellipsoid, Arrow, Trajectory, Camera, Background, ScaleFunction
)
from . import Renderer
import numpy as np
import vtk
from typing import Optional, List, Tuple
import sys
from pathlib import Path


class SimpleRenderer(Renderer):
    """A backend that prints the scene hierarchy and may display a simple 3D view using vedo."""

    def __init__(self, use_vedo: bool = False):
        self.use_vedo = use_vedo

    def render(self, scene: Scene, **kwargs) -> None:
        frame_index = kwargs.get("frame_index", 0)
        total_frames = kwargs.get("total_frames", 1)
        output_dir = kwargs.get("output_dir", None)

        print(f"Rendering scene: {scene.name}  time = {scene.time:.3f} s")
        self._print_node(scene.root, indent=0)

        if not self.use_vedo:
            return

        if output_dir:
            self._vedo_render_frame(scene, frame_index, output_dir)
        else:
            if total_frames > 1:
                print("Multi‑frame animation requires --output to save images. Skipping vedo.",
                      file=sys.stderr)
            else:
                self._vedo_render_interactive(scene)

    # ------------------------------------------------------------------
    # Tree printing
    # ------------------------------------------------------------------
    def _print_node(self, node: SceneNode, indent: int = 0):
        prefix = "  " * indent
        type_name = type(node).__name__
        print(f"{prefix}{type_name}: '{node.name}'")
        if isinstance(node, Ellipsoid):
            print(f"{prefix}  radii = {node.radii}")
        elif isinstance(node, Arrow):
            print(f"{prefix}  dir = {node.direction}, length = {node.length}, color = {node.color}")
        elif isinstance(node, Trajectory):
            print(f"{prefix}  points count = {len(node.points)}")
        elif isinstance(node, ScaledGroup):
            print(f"{prefix}  scale function = {node.scale_function.__class__.__name__}")
        elif isinstance(node, Camera):
            print(f"{prefix}  target = {node.target}, fov = {node.fov}")
        for child in node.children:
            self._print_node(child, indent + 1)

    # ------------------------------------------------------------------
    # Node collection (identical to debug.py logic)
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_ellipsoid_nodes(node: SceneNode, parent_pos: np.ndarray,
                                 out: List[Tuple[str, np.ndarray, str]]):
        """Depth‑first traversal that accumulates world positions (same as debug.py)."""
        world_pos = parent_pos + node.transform.position

        if isinstance(node, Ellipsoid):
            colour = getattr(node, 'color', 'white')
            out.append((node.name, world_pos, colour))
        for child in node.children:
            SimpleRenderer._collect_ellipsoid_nodes(child, world_pos, out)

    # ------------------------------------------------------------------
    # Vedo – interactive single frame
    # ------------------------------------------------------------------
    def _vedo_render_interactive(self, scene: Scene):
        import vedo
        plotter = vedo.Plotter(bg="black")

        # Collect bodies exactly like debug.py
        bodies: List[Tuple[str, np.ndarray, str]] = []
        self._collect_ellipsoid_nodes(scene.root, np.zeros(3), bodies)

        if not bodies:
            print("[SIMPLE] No Ellipsoid nodes found.", file=sys.stderr)
            plotter.show(interactive=True)
            return

        # Determine unit and limits like debug.py
        scene_lower = scene.name.lower()
        if "solar" in scene_lower:
            unit_factor = 149597870700.0   # 1 AU in metres
            view_limit = 15.0              # ±15 AU
        else:
            unit_factor = 3.844e8          # 1 LD in metres
            view_limit = 5.0               # ±5 LD

        # Create vedo spheres at display positions (full 3D)
        for name, pos, color in bodies:
            x = pos[0] / unit_factor
            y = pos[1] / unit_factor
            z = pos[2] / unit_factor
            # Size for visibility (not to real scale)
            radius = 0.5 if name.lower() == 'sun' else 0.2
            sph = vedo.Sphere(pos=(x, y, z), r=radius, c=color, res=24)
            plotter.add(sph)
            # Label at same depth as body
            lbl = vedo.Text3D(name, pos=(x+0.3, y+0.3, z), s=0.3, c='white')
            plotter.add(lbl)

        # Camera: top-down orthographic, strictly matching debug.py's xlim/ylim
        plotter.camera.SetParallelProjection(True)
        plotter.camera.SetParallelScale(view_limit)   # shows -15..15 or -5..5
        plotter.camera.SetPosition(0.0, 0.0, 10.0)    # above XY plane
        plotter.camera.SetFocalPoint(0.0, 0.0, 0.0)   # look at origin (Sun)
        plotter.camera.SetViewUp(0.0, 1.0, 0.0)
        plotter.camera.SetClippingRange(0.1, 100.0)
        plotter.size = (800, 800)
        plotter.show(interactive=True, resetcam=False)

    # ------------------------------------------------------------------
    # Vedo – save a single frame (non‑interactive)
    # ------------------------------------------------------------------
    def _vedo_render_frame(self, scene: Scene, frame_index: int, output_dir: str):
        import vedo
        from pathlib import Path
        plotter = vedo.Plotter(offscreen=True, bg="black")

        # Collect bodies exactly like debug.py
        bodies: List[Tuple[str, np.ndarray, str]] = []
        self._collect_ellipsoid_nodes(scene.root, np.zeros(3), bodies)

        if not bodies:
            print("[SIMPLE] No Ellipsoid nodes found.", file=sys.stderr)
            plotter.close()
            return

        # Determine unit and limits like debug.py
        scene_lower = scene.name.lower()
        if "solar" in scene_lower:
            unit_factor = 149597870700.0
            view_limit = 15.0
        else:
            unit_factor = 3.844e8
            view_limit = 5.0

        # Create vedo spheres at display positions (full 3D)
        for name, pos, color in bodies:
            x = pos[0] / unit_factor
            y = pos[1] / unit_factor
            z = pos[2] / unit_factor
            radius = 0.5 if name.lower() == 'sun' else 0.2
            sph = vedo.Sphere(pos=(x, y, z), r=radius, c=color, res=24)
            plotter.add(sph)
            lbl = vedo.Text3D(name, pos=(x+0.3, y+0.3, z), s=0.3, c='white')
            plotter.add(lbl)

        # Camera setup identical to interactive mode
        plotter.camera.SetParallelProjection(True)
        plotter.camera.SetParallelScale(view_limit)
        plotter.camera.SetPosition(0.0, 0.0, 10.0)
        plotter.camera.SetFocalPoint(0.0, 0.0, 0.0)
        plotter.camera.SetViewUp(0.0, 1.0, 0.0)
        plotter.camera.SetClippingRange(0.1, 100.0)
        plotter.size = (800, 800)

        filename = Path(output_dir) / f"frame_{frame_index:04d}.png"
        plotter.show(interactive=False, resetcam=False)
        plotter.screenshot(str(filename))
        print(f"Saved {filename}", file=sys.stderr)
        plotter.close()

    # ------------------------------------------------------------------
    # Common drawing routine used by both interactive and frame modes
    # ------------------------------------------------------------------
    def _draw_scene(self, scene: Scene, plotter):
        import vedo

        def _local_to_world_mat(rot, pos):
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = pos
            return mat

        def traverse(node: SceneNode,
                     parent_world_pos: np.ndarray,
                     parent_world_rot: np.ndarray,
                     parent_scale_function: Optional[ScaleFunction]):
            local_pos = node.transform.position.copy()

            # Apply parent's nonlinear scale function if present
            if parent_scale_function is not None:
                scaled_local_pos = parent_scale_function.map_vector(local_pos)
            else:
                scaled_local_pos = local_pos

            world_pos = parent_world_rot @ scaled_local_pos + parent_world_pos
            world_rot = parent_world_rot @ node.transform.rotation

            own_scale_function = None
            if isinstance(node, ScaledGroup):
                own_scale_function = node.scale_function

            mat = _local_to_world_mat(world_rot, world_pos)

            # ---------- Draw geometry ----------
            if isinstance(node, Ellipsoid):
                # Use colour attribute if present, fall back to white
                colour = getattr(node, 'color', 'white')

                effective_radii = node.radii * node.transform.scale
                S = np.eye(4)
                S[0, 0] = effective_radii[0]
                S[1, 1] = effective_radii[1]
                S[2, 2] = effective_radii[2]

                full_transform = mat @ S

                sph = vedo.Sphere(pos=(0, 0, 0), r=1.0, c=colour, res=24)
                vtk_mat = vtk.vtkMatrix4x4()
                for i in range(4):
                    for j in range(4):
                        vtk_mat.SetElement(i, j, full_transform[i, j])
                sph.apply_transform(vtk_mat)
                plotter.add(sph)

            elif isinstance(node, Arrow):
                start = world_pos
                direction = node.direction
                norm = np.linalg.norm(direction)
                if norm > 0:
                    end = start + direction / norm * node.length
                else:
                    end = start
                arrow = vedo.Arrow(start, end, c=node.color)
                plotter.add(arrow)

            elif isinstance(node, Trajectory):
                if len(node.points) > 1:
                    transformed = np.array([world_rot @ p + world_pos for p in node.points])
                    # Use trajectory's own color, default to white
                    line_color = getattr(node, 'color', 'white')
                    line = vedo.Line(transformed, c=line_color)
                    plotter.add(line)

            # Recurse
            for child in node.children:
                traverse(child, world_pos, world_rot, own_scale_function)

        # Start traversal from root
        root_scale_function = scene.root.scale_function if isinstance(scene.root, ScaledGroup) else None
        traverse(scene.root, np.zeros(3), np.eye(3), root_scale_function)
