# -*- coding: utf-8 -*-
"""
Debug Backend – prints the scene tree and optionally opens a 3D view via vedo.
"""

from ..base import (
    Scene, SceneNode, Group, ScaledGroup,
    Ellipsoid, Arrow, Trajectory, Camera, Background
)
from . import Renderer
import numpy as np
from typing import Optional


class DebugRenderer(Renderer):
    """A backend that prints the scene hierarchy and may display a simple 3D view using vedo."""

    def __init__(self, use_vedo: bool = False):
        self.use_vedo = use_vedo

    def render(self, scene: Scene, **kwargs) -> None:
        print(f"Rendering scene: {scene.name}  time = {scene.time:.3f} s")
        self._print_node(scene.root, indent=0)
        if self.use_vedo:
            self._vedo_render(scene)

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
    # Vedo rendering – uses homogeneous 4x4 transforms, no .center() call
    # ------------------------------------------------------------------
    def _vedo_render(self, scene: Scene):
        try:
            import vedo
        except ImportError:
            print("vedo is not installed. Install it with: pip install vedo", file=__import__('sys').stderr)
            return

        plotter = vedo.Plotter()

        # Helper to build a 4x4 homogeneous transform (rotation + translation only)
        def _local_to_world_mat(rot, pos):
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = pos
            return mat

        def traverse(node: SceneNode,
                     parent_world_pos: np.ndarray,
                     parent_world_rot: np.ndarray,
                     parent_scale_func: Optional[callable]):
            # Local position of this node relative to parent's origin
            local_pos = node.transform.position.copy()

            # Apply parent's nonlinear scale function if present
            if parent_scale_func is not None:
                scaled_local_pos = parent_scale_func.map_vector(local_pos)
            else:
                scaled_local_pos = local_pos

            # World position of this node's origin
            world_pos = parent_world_rot @ scaled_local_pos + parent_world_pos

            # World rotation of this node
            world_rot = parent_world_rot @ node.transform.rotation

            # Own scale function (for children)
            own_scale_func = None
            if isinstance(node, ScaledGroup):
                own_scale_func = node.scale_function.map_vector

            # Homogeneous matrix that rotates and translates points from this node's
            # local coordinate system into world coordinates (no linear scaling).
            mat = _local_to_world_mat(world_rot, world_pos)

            # ---------- Draw geometry ----------
            if isinstance(node, Ellipsoid):
                # Combine node radii with its local scale factor
                effective_radii = node.radii * node.transform.scale
                S = np.eye(4)
                S[0, 0] = effective_radii[0]
                S[1, 1] = effective_radii[1]
                S[2, 2] = effective_radii[2]

                # Full transform = mat @ S  (first scale, then rotate/translate)
                full_transform = mat @ S

                sph = vedo.Sphere(pos=(0, 0, 0), r=1.0, c='white', res=24)
                sph.apply_transform(full_transform)
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
                    # Transform each point from local (node) coordinates to world
                    transformed = np.array([world_rot @ p + world_pos for p in node.points])
                    line = vedo.Line(transformed)
                    plotter.add(line)

            elif isinstance(node, Camera):
                # The camera is set up later; nothing to draw here
                pass

            elif isinstance(node, Background):
                # Not rendered in debug back‑end
                pass

            # ---------- Recurse children ----------
            for child in node.children:
                traverse(child, world_pos, world_rot, own_scale_func)

        # Start traversal from the root (identity world transform)
        root_scale_func = (scene.root.scale_function.map_vector
                           if isinstance(scene.root, ScaledGroup) else None)
        traverse(scene.root, np.zeros(3), np.eye(3), root_scale_func)

        # Set camera if present
        if scene.camera:
            # For simplicity, use camera local position (no full traversal)
            cam_world_pos = scene.camera.transform.position  # approximate
            plotter.show(interactive=True,
                         viewup=scene.camera.up,
                         pos=cam_world_pos,
                         focal_point=scene.camera.target)
        else:
            plotter.show(interactive=True)
