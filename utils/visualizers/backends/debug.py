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
    # Vedo rendering
    # ------------------------------------------------------------------
    def _vedo_render(self, scene: Scene):
        try:
            import vedo
        except ImportError:
            print("vedo is not installed. Install it with: pip install vedo", file=__import__('sys').stderr)
            return

        plotter = vedo.Plotter()

        # Traverse the scene tree and compute world transforms
        def traverse(node: SceneNode,
                     parent_world_pos: np.ndarray,
                     parent_world_rot: np.ndarray,
                     parent_scale_func: Optional[callable]):
            # Current node's local position relative to parent's origin
            local_pos = node.transform.position.copy()

            # Apply parent's scale function if present
            if parent_scale_func is not None:
                scaled_local_pos = parent_scale_func.map_vector(local_pos)
            else:
                scaled_local_pos = local_pos

            # World position: parent_world_rot @ scaled_local + parent_world_pos
            world_pos = parent_world_rot @ scaled_local_pos + parent_world_pos

            # World rotation: parent_world_rot @ node_local_rotation
            world_rot = parent_world_rot @ node.transform.rotation

            # Determine scale function to pass to children
            own_scale_func = None
            if isinstance(node, ScaledGroup):
                own_scale_func = node.scale_function.map_vector

            # Draw geometry
            if isinstance(node, Ellipsoid):
                # Create a sphere and scale it anisotropically
                vis_radii = node.radii * node.transform.scale
                sph = vedo.Sphere(pos=world_pos, r=1.0, c='white', res=24)
                sph.scale(vis_radii, origin=sph.center())
                # Rotate: apply world rotation matrix (vedo works with orientation vectors)
                # Convert rotation matrix to axis-angle or Euler
                # Use vedo's apply_transform
                # For simplicity use a homogeneous matrix
                transform_mat = np.eye(4)
                transform_mat[:3, :3] = world_rot
                transform_mat[:3, 3] = world_pos
                sph.apply_transform(transform_mat)
                plotter.add(sph)

            elif isinstance(node, Arrow):
                # Draw as a thin cylinder + cone
                start = world_pos
                end = world_pos + node.direction / np.linalg.norm(node.direction) * node.length
                arrow = vedo.Arrow(start, end, c=node.color)
                plotter.add(arrow)

            elif isinstance(node, Trajectory):
                if len(node.points) > 1:
                    # Points are in local coordinates of the node; we need to transform them to world
                    transformed = np.array([parent_world_rot @ p + world_pos for p in node.points])
                    line = vedo.Line(transformed)
                    plotter.add(line)

            elif isinstance(node, Camera):
                # For vedo, we may set camera interactively; skip for now
                pass

            elif isinstance(node, Background):
                # Not rendering background in debug
                pass

            # Recurse
            for child in node.children:
                traverse(child, world_pos, world_rot, own_scale_func)

        # Start traversal from root (identity world)
        root_scale_func = (scene.root.scale_function.map_vector
                           if isinstance(scene.root, ScaledGroup) else None)
        traverse(scene.root, np.zeros(3), np.eye(3), root_scale_func)

        # Set camera if present (optional)
        if scene.camera:
            # Compute world position of camera using the same traversal; but for now,
            # we manually set a reasonable viewing point if the camera is not in the graph
            cam_world_pos = scene.camera.transform.position  # simplified
            plotter.show(interactive=True, viewup=scene.camera.up,
                         pos=cam_world_pos, focal_point=scene.camera.target)
        else:
            plotter.show(interactive=True)
