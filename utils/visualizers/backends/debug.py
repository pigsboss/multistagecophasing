# -*- coding: utf-8 -*-
"""
Debug tool for the **visualizers** module itself.
Prints the scene‑graph hierarchy and, for solar‑system scenes,
creates a 2D scatter plot of celestial body **display positions** using matplotlib.
No 3D rendering or interactive GUI window (matplotlib window is optional).
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from ..base import (
    Scene, SceneNode, Group, ScaledGroup,
    Ellipsoid, Arrow, Trajectory, Camera, Background,
)
from . import Renderer


class DebugRenderer(Renderer):
    """
    Pure debug backend:
    1. Always prints the scene tree.
    2. For solar‑system scenes, shows a top‑down matplotlib scatter plot
       of the (display‑space) positions.
    Does NOT require vedo and does NOT open any 3D window.
    """

    def render(self, scene: Scene, **kwargs) -> None:
        print(f"DEBUG RENDERING: scene = '{scene.name}'  time = {scene.time:.3f} s")
        self._print_node(scene.root, indent=0)

        # Only plot for solar‑system scenes
        if scene.name.lower().startswith("solar"):
            self._plot_ellipsoid_positions(scene)

    # ------------------------------------------------------------------
    # Tree printing (unchanged)
    # ------------------------------------------------------------------
    def _print_node(self, node: SceneNode, indent: int = 0):
        prefix = "  " * indent
        type_name = type(node).__name__
        print(f"{prefix}{type_name}: '{node.name}'")
        if isinstance(node, Ellipsoid):
            print(f"{prefix}  radii = {node.radii}")
        elif isinstance(node, Arrow):
            print(f"{prefix}  dir = {node.direction}, "
                  f"length = {node.length}, color = {node.color}")
        elif isinstance(node, Trajectory):
            print(f"{prefix}  points count = {len(node.points)}")
        elif isinstance(node, ScaledGroup):
            print(f"{prefix}  scale function = "
                  f"{node.scale_function.__class__.__name__}")
        elif isinstance(node, Camera):
            print(f"{prefix}  target = {node.target}, fov = {node.fov}")
        for child in node.children:
            self._print_node(child, indent + 1)

    # ------------------------------------------------------------------
    # Matplotlib scatter plot (no GUI dependency, just display)
    # ------------------------------------------------------------------
    def _plot_ellipsoid_positions(self, scene: Scene):
        """Collect all Ellipsoid nodes directly under the root and plot their
        display‑space (X,Y) positions as a 2D scatter plot."""
        import matplotlib.pyplot as plt

        bodies: List[Tuple[str, np.ndarray, str]] = []
        for child in scene.root.children:
            if not isinstance(child, Ellipsoid):
                continue
            pos = child.transform.position
            colour = getattr(child, 'color', 'white')
            bodies.append((child.name, pos, colour))

        if not bodies:
            print("[DEBUG] No Ellipsoid nodes found under root – nothing to plot.")
            return

        names = [b[0] for b in bodies]
        xs = [b[1][0] for b in bodies]
        ys = [b[1][1] for b in bodies]
        cols = [b[2] for b in bodies]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Solar System Bodies (Scaled Positions)", fontsize=14)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_aspect('equal')

        # Plot Sun larger, others smaller
        for name, x, y, col in zip(names, xs, ys, cols):
            if name.lower() == 'sun':
                ax.scatter(x, y, c=col, s=200, edgecolors='white',
                           linewidth=0.5, label=name, zorder=10)
            else:
                ax.scatter(x, y, c=col, s=80, edgecolors='black',
                           linewidth=0.5, label=name, zorder=5)

        # Labels slightly offset
        for name, x, y in zip(names, xs, ys):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, alpha=0.9)

        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        fig.tight_layout()
        plt.show()
