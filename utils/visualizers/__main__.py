# -*- coding: utf-8 -*-
"""
MCPC Visualizer CLI

Usage:
    python -m utils.visualizers --time "2026-04-10T12:00:00" --spice-kernels ./spice_kernels [--vedo]
"""

import argparse
import sys
from pathlib import Path
import os

# Ensure the project root is importable (in case running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualizers.base import SceneBuilder, LogScale
from utils.visualizers.backends.debug import DebugRenderer

# Import MCPC high precision ephemeris
try:
    from mission_sim.core.spacetime.ephemeris.high_precision import (
        HighPrecisionEphemeris, EphemerisMode, EphemerisConfig
    )
except ImportError as e:
    print("Fatal: could not import MCPC high precision ephemeris module.", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MCPC 3D Scene Visualizer (Sun-Earth-Moon demo)")
    parser.add_argument("--time", type=str, default="2026-04-10T12:00:00",
                        help="UTC start time in ISO format (default: 2026-04-10T12:00:00)")
    parser.add_argument("--spice-kernels", type=str, default="./spice_kernels",
                        help="Path to SPICE kernel directory (default: ./spice_kernels)")
    parser.add_argument("--vedo", action="store_true",
                        help="Use vedo for interactive 3D view (requires vedo)")
    args = parser.parse_args()

    kernel_path = Path(args.spice_kernels).resolve()
    if not kernel_path.exists():
        print(f"Error: SPICE kernel directory not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    # Configure HighPrecisionEphemeris in SPICE mode
    config = EphemerisConfig(
        mode=EphemerisMode.SPICE,
        spice_kernels_path=kernel_path,
        verbose=False
    )

    eph = HighPrecisionEphemeris(config=config)

    # Convert UTC string to ephemeris time (ET) using SPICE
    try:
        epoch = eph.utc_to_et(args.time)
    except Exception as exc:
        print(f"Failed to convert UTC to ET: {exc}", file=sys.stderr)
        eph.shutdown()
        sys.exit(1)

    print(f"Initialized SPICE, epoch = {epoch:.3f} s", file=sys.stdout)

    # Build scene
    builder = SceneBuilder(scale_function=LogScale(linear_threshold=3.8e8, compression=5e8))
    scene = builder.build_solar_system_demo(epoch, eph)

    # Render
    renderer = DebugRenderer(use_vedo=args.vedo)
    renderer.render(scene)

    # Clean up
    eph.shutdown()
    print("Done.", file=sys.stdout)


if __name__ == "__main__":
    main()
