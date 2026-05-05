# -*- coding: utf-8 -*-
"""
MCPC Visualizer CLI

Usage:
    python -m utils.visualizers --scene earth_moon --time "2026-04-10T12:00:00" [--vedo] ...
    python -m utils.visualizers --scene solar_system --time "2026-04-10T12:00:00" [--vedo] ...
"""

import argparse
import sys
from pathlib import Path
import os
import numpy as np

# Ensure the project root is importable (in case running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.visualizers.base import SceneBuilder, LogScale
from utils.visualizers.backends.simple import SimpleRenderer
from mission_sim.core.spacetime.ids import CoordinateFrame

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
        description="MCPC 3D Scene Visualizer")
    parser.add_argument("--scene", type=str, default="earth_moon",
                        choices=["earth_moon", "solar_system"],
                        help="Which scene to render (default: earth_moon)")
    parser.add_argument("--time", type=str, default="2026-04-10T12:00:00",
                        help="UTC start time in ISO format")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total simulation duration in seconds")
    parser.add_argument("--step", type=float, default=3600,
                        help="Time step between frames in seconds (default: 3600)")
    parser.add_argument("--vedo", action="store_true",
                        help="Use vedo for 3D output")
    parser.add_argument("--debug", action="store_true",
                        help="Use debug backend (prints scene tree, optionally renders with vedo)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save PNG frames (vedo required)")
    args = parser.parse_args()

    if args.duration is not None and args.duration <= 0:
        print("Error: --duration must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.step <= 0:
        print("Error: --step must be positive.", file=sys.stderr)
        sys.exit(1)

    # Configure HighPrecisionEphemeris in SPICE mode with appropriate mission type.
    mission_type = "interplanetary" if args.scene == "solar_system" else "earth_moon"
    config = EphemerisConfig(
        mode=EphemerisMode.SPICE,
        spice_mission_type=mission_type,
        verbose=True      # enable detailed SPICE kernel loading output
    )

    eph = HighPrecisionEphemeris(config=config)

    # Convert UTC string to ephemeris time (ET) using SPICE (or fallback if SPICE unavailable)
    try:
        start_epoch = eph.utc_to_et(args.time)
    except Exception as exc:
        print(f"Failed to convert UTC to ET: {exc}", file=sys.stderr)
        eph.shutdown()
        sys.exit(1)

    print(f"Initialized SPICE, start epoch = {start_epoch:.3f} s", file=sys.stdout)

    # SPICE health check: query Earth to ensure SPICE is actually operational
    try:
        test_state = eph.get_state("earth", start_epoch, observer_body="sun",
                                   frame=CoordinateFrame.J2000_ECI)
        if np.linalg.norm(test_state[:3]) < 1e6:
            raise RuntimeError("SPICE returned near-zero Earth position – kernels likely missing")
    except Exception as e:
        print(f"SPICE self-test failed: {e}", file=sys.stderr)
        eph.shutdown()
        sys.exit(1)
    print("SPICE self-test passed (Earth position valid).", file=sys.stdout)

    # Build time sequence
    if args.duration is not None:
        times = [start_epoch + i * args.step
                 for i in range(int(args.duration / args.step) + 1)]
    else:
        times = [start_epoch]

    # Prepare output directory if needed
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Choose scale function per scene
    if args.scene == "solar_system":
        # Wider threshold for outer planets
        scale = LogScale(linear_threshold=1.5e11, compression=5e11)
    else:
        scale = LogScale(linear_threshold=3.8e8, compression=5e8)

    builder = SceneBuilder(scale_function=scale)

    # Enable vedo backend if --output is given even without --vedo
    use_vedo = args.vedo or (args.output is not None)

    # Select renderer based on flags
    if args.debug:
        from utils.visualizers.backends.debug import DebugRenderer
        renderer = DebugRenderer()   # no arguments – pure matplotlib debug
    else:
        renderer = SimpleRenderer(use_vedo=use_vedo)

    total_frames = len(times)
    for idx, epoch in enumerate(times):
        if args.scene == "solar_system":
            scene = builder.build_solar_system(epoch, eph)
        else:
            scene = builder.build_solar_system_demo(epoch, eph)

        print(f"\n--- Frame {idx+1}/{total_frames} : epoch = {epoch:.3f} s ---")
        renderer.render(
            scene,
            frame_index=idx,
            total_frames=total_frames,
            output_dir=str(output_dir) if output_dir else None
        )

    # Clean up
    eph.shutdown()
    print("\nDone.", file=sys.stdout)


if __name__ == "__main__":
    main()
