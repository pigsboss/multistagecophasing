#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPICE Kernel Diagnostic Tool for MCPC

Inspects raw SPK file contents and verifies solar-system body queries.
Run directly: python tests/diagnose_spice_kernels.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import spiceypy as spice
    from mission_sim.core.spacetime.ephemeris.spice_interface import (
        SPICEInterface, SPICEConfig
    )
    from mission_sim.core.spacetime.ids import CoordinateFrame
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)


def find_kernel_path() -> Path:
    """Locate the SPICE kernels directory."""
    import os
    if env := os.environ.get('SPICE_KERNELS'):
        return Path(env)
    for cand in [Path('./spice_kernels'), Path(__file__).parent.parent / 'spice_kernels']:
        if cand.exists():
            return cand
    raise FileNotFoundError("SPICE kernels directory not found")


def inspect_spk_file(spk_path: Path):
    """List every NAIF ID and its time coverage inside a single SPK file."""
    print(f"\n{'='*60}")
    print(f"INSPECTING SPK FILE: {spk_path.name}")
    print(f"{'='*60}")
    try:
        ids = spice.spkobj(str(spk_path))
        if len(ids) == 0:
            print("  (no objects found - file may be empty or invalid SPK)")
            return
        for obj_id in ids:
            cover = spice.spkcov(str(spk_path), obj_id)
            n = spice.wncard(cover)
            intervals = []
            for i in range(n):
                b, e = spice.wnfetd(cover, i)
                try:
                    # Requires LSK to be loaded in kernel pool
                    b_utc = spice.et2utc(b, 'C', 0)
                    e_utc = spice.et2utc(e, 'C', 0)
                    intervals.append(f"{b_utc}  to  {e_utc}")
                except Exception:
                    # Fallback to raw ET seconds if LSK unavailable
                    intervals.append(f"ET {b:.3f} s  to  ET {e:.3f} s")
            print(f"  NAIF ID {obj_id:4d}: {n} interval(s)")
            for iv in intervals:
                print(f"                 {iv}")
    except Exception as e:
        print(f"  ERROR inspecting file: {e}")


def test_body_queries(iface, epoch: float):
    """Query every solar-system body and print a formatted report."""
    bodies = [
        ("mercury", 1),
        ("venus", 2),
        ("earth", 399),
        ("moon", 301),
        ("mars", 4),
        ("jupiter", 5),
        ("saturn", 6),
        ("uranus", 7),
        ("neptune", 8),
    ]

    print(f"\n{'='*60}")
    print(f"QUERY TEST: body state relative to SUN @ epoch = {epoch:.3f} s")
    print(f"{'='*60}")
    print(f"{'Body':<10} {'ExpectedID':<10} {'ResolvedID':<12} {'Status':<6} {'Dist(km)':>14} {'Vel(km/s)':>12}")
    print("-" * 70)

    calc = iface._calc
    all_ok = True

    for name, expected_id in bodies:
        try:
            resolved = int(calc._to_naif_id(name))
            state = iface.get_state(
                name, epoch, "sun", CoordinateFrame.J2000_ECI, "NONE"
            )
            dist = np.linalg.norm(state[:3]) / 1e3
            vel = np.linalg.norm(state[3:6]) / 1e3
            ok = resolved == expected_id
            status = "OK" if ok else "ID_MISMATCH"
            print(f"{name:<10} {expected_id:<10} {resolved:<12} {status:<6} {dist:>14.2f} {vel:>12.4f}")
            if not ok:
                all_ok = False
        except Exception as e:
            all_ok = False
            print(f"{name:<10} {expected_id:<10} {'--':<12} {'FAIL':<6} {'--':>14} {'--':>12}")
            print(f"         -> {type(e).__name__}: {str(e)[:90]}")

    print("-" * 70)
    return all_ok


def main():
    kernel_path = find_kernel_path()
    print(f"[INFO] Kernel path: {kernel_path.resolve()}")

    # 1. Initialize SPICE FIRST to ensure LSK is loaded into the kernel pool.
    #    ET→UTC conversion (et2utc) requires leapsecond data.
    config = SPICEConfig(mission_type="interplanetary", verbose=True)
    iface = SPICEInterface(kernel_path, config)

    print("\n[INFO] Initializing SPICE...")
    if not iface.initialize():
        print("[FAIL] SPICE initialization failed.")
        sys.exit(1)

    print(f"[INFO] Loaded kernels: {[k.name for k in iface._km.get_loaded_kernels()]}")

    # 2. Inspect raw SPK contents AFTER LSK is loaded, so UTC display works.
    spk_files = sorted(
        set(list(kernel_path.rglob("de440.bsp")) + list(kernel_path.rglob("de44*.bsp")))
    )
    if not spk_files:
        print("[FAIL] No de440*.bsp files found!")
        sys.exit(1)

    for spk in spk_files:
        inspect_spk_file(spk)

    # 3. Run query tests (internal ET epoch is acceptable here as it stays inside spacetime)
    epoch = 831211269.185
    all_ok = test_body_queries(iface, epoch)

    iface.shutdown()

    print("\n" + "=" * 60)
    if all_ok:
        print("[SUCCESS] All bodies resolved correctly and returned valid states.")
        sys.exit(0)
    else:
        print("[FAILURE] Some bodies failed. Compare ResolvedID with ExpectedID above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
