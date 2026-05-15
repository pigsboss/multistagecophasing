#!/usr/bin/env python3
"""
N‑body ephemeris accuracy & performance benchmark
======================================================
Uses the existing N‑Body propagator (analytical.NBodyPropagator)
with selectable integrators, and compares results against
SPICE truth obtained via HighPrecisionEphemeris.

Two test methods:
  Method 1 – start from JPL mean Kepler elements at J2000,
             perturbed by Gaussian noise, integrate, compare to SPICE.
  Method 2 – start from SPICE ICs at J2000,
             perturbed by Gaussian noise, integrate, compare to SPICE.

Both methods use the same `--n-samples` and `--sigma-pos/vel`.
No direct SPICE kernel handling is required – HighPrecisionEphemeris
takes care of loading kernels from default locations.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Optional dependencies
# ----------------------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ----------------------------------------------------------------------
# Mission‑Sim imports
# ----------------------------------------------------------------------
from mission_sim.core.spacetime.ephemeris.analytical import NBodyPropagator, _nbody_derivs
from mission_sim.core.spacetime.ephemeris.high_precision import (
    HighPrecisionEphemeris,
    CelestialBody,
    EphemerisConfig,
    EphemerisMode,
    CoordinateFrame,
)
from mission_sim.core.spacetime.ephemeris.jpl_ssb_keplerian_elements import (
    get_all_planet_states,
)
from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
# Rotation from J2000 ecliptic to equatorial (ICRF)
_OBLIQUITY_J2000 = math.radians(23.4392911111)
_R_ECL2EQ = np.array([
    [1.0, 0.0,                       0.0                     ],
    [0.0, math.cos(_OBLIQUITY_J2000), math.sin(_OBLIQUITY_J2000)],
    [0.0, -math.sin(_OBLIQUITY_J2000), math.cos(_OBLIQUITY_J2000)]
])

# Body ordering (kept consistent with GM list)
BODIES = ["SUN", "MERCURY", "VENUS", "EARTH", "MARS",
          "JUPITER", "SATURN", "URANUS", "NEPTUNE"]

# GM in m^3/s^2 (DE440)
GM_DICT = {
    "SUN":      1.32712440041279419e20,
    "MERCURY":  2.203186855140000e13,
    "VENUS":    3.248585920000000e14,
    "EARTH":    3.986004354360959e14,
    "MARS":     4.282831425806000e13,
    "JUPITER":  1.266865361931886e17,
    "SATURN":   3.793120749865224e16,
    "URANUS":   5.793951322279009e15,
    "NEPTUNE":  6.835100718083999e15,
}

# Mapping from Kepler element planet names to uppercase keys
_MEAN_NAME_MAP = {
    "Mercury": "MERCURY",
    "Venus":   "VENUS",
    "EM Bary": "EARTH",
    "Mars":    "MARS",
    "Jupiter": "JUPITER",
    "Saturn":  "SATURN",
    "Uranus":  "URANUS",
    "Neptune": "NEPTUNE",
}

# ----------------------------------------------------------------------
# Helper: convert J2000 ecliptic state dict to ICRF equatorial flat array
# ----------------------------------------------------------------------
def ecliptic_dict_to_icrf_flat(state_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Takes a dict of 6‑vectors in J2000 ecliptic (m, m/s),
    rotates to ICRF equatorial, adds Sun via momentum conservation,
    and returns a flat 54‑element array in the order of BODIES.
    """
    # Rotate all planet states
    eq_states = {}
    for kepler_name, st in state_dict.items():
        eq_states[kepler_name] = np.concatenate([
            _R_ECL2EQ @ st[:3],
            _R_ECL2EQ @ st[3:6]
        ])

    # Sun from barycenter condition
    sun_pos = np.zeros(3)
    sun_vel = np.zeros(3)
    for kepler_name, st in eq_states.items():
        body_name = _MEAN_NAME_MAP[kepler_name]
        mass = GM_DICT[body_name]
        sun_pos -= mass * st[:3]
        sun_vel -= mass * st[3:6]
    sun_pos /= GM_DICT["SUN"]
    sun_vel /= GM_DICT["SUN"]

    # Assemble flat array
    flat = np.empty(54)
    flat[0:6] = np.concatenate([sun_pos, sun_vel])
    for idx, body in enumerate(BODIES[1:]):
        # Find the corresponding kepler name
        for kname, bname in _MEAN_NAME_MAP.items():
            if bname == body:
                flat[6*(idx+1):6*(idx+2)] = eq_states[kname]
                break
        else:
            raise KeyError(f"No Kepler element data for {body}")
    return flat

# ----------------------------------------------------------------------
# Truth provider via HighPrecisionEphemeris
# ----------------------------------------------------------------------
def build_truth_provider() -> HighPrecisionEphemeris:
    """Create a SPICE‑mode ephemeris. Raises if SPICE unavailable."""
    config = EphemerisConfig(mode=EphemerisMode.SPICE)
    eph = HighPrecisionEphemeris(config=config)
    if not eph._spice_initialized:
        raise RuntimeError(
            "SPICE kernels not found. Please set SPICE_KERNELS env var "
            "or place de440.bsp and naif0012.tls in a default path."
        )
    return eph

def get_truth_state(eph: HighPrecisionEphemeris, et_seconds: float,
                    bodies: List[str],
                    frame: CoordinateFrame = CoordinateFrame.J2000_ECI) -> np.ndarray:
    """
    Return concatenated state for the given bodies (order preserved) wrt SSB.
    et_seconds: TDB seconds since J2000.
    """
    states = []
    for bname in bodies:
        body = CelestialBody(bname.lower())
        state = eph.get_state(
            target_body=body,
            epoch=et_seconds,
            observer_body=CelestialBody.SSB,
            frame=frame,
        )
        states.append(state)

    return np.concatenate(states)

# ----------------------------------------------------------------------
# Timing / memory measurement
# ----------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t_start = time.perf_counter()
        if _HAS_PSUTIL:
            self.proc = psutil.Process()
            self.mem_before = self.proc.memory_info().rss
        return self

    def __exit__(self, *args):
        self.t_end = time.perf_counter()
        self.elapsed = self.t_end - self.t_start
        if _HAS_PSUTIL:
            self.mem_after = self.proc.memory_info().rss
            self.mem_peak = max(self.mem_after - self.mem_before, 0)
        else:
            self.mem_peak = None

# ----------------------------------------------------------------------
# Error statistics for a single integration
# ----------------------------------------------------------------------
def compute_errors(pred: np.ndarray, truth: np.ndarray, bodies: List[str]) -> Dict[str, float]:
    """Return per‑body position error (m), RMS, and max."""
    errs = {}
    all_err = []
    for i, name in enumerate(bodies):
        dpos = pred[6*i:6*i+3] - truth[6*i:6*i+3]
        e = np.linalg.norm(dpos)
        errs[name] = e
        all_err.append(e)
    errs["RMS"] = math.sqrt(sum(x*x for x in all_err) / len(all_err))
    errs["MAX"] = max(all_err)
    return errs

# ----------------------------------------------------------------------
# Method dispatch
# ----------------------------------------------------------------------
def run_method_1_or_2(
    method: str,        # "kepler" or "spice"
    integrator: str,
    delta_sec: float,
    n_samples: int,
    sigma_pos: float,
    sigma_vel: float,
    rtol: float,
    atol: float,
    max_step: float,
    truth_eph: HighPrecisionEphemeris,
    bodies: List[str],
    gm_dict: Dict[str, float],
) -> List[Dict[str, float]]:
    """
    Generic runner for method 1 (kepler) or 2 (spice).
    Returns list of per‑sample error dicts.
    The bodies list must contain exactly the bodies to propagate,
    and gm_dict must contain their GM values.
    """
    # Reference state at J2000 (0 seconds) – only needed bodies
    y_spice0 = get_truth_state(truth_eph, 0.0, bodies)

    # Target end state
    y_spice_end = get_truth_state(truth_eph, delta_sec, bodies)

    # Base initial state for each method
    if method == "kepler":
        # Get Kepler initial state for all bodies, then extract subset
        t_cy = 0.0  # J2000
        states_ecl = get_all_planet_states(t_cy)
        full_y0 = ecliptic_dict_to_icrf_flat(states_ecl)  # 54 elements in BODIES order
        # Extract only needed bodies
        base_y0 = np.empty(6 * len(bodies))
        for idx, bname in enumerate(bodies):
            global_idx = BODIES.index(bname)
            base_y0[6*idx:6*idx+6] = full_y0[6*global_idx:6*global_idx+6]
    else:  # "spice"
        base_y0 = y_spice0.copy()

    # Prepare propagator factory
    mu_list = [gm_dict[b] for b in bodies]

    results = []
    for sample in range(n_samples):
        # Generate noise
        pos_len = 3 * len(bodies)
        noise_pos = np.random.normal(0, sigma_pos, pos_len)
        noise_vel = np.random.normal(0, sigma_vel, pos_len)
        y_init = base_y0.copy()
        # Apply noise to correct position/velocity slots
        for i in range(len(bodies)):
            y_init[6*i:6*i+3] += noise_pos[3*i:3*i+3]
            y_init[6*i+3:6*i+6] += noise_vel[3*i:3*i+3]

        # Build initial dict
        init_dict = {}
        for idx, bname in enumerate(bodies):
            init_dict[bname] = y_init[6*idx:6*idx+6]

        # FIX: Force Sun state from SPICE truth
        if "SUN" in bodies:
            init_dict["SUN"] = get_truth_state(truth_eph, 0.0, ["SUN"])

        # Propagate
        with Timer() as tmr:
            prop = NBodyPropagator(
                bodies=bodies,
                mu_list=mu_list,
                initial_states_dict=init_dict,
                epoch_tdb=0.0,
                integrator=integrator,
                rtol=rtol,
                atol=atol,
                max_step=max_step,   # 强制最大步长
            )
            # Overwrite Sun state directly in propagator's buffer
            if "SUN" in bodies:
                prop.Y[:6] = get_truth_state(truth_eph, 0.0, ["SUN"])
            prop.propagate_to(delta_sec)

        # Collect final state
        y_final = np.empty(6 * len(bodies))
        for idx, bname in enumerate(bodies):
            y_final[6*idx:6*idx+6] = prop.get_body_state(bname, delta_sec)

        # Compute errors
        err = compute_errors(y_final, y_spice_end, bodies)
        err["_time"] = tmr.elapsed
        err["_mem"] = tmr.mem_peak if tmr.mem_peak is not None else 0.0
        results.append(err)

    return results

# ----------------------------------------------------------------------
# Aggregate and display
# ----------------------------------------------------------------------
def aggregate_results(errors_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute mean/std for all numeric fields."""
    keys = list(errors_list[0].keys())
    agg = {}
    for k in keys:
        vals = [d[k] for d in errors_list]
        agg[f"{k}_mean"] = statistics.mean(vals)
        if len(vals) > 1:
            agg[f"{k}_std"] = statistics.stdev(vals)
        else:
            agg[f"{k}_std"] = 0.0
    return agg

def print_summary(method_name: str, integrator: str, agg: Dict[str, Any],
                  bodies: List[str]):
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name} | INTEGRATOR: {integrator}")
    print(f"{'='*60}")
    print("Position errors (mean ± std, meters):")
    for b in bodies:
        mean = agg.get(f"{b}_mean", 0)
        std = agg.get(f"{b}_std", 0)
        print(f"  {b:10s}: {mean:12.3f} ± {std:12.3f}")
    print(f"  {'RMS':10s}: {agg['RMS_mean']:12.3f} ± {agg.get('RMS_std',0):12.3f}")
    print(f"  {'MAX':10s}: {agg['MAX_mean']:12.3f} ± {agg.get('MAX_std',0):12.3f}")
    print(f"Time (s): {agg['_time_mean']:.4f} ± {agg.get('_time_std',0):.4f}")
    if _HAS_PSUTIL:
        mem_mean = agg['_mem_mean'] / 1e6
        print(f"Memory peak (MB): {mem_mean:.2f}")

def save_json(data: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="N‑body ephemeris benchmark")
    parser.add_argument("--method", choices=["kepler", "spice", "all"], default="all")
    parser.add_argument("--integrators", nargs="+", default=["dop853"],
                        help="Integrators to test (dp8, rk45, dop853, scipy:rk45, scipy:dop853)")
    parser.add_argument("--delta-years", type=float, default=1.0,
                        help="Integration duration in years")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of perturbed samples")
    parser.add_argument("--sigma-pos", type=float, default=1e3,
                        help="Position perturbation std (meters)")
    parser.add_argument("--sigma-vel", type=float, default=1e-3,
                        help="Velocity perturbation std (m/s)")
    parser.add_argument("--max-step", type=float, default=86400.0,
                        help="Max integration step size in seconds (default: 86400)")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, help="Save results as JSON")
    # NEW: debug-two-body flag
    parser.add_argument("--debug-two-body", action="store_true",
                        help="Run Sun+Earth only for debugging")
    args = parser.parse_args()

    # Initialize truth
    try:
        truth_eph = build_truth_provider()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print("SPICE truth provider initialized.")

    # Time span in seconds
    delta_sec = args.delta_years * 365.25 * 86400.0

    # ------------------------------------------------------------------
    # Debug: analytic two‑body comparison (Sun‑Earth only)
    # ------------------------------------------------------------------
    if args.debug_two_body:
        mu_sun = GM_DICT["SUN"]
        # Obtain J2000 relative state
        sun_state = get_truth_state(truth_eph, 0.0, ["SUN"])
        earth_state = get_truth_state(truth_eph, 0.0, ["EARTH"])
        r_rel = earth_state[:3] - sun_state[:3]
        v_rel = earth_state[3:6] - sun_state[3:6]

        # Convert to Kepler elements (simple inverse)
        from mission_sim.utils.solvers.keplerian import cartesian_to_kepler_elements
        a, e, i, Omega, omega, M0 = cartesian_to_kepler_elements(
            r_rel, v_rel, mu_sun
        )

        n = math.sqrt(mu_sun / a ** 3)
        dt = delta_sec
        M_end = M0 + n * dt

        # Propagate analytically
        state_rel_end = kepler_elements_to_cartesian_batch(
            np.array([a]), np.array([e]), np.array([i]),
            np.array([Omega]), np.array([omega]), np.array([M_end]),
            mu_sun
        )[0]

        # SPICE end state
        sun_end = get_truth_state(truth_eph, dt, ["SUN"])
        earth_end = get_truth_state(truth_eph, dt, ["EARTH"])
        r_rel_end_spice = earth_end[:3] - sun_end[:3]

        pos_err = np.linalg.norm(state_rel_end[:3] - r_rel_end_spice)
        print(f"Analytic two‑body pos error after {args.delta_years:.4f} yr: {pos_err:.3f} m")
        sys.exit(0)

    # Determine bodies and gm dict based on debug flag
    if args.debug_two_body:
        use_bodies = ["SUN", "EARTH"]
        use_gm = {k: GM_DICT[k] for k in use_bodies}
        print("DEBUG: Two-body mode (Sun+Earth only)")
    else:
        use_bodies = BODIES
        use_gm = GM_DICT

    all_output = {}
    for integrator in args.integrators:
        if args.method in ("kepler", "all"):
            print(f"\nRunning method 1 (Kepler + noise) with {integrator}...")
            errs = run_method_1_or_2(
                "kepler", integrator, delta_sec, args.n_samples,
                args.sigma_pos, args.sigma_vel, args.rtol, args.atol,
                args.max_step, truth_eph,
                bodies=use_bodies, gm_dict=use_gm
            )
            agg = aggregate_results(errs)
            key = f"kepler_{integrator}"
            all_output[key] = agg
            print_summary("Kepler (Method 1)", integrator, agg, bodies=use_bodies)

        if args.method in ("spice", "all"):
            print(f"\nRunning method 2 (SPICE + noise) with {integrator}...")
            errs = run_method_1_or_2(
                "spice", integrator, delta_sec, args.n_samples,
                args.sigma_pos, args.sigma_vel, args.rtol, args.atol,
                args.max_step, truth_eph,
                bodies=use_bodies, gm_dict=use_gm
            )
            agg = aggregate_results(errs)
            key = f"spice_{integrator}"
            all_output[key] = agg
            print_summary("SPICE (Method 2)", integrator, agg, bodies=use_bodies)

    if args.output:
        save_json(all_output, args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
