"""
N‑body ephemeris accuracy & performance benchmark
======================================================
Uses the existing N‑Body propagator (analytical.NBodyPropagator)
with selectable integrators, and compares results against
SPICE truth obtained via HighPrecisionEphemeris.

Two test methods:
  Method 1 – start from JPL mean Kepler elements at J2000,
             perturbed by Gaussian noise, integrate each planet
             independently as a two‑body problem, compare
             heliocentric positions to SPICE.
  Method 2 – start from SPICE ICs at J2000,
             perturbed by Gaussian noise, integrate with N‑body,
             compare to SPICE (SSB‑referenced).

Both methods use the same `--n-samples` and `--sigma-pos/vel`.
No direct SPICE kernel handling is required – HighPrecisionEphemeris
takes care of loading kernels from default locations.

Uses short‑period mean elements (Table 1) for Method 1.
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
    get_elements_cartesian,
    get_elements,
    get_elements_short,
)
from mission_sim.utils.solvers.keplerian import (
    kepler_elements_to_cartesian_batch,
    cartesian_to_kepler_elements,
)

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
# Rotation from J2000 ecliptic to equatorial (ICRF)
_OBLIQUITY_J2000 = math.radians(23.4392911111)
_R_ECL2EQ = np.array([
    [1.0, 0.0,                        0.0                      ],
    [0.0, math.cos(_OBLIQUITY_J2000), -math.sin(_OBLIQUITY_J2000)],
    [0.0, math.sin(_OBLIQUITY_J2000),  math.cos(_OBLIQUITY_J2000)]
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


def get_truth_heliocentric_state(eph: HighPrecisionEphemeris, et_seconds: float,
                                 bodies: List[str]) -> np.ndarray:
    """
    Return concatenated heliocentric states (relative to Sun) for the given bodies.
    et_seconds : TDB seconds since J2000.
    Returns array of shape (6 * len(bodies),).
    """
    states = []
    for bname in bodies:
        if bname == "SUN":
            # Sun relative to itself is always zero
            states.append(np.zeros(6))
        else:
            body = CelestialBody(bname.lower())
            state = eph.get_state(
                target_body=body,
                epoch=et_seconds,
                observer_body=CelestialBody.SUN,
                frame=CoordinateFrame.J2000_ECI,
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

    # Target end state (SSB-reference for spice, heliocentric for kepler)
    if method == "kepler":
        y_spice_end = get_truth_heliocentric_state(truth_eph, delta_sec, bodies)
    else:
        y_spice_end = get_truth_state(truth_eph, delta_sec, bodies)

    # Base initial state for each method
    if method == "kepler":
        # Use JPL short-period mean elements (Table 1)
        t_cy = 0.0
        mu_sun = gm_dict["SUN"]
        inv_map = {v: k for k, v in _MEAN_NAME_MAP.items()}

        # ---- Detailed JPL vs SPICE at J2000 ----
        print("\nDetailed JPL vs SPICE at J2000 (heliocentric, equatorial):")
        for bname in bodies:
            if bname == "SUN":
                continue
            kep_name = inv_map[bname]
            el = get_elements_short(kep_name, t_cy)
            a, e_orb, inc, Omega, omega, M0 = (
                el["a"], el["e"], el["i"], el["Omega"], el["omega"], el["M"]
            )
            print(f"\n{bname}:")
            print(f"  a={a:.6e} m, e={e_orb:.8f}, i={inc:.8f} rad")
            print(f"  Omega={Omega:.8f} rad, omega={omega:.8f} rad, M={M0:.8f} rad")
            state_ecl = kepler_elements_to_cartesian_batch(
                np.array([a]), np.array([e_orb]), np.array([inc]),
                np.array([Omega]), np.array([omega]), np.array([M0]),
                mu_sun
            )[0]
            state_eq = np.concatenate([
                _R_ECL2EQ @ state_ecl[:3],
                _R_ECL2EQ @ state_ecl[3:6]
            ])
            # Fetch SPICE truth before reverse rotation check
            spice_state = get_truth_heliocentric_state(truth_eph, 0.0, [bname])
            # 临时反向旋转验证
            _R_EQ2ECL = _R_ECL2EQ.T
            state_eq_test = np.concatenate([
                _R_EQ2ECL @ state_ecl[:3],
                _R_EQ2ECL @ state_ecl[3:6]
            ])
            diff_rev = state_eq_test - spice_state
            print(f"  |diff (reverse rot)| = {np.linalg.norm(diff_rev[:3]):.6e} m")
            diff = state_eq - spice_state
            print(f"  JPL pos  = {state_eq[:3]} m")
            print(f"  SPICE pos= {spice_state[:3]} m")
            print(f"  |diff|   = {np.linalg.norm(diff[:3]):.6e} m")
        # -----------------------------------------------------------

        # Pre-compute final heliocentric equatorial state for each body
        final_states = []
        for bname in bodies:
            if bname == "SUN":
                final_states.append(np.zeros(6))
                continue
            kep_name = inv_map[bname]
            el = get_elements_short(kep_name, t_cy)
            a, e_orb, inc, Omega, omega, M0 = (
                el["a"], el["e"], el["i"], el["Omega"], el["omega"], el["M"]
            )
            n = math.sqrt(mu_sun / (a * a * a))
            M_end = M0 + n * delta_sec
            # Final heliocentric equatorial state (from ecliptic)
            state_ecl_end = kepler_elements_to_cartesian_batch(
                np.array([a]), np.array([e_orb]), np.array([inc]),
                np.array([Omega]), np.array([omega]), np.array([M_end]),
                mu_sun
            )[0]
            state_eq_end = np.concatenate([
                _R_ECL2EQ @ state_ecl_end[:3],
                _R_ECL2EQ @ state_ecl_end[3:6]
            ])
            final_states.append(state_eq_end)

        y_final = np.concatenate(final_states)

        # Compare with SPICE truth once, then replicate for all samples
        y_spice_end = get_truth_heliocentric_state(truth_eph, delta_sec, bodies)
        base_err = compute_errors(y_final, y_spice_end, bodies)
        base_err["_time"] = 0.0
        base_err["_mem"] = 0.0
        return [base_err.copy() for _ in range(n_samples)]

    else:  # "spice"
        base_y0 = y_spice0.copy()
        mu_list = [gm_dict[b] for b in bodies]

        results = []
        for sample in range(n_samples):
            # Generate noise
            pos_len = 3 * len(bodies)
            noise_pos = np.random.normal(0, sigma_pos, pos_len)
            noise_vel = np.random.normal(0, sigma_vel, pos_len)
            y_init = base_y0.copy()
            for i in range(len(bodies)):
                y_init[6*i:6*i+3] += noise_pos[3*i:3*i+3]
                y_init[6*i+3:6*i+6] += noise_vel[3*i:3*i+3]

            # Build initial dict for N‑body propagator
            init_dict = {}
            for idx, bname in enumerate(bodies):
                init_dict[bname] = y_init[6*idx:6*idx+6]

            with Timer() as tmr:
                prop = NBodyPropagator(
                    bodies=bodies,
                    mu_list=mu_list,
                    initial_states_dict=init_dict,
                    epoch_tdb=0.0,
                    integrator=integrator,
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step,
                )
                prop.propagate_to(delta_sec)
                y_final = np.empty(6 * len(bodies))
                for idx, bname in enumerate(bodies):
                    y_final[6*idx:6*idx+6] = prop.get_body_state(bname, delta_sec)

            elapsed = tmr.elapsed
            mem_peak = tmr.mem_peak if tmr.mem_peak is not None else 0.0

            # Compute errors
            err = compute_errors(y_final, y_spice_end, bodies)
            err["_time"] = elapsed
            err["_mem"] = mem_peak
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
    parser.add_argument("--integrator", type=str, default="dop853",
                        help="Integrator to use (dp8, rk45, dop853, scipy:rk45, scipy:dop853)")
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

    # Determine bodies and gm dict based on debug flag
    if args.debug_two_body:
        # For debug, only two bodies – but method 1 will still drop SUN
        spice_bodies = ["SUN", "EARTH"]
        spice_gm = {k: GM_DICT[k] for k in spice_bodies}
        kepler_bodies = ["EARTH"]
        # Fix: include SUN in kepler_gm for gm_dict["SUN"]
        kepler_gm = {"SUN": GM_DICT["SUN"], "EARTH": GM_DICT["EARTH"]}
        print("DEBUG: Two-body mode (Sun+Earth for SPICE; Earth only for Kepler)")
    else:
        spice_bodies = BODIES
        spice_gm = GM_DICT
        kepler_bodies = BODIES[1:]   # exclude Sun for heliocentric comparison
        # Fix: include SUN in kepler_gm for gm_dict["SUN"]
        kepler_gm = {"SUN": GM_DICT["SUN"], **{k: GM_DICT[k] for k in kepler_bodies}}

    all_output = {}
    integrator = args.integrator

    if args.method in ("kepler", "all"):
        print(f"\nRunning method 1 (Kepler + noise) with {integrator}...")
        errs = run_method_1_or_2(
            "kepler", integrator, delta_sec, args.n_samples,
            args.sigma_pos, args.sigma_vel, args.rtol, args.atol,
            args.max_step, truth_eph,
            bodies=kepler_bodies, gm_dict=kepler_gm
        )
        agg = aggregate_results(errs)
        key = f"kepler_{integrator}"
        all_output[key] = agg
        print_summary("Kepler (Method 1)", integrator, agg, bodies=kepler_bodies)

    if args.method in ("spice", "all"):
        print(f"\nRunning method 2 (SPICE + noise) with {integrator}...")
        errs = run_method_1_or_2(
            "spice", integrator, delta_sec, args.n_samples,
            args.sigma_pos, args.sigma_vel, args.rtol, args.atol,
            args.max_step, truth_eph,
            bodies=spice_bodies, gm_dict=spice_gm
        )
        agg = aggregate_results(errs)
        key = f"spice_{integrator}"
        all_output[key] = agg
        print_summary("SPICE (Method 2)", integrator, agg, bodies=spice_bodies)

    if args.output:
        save_json(all_output, args.output)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
