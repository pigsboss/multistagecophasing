# analysis/fuel_analysis.py
"""
Fuel Consumption Analysis Script
Analyzes delta-V consumption under different orbit types, control gains, blind intervals.
Supports parameter scanning and result visualization.
"""

import os
import sys
import argparse
import json
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

# Set matplotlib backend to non-interactive (avoid plt.show() warnings)
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.core.spacetime.ids import CoordinateFrame


@dataclass
class FuelMetrics:
    """Fuel-related metrics for a single simulation run."""
    mission_id: str
    orbit_type: str              # Orbit type: "Halo", "Keplerian", "J2"
    control_gain_scale: float    # Control gain scaling factor
    blind_interval_days: float   # Blind interval duration (days)
    simulation_days: float       # Simulation duration (days)
    total_dv: float              # Total delta-V (m/s)
    avg_dv_per_day: float        # Average delta-V per day (m/s/day)
    max_control_force: float     # Maximum control force (N)
    final_position_error: float  # Final position error (m)
    simulation_time: float       # Actual simulation elapsed time (s)
    parameters: Dict[str, Any]   # Complete parameter copy


def _run_single_simulation(args: Tuple[Dict[str, Any], str]) -> Optional[FuelMetrics]:
    """
    Run a single simulation and return fuel-related metrics.
    Designed to be executed in a worker process.

    Args:
        args: Tuple of (config, mission_id)

    Returns:
        FuelMetrics if successful, else None
    """
    config, mission_id = args

    # Inject mission_id and force silent mode for parallel runs
    config["mission_id"] = mission_id
    config["verbose"] = False
    config["save_fuel_bill"] = False
    config["log_backup"] = False

    try:
        sim = SunEarthL2L1Simulation(config)
        start_time = time.time()
        success = sim.run()
        elapsed = time.time() - start_time

        if not success:
            return None

        # Get basic statistics from simulation object
        stats = sim.get_statistics()
        total_dv = stats.get("accumulated_dv", np.nan)
        final_pos_err = stats.get("final_position_error", np.nan)

        # Load maximum control force from HDF5 file using absolute path
        h5_file_abs = os.path.abspath(sim.h5_file)
        max_force = np.nan
        if os.path.exists(h5_file_abs):
            try:
                with h5py.File(h5_file_abs, 'r') as f:
                    forces = f['control_forces'][:]
                    if len(forces) > 0:
                        max_force = np.max(np.linalg.norm(forces, axis=1))
            except Exception:
                pass

        # Calculate average delta-V per day
        sim_days = config.get("simulation_days", 30)
        avg_dv_per_day = total_dv / sim_days if sim_days > 0 else np.nan

        metrics = FuelMetrics(
            mission_id=sim.mission_id,
            orbit_type=config.get("orbit_type", "Unknown"),
            control_gain_scale=config.get("control_gain_scale", 1.0),
            blind_interval_days=config.get("blind_interval_days", 0),
            simulation_days=sim_days,
            total_dv=total_dv,
            avg_dv_per_day=avg_dv_per_day,
            max_control_force=max_force,
            final_position_error=final_pos_err,
            simulation_time=elapsed,
            parameters=config
        )
        return metrics

    except Exception:
        return None


class FuelAnalyzer:
    """
    Fuel consumption analyzer.
    Scans parameters to evaluate fuel consumption under different conditions,
    generates reports and charts.
    """

    # Default base configuration (silent, no backup)
    DEFAULT_BASE_CONFIG = {
        "mission_name": "Fuel_Analysis",
        "time_step": 10.0,
        "log_buffer_size": 500,
        "log_compression": True,
        "enable_visualization": False,
        "data_dir": "data/fuel_analysis",
        "log_level": "WARNING",
        "integrator": "rk4",            # Default integrator
        "verbose": False,               # Suppress simulation internal output
        "save_fuel_bill": False,        # Do not generate CSV fuel bills
        "log_backup": False,            # Disable file backup for parallel runs
    }

    # Default scan parameters
    DEFAULT_SCAN_PARAMS = {
        "orbit_type": ["Halo", "Keplerian"],          # Orbit type
        "control_gain_scale": [0.5, 1.0, 2.0],        # Control gain scaling factor
        "blind_interval_days": [0, 1, 3, 7],          # Blind interval duration (days)
        "simulation_days": [30, 90]                   # Simulation duration (days)
    }

    def __init__(self,
                 base_config: Optional[Dict] = None,
                 scan_params: Optional[Dict] = None,
                 output_dir: str = "analysis_results",
                 n_repeats: int = 3,
                 n_processes: int = None):
        """
        Initialize fuel analyzer.

        Args:
            base_config: Base simulation configuration (overrides default)
            scan_params: Parameter scan dictionary
            output_dir: Output directory for results
            n_repeats: Number of repeated runs per parameter combination (for statistical stability)
            n_processes: Number of parallel processes (default: None -> cpu_count)
        """
        self.base_config = {**self.DEFAULT_BASE_CONFIG, **(base_config or {})}
        self.scan_params = scan_params or self.DEFAULT_SCAN_PARAMS
        self.output_dir = output_dir
        self.n_repeats = n_repeats
        self.n_processes = n_processes or mp.cpu_count()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Store all metrics
        self.metrics_list: List[FuelMetrics] = []

        print(f"[FuelAnalyzer] Initialized, output directory: {output_dir}")
        print(f"[FuelAnalyzer] Scan parameters: {list(self.scan_params.keys())}")
        print(f"[FuelAnalyzer] Repeats per combination: {n_repeats}")
        print(f"[FuelAnalyzer] Parallel processes: {self.n_processes}")

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build simulation configuration from parameters.
        """
        config = self.base_config.copy()
        config.update(params)

        # Set orbit-specific parameters
        orbit_type = params.get("orbit_type", "Halo")
        if orbit_type == "Halo":
            # Halo orbit default configuration (Sun-Earth L2)
            config.setdefault("Az", 0.05)
            config.setdefault("dt", 0.001)
            config.setdefault("initial_guess", [1.01106, 0.05, 0.0105])
        elif orbit_type == "Keplerian":
            # Keplerian circular orbit (Earth-centered, 7000km radius)
            config.setdefault("elements", [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])
            config.setdefault("simulation_days", params.get("simulation_days", 30))
        elif orbit_type == "J2":
            # Orbit with J2 perturbation (Earth-centered)
            config.setdefault("elements", [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])
            config.setdefault("simulation_days", params.get("simulation_days", 30))
            config["enable_j2"] = True   # Flag to register J2 model during simulation
        else:
            raise ValueError(f"Unknown orbit type: {orbit_type}")

        # Blind interval configuration
        blind_days = params.get("blind_interval_days", 0)
        if blind_days > 0:
            # Construct visibility window: assume first blind_days days are invisible, then visible
            blind_seconds = blind_days * 86400
            sim_seconds = params.get("simulation_days", 30) * 86400
            config["visibility_windows"] = [(blind_seconds, sim_seconds)]
        else:
            config["visibility_windows"] = []   # Always visible

        # Control gain scaling factor: pass to simulation class (if supported)
        # SunEarthL2L1Simulation currently does not support gain scaling,
        # but we pass the parameter for future compatibility.
        config["control_gain_scale"] = params.get("control_gain_scale", 1.0)

        return config

    def _generate_param_combinations(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Generate all parameter combinations (Cartesian product) and repeat n_repeats times.
        Returns a list of (config, mission_id) pairs.
        """
        keys = list(self.scan_params.keys())
        values = [self.scan_params[k] for k in keys]
        combinations = []
        idx = 0
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            for _ in range(self.n_repeats):
                mission_id = f"fuel_analysis_{idx:06d}"
                config = self._build_config(params)
                combinations.append((config, mission_id))
                idx += 1
        return combinations

    def run_scan(self):
        """
        Execute parameter scan simulations in parallel with real-time progress.
        """
        param_combos = self._generate_param_combinations()
        total_runs = len(param_combos)
        print(f"\n[FuelAnalyzer] Starting parameter scan, total runs: {total_runs}")
        print(f"[FuelAnalyzer] Using {self.n_processes} parallel processes...")

        with Pool(processes=self.n_processes) as pool:
            results = []
            with tqdm(total=total_runs, desc="Running simulations", unit="run") as pbar:
                for result in pool.imap_unordered(_run_single_simulation, param_combos):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

                    # Periodically save intermediate results (every 50 runs)
                    if len(results) % 50 == 0:
                        self.metrics_list = results
                        self._save_results(partial=True)

        self.metrics_list = results
        self._save_results(final=True)
        print(f"\n[FuelAnalyzer] Scan finished, successful runs: {len(self.metrics_list)}/{total_runs}")

    def _save_results(self, partial: bool = False, final: bool = False):
        """
        Save results to JSON file.
        """
        if not self.metrics_list:
            return
        if final:
            filename = "fuel_analysis_results_final.json"
        elif partial:
            filename = "fuel_analysis_results_partial.json"
        else:
            filename = "fuel_analysis_results.json"
        filepath = os.path.join(self.output_dir, filename)
        data = [asdict(m) for m in self.metrics_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[FuelAnalyzer] Saved {len(self.metrics_list)} results to {filepath}")

    def plot_results(self):
        """
        Generate fuel analysis charts.
        """
        if not self.metrics_list:
            print("[FuelAnalyzer] No data, cannot plot.")
            return

        # Group by orbit type, blind interval, gain scale
        orbit_types = sorted(set(m.orbit_type for m in self.metrics_list))
        blind_intervals = sorted(set(m.blind_interval_days for m in self.metrics_list))
        gain_scales = sorted(set(m.control_gain_scale for m in self.metrics_list))

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Fuel Consumption Analysis", fontsize=14, fontweight='bold')

        # 1. Average delta-V per day vs orbit type and blind interval (boxplot)
        ax = axes[0, 0]
        data_by_orb_blind = {}
        for orb in orbit_types:
            for blind in blind_intervals:
                key = f"{orb}_{blind}"
                dv_vals = [m.avg_dv_per_day for m in self.metrics_list
                           if m.orbit_type == orb and m.blind_interval_days == blind]
                if dv_vals:
                    data_by_orb_blind[key] = dv_vals
        labels = [f"{k}" for k in data_by_orb_blind.keys()]
        positions = range(len(data_by_orb_blind))
        bp = ax.boxplot(data_by_orb_blind.values(), positions=positions, widths=0.6, patch_artist=True)
        ax.set_xticks(positions, labels, rotation=45, ha='right')
        ax.set_ylabel('Average delta-V per day (m/s/day)')
        ax.set_title('Fuel Consumption by Orbit Type and Blind Interval')
        ax.grid(True, alpha=0.3)

        # 2. Total delta-V vs control gain (line plot, grouped by orbit type)
        ax = axes[0, 1]
        for orb in orbit_types:
            gain_vals = []
            dv_means = []
            dv_stds = []
            for gain in gain_scales:
                dv_vals = [m.total_dv for m in self.metrics_list
                           if m.orbit_type == orb and m.control_gain_scale == gain]
                if dv_vals:
                    gain_vals.append(gain)
                    dv_means.append(np.mean(dv_vals))
                    dv_stds.append(np.std(dv_vals))
            if gain_vals:
                ax.errorbar(gain_vals, dv_means, yerr=dv_stds, marker='o', capsize=3, label=orb)
        ax.set_xlabel('Control Gain Scaling Factor')
        ax.set_ylabel('Total delta-V (m/s)')
        ax.set_title('Total delta-V vs Control Gain')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Total delta-V vs blind interval (boxplot, grouped by orbit type)
        ax = axes[1, 0]
        for orb in orbit_types:
            blind_vals = []
            dv_vals_by_blind = []
            for blind in blind_intervals:
                dv_vals = [m.total_dv for m in self.metrics_list
                           if m.orbit_type == orb and m.blind_interval_days == blind]
                if dv_vals:
                    blind_vals.append(blind)
                    dv_vals_by_blind.append(dv_vals)
            if blind_vals:
                bp = ax.boxplot(dv_vals_by_blind, positions=blind_vals, widths=0.6, patch_artist=True)
                # Connect means with a line
                means = [np.mean(vals) for vals in dv_vals_by_blind]
                ax.plot(blind_vals, means, 'o-', label=orb)
        ax.set_xlabel('Blind Interval (days)')
        ax.set_ylabel('Total delta-V (m/s)')
        ax.set_title('Total delta-V vs Blind Interval')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Distribution of average delta-V per day
        ax = axes[1, 1]
        all_dv = [m.avg_dv_per_day for m in self.metrics_list if not np.isnan(m.avg_dv_per_day)]
        ax.hist(all_dv, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Average delta-V per day (m/s/day)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Fuel Consumption')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "fuel_analysis_plots.png")
        plt.savefig(save_path, dpi=150)
        print(f"[FuelAnalyzer] Plot saved to {save_path}")
        # Do not call plt.show() to avoid warnings

    def generate_report(self):
        """
        Generate fuel analysis report (text).
        """
        if not self.metrics_list:
            print("[FuelAnalyzer] No data, cannot generate report.")
            return

        # Group by orbit type, blind interval, gain scale
        orbit_types = sorted(set(m.orbit_type for m in self.metrics_list))
        blind_intervals = sorted(set(m.blind_interval_days for m in self.metrics_list))
        gain_scales = sorted(set(m.control_gain_scale for m in self.metrics_list))

        lines = []
        lines.append("=" * 70)
        lines.append("Fuel Consumption Analysis Report")
        lines.append("=" * 70)
        lines.append(f"Total runs: {len(self.metrics_list)}")
        lines.append(f"Orbit types: {orbit_types}")
        lines.append(f"Blind intervals (days): {blind_intervals}")
        lines.append(f"Control gain scales: {gain_scales}")
        lines.append("")

        # Summary by orbit type
        lines.append("--- Summary by Orbit Type ---")
        for orb in orbit_types:
            dv_vals = [m.total_dv for m in self.metrics_list if m.orbit_type == orb and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"{orb}:")
                lines.append(f"  Mean total delta-V: {np.mean(dv_vals):.4f} +/- {np.std(dv_vals):.4f} m/s")
                lines.append(f"  Median total delta-V: {np.median(dv_vals):.4f} m/s")
                lines.append(f"  Min/Max: {np.min(dv_vals):.4f} / {np.max(dv_vals):.4f} m/s")
                lines.append(f"  Sample size: {len(dv_vals)}")
                lines.append("")

        # Summary by blind interval (all orbits)
        lines.append("--- Summary by Blind Interval (All Orbits) ---")
        for blind in blind_intervals:
            dv_vals = [m.total_dv for m in self.metrics_list
                       if m.blind_interval_days == blind and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"Blind {blind} days:")
                lines.append(f"  Mean total delta-V: {np.mean(dv_vals):.4f} +/- {np.std(dv_vals):.4f} m/s")
                lines.append(f"  Sample size: {len(dv_vals)}")
                lines.append("")

        # Summary by control gain (all orbits)
        lines.append("--- Summary by Control Gain (All Orbits) ---")
        for gain in gain_scales:
            dv_vals = [m.total_dv for m in self.metrics_list
                       if m.control_gain_scale == gain and not np.isnan(m.total_dv)]
            if dv_vals:
                lines.append(f"Gain {gain}:")
                lines.append(f"  Mean total delta-V: {np.mean(dv_vals):.4f} +/- {np.std(dv_vals):.4f} m/s")
                lines.append(f"  Sample size: {len(dv_vals)}")
                lines.append("")

        lines.append("=" * 70)

        report_str = "\n".join(lines)
        print(report_str)

        # Save report
        report_path = os.path.join(self.output_dir, "fuel_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        print(f"[FuelAnalyzer] Report saved to {report_path}")


def main():
    """
    Example run with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Fuel Consumption Parameter Scan")
    parser.add_argument("--n_processes", type=int, default=None,
                        help="Number of parallel processes (default: cpu_count)")
    parser.add_argument("--n_repeats", type=int, default=2,
                        help="Number of repeated runs per parameter combination (default: 2)")
    parser.add_argument("--simulation_days", type=float, default=30,
                        help="Simulation duration in days (default: 30)")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                        help="Output directory for results (default: analysis_results)")
    parser.add_argument("--data_dir", type=str, default="data/fuel_analysis",
                        help="Data directory for simulation outputs (default: data/fuel_analysis)")
    args = parser.parse_args()

    # Custom scan parameters (adjust as needed)
    scan_params = {
        "orbit_type": ["Halo", "Keplerian"],          # Orbit types
        "control_gain_scale": [0.5, 1.0, 2.0],        # Control gain scaling factor
        "blind_interval_days": [0, 3, 7],             # Blind interval duration (days)
        "simulation_days": [args.simulation_days]     # Simulation duration (days)
    }

    # Base configuration (suppress verbose output, disable CSV, disable backup)
    base_config = {
        "time_step": 10.0,
        "log_buffer_size": 200,
        "enable_visualization": False,
        "data_dir": args.data_dir,
        "log_level": "WARNING",
        "integrator": "rk4",   # Can be changed to "rk45" for variable-step integration
        "verbose": False,      # Suppress simulation internal output
        "save_fuel_bill": False,  # Do not generate CSV fuel bills
        "log_backup": False,   # Disable file backup for parallel runs
    }

    # Create analyzer
    analyzer = FuelAnalyzer(
        base_config=base_config,
        scan_params=scan_params,
        output_dir=args.output_dir,
        n_repeats=args.n_repeats,
        n_processes=args.n_processes
    )

    # Run scan
    analyzer.run_scan()

    # Generate plots and report
    analyzer.plot_results()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
