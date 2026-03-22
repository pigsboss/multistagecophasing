# analysis/control_robustness_analysis.py
"""
Control Algorithm Robustness Monte Carlo Analysis
By varying parameters such as initial errors, measurement noise, model errors, etc.,
statistics of control convergence time, steady-state errors, fuel consumption are collected.
Statistical charts and performance reports are output.
"""

import os
import sys
import time
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

# Set matplotlib backend to non-interactive (avoid plt.show() warnings)
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.utils.logger import HDF5Logger


@dataclass
class RobustnessMetrics:
    """Performance metrics for a single simulation run."""
    mission_id: str
    position_error_final: float      # Final position error (m)
    velocity_error_final: float      # Final velocity error (m/s)
    accumulated_dv: float            # Total delta-V (m/s)
    max_control_force: float         # Maximum control force (N)
    rms_position_error: float        # RMS position error (m)
    rms_velocity_error: float        # RMS velocity error (m/s)
    convergence_time: float          # Convergence time (s), when error falls below threshold
    simulation_time: float           # Actual simulation elapsed time (s)
    parameters: Dict[str, Any] = field(default_factory=dict)


def _run_single_simulation(config: Dict[str, Any], mission_id: str) -> Optional[RobustnessMetrics]:
    """
    Run a single simulation and return performance metrics.
    This function is designed to be executed in a worker process.

    Args:
        config: Simulation configuration dictionary
        mission_id: Unique identifier for this simulation run (to avoid filename conflicts)

    Returns:
        RobustnessMetrics if successful, else None
    """
    # Inject mission_id into config (simulation will use it if BaseSimulation supports it)
    config["mission_id"] = mission_id

    try:
        sim = SunEarthL2L1Simulation(config)
        start_time = time.time()
        success = sim.run()
        elapsed = time.time() - start_time

        if not success:
            # Use logging instead of print to avoid cluttering in parallel runs
            # Since this runs in worker, we can just return None and rely on main progress bar
            return None

        # Get basic statistics from simulation object
        stats = sim.get_statistics()
        final_pos_err = stats.get("final_position_error", np.nan)
        final_vel_err = stats.get("final_velocity_error", np.nan)
        total_dv = stats.get("accumulated_dv", np.nan)

        # Load additional metrics from HDF5 file using absolute path
        h5_file_abs = os.path.abspath(sim.h5_file)
        if os.path.exists(h5_file_abs):
            try:
                with h5py.File(h5_file_abs, 'r') as f:
                    # Tracking errors
                    errors = f['tracking_errors'][:]
                    if len(errors) > 0:
                        rms_pos = np.sqrt(np.mean(errors[:, 0:3]**2))
                        rms_vel = np.sqrt(np.mean(errors[:, 3:6]**2))
                        # Convergence time: first time position error < 1000m
                        threshold = 1000.0
                        times = f['epochs'][:]
                        idx = np.where(np.linalg.norm(errors[:, 0:3], axis=1) < threshold)[0]
                        conv_time = times[idx[0]] if len(idx) > 0 else np.nan
                    else:
                        rms_pos = rms_vel = conv_time = np.nan

                    # Control forces
                    forces = f['control_forces'][:]
                    max_force = np.max(np.linalg.norm(forces, axis=1)) if len(forces) > 0 else np.nan
            except Exception as e:
                # Silent fail in worker, just set to nan
                rms_pos = rms_vel = conv_time = max_force = np.nan
        else:
            rms_pos = rms_vel = conv_time = max_force = np.nan

        metrics = RobustnessMetrics(
            mission_id=sim.mission_id,
            position_error_final=final_pos_err,
            velocity_error_final=final_vel_err,
            accumulated_dv=total_dv,
            max_control_force=max_force,
            rms_position_error=rms_pos,
            rms_velocity_error=rms_vel,
            convergence_time=conv_time,
            simulation_time=elapsed,
            parameters=config
        )
        return metrics

    except Exception as e:
        # Silent fail
        return None


class ControlRobustnessAnalyzer:
    """
    Control robustness analyzer.
    Runs Monte Carlo simulations, collects performance metrics,
    generates statistical reports and charts.
    """

    # Default simulation configuration
    DEFAULT_CONFIG = {
        "mission_name": "Robustness_Analysis",
        "simulation_days": 30,
        "time_step": 10.0,
        "log_buffer_size": 500,
        "log_compression": True,
        "enable_visualization": False,   # Disable visualization for speed
        "data_dir": "data/robustness_analysis",
        "log_level": "WARNING",
        "integrator": "rk4",             # Default integrator
        "verbose": False,                # Suppress verbose output in workers
        "save_fuel_bill": False,         # Do not generate separate CSV files
    }

    # Default parameter variation ranges
    DEFAULT_PARAM_VARY = {
        "initial_pos_error_scale": [0.5, 1.0, 2.0],      # Scale of initial position error (baseline 2000m)
        "initial_vel_error_scale": [0.5, 1.0, 2.0],      # Scale of initial velocity error (baseline 0.01m/s)
        "pos_noise_std": [2.5, 5.0, 10.0],               # Position noise standard deviation (m)
        "vel_noise_std": [0.0025, 0.005, 0.01],          # Velocity noise standard deviation (m/s)
        "control_gain_scale": [0.5, 1.0, 2.0]            # Control gain scaling factor
    }

    def __init__(self,
                 base_config: Optional[Dict] = None,
                 param_vary: Optional[Dict] = None,
                 output_dir: str = "analysis_results",
                 n_runs: int = 10,
                 n_processes: int = None):
        """
        Initialize the analyzer.

        Args:
            base_config: Base simulation configuration (overrides default)
            param_vary: Parameter variation dictionary, key = parameter name, value = list or array
            output_dir: Output directory for results
            n_runs: Number of repeated runs per parameter combination (for statistical stability)
            n_processes: Number of parallel processes (default: None -> cpu_count)
        """
        self.base_config = {**self.DEFAULT_CONFIG, **(base_config or {})}
        self.param_vary = param_vary or self.DEFAULT_PARAM_VARY
        self.output_dir = output_dir
        self.n_runs = n_runs
        self.n_processes = n_processes or mp.cpu_count()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Store all metrics
        self.metrics_list: List[RobustnessMetrics] = []

        print(f"[Analyzer] Initialized, output directory: {output_dir}")
        print(f"[Analyzer] Parameter variation keys: {list(self.param_vary.keys())}")
        print(f"[Analyzer] Runs per combination: {n_runs}")
        print(f"[Analyzer] Parallel processes: {self.n_processes}")

    def _build_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build simulation configuration from parameters.
        """
        config = self.base_config.copy()
        for key, value in params.items():
            config[key] = value
        return config

    def _generate_param_combinations(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Generate all parameter combinations (Cartesian product) and repeat n_runs times.
        Returns a list of (config, mission_id) pairs.
        """
        keys = list(self.param_vary.keys())
        values = [self.param_vary[k] for k in keys]
        combinations = []
        idx = 0
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            for _ in range(self.n_runs):
                # Generate a unique mission_id for each run
                mission_id = f"robustness_{idx:06d}"
                config = self._build_config(params)
                combinations.append((config, mission_id))
                idx += 1
        return combinations

    def run_monte_carlo(self):
        """
        Execute Monte Carlo simulations in parallel.
        """
        param_combos = self._generate_param_combinations()
        total_runs = len(param_combos)
        print(f"\n[Analyzer] Starting Monte Carlo simulation, total runs: {total_runs}")
        print(f"[Analyzer] Using {self.n_processes} parallel processes...")

        # Use a process pool to run simulations in parallel
        with Pool(processes=self.n_processes) as pool:
            # Prepare arguments for starmap (each item is (config, mission_id))
            args = [(config, mission_id) for config, mission_id in param_combos]

            # Use imap_unordered for progress bar
            results = []
            with tqdm(total=total_runs, desc="Running simulations", unit="run") as pbar:
                for result in pool.starmap(_run_single_simulation, args):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

                    # Periodically save intermediate results (every 50 completed runs)
                    if len(results) % 50 == 0:
                        self.metrics_list = results
                        self._save_intermediate_results(partial=True)

        # After all workers finish, save final results
        self.metrics_list = results
        self._save_intermediate_results(final=True)
        print(f"\n[Analyzer] Simulation finished, successful runs: {len(self.metrics_list)}/{total_runs}")

    def _save_intermediate_results(self, final: bool = False, partial: bool = False):
        """
        Save current results to a JSON file.
        """
        if not self.metrics_list:
            return
        if final:
            filename = "robustness_results_final.json"
        elif partial:
            filename = "robustness_results_partial.json"
        else:
            filename = "robustness_results.json"
        filepath = os.path.join(self.output_dir, filename)
        data = [asdict(m) for m in self.metrics_list]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n[Analyzer] Saved {len(self.metrics_list)} results to {filepath}")

    def plot_results(self):
        """
        Generate statistical charts.
        """
        if not self.metrics_list:
            print("[Analyzer] No data, cannot plot.")
            return

        # Extract key metrics
        metrics = self.metrics_list
        pos_err_scales = []
        vel_err_scales = []
        final_pos_err = []
        final_vel_err = []
        total_dv = []
        max_force = []

        for m in metrics:
            pos_scale = m.parameters.get("initial_pos_error_scale", 1.0)
            vel_scale = m.parameters.get("initial_vel_error_scale", 1.0)
            pos_err_scales.append(pos_scale)
            vel_err_scales.append(vel_scale)
            final_pos_err.append(m.position_error_final)
            final_vel_err.append(m.velocity_error_final)
            total_dv.append(m.accumulated_dv)
            max_force.append(m.max_control_force)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Control Algorithm Robustness Analysis", fontsize=14, fontweight='bold')

        # 1. Final position error vs initial position error scaling factor
        ax = axes[0, 0]
        unique_scales = sorted(set(pos_err_scales))
        pos_err_by_scale = [np.array([final_pos_err[i] for i, s in enumerate(pos_err_scales) if s == sc]) for sc in unique_scales]
        bp = ax.boxplot(pos_err_by_scale, positions=unique_scales, widths=0.6, patch_artist=True)
        ax.set_xlabel('Initial Position Error Scaling Factor')
        ax.set_ylabel('Final Position Error (m)')
        ax.set_title('Final Position Error vs Initial Position Error')
        ax.grid(True, alpha=0.3)

        # 2. Total delta-V vs initial velocity error scaling factor
        ax = axes[0, 1]
        unique_vel_scales = sorted(set(vel_err_scales))
        dv_by_vel = [np.array([total_dv[i] for i, s in enumerate(vel_err_scales) if s == sc]) for sc in unique_vel_scales]
        bp = ax.boxplot(dv_by_vel, positions=unique_vel_scales, widths=0.6, patch_artist=True)
        ax.set_xlabel('Initial Velocity Error Scaling Factor')
        ax.set_ylabel('Total delta-V (m/s)')
        ax.set_title('Fuel Consumption vs Initial Velocity Error')
        ax.grid(True, alpha=0.3)

        # 3. Final position error histogram
        ax = axes[1, 0]
        ax.hist(final_pos_err, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Final Position Error (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Position Error')
        ax.grid(True, alpha=0.3)

        # 4. Total delta-V histogram
        ax = axes[1, 1]
        ax.hist(total_dv, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
        ax.set_xlabel('Total delta-V (m/s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Fuel Consumption')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "control_robustness_plots.png")
        plt.savefig(save_path, dpi=150)
        print(f"[Analyzer] Plot saved to {save_path}")
        # Do not call plt.show() to avoid warnings

    def generate_report(self):
        """
        Generate a statistical report.
        """
        if not self.metrics_list:
            print("[Analyzer] No data, cannot generate report.")
            return

        # Extract metrics with non-NaN values
        final_pos_err = np.array([m.position_error_final for m in self.metrics_list if not np.isnan(m.position_error_final)])
        final_vel_err = np.array([m.velocity_error_final for m in self.metrics_list if not np.isnan(m.velocity_error_final)])
        total_dv = np.array([m.accumulated_dv for m in self.metrics_list if not np.isnan(m.accumulated_dv)])
        max_force = np.array([m.max_control_force for m in self.metrics_list if not np.isnan(m.max_control_force)])
        rms_pos = np.array([m.rms_position_error for m in self.metrics_list if not np.isnan(m.rms_position_error)])
        conv_time = np.array([m.convergence_time for m in self.metrics_list if not np.isnan(m.convergence_time)])

        report = []
        report.append("=" * 60)
        report.append("Control Algorithm Robustness Analysis Report")
        report.append("=" * 60)
        report.append(f"Total runs: {len(self.metrics_list)}")
        report.append(f"Valid data points: {len(final_pos_err)}")
        report.append("")

        # Final position error
        report.append("--- Final Position Error (m) ---")
        if len(final_pos_err) > 0:
            report.append(f"  Mean: {np.mean(final_pos_err):.2f}")
            report.append(f"  Std: {np.std(final_pos_err):.2f}")
            report.append(f"  Min: {np.min(final_pos_err):.2f}")
            report.append(f"  Max: {np.max(final_pos_err):.2f}")
            report.append(f"  Median: {np.median(final_pos_err):.2f}")
        else:
            report.append("  No valid data")
        report.append("")

        # Final velocity error
        report.append("--- Final Velocity Error (m/s) ---")
        if len(final_vel_err) > 0:
            report.append(f"  Mean: {np.mean(final_vel_err):.4f}")
            report.append(f"  Std: {np.std(final_vel_err):.4f}")
        else:
            report.append("  No valid data")
        report.append("")

        # Total delta-V
        report.append("--- Total delta-V (m/s) ---")
        if len(total_dv) > 0:
            report.append(f"  Mean: {np.mean(total_dv):.4f}")
            report.append(f"  Std: {np.std(total_dv):.4f}")
            report.append(f"  Min: {np.min(total_dv):.4f}")
            report.append(f"  Max: {np.max(total_dv):.4f}")
        else:
            report.append("  No valid data")
        report.append("")

        # Maximum control force
        report.append("--- Maximum Control Force (N) ---")
        if len(max_force) > 0:
            report.append(f"  Mean: {np.mean(max_force):.2f}")
            report.append(f"  Max: {np.max(max_force):.2f}")
        else:
            report.append("  No valid data")
        report.append("")

        # RMS position error
        report.append("--- RMS Position Error (m) ---")
        if len(rms_pos) > 0:
            report.append(f"  Mean: {np.mean(rms_pos):.2f}")
            report.append(f"  Std: {np.std(rms_pos):.2f}")
        else:
            report.append("  No valid data")
        report.append("")

        # Convergence time
        report.append("--- Convergence Time (days) ---")
        if len(conv_time) > 0:
            report.append(f"  Mean: {np.mean(conv_time)/86400:.2f}")
            report.append(f"  Std: {np.std(conv_time)/86400:.2f}")
        else:
            report.append("  No valid data")
        report.append("")

        report.append("=" * 60)

        report_str = "\n".join(report)
        print(report_str)

        # Save report
        report_path = os.path.join(self.output_dir, "robustness_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        print(f"[Analyzer] Report saved to {report_path}")


def main():
    """
    Example run.
    """
    # Custom parameter variation ranges
    param_vary = {
        "initial_pos_error_scale": [0.5, 1.0, 2.0],
        "initial_vel_error_scale": [0.5, 1.0, 2.0],
        "pos_noise_std": [2.5, 5.0, 10.0],
        "vel_noise_std": [0.0025, 0.005, 0.01],
        # "control_gain_scale": [0.5, 1.0, 2.0]   # Uncomment if control gain scaling is supported
    }

    # Base configuration (suppress verbose output, disable CSV files)
    base_config = {
        "simulation_days": 10,
        "time_step": 10.0,
        "log_buffer_size": 100,
        "enable_visualization": False,
        "data_dir": "data/robustness_analysis",
        "integrator": "rk4",   # Can be changed to "rk45" for variable-step integration
        "verbose": False,      # Suppress simulation internal output
        "save_fuel_bill": False,  # Do not generate CSV fuel bills
    }

    # Create analyzer
    analyzer = ControlRobustnessAnalyzer(
        base_config=base_config,
        param_vary=param_vary,
        output_dir="analysis_results",
        n_runs=3,               # Runs per combination (increase for real analysis)
        n_processes=4           # Adjust based on available CPU cores
    )

    # Run Monte Carlo
    analyzer.run_monte_carlo()

    # Generate plots and report
    analyzer.plot_results()
    analyzer.generate_report()


if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows
    main()