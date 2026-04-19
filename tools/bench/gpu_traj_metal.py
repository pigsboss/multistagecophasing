#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Path-Integral Benchmark – JAX/Metal backend
Algorithmically identical to gpu_traj_cl.py (OpenCL) and cpu.py
All stdout in English per MCPC standards.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Error: JAX not installed.  pip install jax jaxlib", file=sys.stderr)
    sys.exit(1)

# ---------- shared result container (identical to cpu.py) ---------- #
@dataclass
class BenchmarkResult:
    """Benchmark result data class"""
    task_name: str
    implementation: str
    execution_times: List[float]  # seconds
    min_time: float = field(init=False)
    max_time: float = field(init=False)
    avg_time: float = field(init=False)
    median_time: float = field(init=False)
    std_time: float = field(init=False)
    memory_usage: Optional[float] = None  # MB, optional
    notes: str = ""
    precision: str = "fp32"
    result_value: Optional[float] = None

    def __post_init__(self):
        self.min_time = min(self.execution_times)
        self.max_time = max(self.execution_times)
        self.avg_time = float(np.mean(self.execution_times))
        self.median_time = float(np.median(self.execution_times))
        if len(self.execution_times) > 1:
            self.std_time = float(np.std(self.execution_times))
        else:
            self.std_time = 0.0

    @property
    def iterations_per_second(self) -> float:
        return 1.0 / self.avg_time if self.avg_time > 0 else float('inf')
    
    @property
    def paths_per_second(self) -> float:
        """Return number of paths processed per second"""
        import re
        notes = self.notes or ""
        match = re.search(r'Paths=(\d+)', notes)
        if match:
            paths = int(match.group(1))
            return paths / self.avg_time if self.avg_time > 0 else 0.0
        return 0.0
    
    @property
    def total_ops_per_second(self) -> float:
        """Return steps * paths / time"""
        import re
        notes = self.notes or ""
        steps_match = re.search(r'Steps=(\d+)', notes)
        paths_match = re.search(r'Paths=(\d+)', notes)
        if steps_match and paths_match:
            steps = int(steps_match.group(1))
            paths = int(paths_match.group(1))
            return (steps * paths) / self.avg_time if self.avg_time > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "implementation": self.implementation,
            "precision": self.precision,
            "execution_times": self.execution_times,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "std_time": self.std_time,
            "iterations_per_second": self.iterations_per_second,
            "paths_per_second": self.paths_per_second,
            "total_ops_per_second": self.total_ops_per_second,
            "memory_usage": self.memory_usage,
            "notes": self.notes,
            "result_value": self.result_value,
        }


# ---------- core computation – single path, JAX jit ---------- #
from functools import partial

# LCG implementation matching OpenCL's random number generator
def _lcg_next(state: jnp.uint32) -> jnp.uint32:
    """LCG matching OpenCL implementation: state * 1103515245 + 12345"""
    return state * jnp.uint32(1103515245) + jnp.uint32(12345)

def _lcg_random(state: jnp.uint32) -> jnp.float32:
    """Generate random float in [0, 1) matching OpenCL implementation"""
    new_state = _lcg_next(state)
    # Take lower 31 bits (0x7fffffffu in OpenCL)
    val = new_state & jnp.uint32(0x7fffffffu)
    return val.astype(jnp.float32) / jnp.float32(0x7fffffffu)

@partial(jax.jit, static_argnames=('steps', 'use_lcg'))
def _integrate_single_path(steps: int, key: jax.Array, use_lcg: bool = False) -> jnp.float32:
    """Integrate a single path; 100 % match to OpenCL kernel logic."""
    def body_fn_jax(carry, step):
        x, integral, rng_key = carry
        rng_key = jax.random.fold_in(rng_key, step)

        # ---- branching logic identical to OpenCL ---- #
        weight = jnp.where(
            x < -1.0,
            0.1,
            jnp.where(
                x < 0.0,
                0.3 * x + 0.4,
                jnp.where(x < 1.0, 0.5 * (1.0 - x * x), 0.2),
            ),
        )

        # ---- alternating factor ---- #
        factor = jnp.where(
            step % 2 == 0,
            1.0 + 0.1 * x,
            1.0 - 0.1 * x,
        )

        delta = 1.0 / steps
        integral += weight * delta * factor

        # ---- random walk + boundary clip ---- #
        rand_val = jax.random.uniform(rng_key, dtype=jnp.float32)
        x += rand_val * 0.1
        x = jnp.clip(x, -2.0, 2.0)

        return (x, integral, rng_key), None

    def body_fn_lcg(carry, step):
        x, integral, rng_state = carry
        
        # ---- branching logic identical to OpenCL ---- #
        weight = jnp.where(
            x < -1.0,
            0.1,
            jnp.where(
                x < 0.0,
                0.3 * x + 0.4,
                jnp.where(x < 1.0, 0.5 * (1.0 - x * x), 0.2),
            ),
        )

        # ---- alternating factor ---- #
        factor = jnp.where(
            step % 2 == 0,
            1.0 + 0.1 * x,
            1.0 - 0.1 * x,
        )

        delta = 1.0 / steps
        integral += weight * delta * factor

        # ---- random walk + boundary clip (using LCG) ---- #
        rand_val = _lcg_random(rng_state)
        x += rand_val * 0.1
        x = jnp.clip(x, -2.0, 2.0)
        
        # Update LCG state for next iteration
        rng_state = _lcg_next(rng_state)

        return (x, integral, rng_state), None
    
    if use_lcg:
        # For LCG, key is used as initial state
        init_state = key.astype(jnp.uint32)
        carry, _ = jax.lax.scan(body_fn_lcg, (0.0, 0.0, init_state), jnp.arange(steps))
    else:
        # For JAX RNG
        carry, _ = jax.lax.scan(body_fn_jax, (0.0, 0.0, key), jnp.arange(steps))
    
    return carry[1]


def _create_vmap_integrate(use_lcg: bool = False):
    """Create vmap function with specified RNG method"""
    def vmap_func(steps: int, keys: jax.Array) -> jnp.float32:
        if use_lcg:
            # For LCG, keys are initial states
            integrals = jax.vmap(_integrate_single_path, in_axes=(None, 0, None))(
                steps, keys, use_lcg
            )
        else:
            # For JAX RNG
            integrals = jax.vmap(_integrate_single_path, in_axes=(None, 0, None))(
                steps, keys, use_lcg
            )
        return jnp.mean(integrals)
    return vmap_func

def _run_paths(steps: int, paths: int, key: jax.Array, use_lcg: bool = False) -> jnp.float32:
    """Run all paths and return mean integral."""
    if use_lcg:
        # For LCG, create initial states: base_seed + path_id
        base_seed = key.astype(jnp.uint32)
        # Create array of initial states: [base_seed, base_seed+1, ..., base_seed+paths-1]
        states = base_seed + jnp.arange(paths, dtype=jnp.uint32)
        vmap_func = _create_vmap_integrate(use_lcg=True)
        return vmap_func(steps, states)
    else:
        # For JAX RNG
        keys = jax.random.split(key, paths)
        vmap_func = _create_vmap_integrate(use_lcg=False)
        return vmap_func(steps, keys)

_run_paths_jit = jax.jit(_run_paths, static_argnums=(0, 1, 3))


# ---------- public benchmark routine ---------- #
def run_benchmark(
    steps: int = 10_000,
    paths: int = 1_000,
    warmup_iterations: int = 3,
    test_iterations: int = 10,
    use_lcg: bool = False,
) -> BenchmarkResult:
    """Return BenchmarkResult matching gpu_traj_cl.py signature."""
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    device = jax.devices()[0]  # Metal or fallback
    rng_method = "LCG" if use_lcg else "JAX-RNG"
    impl_name = f"JAX {device.platform.upper()} ({device.device_kind}, {rng_method})"

    # Warm-up
    if use_lcg:
        key = jnp.uint32(42)
    else:
        key = random.PRNGKey(42)
    
    for _ in range(warmup_iterations):
        _ = _run_paths_jit(steps, paths, key, use_lcg).block_until_ready()

    # Timed runs
    times: List[float] = []
    results_list: List[float] = []
    
    for i in range(test_iterations):
        if use_lcg:
            key = jnp.uint32(12345 + i)
        else:
            key = random.PRNGKey(12345 + i)
        
        start = time.perf_counter()
        result = _run_paths_jit(steps, paths, key, use_lcg).block_until_ready()
        times.append(time.perf_counter() - start)
        results_list.append(float(result))

    # Calculate throughput metrics
    avg_time = float(np.mean(times)) if times else 0.0
    paths_per_sec = paths / avg_time if avg_time > 0 else 0.0
    total_ops_per_sec = (steps * paths) / avg_time if avg_time > 0 else 0.0
    
    # Use the last result
    final_result = results_list[-1] if results_list else 0.0

    return BenchmarkResult(
        task_name="Trajectory Integral (GPU)",
        implementation=impl_name,
        execution_times=times,
        precision="fp32",
        notes=f"Steps={steps}, Paths={paths}, RNG={rng_method}, "
              f"Throughput={paths_per_sec:.0f} paths/sec, "
              f"Ops={total_ops_per_sec:.0f} steps·paths/sec",
        result_value=final_result,
    )


# ---------- console / JSON reporter (shared schema) ---------- #
class BenchmarkReporter:
    @staticmethod
    def print_results(results: List[BenchmarkResult], output_file: Optional[str] = None) -> None:
        output_data = {
            "benchmark_results": [r.to_dict() for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "jax_version": jax.__version__,
                "metal_backend": jax.devices()[0].platform == "metal",
            },
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("\n" + "=" * 80)
            print("GPU Path Integral Benchmark Results – Metal/JAX")
            print("=" * 80)
            for r in results:
                print(f"\nPrecision: {r.precision.upper()}")
                print("-" * 60)
                print(f"  Device: {r.implementation}")
                print(f"  Time: {r.avg_time:.4f}s (min:{r.min_time:.4f}s, "
                      f"max:{r.max_time:.4f}s, med:{r.median_time:.4f}s)")
                print(f"  Iter/s: {r.iterations_per_second:.2f}")
                if r.std_time > 0:
                    print(f"  Std dev: {r.std_time:.6f}s")
                if r.result_value is not None:
                    print(f"  Result value: {r.result_value:.6f}")
                if r.notes:
                    print(f"  Notes: {r.notes}")

    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str) -> None:
        BenchmarkReporter.print_results(results, filename)


# ---------- CLI ---------- #
def main():
    parser = argparse.ArgumentParser(
        description="GPU Path-Integral Benchmark – JAX/Metal (algorithmically identical to OpenCL version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Default: fp32, (10000,1000), 10 repeats
  %(prog)s --size (5000,500)         # Custom problem size
  %(prog)s --repeats 20              # More timing repeats
  %(prog)s --output metal.json       # Save JSON for later comparison
        """,
    )

    parser.add_argument(
        "--size",
        type=str,
        default="(10000,1000)",
        help='Problem scale as (steps,paths) tuple, default (10000,1000)',
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed iterations (default 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default 3)",
    )
    parser.add_argument(
        "--use-lcg",
        action="store_true",
        help="Use LCG random number generator (matches OpenCL implementation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON output file",
    )

    args = parser.parse_args()

    # Parse size tuple
    size_str = args.size.strip()
    if size_str.startswith("(") and size_str.endswith(")"):
        size_str = size_str[1:-1]
    steps, paths = map(int, size_str.split(","))

    print("Starting GPU Path-Integral Benchmark – Metal/JAX")
    print(f"Configuration: steps={steps}, paths={paths}, repeats={args.repeats}")
    print(f"JAX backend: {jax.devices()[0].platform} ({jax.devices()[0].device_kind})")

    result = run_benchmark(
        steps=steps,
        paths=paths,
        warmup_iterations=args.warmup,
        test_iterations=args.repeats,
        use_lcg=args.use_lcg,
    )

    BenchmarkReporter.print_results([result], output_file=args.output)

    # Quick validation vs CPU reference (small scale)
    print("\n" + "=" * 80)
    print("Validating against CPU reference (small scale)...")
    ref_steps, ref_paths = min(steps, 1000), min(paths, 100)
    from cpu import PathIntegralBenchmark  # local import to avoid circular deps
    ref = PathIntegralBenchmark.python_implementation(steps=ref_steps, paths=ref_paths)
    
    if args.use_lcg:
        gpu_key = jnp.uint32(999)
    else:
        gpu_key = random.PRNGKey(999)
    
    gpu = float(_run_paths_jit(ref_steps, ref_paths, gpu_key, args.use_lcg).block_until_ready())
    diff = abs(ref - gpu)
    print(f"  Reference: {ref:.6f}")
    print(f"  GPU:       {gpu:.6f}")
    print(f"  Diff:      {diff:.2e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
