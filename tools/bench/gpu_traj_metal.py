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
            "memory_usage": self.memory_usage,
            "notes": self.notes,
            "result_value": self.result_value,
        }


# ---------- core computation – single path, JAX jit ---------- #
from functools import partial

@partial(jax.jit, static_argnames=('steps',))
def _integrate_single_path(steps: int, key: jax.Array) -> jnp.float32:
    """Integrate a single path; 100 % match to OpenCL kernel logic."""
    def body_fn(carry, step):
        x, integral = carry
        rng_key = jax.random.fold_in(key, step)

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

        return (x, integral), None

    _, integral = jax.lax.scan(body_fn, (0.0, 0.0), jnp.arange(steps))
    return integral


_vmap_integrate = jax.vmap(_integrate_single_path, in_axes=(None, 0))


def _run_paths(steps: int, paths: int, key: jax.Array) -> jnp.float32:
    """Run all paths and return mean integral."""
    keys = jax.random.split(key, paths)
    integrals = _vmap_integrate(steps, keys)
    return jnp.mean(integrals)


_run_paths_jit = jax.jit(_run_paths, static_argnums=(0, 1))


# ---------- public benchmark routine ---------- #
def run_benchmark(
    steps: int = 10_000,
    paths: int = 1_000,
    warmup_iterations: int = 3,
    test_iterations: int = 10,
) -> BenchmarkResult:
    """Return BenchmarkResult matching gpu_traj_cl.py signature."""
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX not available")

    device = jax.devices()[0]  # Metal or fallback
    impl_name = f"JAX {device.platform.upper()} ({device.device_kind})"

    # Warm-up
    key = random.PRNGKey(42)
    for _ in range(warmup_iterations):
        _ = _run_paths_jit(steps, paths, key).block_until_ready()

    # Timed runs
    times: List[float] = []
    for i in range(test_iterations):
        key = random.PRNGKey(12345 + i)
        start = time.perf_counter()
        result = _run_paths_jit(steps, paths, key).block_until_ready()
        times.append(time.perf_counter() - start)

    return BenchmarkResult(
        task_name="Trajectory Integral (GPU)",
        implementation=impl_name,
        execution_times=times,
        precision="fp32",
        notes=f"Steps={steps}, Paths={paths}",
        result_value=float(result),
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
    )

    BenchmarkReporter.print_results([result], output_file=args.output)

    # Quick validation vs CPU reference (small scale)
    print("\n" + "=" * 80)
    print("Validating against CPU reference (small scale)...")
    ref_steps, ref_paths = min(steps, 1000), min(paths, 100)
    from cpu import PathIntegralBenchmark  # local import to avoid circular deps
    ref = PathIntegralBenchmark.python_implementation(steps=ref_steps, paths=ref_paths)
    gpu = float(_run_paths_jit(ref_steps, ref_paths, random.PRNGKey(999)).block_until_ready())
    diff = abs(ref - gpu)
    print(f"  Reference: {ref:.6f}")
    print(f"  GPU:       {gpu:.6f}")
    print(f"  Diff:      {diff:.2e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
