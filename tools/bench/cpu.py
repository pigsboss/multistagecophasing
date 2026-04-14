#!/usr/bin/env python3
"""
CPU性能基准测试工具
测试三类计算任务在不同实现下的性能表现
"""

import time
import statistics
import json
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import sys

# 可选依赖处理
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available, related tests will be skipped")

try:
    from scipy import integrate, special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available, related tests will be skipped")

try:
    import numba
    from numba import jit, prange, vectorize, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, related tests will be skipped")


@dataclass
class BenchmarkResult:
    """Benchmark result data class"""
    task_name: str
    implementation: str
    execution_times: List[float]  # Store all execution times in seconds
    min_time: float = field(init=False)
    max_time: float = field(init=False)
    avg_time: float = field(init=False)
    median_time: float = field(init=False)
    std_time: float = field(init=False)
    memory_usage: Optional[float] = None  # MB
    notes: str = ""
    
    def __post_init__(self):
        """Calculate statistics from execution times"""
        self.min_time = min(self.execution_times)
        self.max_time = max(self.execution_times)
        self.avg_time = statistics.mean(self.execution_times)
        self.median_time = statistics.median(self.execution_times)
        self.std_time = statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0
    
    @property
    def iterations_per_second(self) -> float:
        """Iterations per second based on average time"""
        return 1.0 / self.avg_time if self.avg_time > 0 else float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "implementation": self.implementation,
            "execution_times": self.execution_times,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "std_time": self.std_time,
            "iterations_per_second": self.iterations_per_second,
            "memory_usage": self.memory_usage,
            "notes": self.notes
        }


class CPUBenchmark:
    """Main CPU benchmark class"""
    
    def __init__(self, warmup_iterations: int = 3, test_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, func: Callable, task_name: str, 
                     impl_name: str, **kwargs) -> BenchmarkResult:
        """
        Run benchmark for a given function
        
        Args:
            func: Function to benchmark
            task_name: Name of the task
            impl_name: Name of the implementation
            **kwargs: Arguments passed to the function
        """
        # Warm-up runs
        for _ in range(self.warmup_iterations):
            func(**kwargs)
        
        # Actual test runs, collect all execution times
        execution_times = []
        for _ in range(self.test_iterations):
            start_time = time.perf_counter()
            func(**kwargs)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
        
        return BenchmarkResult(
            task_name=task_name,
            implementation=impl_name,
            execution_times=execution_times,
            notes=f"Based on {self.test_iterations} iterations"
        )


class PathIntegralBenchmark:
    """路径积分问题的基准测试"""
    
    @staticmethod
    def python_implementation(steps: int = 10000, paths: int = 1000) -> float:
        """纯Python实现"""
        result = 0.0
        for path in range(paths):
            integral = 0.0
            x = 0.0
            for step in range(steps):
                # 模拟路径积分中的复杂分支逻辑
                if x < -1.0:
                    weight = 0.1
                elif x < 0.0:
                    weight = 0.3 * x + 0.4
                elif x < 1.0:
                    weight = 0.5 * (1.0 - x**2)
                else:
                    weight = 0.2
                
                # 更新积分值（包含复杂条件判断）
                delta = 1.0 / steps
                if step % 2 == 0:
                    integral += weight * delta * (1.0 + 0.1 * x)
                else:
                    integral += weight * delta * (1.0 - 0.1 * x)
                
                # 更新路径（带边界检查）
                x += random.random() * 0.1
                if x > 2.0:
                    x = 2.0
                elif x < -2.0:
                    x = -2.0
            
            result += integral / paths
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=False)
    def numba_implementation(steps: int = 10000, paths: int = 1000) -> float:
        """Numba实现（串行）"""
        result = 0.0
        for path in range(paths):
            integral = 0.0
            x = 0.0
            for step in range(steps):
                # 与Python实现相同的逻辑
                if x < -1.0:
                    weight = 0.1
                elif x < 0.0:
                    weight = 0.3 * x + 0.4
                elif x < 1.0:
                    weight = 0.5 * (1.0 - x**2)
                else:
                    weight = 0.2
                
                delta = 1.0 / steps
                if step % 2 == 0:
                    integral += weight * delta * (1.0 + 0.1 * x)
                else:
                    integral += weight * delta * (1.0 - 0.1 * x)
                
                x += random.random() * 0.1
                if x > 2.0:
                    x = 2.0
                elif x < -2.0:
                    x = -2.0
            
            result += integral / paths
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def numba_parallel_implementation(steps: int = 10000, paths: int = 1000) -> float:
        """Numba实现（并行）"""
        result = 0.0
        # 使用prange进行并行化
        for path in prange(paths):
            integral = 0.0
            x = 0.0
            for step in range(steps):
                if x < -1.0:
                    weight = 0.1
                elif x < 0.0:
                    weight = 0.3 * x + 0.4
                elif x < 1.0:
                    weight = 0.5 * (1.0 - x**2)
                else:
                    weight = 0.2
                
                delta = 1.0 / steps
                if step % 2 == 0:
                    integral += weight * delta * (1.0 + 0.1 * x)
                else:
                    integral += weight * delta * (1.0 - 0.1 * x)
                
                x += random.random() * 0.1
                if x > 2.0:
                    x = 2.0
                elif x < -2.0:
                    x = -2.0
            
            result += integral / paths
        
        return result
    
    @staticmethod
    def numpy_scipy_implementation(steps: int = 10000, paths: int = 1000) -> float:
        """NumPy/SciPy实现"""
        if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
            return 0.0
        
        import numpy as np
        from scipy import integrate
        
        # 使用向量化操作
        def integrand(x):
            conditions = [
                (x < -1.0, 0.1),
                (x < 0.0, 0.3 * x + 0.4),
                (x < 1.0, 0.5 * (1.0 - x**2)),
                (True, 0.2)
            ]
            
            result = np.select([c[0] for c in conditions], [c[1] for c in conditions])
            return result
        
        results = []
        for _ in range(paths):
            # 生成随机路径
            random_walk = np.cumsum(np.random.randn(steps) * 0.1)
            random_walk = np.clip(random_walk, -2.0, 2.0)
            
            # 计算积分
            integral = integrate.simpson(integrand(random_walk), dx=1.0/steps)
            results.append(integral)
        
        return np.mean(results)


class MonteCarloBenchmark:
    """蒙特卡洛模拟的基准测试"""
    
    @staticmethod
    def python_implementation(samples: int = 10000000) -> float:
        """纯Python实现"""
        inside = 0
        
        for _ in range(samples):
            x = random.random()
            y = random.random()
            if x*x + y*y <= 1.0:
                inside += 1
        
        return 4.0 * inside / samples
    
    @staticmethod
    def numpy_implementation(samples: int = 10000000) -> float:
        """NumPy实现（使用OpenBLAS/GSL）"""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        import numpy as np
        x = np.random.random(samples)
        y = np.random.random(samples)
        
        # 使用向量化操作
        distances = x**2 + y**2
        inside = np.sum(distances <= 1.0)
        
        return 4.0 * inside / samples
    
    @staticmethod
    @vectorize([float64(float64, float64)], target='cpu')
    def numba_vectorized(x: float, y: float) -> float:
        """Numba向量化函数"""
        return 1.0 if x*x + y*y <= 1.0 else 0.0
    
    @staticmethod
    def numba_implementation(samples: int = 10000000) -> float:
        """Numba实现"""
        if not NUMBA_AVAILABLE or not NUMPY_AVAILABLE:
            return 0.0
        
        import numpy as np
        x = np.random.random(samples)
        y = np.random.random(samples)
        
        # 使用向量化函数
        results = MonteCarloBenchmark.numba_vectorized(x, y)
        inside = np.sum(results)
        
        return 4.0 * inside / samples
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def numba_parallel_implementation(samples: int = 10000000) -> float:
        """Numba并行实现"""
        inside = 0
        
        # 使用prange并行循环
        for i in prange(samples):
            x = random.random()
            y = random.random()
            if x*x + y*y <= 1.0:
                inside += 1
        
        return 4.0 * inside / samples


class NBodyBenchmark:
    """N体动力学模拟的基准测试"""
    
    def __init__(self, num_bodies: int = 1000, steps: int = 100):
        self.num_bodies = num_bodies
        self.steps = steps
        
        # 初始化位置和速度
        if NUMPY_AVAILABLE:
            import numpy as np
            self.positions = np.random.randn(num_bodies, 3) * 10.0
            self.velocities = np.random.randn(num_bodies, 3) * 0.1
            self.masses = np.random.rand(num_bodies) * 10.0
        else:
            self.positions = [[random.gauss(0, 10) for _ in range(3)] 
                              for _ in range(num_bodies)]
            self.velocities = [[random.gauss(0, 0.1) for _ in range(3)] 
                               for _ in range(num_bodies)]
            self.masses = [random.random() * 10 for _ in range(num_bodies)]
    
    def python_implementation(self) -> None:
        """纯Python实现"""
        positions = [list(p) for p in self.positions]
        velocities = [list(v) for v in self.velocities]
        masses = list(self.masses)
        
        G = 6.67430e-11  # 引力常数
        dt = 0.01
        
        for _ in range(self.steps):
            # 计算加速度
            accelerations = [[0.0, 0.0, 0.0] for _ in range(self.num_bodies)]
            
            for i in range(self.num_bodies):
                for j in range(self.num_bodies):
                    if i != j:
                        # 计算距离
                        dx = positions[j][0] - positions[i][0]
                        dy = positions[j][1] - positions[i][1]
                        dz = positions[j][2] - positions[i][2]
                        
                        r2 = dx*dx + dy*dy + dz*dz
                        r = math.sqrt(r2)
                        
                        # 避免除以零
                        if r > 1e-10:
                            force = G * masses[i] * masses[j] / r2
                            ax = force * dx / (r * masses[i])
                            ay = force * dy / (r * masses[i])
                            az = force * dz / (r * masses[i])
                            
                            accelerations[i][0] += ax
                            accelerations[i][1] += ay
                            accelerations[i][2] += az
            
            # 更新位置和速度
            for i in range(self.num_bodies):
                velocities[i][0] += accelerations[i][0] * dt
                velocities[i][1] += accelerations[i][1] * dt
                velocities[i][2] += accelerations[i][2] * dt
                
                positions[i][0] += velocities[i][0] * dt
                positions[i][1] += velocities[i][1] * dt
                positions[i][2] += velocities[i][2] * dt
    
    def numpy_implementation(self) -> None:
        """NumPy实现（向量化）"""
        if not NUMPY_AVAILABLE:
            return
        
        import numpy as np
        
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        masses = np.array(self.masses)
        
        G = 6.67430e-11
        dt = 0.01
        
        for _ in range(self.steps):
            # 使用广播计算所有粒子对
            # 创建位置差矩阵 (n, n, 3)
            dx = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            
            # 计算距离
            r2 = np.sum(dx**2, axis=2)
            r = np.sqrt(r2)
            
            # 避免自相互作用和对角线元素
            n = len(positions)
            diag_indices = np.diag_indices(n)
            r2[diag_indices] = np.inf  # 设置对角线为无穷大
            
            # 计算力的大小 (n, n)
            forces = G * masses[:, np.newaxis] * masses[np.newaxis, :] / r2
            
            # 计算加速度 (n, n, 3)
            # 使用np.where避免对角线上的除零问题
            # 当r为0时（对角线），加速度应为0
            with np.errstate(invalid='ignore', divide='ignore'):
                accelerations_pair = np.where(
                    r[:, :, np.newaxis] > 0,
                    forces[:, :, np.newaxis] * dx / (r[:, :, np.newaxis] * masses[:, np.newaxis, np.newaxis]),
                    0.0
                )
            
            # 对所有j求和得到每个粒子的总加速度
            accelerations = np.sum(accelerations_pair, axis=1)
            
            # 更新速度和位置
            velocities += accelerations * dt
            positions += velocities * dt
    
    def numba_implementation(self) -> None:
        """Numba实现（串行）"""
        if not NUMBA_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        import numpy as np
        
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        masses = np.array(self.masses)
        
        @jit(nopython=True)
        def nbody_step(positions, velocities, masses, steps, dt, G):
            n = len(positions)
            
            for _ in range(steps):
                accelerations = np.zeros_like(positions)
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            dx = positions[j, 0] - positions[i, 0]
                            dy = positions[j, 1] - positions[i, 1]
                            dz = positions[j, 2] - positions[i, 2]
                            
                            r2 = dx*dx + dy*dy + dz*dz
                            r = np.sqrt(r2)
                            
                            if r > 1e-10:
                                force = G * masses[i] * masses[j] / r2
                                ax = force * dx / (r * masses[i])
                                ay = force * dy / (r * masses[i])
                                az = force * dz / (r * masses[i])
                                
                                accelerations[i, 0] += ax
                                accelerations[i, 1] += ay
                                accelerations[i, 2] += az
                
                velocities += accelerations * dt
                positions += velocities * dt
        
        dt = 0.01
        G = 6.67430e-11
        nbody_step(positions, velocities, masses, self.steps, dt, G)
    
    def numba_parallel_implementation(self) -> None:
        """Numba实现（并行）"""
        if not NUMBA_AVAILABLE or not NUMPY_AVAILABLE:
            return
        
        import numpy as np
        
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        masses = np.array(self.masses)
        
        @jit(nopython=True, parallel=True)
        def nbody_step_parallel(positions, velocities, masses, steps, dt, G):
            n = len(positions)
            
            for _ in range(steps):
                accelerations = np.zeros_like(positions)
                
                # 外层循环并行化
                for i in prange(n):
                    for j in range(n):
                        if i != j:
                            dx = positions[j, 0] - positions[i, 0]
                            dy = positions[j, 1] - positions[i, 1]
                            dz = positions[j, 2] - positions[i, 2]
                            
                            r2 = dx*dx + dy*dy + dz*dz
                            r = np.sqrt(r2)
                            
                            if r > 1e-10:
                                force = G * masses[i] * masses[j] / r2
                                ax = force * dx / (r * masses[i])
                                ay = force * dy / (r * masses[i])
                                az = force * dz / (r * masses[i])
                                
                                accelerations[i, 0] += ax
                                accelerations[i, 1] += ay
                                accelerations[i, 2] += az
                
                velocities += accelerations * dt
                positions += velocities * dt
        
        dt = 0.01
        G = 6.67430e-11
        nbody_step_parallel(positions, velocities, masses, self.steps, dt, G)


class BenchmarkReporter:
    """Benchmark results reporter"""
    
    @staticmethod
    def print_results(results: List[BenchmarkResult], output_file: Optional[str] = None) -> None:
        """Print results to console or save to file"""
        output_data = {
            "benchmark_results": [r.to_dict() for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        else:
            # Print to console
            print("\n" + "="*80)
            print("CPU Benchmark Results")
            print("="*80)
            
            # Group by task
            tasks = {}
            for result in results:
                if result.task_name not in tasks:
                    tasks[result.task_name] = []
                tasks[result.task_name].append(result)
            
            for task_name, task_results in tasks.items():
                print(f"\n{task_name}:")
                print("-" * 60)
                
                # Find Python implementation as baseline
                baseline = None
                for result in task_results:
                    if result.implementation in ["Pure Python", "Python"]:
                        baseline = result
                        break
                
                # If no baseline found, use the fastest as baseline
                if baseline is None:
                    baseline = min(task_results, key=lambda x: x.avg_time)
                    baseline_is_python = False
                else:
                    baseline_is_python = True
                
                # Sort by average execution time from fastest to slowest
                sorted_results = sorted(task_results, key=lambda x: x.avg_time)
                
                for result in sorted_results:
                    if result is baseline:
                        # Baseline marker
                        if baseline_is_python:
                            marker = " [Baseline - Pure Python]"
                        else:
                            marker = " [Baseline - Fastest]"
                    else:
                        # Calculate speedup relative to baseline
                        speedup = baseline.avg_time / result.avg_time
                        marker = f" (Speedup vs baseline: {speedup:.2f}x)"
                    
                    print(f"  {result.implementation:25s}")
                    print(f"    Time: {result.avg_time:.4f}s (min:{result.min_time:.4f}s, "
                          f"max:{result.max_time:.4f}s, med:{result.median_time:.4f}s)")
                    print(f"    Iter/s: {result.iterations_per_second:.2f}{marker}")
                    if result.std_time > 0:
                        print(f"    Std dev: {result.std_time:.6f}s")
                    if result.notes:
                        print(f"    {result.notes}")
    
    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str = "cpu_benchmark_results.json") -> None:
        """Save results to JSON file (backward compatibility)"""
        BenchmarkReporter.print_results(results, filename)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CPU performance benchmark tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --size (5000,500)                     # Set same scale for all tests
  %(prog)s --size-path (5000,500) --size-mc 1000000 --size-nbody (200,20)
  %(prog)s --repeats 10 --output results.json    # Run 10 repeats and save to JSON
  %(prog)s --help                                # Show this help message
  
Scale parameters:
  --size (m,n):        Set uniform scale for all tests
  --size-path (m,n):   Path integral test, m=steps, n=paths
  --size-mc m:         Monte Carlo test, m=samples
  --size-nbody (m,n):  N-body test, m=bodies, n=steps
        """
    )
    
    parser.add_argument(
        "--size", type=str,
        help="Set uniform scale for all tests, format (m,n)"
    )
    
    parser.add_argument(
        "--size-path", type=str,
        help="Set path integral test scale, format (m,n) where m=steps, n=paths"
    )
    
    parser.add_argument(
        "--size-mc", type=str,
        help="Set Monte Carlo test scale, m=samples"
    )
    
    parser.add_argument(
        "--size-nbody", type=str,
        help="Set N-body test scale, format (m,n) where m=bodies, n=steps"
    )
    
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of repeats for each test (default: 5)"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Path to output JSON file, if not specified print to console"
    )
    
    args = parser.parse_args()
    
    # Parse scale parameters
    def parse_tuple(s: str) -> Tuple[int, int]:
        """Parse (m,n) format string to integer tuple"""
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1]
        parts = s.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid tuple format: {s}")
        return (int(parts[0].strip()), int(parts[1].strip()))
    
    def parse_int(s: str) -> int:
        """Parse integer"""
        return int(s.strip())
    
    # Determine test scales
    if args.size:
        # Uniform scale
        base_m, base_n = parse_tuple(args.size)
        path_steps, path_paths = base_m, base_n
        mc_samples = base_m
        nbody_bodies, nbody_steps = base_m, base_n
    else:
        # Individual settings or defaults
        path_steps, path_paths = (5000, 500) if not args.size_path else parse_tuple(args.size_path)
        mc_samples = 1000000 if not args.size_mc else parse_int(args.size_mc)
        nbody_bodies, nbody_steps = (200, 20) if not args.size_nbody else parse_tuple(args.size_nbody)
    
    print("Starting CPU benchmark...")
    print(f"Configuration:")
    print(f"  Path Integral: steps={path_steps}, paths={path_paths}")
    print(f"  Monte Carlo: samples={mc_samples}")
    print(f"  N-Body: bodies={nbody_bodies}, steps={nbody_steps}")
    print(f"  Repeats: {args.repeats}")
    
    # Create benchmark instance
    benchmark = CPUBenchmark(warmup_iterations=2, test_iterations=args.repeats)
    
    all_results = []
    
    # Test 1: Path Integral
    print("\n1. Testing complex branching and loops (Path Integral)...")
    
    path_integral = PathIntegralBenchmark()
    
    # Pure Python implementation
    result = benchmark.run_benchmark(
        path_integral.python_implementation,
        task_name="Path Integral",
        impl_name="Pure Python",
        steps=path_steps,
        paths=path_paths
    )
    all_results.append(result)
    
    # NumPy/SciPy implementation
    if NUMPY_AVAILABLE and SCIPY_AVAILABLE:
        result = benchmark.run_benchmark(
            path_integral.numpy_scipy_implementation,
            task_name="Path Integral",
            impl_name="NumPy/SciPy",
            steps=path_steps,
            paths=path_paths
        )
        all_results.append(result)
    
    # Numba serial implementation
    if NUMBA_AVAILABLE:
        result = benchmark.run_benchmark(
            path_integral.numba_implementation,
            task_name="Path Integral",
            impl_name="Numba (Serial)",
            steps=path_steps,
            paths=path_paths
        )
        all_results.append(result)
        
        # Numba parallel implementation
        result = benchmark.run_benchmark(
            path_integral.numba_parallel_implementation,
            task_name="Path Integral",
            impl_name="Numba (Parallel)",
            steps=path_steps,
            paths=path_paths
        )
        all_results.append(result)
    
    # Test 2: Monte Carlo simulation
    print("\n2. Testing large-scale vector operations (Monte Carlo)...")
    
    monte_carlo = MonteCarloBenchmark()
    
    # Pure Python implementation
    result = benchmark.run_benchmark(
        monte_carlo.python_implementation,
        task_name="Monte Carlo",
        impl_name="Pure Python",
        samples=mc_samples
    )
    all_results.append(result)
    
    # NumPy implementation
    if NUMPY_AVAILABLE:
        result = benchmark.run_benchmark(
            monte_carlo.numpy_implementation,
            task_name="Monte Carlo",
            impl_name="NumPy",
            samples=mc_samples
        )
        all_results.append(result)
    
    # Numba implementation
    if NUMBA_AVAILABLE and NUMPY_AVAILABLE:
        result = benchmark.run_benchmark(
            monte_carlo.numba_implementation,
            task_name="Monte Carlo",
            impl_name="Numba",
            samples=mc_samples
        )
        all_results.append(result)
        
        # Numba parallel implementation
        result = benchmark.run_benchmark(
            monte_carlo.numba_parallel_implementation,
            task_name="Monte Carlo",
            impl_name="Numba (Parallel)",
            samples=mc_samples
        )
        all_results.append(result)
    
    # Test 3: N-body dynamics
    print("\n3. Testing large-scale data and synchronization (N-Body)...")
    
    nbody = NBodyBenchmark(num_bodies=nbody_bodies, steps=nbody_steps)
    
    # Pure Python implementation
    result = benchmark.run_benchmark(
        nbody.python_implementation,
        task_name="N-Body",
        impl_name="Pure Python"
    )
    all_results.append(result)
    
    # NumPy implementation
    if NUMPY_AVAILABLE:
        result = benchmark.run_benchmark(
            nbody.numpy_implementation,
            task_name="N-Body",
            impl_name="NumPy"
        )
        all_results.append(result)
    
    # Numba implementation
    if NUMBA_AVAILABLE and NUMPY_AVAILABLE:
        result = benchmark.run_benchmark(
            nbody.numba_implementation,
            task_name="N-Body",
            impl_name="Numba (Serial)"
        )
        all_results.append(result)
        
        # Numba parallel implementation
        result = benchmark.run_benchmark(
            nbody.numba_parallel_implementation,
            task_name="N-Body",
            impl_name="Numba (Parallel)"
        )
        all_results.append(result)
    
    # Output results
    BenchmarkReporter.print_results(all_results, output_file=args.output)
    
    # Output system information
    print("\n" + "="*80)
    print("System Information:")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    
    if NUMPY_AVAILABLE:
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        # Detect BLAS backend
        try:
            import numpy.distutils.system_info as sysinfo
            blas_info = sysinfo.get_info('blas_opt')
            if blas_info and 'libraries' in blas_info:
                print(f"  BLAS backend: {blas_info.get('libraries', ['Unknown'])[0]}")
        except:
            pass
    
    if SCIPY_AVAILABLE:
        from scipy import __version__ as scipy_version
        print(f"  SciPy version: {scipy_version}")
    
    if NUMBA_AVAILABLE:
        print(f"  Numba version: {numba.__version__}")
    
    print("="*80)


if __name__ == "__main__":
    main()
