#!/usr/bin/env python3
"""
GPU Path Integral Benchmark using OpenCL
测试不同浮点精度(FP16/FP32/FP64)在GPU上的路径积分性能
"""

import time
import statistics
import json
import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# OpenCL optional dependency
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    print("Warning: PyOpenCL not available, GPU benchmark will be skipped")


@dataclass
class BenchmarkResult:
    """Benchmark result data class (consistent with cpu.py)"""
    task_name: str
    implementation: str
    execution_times: List[float]  # in seconds
    min_time: float = field(init=False)
    max_time: float = field(init=False)
    avg_time: float = field(init=False)
    median_time: float = field(init=False)
    std_time: float = field(init=False)
    memory_usage: Optional[float] = None
    notes: str = ""
    precision: str = ""  # Additional field for precision info
    result_value: Optional[float] = None  # For validation
    
    def __post_init__(self):
        self.min_time = min(self.execution_times)
        self.max_time = max(self.execution_times)
        self.avg_time = statistics.mean(self.execution_times)
        self.median_time = statistics.median(self.execution_times)
        self.std_time = statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0
    
    @property
    def iterations_per_second(self) -> float:
        return 1.0 / self.avg_time if self.avg_time > 0 else float('inf')
    
    @property
    def paths_per_second(self) -> float:
        """Return number of paths processed per second"""
        # Parse paths from notes
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
            "result_value": self.result_value
        }


class GPUPathIntegralBenchmark:
    """OpenCL GPU Path Integral Benchmark supporting multiple precisions"""
    
    # Mapping precision names to OpenCL types and constants
    PRECISION_CONFIG = {
        'fp32': {
            'ctype': 'float',
            'suffix': 'f',
            'np_dtype': np.float32,
            'ext_pragma': ''
        },
        'fp64': {
            'ctype': 'double',
            'suffix': '',
            'np_dtype': np.float64,
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
        },
        'fp16': {
            'ctype': 'half',
            'suffix': 'h',
            'np_dtype': np.float16,
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp16 : enable'
        }
    }
    
    def __init__(self, platform_idx: Optional[int] = None, device_idx: Optional[int] = None):
        if not PYOPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL not available")
        
        # Initialize OpenCL context and queue
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        if platform_idx is not None:
            platform = platforms[platform_idx]
        else:
            platform = platforms[0]
        
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            # Fall back to CPU if no GPU available (for testing)
            devices = platform.get_devices()
        
        if device_idx is not None:
            self.device = devices[device_idx]
        else:
            self.device = devices[0]
        
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"Using device: {self.device.name}")
        print(f"Device vendor: {self.device.vendor}")
        print(f"OpenCL version: {self.device.version}")
    
    def check_precision_support(self, precision: str) -> bool:
        """Check if device supports specific floating point precision"""
        if precision == 'fp32':
            return True  # FP32 is mandatory in OpenCL
        
        extensions = self.device.extensions
        if precision == 'fp64':
            return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions
        elif precision == 'fp16':
            return 'cl_khr_fp16' in extensions
        return False
    
    def generate_kernel_source(self, precision: str) -> str:
        """Generate OpenCL kernel source for specific precision"""
        config = self.PRECISION_CONFIG[precision]
        ctype = config['ctype']
        suffix = config['suffix']
        ext_pragma = config['ext_pragma']
        
        # Kernel template with precision-specific types
        kernel = f'''
        {ext_pragma}
        
        // Linear Congruential Generator for random numbers
        inline uint lcg(uint *state) {{
            *state = (*state) * 1103515245u + 12345u;
            return *state;
        }}
        
        inline {ctype} random_{precision}(uint *state) {{
            uint val = lcg(state) & 0x7fffffffu;
            return ({ctype})val / ({ctype})0x7fffffffu;
        }}
        
        __kernel void path_integral(
            const int steps,
            const int paths,
            const uint base_seed,
            __global {ctype} *results
        ) {{
            int path_id = get_global_id(0);
            if (path_id >= paths) return;
            
            // Each path has its own RNG state
            uint rng_state = base_seed + (uint)path_id;
            
            {ctype} integral = 0.0{suffix};
            {ctype} x = 0.0{suffix};
            {ctype} delta = 1.0{suffix} / ({ctype})steps;
            
            for (int step = 0; step < steps; step++) {{
                // Branching logic matching cpu.py implementation
                {ctype} weight;
                if (x < -1.0{suffix}) {{
                    weight = 0.1{suffix};
                }} else if (x < 0.0{suffix}) {{
                    weight = 0.3{suffix} * x + 0.4{suffix};
                }} else if (x < 1.0{suffix}) {{
                    weight = 0.5{suffix} * (1.0{suffix} - x * x);
                }} else {{
                    weight = 0.2{suffix};
                }}
                
                // Alternating factor
                {ctype} factor = (step % 2 == 0) ? 
                    (1.0{suffix} + 0.1{suffix} * x) : 
                    (1.0{suffix} - 0.1{suffix} * x);
                
                integral += weight * delta * factor;
                
                // Update position with random walk
                {ctype} rand_val = random_{precision}(&rng_state);
                x += rand_val * 0.1{suffix};
                
                // Boundary check [-2.0, 2.0]
                if (x > 2.0{suffix}) x = 2.0{suffix};
                if (x < -2.0{suffix}) x = -2.0{suffix};
            }}
            
            results[path_id] = integral;
        }}
        '''
        return kernel
    
    def build_program(self, precision: str):
        """Build OpenCL program for specific precision"""
        source = self.generate_kernel_source(precision)
        program = cl.Program(self.context, source)
        
        try:
            program.build(options=['-cl-fast-relaxed-math'])
        except cl.RuntimeError as e:
            build_log = program.get_build_info(self.device, cl.program_build_info.LOG)
            raise RuntimeError(f"OpenCL build error: {e}\nBuild log:\n{build_log}")
        
        return program
    
    def run_benchmark(self, precision: str = 'fp32', steps: int = 10000, paths: int = 1000,
                     warmup_iterations: int = 3, test_iterations: int = 10) -> BenchmarkResult:
        """
        Run GPU benchmark for specific precision
        
        Args:
            precision: 'fp16', 'fp32', or 'fp64'
            steps: Number of integration steps per trajectory
            paths: Number of trajectories (parallelism)
            warmup_iterations: Number of warmup runs
            test_iterations: Number of timed test runs
        """
        if not self.check_precision_support(precision):
            raise RuntimeError(f"Precision {precision} not supported by device")
        
        # Build program
        program = self.build_program(precision)
        kernel = program.path_integral
        
        # Prepare buffers
        config = self.PRECISION_CONFIG[precision]
        np_dtype = config['np_dtype']
        
        # Determine work group size
        max_wg_size = self.device.max_work_group_size
        wg_size = min(256, max_wg_size)
        global_size = ((paths + wg_size - 1) // wg_size) * wg_size
        
        # Warm-up runs
        for i in range(warmup_iterations):
            seed = np.uint32(12345 + i)
            results_np = np.empty(paths, dtype=np_dtype)
            results_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, results_np.nbytes)
            kernel(self.queue, (global_size,), (wg_size,),
                   np.int32(steps), np.int32(paths), seed, results_buf)
            self.queue.finish()
        
        # Test runs with timing
        execution_times = []
        final_results = []
        
        for i in range(test_iterations):
            seed = np.uint32(12345 + i + warmup_iterations)
            results_np = np.empty(paths, dtype=np_dtype)
            results_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, results_np.nbytes)
            
            # Use Python time for wall-clock time (consistent with cpu.py)
            # Include data transfer in timing for fair comparison with Metal
            start = time.perf_counter()
            kernel(self.queue, (global_size,), (wg_size,),
                   np.int32(steps), np.int32(paths), seed, results_buf)
            cl.enqueue_copy(self.queue, results_np, results_buf)
            self.queue.finish()
            end = time.perf_counter()
            
            execution_times.append(end - start)
            final_results.append(float(np.mean(results_np)))
            
            # Clean up buffer
            results_buf.release()
        
        # Use the last result for validation
        final_result = final_results[-1] if final_results else 0.0
        
        # Calculate throughput metrics
        avg_time = statistics.mean(execution_times) if execution_times else 0.0
        paths_per_sec = paths / avg_time if avg_time > 0 else 0.0
        total_ops_per_sec = (steps * paths) / avg_time if avg_time > 0 else 0.0
        
        return BenchmarkResult(
            task_name="Trajectory Integral (GPU)",
            implementation=f"OpenCL {self.device.name}",
            precision=precision,
            execution_times=execution_times,
            notes=f"Steps={steps}, Paths={paths}, WorkGroup={wg_size}, "
                  f"Throughput={paths_per_sec:.0f} paths/sec, "
                  f"Ops={total_ops_per_sec:.0f} steps·paths/sec",
            result_value=final_result
        )
    
    @staticmethod
    def python_reference(steps: int = 10000, paths: int = 1000) -> float:
        """Pure Python reference implementation for validation"""
        import random
        result = 0.0
        
        for path in range(paths):
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


class BenchmarkReporter:
    """Results reporter (consistent with cpu.py style)"""
    
    @staticmethod
    def print_results(results: List[BenchmarkResult], output_file: Optional[str] = None) -> None:
        """Print results to console or save to file"""
        output_data = {
            "benchmark_results": [r.to_dict() for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "opencl_available": PYOPENCL_AVAILABLE
            }
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("\n" + "="*80)
            print("GPU Path Integral Benchmark Results")
            print("="*80)
            
            # Group by precision
            for result in results:
                print(f"\nPrecision: {result.precision.upper()}")
                print("-" * 60)
                print(f"  Device: {result.implementation}")
                print(f"  Time: {result.avg_time:.4f}s (min:{result.min_time:.4f}s, "
                      f"max:{result.max_time:.4f}s, med:{result.median_time:.4f}s)")
                print(f"  Iter/s: {result.iterations_per_second:.2f}")
                if result.std_time > 0:
                    print(f"  Std dev: {result.std_time:.6f}s")
                if result.result_value is not None:
                    print(f"  Result value: {result.result_value:.6f}")
                if result.notes:
                    print(f"  Notes: {result.notes}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GPU Path Integral Benchmark using OpenCL (FP16/FP32/FP64)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --precision fp32                    # Test FP32 only
  %(prog)s --precision fp64 --size (5000,500)  # Test FP64 with custom scale
  %(prog)s --precision all                     # Test all supported precisions
  %(prog)s --output results.json               # Save results to JSON
        """
    )
    
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp16", "fp32", "fp64", "all"],
        help="Floating point precision to test (default: fp32)"
    )
    
    parser.add_argument(
        "--size", type=str, default="(10000,1000)",
        help="Benchmark scale as (steps,paths), default (10000,1000)"
    )
    
    parser.add_argument(
        "--repeats", type=int, default=10,
        help="Number of test iterations (default: 10)"
    )
    
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (default: 3)"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--platform", type=int, default=0,
        help="OpenCL platform index (default: 0)"
    )
    
    parser.add_argument(
        "--device", type=int, default=0,
        help="OpenCL device index (default: 0)"
    )
    
    args = parser.parse_args()
    
    if not PYOPENCL_AVAILABLE:
        print("Error: PyOpenCL not installed. Please install: pip install pyopencl")
        sys.exit(1)
    
    # Parse size tuple
    size_str = args.size.strip()
    if size_str.startswith('(') and size_str.endswith(')'):
        size_str = size_str[1:-1]
    steps, paths = map(int, size_str.split(','))
    
    print("Starting GPU Path Integral Benchmark...")
    print(f"Configuration: steps={steps}, paths={paths}, repeats={args.repeats}")
    
    # Initialize benchmark
    try:
        benchmark = GPUPathIntegralBenchmark(
            platform_idx=args.platform,
            device_idx=args.device
        )
    except Exception as e:
        print(f"Error initializing OpenCL: {e}")
        sys.exit(1)
    
    # Determine which precisions to test
    if args.precision == 'all':
        precisions_to_test = ['fp16', 'fp32', 'fp64']
    else:
        precisions_to_test = [args.precision]
    
    # Run benchmarks
    results = []
    for precision in precisions_to_test:
        if not benchmark.check_precision_support(precision):
            print(f"\nSkipping {precision}: not supported by device")
            continue
        
        print(f"\nTesting {precision.upper()}...")
        try:
            result = benchmark.run_benchmark(
                precision=precision,
                steps=steps,
                paths=paths,
                warmup_iterations=args.warmup,
                test_iterations=args.repeats
            )
            results.append(result)
            print(f"  Average time: {result.avg_time:.4f}s")
        except Exception as e:
            print(f"  Error running {precision}: {e}")
    
    # Report results
    if results:
        BenchmarkReporter.print_results(results, output_file=args.output)
    else:
        print("\nNo results generated.")
    
    # Compare with reference if single precision test
    if len(results) == 1 and results[0].precision == 'fp32':
        print("\n" + "="*80)
        print("Validating against Python reference...")
        ref_value = GPUPathIntegralBenchmark.python_reference(steps=min(steps, 1000), paths=min(paths, 100))
        gpu_value = results[0].result_value
        diff = abs(ref_value - gpu_value)
        print(f"  Reference: {ref_value:.6f}")
        print(f"  GPU result: {gpu_value:.6f}")
        print(f"  Difference: {diff:.6e}")
        print("="*80)


if __name__ == "__main__":
    main()
