#!/usr/bin/env python3
"""
GPU Monte Carlo Benchmark using OpenCL
Estimate Pi using Monte Carlo method with FP16/FP32/FP64 precision
"""

import time
import statistics
import json
import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

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
    precision: str = ""
    result_value: Optional[float] = None  # Estimated Pi value
    
    def __post_init__(self):
        self.min_time = min(self.execution_times)
        self.max_time = max(self.execution_times)
        self.avg_time = statistics.mean(self.execution_times)
        self.median_time = statistics.median(self.execution_times)
        self.std_time = statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0
    
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
            "result_value": self.result_value
        }


class GPUMonteCarloBenchmark:
    """OpenCL GPU Monte Carlo Benchmark supporting multiple precisions"""
    
    PRECISION_CONFIG = {
        'fp32': {
            'ctype': 'float',
            'suffix': 'f',
            'np_dtype': np.float32,
            'one': '1.0f',
            'ext_pragma': ''
        },
        'fp64': {
            'ctype': 'double',
            'suffix': '',
            'np_dtype': np.float64,
            'one': '1.0',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp64 : enable'
        },
        'fp16': {
            'ctype': 'half',
            'suffix': 'h',
            'np_dtype': np.float16,
            'one': '1.0h',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp16 : enable'
        }
    }
    
    def __init__(self, platform_idx: Optional[int] = None, device_idx: Optional[int] = None):
        if not PYOPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL not available")
        
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        if platform_idx is not None:
            platform = platforms[platform_idx]
        else:
            platform = platforms[0]
        
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices()  # Fallback to any device
        
        if device_idx is not None:
            self.device = devices[device_idx]
        else:
            self.device = devices[0]
        
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        print(f"Using device: {self.device.name}")
        print(f"Device vendor: {self.device.vendor}")
        print(f"OpenCL version: {self.device.version}")
        print(f"Max compute units: {self.device.max_compute_units}")
        print(f"Max work group size: {self.device.max_work_group_size}")
    
    def check_precision_support(self, precision: str) -> bool:
        """Check if device supports specific floating point precision"""
        if precision == 'fp32':
            return True
        
        extensions = self.device.extensions
        if precision == 'fp64':
            return 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions
        elif precision == 'fp16':
            return 'cl_khr_fp16' in extensions
        return False
    
    def generate_kernel_source(self, precision: str) -> str:
        """Generate OpenCL kernel for Monte Carlo Pi estimation"""
        config = self.PRECISION_CONFIG[precision]
        ctype = config['ctype']
        suffix = config['suffix']
        one = config['one']
        ext_pragma = config['ext_pragma']
        
        kernel = f'''
        {ext_pragma}
        
        // Linear Congruential Generator
        inline uint lcg(uint *state) {{
            *state = (*state) * 1103515245u + 12345u;
            return *state;
        }}
        
        inline {ctype} random_{precision}(uint *state) {{
            uint val = lcg(state) & 0x7fffffffu;
            return ({ctype})val / ({ctype})0x7fffffffu;
        }}
        
        inline uint wang_hash(uint seed) {{
            seed = (seed ^ 61) ^ (seed >> 16);
            seed *= 9;
            seed = seed ^ (seed >> 4);
            seed *= 0x27d4eb2d;
            seed = seed ^ (seed >> 15);
            return seed;
        }}
        
        kernel void monte_carlo_pi(
            const ulong samples_per_item,
            const uint total_items,
            const uint base_seed,
            global ulong *partial_counts,
            local ulong *local_counts
        ) {{
            uint gid = get_global_id(0);
            uint lid = get_local_id(0);
            uint group_size = get_local_size(0);
            
            ulong local_count = 0;
            
            // Each work-item processes samples_per_item random points
            uint rng_state = wang_hash(base_seed + gid);
            
            for (ulong i = 0; i < samples_per_item; i++) {{
                {ctype} x = random_{precision}(&rng_state);
                {ctype} y = random_{precision}(&rng_state);
                
                {ctype} dist = x * x + y * y;
                if (dist <= {one}) {{
                    local_count++;
                }}
            }}
            
            // Store in local memory for reduction
            local_counts[lid] = local_count;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Parallel reduction in local memory
            for (uint stride = group_size / 2; stride > 0; stride /= 2) {{
                if (lid < stride) {{
                    local_counts[lid] += local_counts[lid + stride];
                }}
                barrier(CLK_LOCAL_MEM_FENCE);
            }}
            
            // Write result for this work-group
            if (lid == 0) {{
                partial_counts[get_group_id(0)] = local_counts[0];
            }}
        }}
        
        kernel void sum_reduction(
            global const ulong *input,
            global ulong *output,
            const uint n
        ) {{
            // Simple parallel sum (assumes single work-group for final reduction)
            uint gid = get_global_id(0);
            if (gid >= n) return;
            
            // In practice, we perform this on CPU for small arrays
            // or use another reduction kernel
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
    
    def run_benchmark(self, precision: str = 'fp32', total_samples: int = 10000000,
                     warmup_iterations: int = 3, test_iterations: int = 10) -> BenchmarkResult:
        """
        Run GPU Monte Carlo benchmark
        
        Args:
            precision: 'fp16', 'fp32', or 'fp64'
            total_samples: Total number of random samples
            warmup_iterations: Number of warmup runs
            test_iterations: Number of timed test runs
        """
        if not self.check_precision_support(precision):
            raise RuntimeError(f"Precision {precision} not supported by device")
        
        program = self.build_program(precision)
        kernel = program.monte_carlo_pi
        
        # Determine work size
        max_wg_size = min(256, self.device.max_work_group_size)
        
        # Use many work-items for better GPU utilization
        # Each work-item processes multiple samples
        preferred_wi = self.device.max_compute_units * 64  # Heuristic: 64 items per CU
        num_work_items = min(preferred_wi, (total_samples + 999) // 1000)  # At least 1000 per item
        num_work_items = ((num_work_items + max_wg_size - 1) // max_wg_size) * max_wg_size  # Round up
        
        samples_per_item = (total_samples + num_work_items - 1) // num_work_items
        actual_total = samples_per_item * num_work_items
        
        global_size = num_work_items
        local_size = max_wg_size
        num_groups = global_size // local_size
        
        # Prepare buffers
        partial_counts_np = np.zeros(num_groups, dtype=np.uint64)
        partial_counts_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, partial_counts_np.nbytes)
        
        # Warm-up runs
        for i in range(warmup_iterations):
            kernel(self.queue, (global_size,), (local_size,),
                   np.uint64(samples_per_item), np.uint32(num_work_items), 
                   np.uint32(12345 + i), partial_counts_buf,
                   cl.LocalMemory(np.dtype(np.uint64).itemsize * local_size))
            self.queue.finish()
        
        # Test runs
        execution_times = []
        for i in range(test_iterations):
            seed = np.uint32(12345 + i + warmup_iterations)
            
            start = time.perf_counter()
            event = kernel(self.queue, (global_size,), (local_size,),
                          np.uint64(samples_per_item), np.uint32(num_work_items),
                          seed, partial_counts_buf,
                          cl.LocalMemory(np.dtype(np.uint64).itemsize * local_size))
            event.wait()
            end = time.perf_counter()
            
            execution_times.append(end - start)
        
        # Read results
        cl.enqueue_copy(self.queue, partial_counts_np, partial_counts_buf)
        self.queue.finish()
        
        total_inside = np.sum(partial_counts_np)
        pi_estimate = 4.0 * total_inside / actual_total
        
        return BenchmarkResult(
            task_name="Monte Carlo Pi (GPU)",
            implementation=f"OpenCL {self.device.name}",
            precision=precision,
            execution_times=execution_times,
            notes=f"Samples={actual_total}, WorkItems={num_work_items}, Samples/Item={samples_per_item}",
            result_value=float(pi_estimate)
        )
    
    @staticmethod
    def python_reference(samples: int = 1000000) -> float:
        """Pure Python reference implementation for validation"""
        import random
        inside = 0
        for _ in range(samples):
            x = random.random()
            y = random.random()
            if x*x + y*y <= 1.0:
                inside += 1
        return 4.0 * inside / samples


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
            print("GPU Monte Carlo Benchmark Results")
            print("="*80)
            
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
                    error = abs(result.result_value - np.pi)
                    print(f"  Pi estimate: {result.result_value:.8f}")
                    print(f"  Error vs math.pi: {error:.2e}")
                if result.notes:
                    print(f"  Notes: {result.notes}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GPU Monte Carlo Benchmark using OpenCL (FP16/FP32/FP64)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --precision fp32 --samples 10000000     # Test FP32 with 10M samples
  %(prog)s --precision fp64 --samples 5000000      # Test FP64 with 5M samples
  %(prog)s --precision all                         # Test all supported precisions
  %(prog)s --output results.json                   # Save results to JSON
        """
    )
    
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp16", "fp32", "fp64", "all"],
        help="Floating point precision to test (default: fp32)"
    )
    
    parser.add_argument(
        "--samples", type=int, default=10000000,
        help="Total number of Monte Carlo samples (default: 10000000)"
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
    
    print("Starting GPU Monte Carlo Benchmark...")
    print(f"Configuration: samples={args.samples}, repeats={args.repeats}")
    
    try:
        benchmark = GPUMonteCarloBenchmark(
            platform_idx=args.platform,
            device_idx=args.device
        )
    except Exception as e:
        print(f"Error initializing OpenCL: {e}")
        sys.exit(1)
    
    # Determine precisions to test
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
                total_samples=args.samples,
                warmup_iterations=args.warmup,
                test_iterations=args.repeats
            )
            results.append(result)
            print(f"  Average time: {result.avg_time:.4f}s")
            print(f"  Pi estimate: {result.result_value:.8f}")
        except Exception as e:
            print(f"  Error running {precision}: {e}")
    
    # Report results
    if results:
        BenchmarkReporter.print_results(results, output_file=args.output)
    else:
        print("\nNo results generated.")
    
    # Validation against reference
    if results:
        print("\n" + "="*80)
        print("Validating against Python reference (100,000 samples)...")
        ref_pi = GPUMonteCarloBenchmark.python_reference(samples=100000)
        print(f"  Reference Pi: {ref_pi:.8f}")
        for result in results:
            err = abs(result.result_value - np.pi)
            print(f"  {result.precision.upper()} error vs math.pi: {err:.2e}")
        print("="*80)


if __name__ == "__main__":
    main()
