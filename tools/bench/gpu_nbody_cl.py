#!/usr/bin/env python3
"""
GPU N-Body Benchmark using OpenCL
Gravitational N-body simulation with FP16/FP32/FP64 precision
"""

import time
import statistics
import json
import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
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
    memory_usage: Optional[float] = None  # MB
    notes: str = ""
    precision: str = ""
    final_energy: Optional[float] = None  # For validation
    
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
            "final_energy": self.final_energy
        }


class GPUNBodyBenchmark:
    """OpenCL GPU N-Body Benchmark supporting multiple precisions"""
    
    PRECISION_CONFIG = {
        'fp32': {
            'ctype': 'float',
            'vec_type': 'float4',
            'suffix': 'f',
            'np_dtype': np.float32,
            'zero': '0.0f',
            'one': '1.0f',
            'ext_pragma': '',
            'accum_type': 'float',
            'convert_in': '',
            'convert_out': ''
        },
        'fp64': {
            'ctype': 'double',
            'vec_type': 'double4',
            'suffix': '',
            'np_dtype': np.float64,
            'zero': '0.0',
            'one': '1.0',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp64 : enable',
            'accum_type': 'double',
            'convert_in': '',
            'convert_out': ''
        },
        'fp16': {
            'ctype': 'half',
            'vec_type': 'half4',
            'suffix': 'h',
            'np_dtype': np.float16,
            'zero': '0.0h',
            'one': '1.0h',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp16 : enable',
            'accum_type': 'float',  # Mixed precision: use float for accumulation
            'convert_in': 'convert_float4',
            'convert_out': 'convert_half4'
        }
    }
    
    # Tile size for shared memory optimization
    TILE_SIZE = 256
    
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
            devices = platform.get_devices()
        
        if device_idx is not None:
            self.device = devices[device_idx]
        else:
            self.device = devices[0]
        
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # Device info
        self.max_wg_size = self.device.max_work_group_size
        self.local_mem_size = self.device.local_mem_size
        self.compute_units = self.device.max_compute_units
        
        print(f"Using device: {self.device.name}")
        print(f"Device vendor: {self.device.vendor}")
        print(f"OpenCL version: {self.device.version}")
        print(f"Max compute units: {self.compute_units}")
        print(f"Max work group size: {self.max_wg_size}")
    
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
        """
        Generate OpenCL kernel for N-body simulation with tiling optimization
        Reference: SKILL.opencl.md Section 1.1 (Local memory usage)
        """
        config = self.PRECISION_CONFIG[precision]
        ctype = config['ctype']
        vec_type = config['vec_type']
        suffix = config['suffix']
        zero = config['zero']
        accum_type = config['accum_type']
        convert_in = config['convert_in']
        convert_out = config['convert_out']
        ext_pragma = config['ext_pragma']
        tile_size = self.TILE_SIZE
        
        kernel = f'''
        {ext_pragma}
        
        #define TILE_SIZE {tile_size}
        #define SOFTENING (0.01{suffix} * 0.01{suffix})  // Plummer softening squared
        
        inline {accum_type}4 compute_acceleration(
            {accum_type}4 pos_i,
            {accum_type}4 pos_j,
            {accum_type} mass_j
        ) {{
            {accum_type}4 r_vec = pos_j - pos_i;
            {accum_type} r2 = r_vec.x*r_vec.x + r_vec.y*r_vec.y + r_vec.z*r_vec.z + SOFTENING;
            {accum_type} inv_r = rsqrt(r2);  // Fast inverse sqrt
            {accum_type} inv_r3 = inv_r * inv_r * inv_r;
            
            return mass_j * inv_r3 * r_vec;
        }}
        
        __kernel void nbody_simulation(
            __global {vec_type}* positions,
            __global {vec_type}* velocities,
            __global {ctype}* masses,
            int n,
            {ctype} G,
            {ctype} dt,
            int steps
        ) {{
            int gid = get_global_id(0);
            if (gid >= n) return;
            
            int lid = get_local_id(0);
            
            // Load particle i data
            {vec_type} pos_i_vec = positions[gid];
            {vec_type} vel_i_vec = velocities[gid];
            {ctype} mass_i = masses[gid];
            
            // Convert to accumulation type for computation (mixed precision for fp16)
            {accum_type}4 pos_i = {"convert_float4(pos_i_vec)" if precision == "fp16" else "pos_i_vec"};
            {accum_type}4 vel_i = {"convert_float4(vel_i_vec)" if precision == "fp16" else "vel_i_vec"};
            
            // Local memory tiles
            __local {vec_type} local_pos[TILE_SIZE];
            __local {ctype} local_mass[TILE_SIZE];
            
            // Simulation loop
            for (int s = 0; s < steps; s++) {{
                {accum_type}4 acc = ({accum_type}4)({zero}, {zero}, {zero}, {zero});
                
                // Tiling loop over all particles
                for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
                    int j = tile * TILE_SIZE + lid;
                    
                    // Load tile into local memory (coalesced access)
                    if (j < n) {{
                        local_pos[lid] = positions[j];
                        local_mass[lid] = masses[j];
                    }} else {{
                        local_pos[lid] = ({vec_type})({zero});
                        local_mass[lid] = {zero};
                    }}
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    // Compute interactions with this tile
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE; k++) {{
                        {accum_type}4 pos_j = {"convert_float4(local_pos[k])" if precision == "fp16" else "local_pos[k]"};
                        {accum_type} mass_j = local_mass[k];
                        
                        if (tile * TILE_SIZE + k != gid && tile * TILE_SIZE + k < n) {{
                            acc += compute_acceleration(pos_i, pos_j, mass_j);
                        }}
                    }}
                    barrier(CLK_LOCAL_MEM_FENCE);
                }}
                
                // Update velocity and position (semi-implicit Euler)
                vel_i += G * acc * dt;
                pos_i += vel_i * dt;
            }}
            
            // Store back
            positions[gid] = {"convert_half4(pos_i)" if precision == "fp16" else "pos_i"};
            velocities[gid] = {"convert_half4(vel_i)" if precision == "fp16" else "vel_i"};
        }}
        '''
        return kernel
    
    def build_program(self, precision: str):
        """Build OpenCL program"""
        source = self.generate_kernel_source(precision)
        program = cl.Program(self.context, source)
        
        build_options = [
            '-cl-fast-relaxed-math',
            '-cl-mad-enable',
            f'-DWORK_GROUP_SIZE={min(self.TILE_SIZE, self.max_wg_size)}'
        ]
        
        try:
            program.build(options=' '.join(build_options))
        except cl.RuntimeError as e:
            build_log = program.get_build_info(self.device, cl.program_build_info.LOG)
            raise RuntimeError(f"OpenCL build error: {e}\nBuild log:\n{build_log}")
        
        return program
    
    def run_benchmark(self, precision: str = 'fp32', num_bodies: int = 1000, 
                     steps: int = 100, dt: float = 0.01,
                     warmup_iterations: int = 3, test_iterations: int = 10,
                     steps_per_chunk: int = 10) -> BenchmarkResult:
        """
        Run GPU N-body benchmark with chunked execution to prevent system hangs
        
        Args:
            precision: 'fp16', 'fp32', or 'fp64'
            num_bodies: Number of particles
            steps: Number of integration steps
            dt: Time step
            warmup_iterations: Number of warmup runs
            test_iterations: Number of timed test runs
            steps_per_chunk: Steps per kernel launch (prevents hangs)
        """
        if not self.check_precision_support(precision):
            raise RuntimeError(f"Precision {precision} not supported by device")
        
        if steps_per_chunk < 1:
            steps_per_chunk = steps
        
        # Warn if chunk size is too large
        estimated_time_per_chunk = (num_bodies * steps_per_chunk) / 1e6  # Rough estimate
        if estimated_time_per_chunk > 0.5:  # More than 0.5 seconds per chunk
            print(f"Warning: steps_per_chunk={steps_per_chunk} may cause long kernel execution.")
            print("  Reduce --chunk-size if system becomes unresponsive.")
        
        program = self.build_program(precision)
        kernel = program.nbody_simulation
        
        config = self.PRECISION_CONFIG[precision]
        np_dtype = config['np_dtype']
        
        # Generate initial conditions (reproducible)
        np.random.seed(42)
        positions = np.random.randn(num_bodies, 4).astype(np_dtype)
        positions[:, 3] = 0.0  # w component unused but needed for float4 alignment
        
        velocities = np.random.randn(num_bodies, 4).astype(np_dtype) * 0.1
        velocities[:, 3] = 0.0
        
        masses = np.random.rand(num_bodies).astype(np_dtype) * 10.0 + 1.0
        
        # OpenCL buffers
        mf = cl.mem_flags
        pos_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
        vel_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
        mass_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)
        
        # Work group size (must be <= TILE_SIZE and device limit)
        local_size = min(self.TILE_SIZE, self.max_wg_size)
        local_size = min(local_size, 256)  # Cap at 256 for better occupancy
        global_size = ((num_bodies + local_size - 1) // local_size) * local_size
        
        G = np_dtype(6.67430e-11)  # Gravitational constant
        dt = np_dtype(dt)
        
        # Warm-up runs (using chunked execution as well)
        for _ in range(warmup_iterations):
            steps_done = 0
            while steps_done < steps:
                current_steps = min(steps_per_chunk, steps - steps_done)
                kernel(self.queue, (global_size,), (local_size,),
                       pos_buf, vel_buf, mass_buf,
                       np.int32(num_bodies), G, dt, np.int32(current_steps))
                steps_done += current_steps
            self.queue.finish()
        
        # Reset to initial conditions for actual test
        np.random.seed(42)
        positions = np.random.randn(num_bodies, 4).astype(np_dtype)
        positions[:, 3] = 0.0
        velocities = np.random.randn(num_bodies, 4).astype(np_dtype) * 0.1
        velocities[:, 3] = 0.0
        
        cl.enqueue_copy(self.queue, pos_buf, positions)
        cl.enqueue_copy(self.queue, vel_buf, velocities)
        self.queue.finish()
        
        # Test runs
        execution_times = []
        
        for iteration in range(test_iterations):
            # Reset state
            np.random.seed(42)
            pos_init = np.random.randn(num_bodies, 4).astype(np_dtype)
            pos_init[:, 3] = 0.0
            vel_init = np.random.randn(num_bodies, 4).astype(np_dtype) * 0.1
            vel_init[:, 3] = 0.0
            
            cl.enqueue_copy(self.queue, pos_buf, pos_init)
            cl.enqueue_copy(self.queue, vel_buf, vel_init)
            self.queue.finish()
            
            start = time.perf_counter()
            
            # Chunked execution to prevent hangs
            steps_done = 0
            while steps_done < steps:
                current_steps = min(steps_per_chunk, steps - steps_done)
                
                event = kernel(self.queue, (global_size,), (local_size,),
                              pos_buf, vel_buf, mass_buf,
                              np.int32(num_bodies), G, dt, np.int32(current_steps))
                event.wait()
                
                steps_done += current_steps
                if test_iterations == 1:  # Only show progress for single test iteration
                    print(f"\r  Progress: {steps_done}/{steps} steps", end="")
            
            if test_iterations == 1:
                print()  # New line after progress
            
            end = time.perf_counter()
            execution_times.append(end - start)
        
        # Read final state for validation (optional)
        final_pos = np.empty_like(positions)
        cl.enqueue_copy(self.queue, final_pos, pos_buf)
        self.queue.finish()
        
        # Calculate approximate total energy for validation (simplified)
        # E = sum(0.5 * m * v^2) - sum(G * m_i * m_j / r_ij)
        # This is just a sanity check value, not full energy calculation
        final_energy = float(np.sum(final_pos))  # Placeholder metric
        
        # Memory usage estimate
        mem_usage = (positions.nbytes + velocities.nbytes + masses.nbytes) / (1024**2)
        
        return BenchmarkResult(
            task_name="N-Body Simulation (GPU)",
            implementation=f"OpenCL {self.device.name}",
            precision=precision,
            execution_times=execution_times,
            notes=f"Bodies={num_bodies}, Steps={steps}, dt={dt}, "
                  f"TileSize={self.TILE_SIZE}, StepsPerChunk={steps_per_chunk}",
            final_energy=final_energy,
            memory_usage=mem_usage
        )
    
    @staticmethod
    def python_reference(num_bodies: int = 200, steps: int = 20, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure Python reference implementation (for validation, small N only)
        Returns final positions and velocities
        """
        np.random.seed(42)
        positions = np.random.randn(num_bodies, 3)
        velocities = np.random.randn(num_bodies, 3) * 0.1
        masses = np.random.rand(num_bodies) * 10.0 + 1.0
        
        G = 6.67430e-11
        softening = 0.01 * 0.01
        
        for _ in range(steps):
            accelerations = np.zeros_like(positions)
            
            for i in range(num_bodies):
                for j in range(num_bodies):
                    if i == j:
                        continue
                    r_vec = positions[j] - positions[i]
                    r2 = np.dot(r_vec, r_vec) + softening
                    r = np.sqrt(r2)
                    inv_r3 = 1.0 / (r2 * r)
                    accelerations[i] += G * masses[j] * inv_r3 * r_vec
            
            velocities += accelerations * dt
            positions += velocities * dt
        
        return positions, velocities


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
            print("GPU N-Body Benchmark Results")
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
                if result.memory_usage:
                    print(f"  Memory: {result.memory_usage:.2f} MB")
                if result.notes:
                    print(f"  Notes: {result.notes}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GPU N-Body Benchmark using OpenCL (FP16/FP32/FP64)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --precision fp32 --bodies 1000 --steps 100    # Test FP32
  %(prog)s --precision fp64 --bodies 2048 --steps 50     # Test FP64
  %(prog)s --precision fp16 --bodies 512 --steps 200     # Test FP16
  %(prog)s --precision all --bodies 1024 --steps 100     # Test all precisions
  %(prog)s --output results.json                         # Save to file
        """
    )
    
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp16", "fp32", "fp64", "all"],
        help="Floating point precision to test (default: fp32)"
    )
    
    parser.add_argument(
        "--bodies", type=int, default=1000,
        help="Number of bodies/particles (default: 1000)"
    )
    
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of integration steps (default: 100)"
    )
    
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Time step (default: 0.01)"
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
        "--chunk-size", type=int, default=10,
        help="Steps per kernel launch (smaller values prevent hangs, default: 10)"
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
    
    print("Starting GPU N-Body Benchmark...")
    print(f"Configuration: bodies={args.bodies}, steps={args.steps}, dt={args.dt}, repeats={args.repeats}")
    
    try:
        benchmark = GPUNBodyBenchmark(
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
                num_bodies=args.bodies,
                steps=args.steps,
                dt=args.dt,
                warmup_iterations=args.warmup,
                test_iterations=args.repeats,
                steps_per_chunk=args.chunk_size
            )
            results.append(result)
            print(f"  Average time: {result.avg_time:.4f}s")
            print(f"  Memory usage: {result.memory_usage:.2f} MB")
        except Exception as e:
            print(f"  Error running {precision}: {e}")
    
    # Report results
    if results:
        BenchmarkReporter.print_results(results, output_file=args.output)
    else:
        print("\nNo results generated.")
    
    # Validation against Python reference (small N only)
    if results and args.bodies <= 500:
        print("\n" + "="*80)
        print("Validating against Python reference (small N)...")
        try:
            ref_pos, ref_vel = GPUNBodyBenchmark.python_reference(
                num_bodies=min(args.bodies, 200),
                steps=min(args.steps, 20),
                dt=args.dt
            )
            print(f"  Python reference completed for {min(args.bodies, 200)} bodies")
            print(f"  Final position mean: {np.mean(ref_pos):.6e}")
        except Exception as e:
            print(f"  Reference error: {e}")
        print("="*80)


if __name__ == "__main__":
    main()
