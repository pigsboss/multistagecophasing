#!/usr/bin/env python3
"""
GPU Monte Carlo Benchmark using OpenCL
Estimate Pi using Monte Carlo method with FP16/FP32/FP64 precision
Supports batched execution for handling massive sample counts within memory constraints
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
    total_samples: int = 0  # Actual total samples processed
    num_batches: int = 1    # Number of batches used
    
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
    def samples_per_second(self) -> float:
        """Return samples processed per second"""
        return self.total_samples / self.avg_time if self.avg_time > 0 else 0.0
    
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
            "samples_per_second": self.samples_per_second,
            "memory_usage": self.memory_usage,
            "notes": self.notes,
            "result_value": self.result_value,
            "total_samples": self.total_samples,
            "num_batches": self.num_batches
        }


class GPUMonteCarloBenchmark:
    """OpenCL GPU Monte Carlo Benchmark supporting multiple precisions"""
    
    PRECISION_CONFIG = {
        'fp32': {
            'ctype': 'float',
            'vec_type': 'float4',
            'suffix': 'f',
            'np_dtype': np.float32,
            'one': '1.0f',
            'ext_pragma': '',
            'accum_ctype': 'float',
            'vec_size': 4
        },
        'fp64': {
            'ctype': 'double',
            'vec_type': 'double4',
            'suffix': '',
            'np_dtype': np.float64,
            'one': '1.0',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp64 : enable',
            'accum_ctype': 'double',
            'vec_size': 4
        },
        'fp16': {
            'ctype': 'half',
            'vec_type': 'half4',
            'suffix': 'h',
            'np_dtype': np.float16,
            'one': '1.0h',
            'ext_pragma': '#pragma OPENCL EXTENSION cl_khr_fp16 : enable',
            'accum_ctype': 'float',  # Mixed precision: use fp32 for accumulation
            'vec_size': 4
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
        
        # Query device properties for optimization (from SKILL.opencl.md)
        self.max_wg_size = self.device.max_work_group_size
        self.preferred_wg_size = getattr(self.device, 'preferred_work_group_size_multiple', 64)
        self.local_mem_size = self.device.local_mem_size
        self.compute_units = self.device.max_compute_units
        
        print(f"Using device: {self.device.name}")
        print(f"Device vendor: {self.device.vendor}")
        print(f"OpenCL version: {self.device.version}")
        print(f"Max compute units: {self.compute_units}")
        print(f"Max work group size: {self.max_wg_size}")
        print(f"Preferred work group size multiple: {self.preferred_wg_size}")
        print(f"Local memory size: {self.local_mem_size / 1024:.1f} KB")
    
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
        """Generate optimized OpenCL kernel using techniques from SKILL.opencl.md"""
        config = self.PRECISION_CONFIG[precision]
        ctype = config['ctype']
        vec_type = config['vec_type']
        suffix = config['suffix']
        one = config['one']
        ext_pragma = config['ext_pragma']
        accum_ctype = config['accum_ctype']
        
        kernel = f'''
        {ext_pragma}
        
        // 64-bit atomic add with proper fallback (from SKILL.opencl.md)
        #if defined(cl_khr_int64_base_atomics)
            #define ATOMIC_ADD_ULONG(ptr, val) atom_add(ptr, val)
        #else
            // Fallback implementation using compare-and-swap
            inline void atomic_add_ulong_fallback(global ulong *ptr, ulong val) {{
                union {{
                    ulong u64;
                    uint2 u32;
                }} old, new_val, read_back;
                
                do {{
                    old.u64 = *ptr;
                    new_val.u64 = old.u64 + val;
                    read_back.u32 = atom_cmpxchg((global uint2*)ptr, old.u32, new_val.u32);
                }} while (read_back.u32.x != old.u32.x || read_back.u32.y != old.u32.y);
            }}
            #define ATOMIC_ADD_ULONG(ptr, val) atomic_add_ulong_fallback(ptr, val)
        #endif
        
        // Optimized Philox4x32 RNG (from SKILL.opencl.md best practices)
        inline uint4 philox4x32_round(uint4 ctr, uint4 key) {{
            const uint M0 = 0xD2511F53u;
            const uint M1 = 0xCD9E8D57u;
            
            uint hi0 = mul_hi(M0, ctr.x);
            uint hi1 = mul_hi(M1, ctr.z);
            
            return (uint4)(
                hi1 ^ ctr.y ^ key.x,
                M1 * ctr.z,
                hi0 ^ ctr.w ^ key.y,
                M0 * ctr.x
            );
        }}
        
        inline uint4 philox4x32(uint4 ctr, uint4 key) {{
            // 10 rounds for good statistical quality
            #pragma unroll 5
            for (int i = 0; i < 5; i++) {{
                ctr = philox4x32_round(ctr, key);
                key.xy += (uint2)(0x9E3779B9u, 0xBB67AE85u);
                key.zw += (uint2)(0x9E3779B9u, 0xBB67AE85u);
                ctr = philox4x32_round(ctr, key);
                key.xy += (uint2)(0x9E3779B9u, 0xBB67AE85u);
                key.zw += (uint2)(0x9E3779B9u, 0xBB67AE85u);
            }}
            return ctr;
        }}
        
        // Convert to [0,1) with proper masking (coalesced access pattern)
        inline {vec_type} uint4_to_float4(uint4 u) {{
            const uint MASK = 0x3FFFFFFFu;  // 30 bits for better precision
            const {ctype} SCALE = 1.0{suffix} / ({ctype})(1u << 30);
            
            u &= MASK;
            return ({vec_type})(
                ({ctype})u.x * SCALE,
                ({ctype})u.y * SCALE,
                ({ctype})u.z * SCALE,
                ({ctype})u.w * SCALE
            );
        }}
        
        // Main Monte Carlo kernel with coalesced memory access
        kernel void monte_carlo_pi(
            const ulong samples_per_item,
            const uint base_seed,
            global ulong *global_counter,
            local ulong *local_counter
        ) {{
            uint gid = get_global_id(0);
            uint lid = get_local_id(0);
            uint wg_size = get_local_size(0);
            
            // Coalesced memory access pattern: consecutive global IDs access consecutive memory
            uint4 key = (uint4)(
                base_seed + gid * 4,
                base_seed + gid * 4 + 1,
                base_seed + gid * 4 + 2,
                base_seed + gid * 4 + 3
            );
            
            // Initialize counter with proper distribution
            uint4 counter = (uint4)(
                gid,
                (gid >> 16) | (gid << 16),  // Spread bits for better distribution
                0,
                0
            );
            
            {accum_ctype} local_count = 0{suffix};
            
            // Vectorized processing with loop unrolling
            for (ulong i = 0; i < samples_per_item; i++) {{
                counter.z = (uint)(i & 0xFFFFFFFFu);
                counter.w = (uint)(i >> 32);
                
                // Generate x coordinates
                uint4 rand_x = philox4x32(counter, key);
                {vec_type} x = uint4_to_float4(rand_x);
                
                // Generate y coordinates with different key
                uint4 key_y = key + (uint4)(0xB5AD4ECEu, 0x0F1DBCB3u, 0x876D1B3u, 0xE12047C5u);
                uint4 rand_y = philox4x32(counter, key_y);
                {vec_type} y = uint4_to_float4(rand_y);
                
                // Vectorized distance calculation
                {vec_type} dist = x * x + y * y;
                
                // Branchless counting (from SKILL.opencl.md optimization)
                local_count += ({accum_ctype})(dist.x <= {one}) * 1{suffix};
                local_count += ({accum_ctype})(dist.y <= {one}) * 1{suffix};
                local_count += ({accum_ctype})(dist.z <= {one}) * 1{suffix};
                local_count += ({accum_ctype})(dist.w <= {one}) * 1{suffix};
            }}
            
            // Local reduction using tree pattern (optimized for GPU)
            local_counter[lid] = (ulong)local_count;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Tree reduction with power-of-two stride
            for (uint stride = wg_size >> 1; stride > 0; stride >>= 1) {{
                if (lid < stride) {{
                    local_counter[lid] += local_counter[lid + stride];
                }}
                barrier(CLK_LOCAL_MEM_FENCE);
            }}
            
            // Atomic update to global memory (single operation per work-group)
            if (lid == 0) {{
                ATOMIC_ADD_ULONG(global_counter, local_counter[0]);
            }}
        }}
        '''
        return kernel
    
    def build_program(self, precision: str):
        """Build OpenCL program with optimization flags from SKILL.opencl.md"""
        source = self.generate_kernel_source(precision)
        program = cl.Program(self.context, source)
        
        # Build options optimized for different device types (from SKILL.opencl.md)
        build_options = [
            '-cl-fast-relaxed-math',
            '-cl-mad-enable',
            '-cl-no-signed-zeros',
            '-cl-unsafe-math-optimizations',
            f'-DWORK_GROUP_SIZE={self.preferred_wg_size}',
            f'-DMAX_WORK_GROUP_SIZE={self.max_wg_size}'
        ]
        
        # Add device-specific optimizations
        device_type = self.device.type
        if device_type == cl.device_type.GPU:
            build_options.append('-cl-std=CL1.2')
        elif device_type == cl.device_type.CPU:
            build_options.append('-cl-opt-disable')  # Better for CPU debugging
        
        try:
            program.build(options=' '.join(build_options))
        except cl.RuntimeError as e:
            build_log = program.get_build_info(self.device, cl.program_build_info.LOG)
            # Check for specific errors
            if 'cl_khr_int64_base_atomics' in build_log and precision == 'fp64':
                print("Warning: 64-bit atomics not fully supported, using fallback")
                # Rebuild without 64-bit atomics dependency
                build_options.append('-DNO_64BIT_ATOMICS')
                program = cl.Program(self.context, source)
                program.build(options=' '.join(build_options))
            else:
                raise RuntimeError(f"OpenCL build error: {e}\nBuild log:\n{build_log}")
        
        return program
    
    def run_benchmark(self, precision: str = 'fp32', total_samples: Optional[int] = None,
                     num_batches: int = 1, samples_per_batch: Optional[int] = None,
                     warmup_iterations: int = 3, test_iterations: int = 10) -> BenchmarkResult:
        """
        Run GPU Monte Carlo benchmark with batched execution support
        
        Args:
            precision: 'fp16', 'fp32', or 'fp64'
            total_samples: Total number of random samples (alternative to batches)
            num_batches: Number of batches to run (for memory-constrained execution)
            samples_per_batch: Samples per batch (if None, calculated from total_samples)
            warmup_iterations: Number of warmup runs
            test_iterations: Number of timed test runs
            
        Note:
            Either specify total_samples OR (num_batches, samples_per_batch)
        """
        if not self.check_precision_support(precision):
            raise RuntimeError(f"Precision {precision} not supported by device")
        
        # Calculate batch parameters
        if total_samples is not None:
            # Traditional mode: divide total into batches
            if samples_per_batch is None:
                samples_per_batch = (total_samples + num_batches - 1) // num_batches
            actual_total = samples_per_batch * num_batches
        else:
            # Batch mode: total is product of batches and per-batch
            actual_total = samples_per_batch * num_batches
        
        program = self.build_program(precision)
        kernel = program.monte_carlo_pi
        
        # Optimize work group configuration (from SKILL.opencl.md best practices)
        # First, build the program to get kernel info
        program = self.build_program(precision)
        kernel = program.monte_carlo_pi
        
        # Optimize work group size
        try:
            # Get preferred work group size multiple
            preferred_multiple = self.preferred_wg_size
            
            # Query kernel-specific work group info
            compile_wg_size = kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE,
                self.device
            )
            
            # Use local memory size to determine optimal size
            local_mem_required = np.dtype(np.uint64).itemsize * compile_wg_size
            if local_mem_required > self.local_mem_size:
                # Reduce work group size to fit in local memory
                compile_wg_size = self.local_mem_size // np.dtype(np.uint64).itemsize
            
            # Round to preferred multiple for better performance
            optimal_size = (compile_wg_size // preferred_multiple) * preferred_multiple
            optimal_size = max(optimal_size, preferred_multiple)
            optimal_size = min(optimal_size, self.max_wg_size)
            
            # Ensure it's a power of two for efficient reduction
            optimal_size = 1 << (int(np.log2(optimal_size)))
            local_size = optimal_size
        except Exception as e:
            print(f"Warning: Could not query work group info: {e}")
            # Fallback to device preferred size
            local_size = min(256, self.preferred_wg_size)
        
        # Calculate optimal global size based on compute units and occupancy
        # Aim for 2-4 work-groups per compute unit for good occupancy
        target_work_groups = self.compute_units * 4
        min_global_items = target_work_groups * local_size
        
        # Ensure each work-item processes enough samples for good efficiency
        # Target 1024-4096 samples per work-item for good arithmetic intensity
        target_samples_per_item = 2048
        estimated_items = max(min_global_items, samples_per_batch // target_samples_per_item)
        
        # Round up to nearest multiple of local_size for proper work-group distribution
        num_work_items = ((estimated_items + local_size - 1) // local_size) * local_size
        
        # Vectorized processing: each iteration processes 4 samples
        vec_size = 4
        samples_per_vec_iter = (samples_per_batch + num_work_items * vec_size - 1) // (num_work_items * vec_size)
        
        # Adjust to ensure total samples is multiple of vector size
        adjusted_batch_samples = samples_per_vec_iter * num_work_items * vec_size
        
        global_size = num_work_items
        
        # Validate memory requirements
        required_local_mem = np.dtype(np.uint64).itemsize * local_size
        if required_local_mem > self.local_mem_size:
            print(f"Warning: Local memory requirement ({required_local_mem} bytes) "
                  f"exceeds device capacity ({self.local_mem_size} bytes)")
            # Reduce local size to fit
            local_size = self.local_mem_size // np.dtype(np.uint64).itemsize
            local_size = (local_size // self.preferred_wg_size) * self.preferred_wg_size
            local_size = max(local_size, self.preferred_wg_size)
        
        # Prepare buffers
        # For the kernel, we need a global counter and partial counts
        # The kernel signature expects: samples_per_item, base_seed, global_counter, local_counter
        # So we need to adjust our approach
        
        # Create a buffer for the global counter
        global_counter_np = np.zeros(1, dtype=np.uint64)
        global_counter_buf = cl.Buffer(self.context, 
                                      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=global_counter_np)
        
        # Local memory for reduction
        local_mem_size = np.dtype(np.uint64).itemsize * local_size
        # Add padding for better memory alignment (from SKILL.opencl.md)
        local_mem_size = ((local_mem_size + 63) // 64) * 64
        
        # Calculate samples per item for the kernel
        # The kernel processes 4 samples per iteration, and we have samples_per_vec_iter iterations
        samples_per_item = samples_per_vec_iter
        
        # Warm-up runs (single batch for warmup)
        for i in range(warmup_iterations):
            # Reset global counter to zero
            cl.enqueue_copy(self.queue, global_counter_buf, global_counter_np)
            
            seed = np.uint32(10000 + i * 100000)
            kernel(self.queue, (global_size,), (local_size,),
                   np.uint64(samples_per_item), seed, 
                   global_counter_buf,
                   cl.LocalMemory(local_mem_size))
            self.queue.finish()
        
        # Test runs with batched execution
        execution_times = []
        batch_inside_totals = []  # Store inside counts for validation
        
        for i in range(test_iterations):
            total_inside = 0
            start = time.perf_counter()
            
            # Execute multiple batches
            for batch_idx in range(num_batches):
                # Reset global counter to zero for each batch
                cl.enqueue_copy(self.queue, global_counter_buf, global_counter_np)
                
                SEED_STRIDE = 100000
                seed = np.uint32(12345 + i * num_batches * SEED_STRIDE + batch_idx * SEED_STRIDE)
                
                event = kernel(self.queue, (global_size,), (local_size,),
                              np.uint64(samples_per_item), seed,
                              global_counter_buf,
                              cl.LocalMemory(local_mem_size))
                event.wait()
                
                # Read the global counter
                cl.enqueue_copy(self.queue, global_counter_np, global_counter_buf)
                self.queue.finish()
                total_inside += global_counter_np[0]
            
            end = time.perf_counter()
            execution_times.append(end - start)
            batch_inside_totals.append(total_inside)
        
        # Calculate final statistics
        avg_inside = np.mean(batch_inside_totals)
        pi_estimate = 4.0 * avg_inside / (adjusted_batch_samples * num_batches)
        
        return BenchmarkResult(
            task_name="Monte Carlo Pi (GPU Batched)",
            implementation=f"OpenCL {self.device.name}",
            precision=precision,
            execution_times=execution_times,
            notes=f"TotalSamples={adjusted_batch_samples * num_batches}, "
                  f"Batches={num_batches}, Samples/Batch={adjusted_batch_samples}, "
                  f"WorkItems={num_work_items}, Samples/Item={samples_per_item}",
            result_value=float(pi_estimate),
            total_samples=adjusted_batch_samples * num_batches,
            num_batches=num_batches
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
                print(f"  Total Samples: {result.total_samples:,}")
                print(f"  Batches: {result.num_batches}")
                print(f"  Time: {result.avg_time:.4f}s (min:{result.min_time:.4f}s, "
                      f"max:{result.max_time:.4f}s, med:{result.median_time:.4f}s)")
                print(f"  Iter/s: {result.iterations_per_second:.2f}")
                print(f"  Samples/s: {result.samples_per_second:,.0f}")
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
        description="GPU Monte Carlo Benchmark using OpenCL (FP16/FP32/FP64) with Batched Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode: 10M samples in single batch
  %(prog)s --precision fp32 --samples 10000000
  
  # Batched mode: 1B samples in 100 batches of 10M each (memory efficient)
  %(prog)s --precision fp32 --batches 100 --samples-per-batch 10000000
  
  # Large scale: 100B samples with small batches
  %(prog)s --precision fp64 --batches 10000 --samples-per-batch 10000000
  
  # Compare all precisions
  %(prog)s --precision all --samples 100000000
  
  # Save results
  %(prog)s --output results.json
        """
    )
    
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp16", "fp32", "fp64", "all"],
        help="Floating point precision to test (default: fp32)"
    )
    
    # Mutually exclusive group for sample specification
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        "--samples", type=int,
        help="Total number of Monte Carlo samples (single batch mode)"
    )
    sample_group.add_argument(
        "--batches", type=int,
        help="Number of batches for repeated execution (batch mode)"
    )
    
    parser.add_argument(
        "--samples-per-batch", type=int, default=10000000,
        help="Samples per batch when using --batches (default: 10000000)"
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
    
    # Determine execution mode
    if args.batches is not None:
        print(f"Batched mode: {args.batches} batches x {args.samples_per_batch:,} samples")
        total_samples = None
        num_batches = args.batches
        samples_per_batch = args.samples_per_batch
    else:
        print(f"Single batch mode: {args.samples:,} samples")
        total_samples = args.samples
        num_batches = 1
        samples_per_batch = None
    
    print(f"Repeats: {args.repeats}")
    
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
                total_samples=total_samples,
                num_batches=num_batches,
                samples_per_batch=samples_per_batch,
                warmup_iterations=args.warmup,
                test_iterations=args.repeats
            )
            results.append(result)
            print(f"  Average time: {result.avg_time:.4f}s")
            print(f"  Total samples: {result.total_samples:,}")
            print(f"  Samples/sec: {result.samples_per_second:,.0f}")
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
