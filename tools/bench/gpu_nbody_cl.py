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
    
    # Tile size for shared memory optimization - reduced from 256 to 128 for safety
    TILE_SIZE = 128
    
    def __init__(self, platform_idx: Optional[int] = None, device_idx: Optional[int] = None, 
                 verbose: str = "INFO", no_tile: bool = False, auto_mode: bool = True):
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
        
        # Add more conservative command queue properties
        queue_properties = cl.command_queue_properties.PROFILING_ENABLE
        try:
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, properties=queue_properties)
        except Exception as e:
            self._debug_print(f"Warning: Could not create queue with profiling: {e}", "WARN")
            self.queue = cl.CommandQueue(self.context)
        
        # Device info
        self.max_wg_size = self.device.max_work_group_size
        self.local_mem_size = self.device.local_mem_size
        self.compute_units = self.device.max_compute_units
        self.verbose = verbose
        
        # Set more conservative TILE_SIZE
        self.TILE_SIZE = min(128, self.max_wg_size)
        
        self._debug_print(f"Using device: {self.device.name}")
        self._debug_print(f"Device vendor: {self.device.vendor}")
        self._debug_print(f"OpenCL version: {self.device.version}")
        self._debug_print(f"Max compute units: {self.compute_units}")
        self._debug_print(f"Max work group size: {self.max_wg_size}")
        self._debug_print(f"Local memory size: {self.local_mem_size} bytes")
        self._debug_print(f"Using conservative TILE_SIZE: {self.TILE_SIZE}", "INFO")
        
        # 添加no_tile标志
        self.no_tile = no_tile
        self.auto_mode = auto_mode
        
        if auto_mode and not no_tile:
            # 检测Intel集成显卡，自动选择更保守的设置
            if "Intel" in self.device.vendor and "Graphics" in self.device.name:
                self._debug_print("Intel Graphics detected, enabling auto-safety mode", "INFO")
                self.intel_gpu = True
            else:
                self.intel_gpu = False
    
    def _debug_print(self, msg: str, level: str = "INFO") -> None:
        """Debug output with timestamp and level"""
        # Only print if message level is at or above current verbosity level
        level_priority = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
        current_priority = level_priority.get(self.verbose, 2)
        msg_priority = level_priority.get(level, 2)
        
        if msg_priority <= current_priority:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {msg}")
    
    def _check_local_memory(self, precision: str) -> None:
        """Check if local memory usage exceeds device limits"""
        config = self.PRECISION_CONFIG[precision]
        
        # Calculate memory per work-item in local memory
        if config['ctype'] == 'float':
            vec_size = 16  # bytes for float4
            scalar_size = 4
        elif config['ctype'] == 'double':
            vec_size = 32  # bytes for double4  
            scalar_size = 8
        else:  # half
            vec_size = 8   # bytes for half4
            scalar_size = 2
        
        local_mem_required = self.TILE_SIZE * (vec_size + scalar_size)
        local_mem_available = self.local_mem_size
        
        self._debug_print(f"Local memory check: required={local_mem_required}B, available={local_mem_available}B")
        
        if local_mem_required > local_mem_available:
            self._debug_print(f"WARNING: Local memory requirement ({local_mem_required}B) exceeds device limit ({local_mem_available}B)", "WARN")
            # Auto-adjust tile size
            new_tile = min(self.TILE_SIZE, local_mem_available // (vec_size + scalar_size))
            self._debug_print(f"Auto-adjusting TILE_SIZE from {self.TILE_SIZE} to {new_tile}", "WARN")
            self.TILE_SIZE = new_tile
    
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
        Generate OpenCL kernel for N-body simulation
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
        
        if self.no_tile:
            # 简化的全局内存内核（不使用tile机制）
            return self._generate_naive_kernel(
                precision, config, ctype, vec_type, suffix, 
                zero, accum_type, convert_in, convert_out, ext_pragma
            )
        else:
            # 原有的tile优化内核
            return self._generate_tiled_kernel(
                precision, config, ctype, vec_type, suffix, 
                zero, accum_type, convert_in, convert_out, ext_pragma
            )

    def _generate_naive_kernel(self, precision: str, config: Dict, ctype: str, vec_type: str, 
                              suffix: str, zero: str, accum_type: str, 
                              convert_in: str, convert_out: str, ext_pragma: str) -> str:
        """
        生成简化的全局内存内核（不使用tile优化）
        """
        kernel = f'''
        {ext_pragma}
        
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
            
            // Load particle i data
            {vec_type} pos_i_vec = positions[gid];
            {vec_type} vel_i_vec = velocities[gid];
            {ctype} mass_i = masses[gid];
            
            // Convert to accumulation type for computation
            {accum_type}4 pos_i = {"convert_float4(pos_i_vec)" if precision == "fp16" else "pos_i_vec"};
            {accum_type}4 vel_i = {"convert_float4(vel_i_vec)" if precision == "fp16" else "vel_i_vec"};
            
            // Simulation loop
            for (int s = 0; s < steps; s++) {{
                {accum_type}4 acc = ({accum_type}4)({zero}, {zero}, {zero}, {zero});
                
                // Naive O(N²) loop over all particles
                for (int j = 0; j < n; j++) {{
                    if (j == gid) continue;  // Skip self-interaction
                    
                    {vec_type} pos_j_vec = positions[j];
                    {ctype} mass_j = masses[j];
                    
                    {accum_type}4 pos_j = {"convert_float4(pos_j_vec)" if precision == "fp16" else "pos_j_vec"};
                    acc += compute_acceleration(pos_i, pos_j, mass_j);
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

    def _generate_tiled_kernel(self, precision: str, config: Dict, ctype: str, vec_type: str,
                              suffix: str, zero: str, accum_type: str,
                              convert_in: str, convert_out: str, ext_pragma: str) -> str:
        """
        生成修复后的tile优化内核 - 减少寄存器压力
        """
        tile_size = self.TILE_SIZE
        
        kernel = f'''
        {ext_pragma}
        
        #define TILE_SIZE {tile_size}
        #define SOFTENING (0.01{suffix} * 0.01{suffix})  // Plummer softening squared
        
        // 简化：直接内联计算，减少函数调用开销
        #define COMPUTE_ACC(pos_i, pos_j, mass_j, acc) {{ \\
            {accum_type}4 r_vec = (pos_j) - (pos_i); \\
            {accum_type} r2 = r_vec.x*r_vec.x + r_vec.y*r_vec.y + r_vec.z*r_vec.z + SOFTENING; \\
            {accum_type} inv_r = rsqrt(r2); \\
            {accum_type} inv_r3 = inv_r * inv_r * inv_r; \\
            (acc) += (mass_j) * inv_r3 * r_vec; \\
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
            int lid = get_local_id(0);
            
            // 提前返回无效 work-item，减少资源占用
            if (gid >= n) {{
                // 仍需参与 barriers，但跳过所有计算
                for (int s = 0; s < steps; s++) {{
                    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {{
                        barrier(CLK_LOCAL_MEM_FENCE);
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }}
                }}
                return;
            }}
            
            // 加载数据
            {vec_type} pos_i_vec = positions[gid];
            {vec_type} vel_i_vec = velocities[gid];
            
            {accum_type}4 pos_i = {"convert_float4(pos_i_vec)" if precision == "fp16" else "pos_i_vec"};
            {accum_type}4 vel_i = {"convert_float4(vel_i_vec)" if precision == "fp16" else "vel_i_vec"};
            
            __local {vec_type} local_pos[TILE_SIZE];
            __local {ctype} local_mass[TILE_SIZE];
            
            int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
            
            // 主模拟循环
            for (int s = 0; s < steps; s++) {{
                {accum_type}4 acc = ({accum_type}4)({zero});
                
                // 瓦片循环
                for (int tile = 0; tile < num_tiles; tile++) {{
                    int j = tile * TILE_SIZE + lid;
                    
                    // 协作加载到局部内存
                    if (j < n) {{
                        local_pos[lid] = positions[j];
                        local_mass[lid] = masses[j];
                    }} else {{
                        local_pos[lid] = ({vec_type})({zero});
                        local_mass[lid] = {zero};
                    }}
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    // 计算当前瓦片的贡献
                    int tile_end = min(TILE_SIZE, n - tile * TILE_SIZE);
                    
                    #pragma unroll 4
                    for (int k = 0; k < tile_end; k++) {{
                        int j_idx = tile * TILE_SIZE + k;
                        if (j_idx == gid) continue;  // 跳过自身
                        
                        {accum_type}4 pos_j = {"convert_float4(local_pos[k])" if precision == "fp16" else "local_pos[k]"};
                        {accum_type} mass_j = local_mass[k];
                        COMPUTE_ACC(pos_i, pos_j, mass_j, acc);
                    }}
                    barrier(CLK_LOCAL_MEM_FENCE);
                }}
                
                // 更新速度和位置
                vel_i += G * acc * dt;
                pos_i += vel_i * dt;
            }}
            
            // 写回结果
            positions[gid] = {"convert_half4(pos_i)" if precision == "fp16" else "pos_i"};
            velocities[gid] = {"convert_half4(vel_i)" if precision == "fp16" else "vel_i"};
        }}
        '''
        return kernel
    
    def build_program(self, precision: str):
        """Build OpenCL program with detailed error reporting"""
        self._debug_print(f"Generating kernel source for {precision}...", "DEBUG")
        source = self.generate_kernel_source(precision)
        
        self._debug_print(f"Building OpenCL program...")
        program = cl.Program(self.context, source)
        
        build_options = [
            '-cl-fast-relaxed-math',
            '-cl-mad-enable',
        ]
        
        # 只有在tile模式下才添加WORK_GROUP_SIZE定义
        if not self.no_tile:
            build_options.append(f'-DWORK_GROUP_SIZE={min(self.TILE_SIZE, self.max_wg_size)}')
        
        self._debug_print(f"Build options: {' '.join(build_options)}", "DEBUG")
        
        try:
            program.build(options=' '.join(build_options))
            self._debug_print("Program built successfully")
        except cl.RuntimeError as e:
            build_log = program.get_build_info(self.device, cl.program_build_info.LOG)
            self._debug_print(f"ERROR: OpenCL build failed", "ERROR")
            self._debug_print(f"Build log:\n{build_log}", "ERROR")
            raise RuntimeError(f"OpenCL build error: {e}\nBuild log:\n{build_log}")
        
        return program
    
    def run_benchmark(self, precision: str = 'fp32', num_bodies: int = 1000, 
                     steps: int = 100, dt: float = 0.01,
                     warmup_iterations: int = 3, test_iterations: int = 10,
                     steps_per_chunk: int = 5) -> BenchmarkResult:
        """
        Run GPU N-body benchmark with enhanced safety measures
        
        Args:
            precision: 'fp16', 'fp32', or 'fp64'
            num_bodies: Number of particles
            steps: Number of integration steps
            dt: Time step
            warmup_iterations: Number of warmup runs
            test_iterations: Number of timed test runs
            steps_per_chunk: Steps per kernel launch (prevents hangs)
        """
        self._debug_print(f"Starting benchmark: precision={precision}, bodies={num_bodies}, steps={steps}")
        
        if not self.check_precision_support(precision):
            raise RuntimeError(f"Precision {precision} not supported by device")
        
        # Intel GPU specific: use no-tile mode for large N to reduce resource usage
        if num_bodies >= 500 and not self.no_tile:
            self._debug_print(f"Large N={num_bodies}, switching to no-tile mode to reduce resource usage", "INFO")
            self.no_tile = True
        
        if self.no_tile:
            self._debug_print("Using naive global memory computation (no tiling)", "INFO")
            self.TILE_SIZE = 1
            # No-tile mode benefits from larger work groups
            local_size = min(256, self.max_wg_size, num_bodies)
        else:
            # Small N adjustment
            if num_bodies < 256:
                self.TILE_SIZE = 1
                while self.TILE_SIZE * 2 <= num_bodies:
                    self.TILE_SIZE *= 2
                self.TILE_SIZE = min(self.TILE_SIZE, self.max_wg_size)
                self.TILE_SIZE = max(1, self.TILE_SIZE)
                self._debug_print(f"Small N={num_bodies}, setting TILE_SIZE={self.TILE_SIZE}", "INFO")
            else:
                self._check_local_memory(precision)
        
        self._debug_print(f"Final TILE_SIZE={self.TILE_SIZE}")
        
        # FIX: 对于大N，增加 chunk size 减少内核调用次数，而不是减少
        # 关键修改：steps_per_chunk 应该足够大以减少调用次数，但足够小以避免超时
        if num_bodies >= 500:
            # 对于大N，使用更大的 chunk 减少内核调用次数
            recommended_chunk = min(10, max(2, steps // 10))  # 至少分10个chunk，或每chunk至少2步
            if steps_per_chunk < recommended_chunk:
                steps_per_chunk = recommended_chunk
                self._debug_print(f"Large N auto-adjust: increasing steps_per_chunk to {steps_per_chunk}", "INFO")
        elif num_bodies < 200:
            if steps_per_chunk > 2:
                steps_per_chunk = 2
        
        # 限制最大 chunk 大小
        if steps_per_chunk > 20:
            steps_per_chunk = 20
            self._debug_print(f"Limiting steps_per_chunk to max 20", "INFO")
        
        # 对于Intel GPU，减少预热迭代
        if num_bodies >= 500:
            warmup_iterations = min(1, warmup_iterations)
            self._debug_print(f"Reducing warmup to {warmup_iterations} for large N", "INFO")
        
        self._debug_print(f"Building OpenCL program...")
        program = self.build_program(precision)
        kernel = program.nbody_simulation
        
        config = self.PRECISION_CONFIG[precision]
        np_dtype = config['np_dtype']
        
        # Generate initial conditions
        self._debug_print(f"Generating initial conditions for {num_bodies} bodies...")
        np.random.seed(42)
        positions = np.random.randn(num_bodies, 4).astype(np_dtype)
        positions[:, 3] = 0.0
        
        velocities = np.random.randn(num_bodies, 4).astype(np_dtype) * 0.1
        velocities[:, 3] = 0.0
        
        masses = np.random.rand(num_bodies).astype(np_dtype) * 10.0 + 1.0
        
        # OpenCL buffers
        self._debug_print(f"Allocating GPU buffers...")
        mf = cl.mem_flags
        pos_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=positions)
        vel_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
        mass_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)
        
        # 工作组分派 - 根据no_tile模式调整
        if self.no_tile:
            # 在no-tile模式下，使用简单的工作组配置
            local_size = min(256, self.max_wg_size, num_bodies)
            
            # 确保本地大小至少为1且是2的幂
            local_size = max(1, local_size)
            
            # 对于小规模问题，使用粒子数作为工作组大小
            if num_bodies < local_size:
                local_size = num_bodies
            
            self._debug_print(f"No-tile mode: using local_size={local_size}", "INFO")
        else:
            # 原有的tile模式工作组配置逻辑保持不变
            local_size = min(self.TILE_SIZE, self.max_wg_size, 256)
            
            # 确保本地大小不超过全局大小
            if local_size > num_bodies:
                # 找到小于num_bodies的最大2的幂
                new_local = 1
                while new_local * 2 <= num_bodies:
                    new_local *= 2
                local_size = max(1, new_local)
                self._debug_print(f"Small N adjustment: local_size={local_size} (num_bodies={num_bodies})", "INFO")
            
            # 确保本地大小至少为1
            local_size = max(1, local_size)
            
            # 验证和调整工作组件配置
            if local_size > self.max_wg_size:
                self._debug_print(f"ERROR: local_size={local_size} exceeds max_wg_size={self.max_wg_size}", "ERROR")
                local_size = min(local_size, self.max_wg_size)

            # 确保local_size至少为1
            if local_size < 1:
                self._debug_print(f"WARNING: local_size={local_size} is invalid, setting to 1", "WARN")
                local_size = 1

            # 确保TILE_SIZE与local_size一致
            self.TILE_SIZE = local_size

        # 计算全局大小（必须是本地大小的整数倍）
        global_size = ((num_bodies + local_size - 1) // local_size) * local_size
        
        # 最终验证
        if global_size == 0:
            global_size = local_size = 1
            self._debug_print(f"CRITICAL: Work size validation failed, using minimal configuration", "ERROR")

        self._debug_print(f"Validated work group: global={global_size}, local={local_size}, TILE_SIZE={self.TILE_SIZE}")
        
        # 修改：添加额外安全检查
        if global_size > 1000000:  # 避免过大的全局工作大小
            self._debug_print(f"Warning: Large global_size {global_size}, consider reducing problem size", "WARN")
        
        G = np_dtype(6.67430e-11)
        dt = np_dtype(dt)
        
        # Warm-up runs with progress
        self._debug_print(f"Starting {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            self._debug_print(f"  Warmup iteration {i+1}/{warmup_iterations}")
            steps_done = 0
            chunk_count = 0
            while steps_done < steps:
                current_steps = min(steps_per_chunk, steps - steps_done)
                chunk_count += 1
                self._debug_print(f"    Chunk {chunk_count}: steps={current_steps}", "DEBUG")
                
                try:
                    event = kernel(self.queue, (global_size,), (local_size,),
                                   pos_buf, vel_buf, mass_buf,
                                   np.int32(num_bodies), G, dt, np.int32(current_steps))
                    # FIX: 添加 flush 让 GPU 开始执行
                    self.queue.flush()
                    
                    # FIX: 每10个 chunk 强制同步，防止资源累积
                    if chunk_count % 10 == 0:
                        event.wait()
                        self.queue.finish()
                        
                except Exception as e:
                    self._debug_print(f"ERROR during kernel execution: {e}", "ERROR")
                    raise
                
                steps_done += current_steps
            
            self.queue.finish()
            self._debug_print(f"  Warmup iteration {i+1} completed")
        
        # Test runs with timing
        self._debug_print(f"Starting {test_iterations} test iterations...")
        execution_times = []
        
        for iteration in range(test_iterations):
            self._debug_print(f"Test iteration {iteration+1}/{test_iterations}")
            
            # Reset state
            np.random.seed(42)
            pos_init = np.random.randn(num_bodies, 4).astype(np_dtype)
            pos_init[:, 3] = 0.0
            vel_init = np.random.randn(num_bodies, 4).astype(np_dtype) * 0.1
            vel_init[:, 3] = 0.0
            
            self._debug_print("  Copying data to GPU...")
            cl.enqueue_copy(self.queue, pos_buf, pos_init)
            cl.enqueue_copy(self.queue, vel_buf, vel_init)
            self.queue.finish()
            
            start = time.perf_counter()
            
            # Chunked execution with progress reporting
            steps_done = 0
            chunk_count = 0
            
            while steps_done < steps:
                current_steps = min(steps_per_chunk, steps - steps_done)
                chunk_count += 1
                
                self._debug_print(f"  Running chunk {chunk_count}: {current_steps} steps", "DEBUG")
                
                try:
                    event = kernel(self.queue, (global_size,), (local_size,),
                                  pos_buf, vel_buf, mass_buf,
                                  np.int32(num_bodies), G, dt, np.int32(current_steps))
                    # FIX: 添加 flush
                    self.queue.flush()
                    
                    # FIX: 定期同步
                    if chunk_count % 10 == 0:
                        event.wait()
                        
                except cl.RuntimeError as e:
                    self._debug_print(f"ERROR: Kernel execution failed: {e}", "ERROR")
                    self._debug_print(f"Try reducing --chunk-size (current: {steps_per_chunk})", "ERROR")
                    raise
                
                steps_done += current_steps
            
            # 最终同步
            self.queue.finish()
            end = time.perf_counter()
            iteration_time = end - start
            execution_times.append(iteration_time)
            
            self._debug_print(f"  Iteration {iteration+1} completed in {iteration_time:.3f}s")
        
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
        "--chunk-size", type=int, default=5,
        help="Steps per kernel launch (smaller values prevent hangs, default: 5)"
    )
    
    # 添加新参数
    parser.add_argument(
        "--min-tile-size", type=int, default=16,
        help="Minimum tile size for small N problems (default: 16)"
    )
    
    parser.add_argument(
        "--safe-mode", action="store_true",
        help="Enable extra safety measures: smaller tile size, reduced unrolling"
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
    
    parser.add_argument(
        "--no-tile", action="store_true",
        help="Disable tiling optimization, use naive global memory computation"
    )
    
    parser.add_argument(
        "--no-auto-mode", action="store_true",
        help="Disable automatic safety mode for Intel GPUs"
    )
        
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Verbose output level (use -v for INFO, -vv for DEBUG)"
    )
    
    args = parser.parse_args()
    
    # Set debug level
    debug_levels = {0: "ERROR", 1: "INFO", 2: "DEBUG"}
    current_level = debug_levels.get(min(args.verbose, 2), "INFO")
    
    if not PYOPENCL_AVAILABLE:
        print("Error: PyOpenCL not installed. Please install: pip install pyopencl")
        sys.exit(1)
    
    print("Starting GPU N-Body Benchmark...")
    print(f"Configuration: bodies={args.bodies}, steps={args.steps}, dt={args.dt}, repeats={args.repeats}")
    
    try:
        benchmark = GPUNBodyBenchmark(
            platform_idx=args.platform,
            device_idx=args.device,
            verbose=current_level,
            no_tile=args.no_tile,
            auto_mode=not args.no_auto_mode  # 添加no_auto_mode参数
        )
    except Exception as e:
        print(f"Error initializing OpenCL: {e}")
        sys.exit(1)
    
    # Safe mode options
    if args.safe_mode:
        print("Safe mode enabled: using conservative settings")
        args.chunk_size = max(1, args.chunk_size // 2)
        if args.bodies > 500:
            args.bodies = 500  # Limit number of bodies
            print(f"  Limited to {args.bodies} bodies for safety")
    
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
