#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Path-Integral Benchmark - JAX Universal Backend
===================================================

Supported backends:
- Metal (Apple Silicon)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Intel GPUs
- CPU (fallback)

Two independent computation methods:
1. Scalar Loops - Emulates OpenCL, explicit scalar computation with path/time loops
2. Vectorized - Emulates NumPy, explicit vectorized computation with time loops

All output follows MCPC Coding Standards (English only).
Algorithms consistent with OpenCL and CPU versions for comparable results.

Usage examples:
  python gpu_traj_jax.py --method scalar --backend metal
  python gpu_traj_jax.py --method vectorized --backend cuda
  python gpu_traj_jax.py --method both --use-lcg --output results.json
  python gpu_traj_jax.py --size (100000,10000) --chunk-size (5000,5000)  # Large task with chunking
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np

# ---------- Helper functions for 2D chunking ---------- #
def estimate_optimal_chunk_sizes(steps: int, paths: int, device_memory_gb: float = 16.0) -> Tuple[int, int]:
    """
    Estimate optimal 2D chunk sizes based on device memory constraints.
    
    Returns:
        Tuple of (chunk_steps, chunk_paths)
    """
    # Memory estimation: each path requires ~4 bytes * steps for intermediate arrays
    # Conservative estimate: use only 20% of device memory for working set
    available_bytes = device_memory_gb * 1024**3 * 0.2
    
    # Bytes per path per step (rough estimate for float32 arrays: x, integral, random, weight)
    bytes_per_cell = 4 * 4  # 4 arrays * 4 bytes
    
    total_cells = steps * paths
    memory_needed = total_cells * bytes_per_cell
    
    if memory_needed <= available_bytes:
        # No chunking needed
        return steps, paths
    
    # Need to chunk. Strategy: keep paths as large as possible (for vectorization),
    # chunk time dimension first, then path dimension if necessary.
    
    # Estimate max paths we can handle without time chunking
    max_paths_no_time_chunk = int(available_bytes / (steps * bytes_per_cell))
    
    if max_paths_no_time_chunk >= 128:  # Minimum for GPU efficiency
        # Only chunk time dimension
        optimal_chunk_paths = min(paths, ((max_paths_no_time_chunk // 128) * 128))
        optimal_chunk_steps = steps
    else:
        # Need 2D chunking
        optimal_chunk_paths = 1024  # Standard GPU workgroup size multiple
        cells_per_chunk = int(available_bytes / bytes_per_cell)
        optimal_chunk_steps = max(100, cells_per_chunk // optimal_chunk_paths)
        optimal_chunk_steps = min(optimal_chunk_steps, steps)
    
    print(f"Auto-chunking: steps={optimal_chunk_steps}, paths={optimal_chunk_paths}")
    return optimal_chunk_steps, optimal_chunk_paths

# ---------- JAX环境检查和后端检测 ---------- #
def detect_available_backends() -> List[str]:
    """检测可用的JAX后端"""
    available_backends = []
    
    try:
        import jax
        # 尝试导入各个后端的特定功能
        if jax.default_backend() != 'cpu':
            # 获取实际的后端信息
            try:
                devices = jax.devices()
                if devices:
                    platform = devices[0].platform
                    if platform in ['gpu', 'cuda']:
                        available_backends.append('cuda')
                    elif platform == 'metal':
                        available_backends.append('metal')
                    elif platform == 'rocm':
                        available_backends.append('rocm')
                    else:
                        available_backends.append(platform)
            except:
                pass
        
        # 如果没有检测到GPU后端，添加CPU作为后备
        if not available_backends:
            available_backends.append('cpu')
            
    except ImportError:
        print("Error: JAX not installed. Install with: pip install jax jaxlib", file=sys.stderr)
        sys.exit(1)
    
    return available_backends

def set_jax_backend(backend: str) -> None:
    """设置JAX后端"""
    import jax
    
    # 首先尝试设置环境变量
    os.environ['JAX_PLATFORM_NAME'] = backend
    
    # 重新初始化JAX（某些情况下需要）
    try:
        jax.config.update('jax_platform_name', backend)
    except:
        pass
    
    # 验证后端是否设置成功
    actual_backend = jax.default_backend()
    print(f"JAX backend set to: {actual_backend}")
    
    # 显示设备信息
    try:
        devices = jax.devices()
        print(f"Available devices: {[str(d) for d in devices]}")
    except:
        pass

# 检查JAX是否可用
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Error: JAX not installed.  pip install jax jaxlib", file=sys.stderr)
    sys.exit(1)

# ---------- 权重函数实现 ---------- #
def compute_weight_continuous(x):
    """
    6次多项式连续逼近原分段权重函数（方案C）
    计算量：6次乘法 + 6次加法（霍纳法则）
    无分支，适合GPU/向量化计算
    """
    # 预计算系数（基于最小二乘拟合 [-2, 2] 区间）
    c0 = 0.3125
    c1 = 0.234375
    c2 = -0.2734375
    c3 = -0.05859375
    c4 = 0.13671875
    c5 = 0.01953125
    c6 = -0.03125
    
    # 使用霍纳法则计算：(((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
    result = c6
    result = result * x + c5
    result = result * x + c4
    result = result * x + c3
    result = result * x + c2
    result = result * x + c1
    result = result * x + c0
    
    return result


def compute_weight_piecewise(x):
    """
    原分段权重函数（用于对比）
    """
    # 在JAX中使用jnp.where实现条件分支
    import jax.numpy as jnp
    
    return jnp.where(
        x < -1.0, 0.1,
        jnp.where(
            x < 0.0, 0.3 * x + 0.4,
            jnp.where(
                x < 1.0, 0.5 * (1.0 - x * x),
                0.2
            )
        )
    )


# 全局权重计算方法标志
_use_continuous_weight = False

def set_weight_method(use_continuous: bool = False):
    """设置权重计算方法：True=连续函数, False=分段函数"""
    global _use_continuous_weight
    _use_continuous_weight = use_continuous
    method_name = "continuous (polynomial)" if use_continuous else "piecewise"
    print(f"Weight calculation method set to: {method_name}")
    return use_continuous

# ---------- Detailed Timing Analysis ---------- #
@dataclass
class TimingBreakdown:
    """Four-category timing breakdown for benchmark analysis"""
    # Category I: Pure Python overhead
    python_setup: float = 0.0  # Module import, argument parsing, backend setup
    
    # Category II: JIT compilation
    jit_compilation: float = 0.0  # Tracing + XLA compilation time
    first_execution_total: float = 0.0  # First run (compile + execute)
    
    # Category III: Steady-state execution
    warmup_execution: float = 0.0  # Warmup iterations after first compile
    steady_state_avg: float = 0.0  # Average of test iterations
    steady_state_min: float = 0.0  # Best case execution
    steady_state_max: float = 0.0  # Worst case execution
    
    # Category IV: JAX runtime overhead
    jax_array_init: float = 0.0  # Array creation and initialization
    jax_memory_alloc: float = 0.0  # Memory allocation overhead
    jax_sync_overhead: float = 0.0  # block_until_ready and synchronization
    
    # Per-path and per-operation metrics
    per_path_time_ms: float = 0.0  # Time per path in milliseconds
    per_operation_time_ns: float = 0.0  # Time per (path, step) in nanoseconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_i_python_setup_ms": self.python_setup * 1000,
            "category_ii_jit_compilation_ms": self.jit_compilation * 1000,
            "category_ii_first_execution_ms": self.first_execution_total * 1000,
            "category_iii_steady_state_avg_ms": self.steady_state_avg * 1000,
            "category_iii_steady_state_min_ms": self.steady_state_min * 1000,
            "category_iii_steady_state_max_ms": self.steady_state_max * 1000,
            "category_iv_jax_array_init_ms": self.jax_array_init * 1000,
            "category_iv_jax_memory_alloc_ms": self.jax_memory_alloc * 1000,
            "category_iv_jax_sync_ms": self.jax_sync_overhead * 1000,
            "per_path_time_ms": self.per_path_time_ms,
            "per_operation_time_ns": self.per_operation_time_ns,
        }
    
    def print_summary(self, steps: int, paths: int):
        """Print formatted timing summary"""
        print("\n" + "="*70)
        print("DETAILED TIMING BREAKDOWN")
        print("="*70)
        
        total = (self.python_setup + self.jit_compilation + 
                self.steady_state_avg + self.jax_array_init)
        
        print(f"[I]   Pure Python Setup:        {self.python_setup*1000:8.2f} ms ({self.python_setup/total*100:5.1f}%)")
        print(f"      - Module import, argument parsing, backend initialization")
        
        print(f"\n[II]  JIT Compilation:          {self.jit_compilation*1000:8.2f} ms ({self.jit_compilation/total*100:5.1f}%)")
        print(f"      - First execution total:  {self.first_execution_total*1000:8.2f} ms")
        print(f"      - Second execution (pure):{(self.first_execution_total-self.jit_compilation)*1000:8.2f} ms")
        
        print(f"\n[III] Steady-State Execution:   {self.steady_state_avg*1000:8.2f} ms ({self.steady_state_avg/total*100:5.1f}%)")
        print(f"      - Min: {self.steady_state_min*1000:.2f} ms, Max: {self.steady_state_max*1000:.2f} ms")
        
        print(f"\n[IV]  JAX Runtime Overhead:     {self.jax_array_init*1000:8.2f} ms ({self.jax_array_init/total*100:5.1f}%)")
        print(f"      - Array initialization:   {self.jax_array_init*1000:8.2f} ms")
        print(f"      - Memory allocation:      {self.jax_memory_alloc*1000:8.2f} ms")
        print(f"      - Synchronization:        {self.jax_sync_overhead*1000:8.2f} ms")
        
        print(f"\n{'-'*70}")
        print(f"Per-Path Time:        {self.per_path_time_ms*1000:.3f} us ({self.per_path_time_ms:.6f} ms)")
        print(f"Per-Operation Time:   {self.per_operation_time_ns:.1f} ns  ({self.per_operation_time_ns/1000:.3f} us)")
        print(f"Total Operations:     {steps * paths:,} (steps × paths)")
        print(f"Theoretical Throughput: {1e9/self.per_operation_time_ns:.0e} ops/sec")
        print("="*70)

# ---------- shared result container (identical to cpu.py) ---------- #
@dataclass
class BenchmarkResult:
    """Benchmark result data class"""
    task_name: str
    implementation: str
    method_type: str  # 添加方法类型：'scalar' 或 'vectorized'
    backend: str  # 添加后端信息
    execution_times: List[float]
    min_time: float = field(init=False)
    max_time: float = field(init=False)
    avg_time: float = field(init=False)
    median_time: float = field(init=False)
    std_time: float = field(init=False)
    memory_usage: Optional[float] = None
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
            "method_type": self.method_type,
            "backend": self.backend,
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


# ---------- 技术途径1：标量循环（OpenCL风格） ---------- #
class ScalarOpenCLStyle:
    """标量循环实现，仿效OpenCL显式编写计算过程"""
    
    @staticmethod
    def _integrate_single_path_scalar(steps: int, key, use_lcg: bool = False, use_continuous: bool = False):
        """单个路径的积分计算（标量版本）"""
        import jax
        import jax.numpy as jnp
        
        def body_fn_scalar(carry, step):
            x, integral, rng_state = carry
            
            # 根据设置选择权重计算方法
            if use_continuous:
                # 连续函数（6次多项式，无分支）
                c0, c1, c2, c3, c4, c5, c6 = (
                    0.3125, 0.234375, -0.2734375, 
                    -0.05859375, 0.13671875, 0.01953125, -0.03125
                )
                weight = c6
                weight = weight * x + c5
                weight = weight * x + c4
                weight = weight * x + c3
                weight = weight * x + c2
                weight = weight * x + c1
                weight = weight * x + c0
            else:
                # 分段函数（有分支）
                weight = jnp.where(
                    x < -1.0, 0.1,
                    jnp.where(x < 0.0, 0.3 * x + 0.4,
                        jnp.where(x < 1.0, 0.5 * (1.0 - x * x), 0.2))
                )
            
            # 交替因子
            factor = jnp.where(step % 2 == 0, 1.0 + 0.1 * x, 1.0 - 0.1 * x)
            
            delta = 1.0 / steps
            integral += weight * delta * factor
            
            # 随机游走 + 边界裁剪
            if use_lcg:
                # LCG随机数生成器
                def lcg_next(state):
                    return state * jnp.uint32(1103515245) + jnp.uint32(12345)
                
                def lcg_random(state):
                    new_state = lcg_next(state)
                    val = new_state & jnp.uint32(0x7fffffff)
                    return val.astype(jnp.float32) / jnp.float32(2147483647.0), new_state
                
                rand_val, new_state = lcg_random(rng_state)
                rng_state = new_state
            else:
                # JAX内置随机数生成器
                subkey = jax.random.fold_in(key, step)
                rand_val = jax.random.uniform(subkey, dtype=jnp.float32)
                rng_state = key  # 保持相同的key
            
            x += rand_val * 0.1
            x = jnp.clip(x, -2.0, 2.0)
            
            return (x, integral, rng_state), None
        
        return body_fn_scalar
    
    @staticmethod
    def create_compute_function(steps: int, paths: int, use_lcg: bool = False, use_continuous: bool = False):
        """创建标量计算函数"""
        import jax
        import jax.numpy as jnp
        from jax import random
        
        def compute_scalar(key):
            """标量计算主函数"""
            if use_lcg:
                # 为每个路径创建初始LCG状态
                base_seed = key.astype(jnp.uint32)
                states = base_seed + jnp.arange(paths, dtype=jnp.uint32)
                
                def process_single_path(initial_state):
                    body_fn = ScalarOpenCLStyle._integrate_single_path_scalar(steps, None, use_lcg, use_continuous)
                    (_, integral, _), _ = jax.lax.scan(
                        body_fn, 
                        (0.0, 0.0, initial_state), 
                        jnp.arange(steps)
                    )
                    return integral
                
                # 使用vmap处理所有路径
                integrals = jax.vmap(process_single_path)(states)
            else:
                # 为每个路径创建独立的随机key
                keys = jax.random.split(key, paths)
                
                def process_single_path(path_key):
                    body_fn = ScalarOpenCLStyle._integrate_single_path_scalar(steps, path_key, use_lcg, use_continuous)
                    (_, integral, _), _ = jax.lax.scan(
                        body_fn,
                        (0.0, 0.0, path_key),
                        jnp.arange(steps)
                    )
                    return integral
                
                # 使用vmap处理所有路径
                integrals = jax.vmap(process_single_path)(keys)
            
            # 返回所有路径的平均积分值
            return jnp.mean(integrals)
        
        return jax.jit(compute_scalar)

# ---------- 技术途径2：向量化计算（NumPy风格） ---------- #
class VectorizedNumPyStyle:
    """向量化实现，支持时间维度分块（Chunking）"""
    
    @staticmethod
    def create_compute_function(steps: int, paths: int, use_lcg: bool = False, use_continuous: bool = False):
        """创建向量化计算函数 - 完全匹配cpu.py的计算模式"""
        import jax
        import jax.numpy as jnp
        from jax import random
        
        delta = 1.0 / steps
        
        @jax.jit
        def compute_vectorized(key):
            # 初始化所有路径的状态
            x_array = jnp.zeros((paths,), dtype=jnp.float32)
            integral_array = jnp.zeros((paths,), dtype=jnp.float32)
            
            # 预计算多项式系数（如果使用连续函数）
            if use_continuous:
                c0, c1, c2, c3, c4, c5, c6 = (
                    0.3125, 0.234375, -0.2734375, 
                    -0.05859375, 0.13671875, 0.01953125, -0.03125
                )
            
            if use_lcg:
                # LCG初始化
                base_seed = key.astype(jnp.uint32)
                states = base_seed + jnp.arange(paths, dtype=jnp.uint32)
                
                # 时间步循环 - 使用fori_loop
                def body_fn_lcg(step, carry):
                    x_array, integral_array, states = carry
                    
                    # LCG随机数
                    new_states = states * jnp.uint32(1103515245) + jnp.uint32(12345)
                    rand_vals = (new_states & jnp.uint32(0x7fffffff)).astype(jnp.float32)
                    rand_vals = rand_vals / jnp.float32(2147483647.0) * 0.1
                    
                    # 根据设置选择权重计算方法
                    if use_continuous:
                        # 连续函数（6次多项式，无分支，霍纳法则）
                        weight_array = c6
                        weight_array = weight_array * x_array + c5
                        weight_array = weight_array * x_array + c4
                        weight_array = weight_array * x_array + c3
                        weight_array = weight_array * x_array + c2
                        weight_array = weight_array * x_array + c1
                        weight_array = weight_array * x_array + c0
                    else:
                        # 分段函数（有分支）
                        weight_array = jnp.where(
                            x_array < -1.0, 0.1,
                            jnp.where(
                                x_array < 0.0, 0.3 * x_array + 0.4,
                                jnp.where(
                                    x_array < 1.0, 0.5 * (1.0 - x_array * x_array),
                                    0.2
                                )
                            )
                        )
                    
                    # 交替因子 - 使用jnp.where代替if语句
                    factor_array = jnp.where(
                        step % 2 == 0,
                        1.0 + 0.1 * x_array,
                        1.0 - 0.1 * x_array
                    )
                    
                    # 更新积分
                    integral_array = integral_array + weight_array * delta * factor_array
                    
                    # 更新x
                    x_array = jnp.clip(x_array + rand_vals, -2.0, 2.0)
                    
                    return (x_array, integral_array, new_states)
                
                final_x, final_integral, _ = jax.lax.fori_loop(
                    0, steps, body_fn_lcg, (x_array, integral_array, states)
                )
                
            else:
                # JAX随机数
                # 时间步循环 - 使用scan
                def body_fn_jax(carry, step):
                    x_array, integral_array = carry
                    
                    # 为当前时间步生成随机数
                    step_key = random.fold_in(key, step)
                    subkeys = random.split(step_key, paths)
                    rand_vals = jax.vmap(lambda k: random.uniform(k, dtype=jnp.float32))(subkeys) * 0.1
                    
                    # 根据设置选择权重计算方法
                    if use_continuous:
                        # 连续函数（6次多项式，无分支，霍纳法则）
                        weight_array = c6
                        weight_array = weight_array * x_array + c5
                        weight_array = weight_array * x_array + c4
                        weight_array = weight_array * x_array + c3
                        weight_array = weight_array * x_array + c2
                        weight_array = weight_array * x_array + c1
                        weight_array = weight_array * x_array + c0
                    else:
                        # 分段函数（有分支）
                        weight_array = jnp.where(
                            x_array < -1.0, 0.1,
                            jnp.where(
                                x_array < 0.0, 0.3 * x_array + 0.4,
                                jnp.where(
                                    x_array < 1.0, 0.5 * (1.0 - x_array * x_array),
                                    0.2
                                )
                            )
                        )
                    
                    # 交替因子 - 使用jnp.where代替if语句
                    factor_array = jnp.where(
                        step % 2 == 0,
                        1.0 + 0.1 * x_array,
                        1.0 - 0.1 * x_array
                    )
                    
                    # 更新积分
                    integral_array = integral_array + weight_array * delta * factor_array
                    
                    # 更新x
                    x_array = jnp.clip(x_array + rand_vals, -2.0, 2.0)
                    
                    return (x_array, integral_array), None
                
                (final_x, final_integral), _ = jax.lax.scan(
                    body_fn_jax, (x_array, integral_array), jnp.arange(steps)
                )
            
            # 返回平均值
            return jnp.mean(final_integral)
        
        return compute_vectorized

    @staticmethod
    def create_compute_function_optimized_rng(steps: int, paths: int, use_lcg: bool = False, use_continuous: bool = False):
        """
        Optimized version with pre-generated random numbers for all steps.
        Suitable for smaller problems that fit in memory.
        """
        import jax
        import jax.numpy as jnp
        from jax import random
        
        delta = 1.0 / steps
        
        if use_continuous:
            c0, c1, c2, c3, c4, c5, c6 = (
                0.3125, 0.234375, -0.2734375, 
                -0.05859375, 0.13671875, 0.01953125, -0.03125
            )
        
        @jax.jit
        def compute_vectorized_optimized(key):
            # Pre-generate all random numbers: (steps, paths)
            if use_lcg:
                # LCG generation
                base_seed = key.astype(jnp.uint32)
                step_offsets = jnp.arange(steps, dtype=jnp.uint32)[:, None]
                path_offsets = jnp.arange(paths, dtype=jnp.uint32)[None, :]
                states = base_seed + step_offsets * paths + path_offsets
                
                a, c, mask = jnp.uint32(1103515245), jnp.uint32(12345), jnp.uint32(0x7fffffff)
                new_states = states * a + c
                all_rand = (new_states & mask).astype(jnp.float32) / jnp.float32(2147483647.0) * 0.1
            else:
                # JAX RNG: Generate all at once
                step_keys = random.split(key, steps)
                
                def gen_step(step_k):
                    return random.uniform(step_k, (paths,), dtype=jnp.float32) * 0.1
                
                all_rand = jax.vmap(gen_step)(step_keys)  # (steps, paths)
            
            # Initialize state
            x_array = jnp.zeros((paths,), dtype=jnp.float32)
            integral_array = jnp.zeros((paths,), dtype=jnp.float32)
            
            def body_fn(step, carry):
                x_arr, int_arr = carry
                
                # Use pre-generated random number
                rand_vals = all_rand[step]
                
                # Weight calculation
                if use_continuous:
                    w = c6
                    w = w * x_arr + c5
                    w = w * x_arr + c4
                    w = w * x_arr + c3
                    w = w * x_arr + c2
                    w = w * x_arr + c1
                    w = w * x_arr + c0
                else:
                    w = jnp.where(
                        x_arr < -1.0, 0.1,
                        jnp.where(
                            x_arr < 0.0, 0.3 * x_arr + 0.4,
                            jnp.where(
                                x_arr < 1.0, 0.5 * (1.0 - x_arr * x_arr),
                                0.2
                            )
                        )
                    )
                
                factor = jnp.where(step % 2 == 0, 1.0 + 0.1 * x_arr, 1.0 - 0.1 * x_arr)
                
                int_arr = int_arr + w * delta * factor
                x_arr = jnp.clip(x_arr + rand_vals, -2.0, 2.0)
                
                return (x_arr, int_arr), None
            
            (final_x, final_integral), _ = jax.lax.scan(
                body_fn, (x_array, integral_array), jnp.arange(steps)
            )
            
            return jnp.mean(final_integral)
        
        return compute_vectorized_optimized

    @staticmethod
    def create_compute_function_chunked(steps: int, paths: int, use_lcg: bool = False, 
                                       use_continuous: bool = False, chunk_size: int = 10000):
        """
        Create chunked computation function with optimized RNG pre-generation.
        
        For JAX RNG (non-LCG): Pre-generates all random numbers for the chunk
        to avoid per-step random.split overhead.
        """
        import jax
        import jax.numpy as jnp
        from jax import random
        
        delta = 1.0 / steps
        
        # Precompute polynomial coefficients
        if use_continuous:
            c0, c1, c2, c3, c4, c5, c6 = (
                0.3125, 0.234375, -0.2734375, 
                -0.05859375, 0.13671875, 0.01953125, -0.03125
            )
        
        @jax.jit
        def compute_chunk(x_init, integral_init, key, chunk_start_idx: int, chunk_steps: int):
            """
            Compute a single chunk with pre-generated random numbers.
            """
            # Determine actual tile size (handle remainder chunks)
            actual_paths = x_init.shape[0]
            
            # Generate all random numbers for this chunk at once
            # Shape: (chunk_steps, actual_paths)
            if use_lcg:
                # For LCG, we generate states and convert to random numbers
                # Using vmap for vectorized LCG generation
                base_seed = key.astype(jnp.uint32) if jnp.issubdtype(key.dtype, jnp.integer) else jnp.uint32(12345)
                
                # Create initial states for each path and step
                # state[i, j] = base + i * paths + j
                step_offsets = jnp.arange(chunk_steps, dtype=jnp.uint32)[:, None]
                path_offsets = jnp.arange(actual_paths, dtype=jnp.uint32)[None, :]
                states = base_seed + step_offsets * actual_paths + path_offsets
                
                # LCG parameters
                a = jnp.uint32(1103515245)
                c = jnp.uint32(12345)
                mask = jnp.uint32(0x7fffffff)
                
                # Vectorized LCG
                new_states = states * a + c
                rand_array = (new_states & mask).astype(jnp.float32) / jnp.float32(2147483647.0) * 0.1
            else:
                # For JAX RNG: Use split once to get subkeys for all steps, then vmap
                # More efficient than split in loop
                step_keys = random.split(key, chunk_steps)
                
                # Generate random numbers for all steps and paths at once
                # Using vmap over steps, each generating (paths,) random numbers
                def gen_step_rand(step_key):
                    return random.uniform(step_key, (actual_paths,), dtype=jnp.float32) * 0.1
                
                # Vectorized generation: (chunk_steps, paths)
                rand_array = jax.vmap(gen_step_rand)(step_keys)
            
            # Now scan through steps using pre-generated random numbers
            def body_fn(step_in_chunk, carry):
                x_array, integral_array = carry
                global_step = chunk_start_idx + step_in_chunk
                
                # Use pre-generated random number (no generation overhead)
                rand_vals = rand_array[step_in_chunk]
                
                # Weight calculation (continuous or piecewise)
                if use_continuous:
                    # Horner's method for polynomial
                    w = c6
                    w = w * x_array + c5
                    w = w * x_array + c4
                    w = w * x_array + c3
                    w = w * x_array + c2
                    w = w * x_array + c1
                    w = w * x_array + c0
                else:
                    w = jnp.where(
                        x_array < -1.0, 0.1,
                        jnp.where(
                            x_array < 0.0, 0.3 * x_array + 0.4,
                            jnp.where(
                                x_array < 1.0, 0.5 * (1.0 - x_array * x_array),
                                0.2
                            )
                        )
                    )
                
                # Alternating factor
                factor = jnp.where(
                    global_step % 2 == 0,
                    1.0 + 0.1 * x_array,
                    1.0 - 0.1 * x_array
                )
                
                # Update state
                integral_array = integral_array + w * delta * factor
                x_array = jnp.clip(x_array + rand_vals, -2.0, 2.0)
                
                return (x_array, integral_array), None
            
            # Execute scan using pre-generated random numbers
            (final_x, final_integral), _ = jax.lax.scan(
                body_fn, 
                (x_init, integral_init), 
                jnp.arange(chunk_steps)
            )
            
            return final_x, final_integral
        
        def compute_full_chunked(key):
            """Main execution with chunking and optimized RNG."""
            # Initialize state
            x_array = jnp.zeros((paths,), dtype=jnp.float32)
            integral_array = jnp.zeros((paths,), dtype=jnp.float32)
            
            # Calculate chunks
            num_full_chunks = steps // chunk_size
            remainder = steps % chunk_size
            
            # Generate keys for each chunk
            num_chunks_total = num_full_chunks + (1 if remainder > 0 else 0)
            if use_lcg:
                # For LCG, key is a uint32 scalar; create distinct seeds by offsetting
                chunk_keys = jnp.arange(num_chunks_total, dtype=jnp.uint32) + key
            else:
                # For JAX RNG, use split to create distinct keys
                chunk_keys = random.split(key, num_chunks_total)
            
            # Process full chunks sequentially
            for i in range(num_full_chunks):
                start_idx = i * chunk_size
                x_array, integral_array = compute_chunk(
                    x_array, integral_array, chunk_keys[i], 
                    start_idx, chunk_size
                )
                # Periodic synchronization to manage memory pressure
                if i % 10 == 9:
                    jax.block_until_ready((x_array, integral_array))
            
            # Process remainder
            if remainder > 0:
                start_idx = num_full_chunks * chunk_size
                x_array, integral_array = compute_chunk(
                    x_array, integral_array, chunk_keys[num_full_chunks],
                    start_idx, remainder
                )
            
            return jnp.mean(integral_array)
        
        return compute_full_chunked


# ---------- public benchmark routine ---------- #
def run_benchmark(
    steps: int = 10_000,
    paths: int = 1_000,
    warmup_iterations: int = 3,
    test_iterations: int = 10,
    use_lcg: bool = False,
    method_type: str = "scalar",  # "scalar" 或 "vectorized"
    backend: str = "auto",  # "auto" 或指定后端
    use_continuous: bool = False,  # 新增：使用连续权重函数
    chunk_steps: Optional[int] = None,  # None 表示不分块（传统模式）
    chunk_paths: Optional[int] = None,  # 新增：路径维度分块
    use_jax_profiler: bool = False,  # 新增：启用 JAX profiler
) -> Tuple[BenchmarkResult, TimingBreakdown]:
    """Run path integral benchmark with detailed four-category timing"""
    import jax
    import jax.numpy as jnp
    from jax import random
    
    timing = TimingBreakdown()
    
    # === CATEGORY I: Pure Python Setup ===
    python_start = time.perf_counter()
    
    # Backend setup
    if backend != "auto":
        set_jax_backend(backend)
    else:
        available_backends = detect_available_backends()
        if available_backends:
            selected_backend = available_backends[0]
            set_jax_backend(selected_backend)
    
    devices = jax.devices()
    device = devices[0] if devices else None
    platform = device.platform if device else "unknown"
    device_kind = device.device_kind if device else "unknown"
    
    # Determine chunking
    use_chunking = (
        (chunk_steps is not None and chunk_steps > 0 and chunk_steps < steps) or 
        (chunk_paths is not None and chunk_paths > 0 and chunk_paths < paths)
    )
    
    # Create compute function (Python-only, no XLA yet)
    if method_type == "scalar":
        compute_func = ScalarOpenCLStyle.create_compute_function(steps, paths, use_lcg, use_continuous)
        method_desc = "Scalar (OpenCL-style loops)"
    elif method_type == "vectorized":
        if use_chunking:
            actual_chunk_steps = chunk_steps if chunk_steps is not None else steps
            compute_func = VectorizedNumPyStyle.create_compute_function_chunked(
                steps, paths, use_lcg, use_continuous, actual_chunk_steps
            )
            method_desc = f"Vectorized Chunked (chunk_steps={actual_chunk_steps})"
        else:
            compute_func = VectorizedNumPyStyle.create_compute_function_optimized_rng(
                steps, paths, use_lcg, use_continuous
            )
            method_desc = "Vectorized (monolithic, pre-gen RNG)"
    else:
        raise ValueError(f"Unknown method_type: {method_type}")
    
    python_end = time.perf_counter()
    timing.python_setup = python_end - python_start
    
    print(f"Configuration: steps={steps}, paths={paths}, method={method_desc}")
    print(f"Weight method: {'continuous' if use_continuous else 'piecewise'}")
    print(f"Pure Python setup time: {timing.python_setup*1000:.2f} ms")
    
    # === CATEGORY IV (partial): JAX Array Initialization ===
    jax_init_start = time.perf_counter()
    
    if use_lcg:
        key = jnp.uint32(42)
    else:
        key = random.PRNGKey(42)
    
    # Force array creation to complete
    jax.block_until_ready(key)
    
    jax_init_end = time.perf_counter()
    timing.jax_array_init = jax_init_end - jax_init_start
    
    # === CATEGORY II: JIT Compilation Time ===
    print("\nMeasuring JIT compilation time...")
    
    # First execution: triggers compilation
    first_start = time.perf_counter()
    result_first = compute_func(key).block_until_ready()
    first_end = time.perf_counter()
    timing.first_execution_total = first_end - first_start
    
    # Second execution: already compiled, measure pure execution
    second_start = time.perf_counter()
    result_second = compute_func(key).block_until_ready()
    second_end = time.perf_counter()
    second_execution_time = second_end - second_start
    
    # JIT compilation time = first - second
    timing.jit_compilation = timing.first_execution_total - second_execution_time
    timing.jax_sync_overhead = second_execution_time * 0.1  # Estimate sync as 10% of execution
    
    print(f"  First execution (compile+run):  {timing.first_execution_total*1000:.2f} ms")
    print(f"  Second execution (pure run):    {second_execution_time*1000:.2f} ms")
    print(f"  JIT compilation time:           {timing.jit_compilation*1000:.2f} ms")
    print(f"  Compilation overhead factor:    {timing.first_execution_total/second_execution_time:.1f}x")
    
    # === CATEGORY III: Warmup and Steady-State ===
    print(f"\nWarmup iterations ({warmup_iterations})...")
    warmup_start = time.perf_counter()
    
    for i in range(warmup_iterations):
        if use_lcg:
            test_key = jnp.uint32((i + 2) * 1000)  # Avoid reusing keys
        else:
            test_key = random.PRNGKey((i + 2) * 1000)
        compute_func(test_key).block_until_ready()
    
    warmup_end = time.perf_counter()
    timing.warmup_execution = warmup_end - warmup_start
    
    # === CATEGORY III: Test Iterations (Steady State) ===
    print(f"Test iterations ({test_iterations})...")
    
    # Optional JAX profiler
    if use_jax_profiler:
        try:
            import jax.profiler
            print("Starting JAX profiler...")
            jax.profiler.start_trace("/tmp/jax_profile")
        except ImportError:
            print("Warning: JAX profiler not available")
            use_jax_profiler = False
    
    times = []
    results_list = []
    
    for i in range(test_iterations):
        if use_lcg:
            iter_key = jnp.uint32(12345 + i)
        else:
            iter_key = random.PRNGKey(12345 + i)
        
        iter_start = time.perf_counter()
        result = compute_func(iter_key).block_until_ready()
        iter_end = time.perf_counter()
        
        times.append(iter_end - iter_start)
        results_list.append(float(result))
    
    if use_jax_profiler:
        jax.profiler.stop_trace()
        print("JAX profiler trace saved to /tmp/jax_profile")
    
    timing.steady_state_avg = float(np.mean(times))
    timing.steady_state_min = float(np.min(times))
    timing.steady_state_max = float(np.max(times))
    
    # === Calculate Per-Path and Per-Operation Metrics ===
    total_operations = steps * paths
    timing.per_path_time_ms = timing.steady_state_avg * 1000 / paths  # ms per path
    timing.per_operation_time_ns = timing.steady_state_avg * 1e9 / total_operations  # ns per (path,step)
    
    # Print detailed summary
    timing.print_summary(steps, paths)
    
    # Create BenchmarkResult
    weight_label = "Continuous" if use_continuous else "Piecewise"
    chunk_label = f"Chunk({chunk_steps},{chunk_paths})" if use_chunking else "Mono"
    impl_name = f"JAX {platform.upper()} ({device_kind}, {method_desc}, {chunk_label})"
    
    result = BenchmarkResult(
        task_name="Trajectory Integral (GPU)",
        implementation=impl_name,
        method_type=method_type,
        backend=platform,
        execution_times=times,
        precision="fp32",
        notes=(f"Steps={steps}, Paths={paths}, "
               f"JIT_Compile_ms={timing.jit_compilation*1000:.1f}, "
               f"Per_Path_us={timing.per_path_time_ms*1000:.2f}, "
               f"Per_Op_ns={timing.per_operation_time_ns:.1f}"),
        result_value=results_list[-1] if results_list else None,
    )
    
    return result, timing


# ---------- console / JSON reporter (shared schema) ---------- #
class BenchmarkReporter:
    @staticmethod
    def print_results(results: List[BenchmarkResult], 
                     timings: Optional[List[TimingBreakdown]] = None,
                     output_file: Optional[str] = None) -> None:
        import jax
        
        output_data = {
            "benchmark_results": [r.to_dict() for r in results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "jax_version": jax.__version__,
                "default_backend": jax.default_backend(),
                "available_devices": [str(d) for d in jax.devices()],
            },
        }

        if timings:
            output_data["detailed_timings"] = [t.to_dict() for t in timings]
            
            # Calculate aggregate statistics
            if timings:
                avg_jit = np.mean([t.jit_compilation for t in timings])
                avg_steady = np.mean([t.steady_state_avg for t in timings])
                avg_per_path = np.mean([t.per_path_time_ms for t in timings])
                avg_per_op = np.mean([t.per_operation_time_ns for t in timings])
                
                output_data["aggregate_timing"] = {
                    "avg_jit_compilation_ms": avg_jit * 1000,
                    "avg_steady_state_ms": avg_steady * 1000,
                    "avg_per_path_time_ms": avg_per_path,
                    "avg_per_operation_time_ns": avg_per_op,
                }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("\n" + "=" * 80)
            print("GPU Path Integral Benchmark Results - JAX Universal Backend")
            print("=" * 80)
            
            # Group by method type
            method_groups = {}
            for r in results:
                if r.method_type not in method_groups:
                    method_groups[r.method_type] = []
                method_groups[r.method_type].append(r)
            
            for method_type, method_results in method_groups.items():
                print(f"\nMethod Type: {method_type.upper()}")
                print("-" * 60)
                
                for r in method_results:
                    print(f"  Backend: {r.backend.upper()}")
                    print(f"  Implementation: {r.implementation}")
                    print(f"  Time: {r.avg_time:.4f}s (min:{r.min_time:.4f}s, "
                          f"max:{r.max_time:.4f}s, med:{r.median_time:.4f}s)")
                    print(f"  Iter/s: {r.iterations_per_second:.2f}")
                    print(f"  Throughput: {r.paths_per_second:.0f} paths/sec")
                    if r.std_time > 0:
                        print(f"  Std dev: {r.std_time:.6f}s")
                    if r.result_value is not None:
                        print(f"  Result value: {r.result_value:.6f}")
                    print()
            
            # Print aggregate timing statistics if available
            if timings:
                print("\n" + "="*70)
                print("AGGREGATE TIMING STATISTICS")
                print("="*70)
                
                avg_jit = np.mean([t.jit_compilation for t in timings])
                avg_steady = np.mean([t.steady_state_avg for t in timings])
                avg_per_path = np.mean([t.per_path_time_ms for t in timings])
                avg_per_op = np.mean([t.per_operation_time_ns for t in timings])
                
                print(f"Average JIT Compilation:     {avg_jit*1000:.2f} ms")
                print(f"Average Steady-State:        {avg_steady*1000:.2f} ms")
                print(f"Average Per-Path Time:       {avg_per_path*1000:.2f} us")
                print(f"Average Per-Operation Time:  {avg_per_op:.1f} ns")

    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str) -> None:
        BenchmarkReporter.print_results(results, filename)


# ---------- CLI ---------- #
def main():
    import jax
    
    # Detect available backends
    available_backends = detect_available_backends()
    
    parser = argparse.ArgumentParser(
        description="GPU Path-Integral Benchmark - JAX Universal Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                                      # Default parameters
  %(prog)s --size (5000,500)                    # Custom problem scale
  %(prog)s --method scalar                      # Use scalar loop method
  %(prog)s --method vectorized                  # Use vectorized method
  %(prog)s --backend {available_backends[0] if available_backends else 'cpu'}  # Specify backend
  %(prog)s --use-lcg                           # Use LCG random number generator
  %(prog)s --weight-method continuous          # Use continuous weight function (polynomial)
  %(prog)s --weight-method piecewise           # Use piecewise weight function (default)
  %(prog)s --chunk-size (5000,10000)           # Enable 2D chunking with specific tile size
  %(prog)s --no-chunk                          # Disable auto-chunking (force monolithic)
  %(prog)s --output benchmark.json              # Save results to JSON file
  %(prog)s --list-backends                     # List available backends

Available backends: {', '.join(available_backends)}

Weight Methods:
  piecewise   Original piecewise weight function (with branches)
  continuous  6th-degree polynomial approximation (branch-free)

Chunking Strategy:
  By default, auto-chunking is enabled for large tasks (steps*paths > 10M).
  Use --chunk-size (steps,paths) to manually specify tile dimensions.
  Use --no-chunk to force single-kernel execution (may cause TDR on large tasks).
        """,
    )

    parser.add_argument(
        "--size",
        type=str,
        default="(10000,1000)",
        help='Problem scale as (steps,paths) tuple. Default: (10000,1000)',
    )
    
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of test iterations (default: 10)",
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["scalar", "vectorized", "both"],
        default="both",
        help="Computation method: scalar (OpenCL-style loops), vectorized (NumPy-style), or both",
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help=f"Specify JAX backend (auto, {', '.join(available_backends)})",
    )
    
    parser.add_argument(
        "--use-lcg",
        action="store_true",
        help="Use LCG random number generator (matches OpenCL implementation)",
    )
    
    parser.add_argument(
        "--weight-method",
        type=str,
        choices=["piecewise", "continuous", "both"],
        default="piecewise",
        help="Weight calculation method: piecewise (default, with branches), continuous (polynomial, branch-free), or both (compare)",
    )
    
    # Modified to Tuple style, consistent with --size
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help='Chunk size as (steps,paths) tuple. Enables 2D chunking when set. Example: (5000,10000). Default: auto-determined for large tasks.',
    )
    
    # Add option to disable chunking
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable auto-chunking and force monolithic kernel execution",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON output file path",
    )
    
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List all available JAX backends and exit",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (uses smaller problem scale)",
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable detailed timing breakdown with four-category analysis",
    )
    
    parser.add_argument(
        "--jax-profiler",
        action="store_true",
        help="Enable JAX built-in profiler (saves trace to /tmp/jax_profile)",
    )
    
    args = parser.parse_args()
    
    # List backend options
    if args.list_backends:
        print("Available JAX backends:")
        for i, backend in enumerate(available_backends):
            print(f"  {i+1}. {backend}")
        return
    
    # Parse scale parameters
    size_str = args.size.strip()
    if size_str.startswith("(") and size_str.endswith(")"):
        size_str = size_str[1:-1]
    steps, paths = map(int, size_str.split(","))
    
    # Quick test mode
    if args.quick:
        print("Quick mode enabled - reducing problem scale")
        steps, paths = min(steps, 1000), min(paths, 100)
    
    print("Starting GPU Path-Integral Benchmark - JAX Universal Backend")
    print(f"Configuration: steps={steps}, paths={paths}, repeats={args.repeats}")
    print(f"Method: {args.method}")
    print(f"Backend: {args.backend}")
    print(f"RNG: {'LCG' if args.use_lcg else 'JAX-RNG'}")
    print(f"Weight method: {args.weight_method}")
    if args.quick:
        print("Quick mode: ON")
    
    # Backend setup
    if args.backend != "auto":
        set_jax_backend(args.backend)
    else:
        if available_backends:
            selected_backend = available_backends[0]
            print(f"Auto-selected backend: {selected_backend}")
            set_jax_backend(selected_backend)
    
    # Get device info
    devices = jax.devices()
    if devices:
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Available devices: {[str(d) for d in devices]}")
    
    # Determine chunking strategy (unified style parsing)
    chunk_steps = None
    chunk_paths = None
    
    if args.no_chunk:
        print("Chunking disabled (monolithic mode)")
        chunk_steps = None
        chunk_paths = None
    elif args.chunk_size:
        # Parse tuple format (steps, paths)
        chunk_str = args.chunk_size.strip()
        if chunk_str.startswith("(") and chunk_str.endswith(")"):
            chunk_str = chunk_str[1:-1]
        try:
            chunk_steps, chunk_paths = map(int, chunk_str.split(","))
            print(f"Manual chunking enabled: tile size ({chunk_steps}, {chunk_paths})")
        except ValueError:
            print(f"Error: Invalid chunk-size format '{args.chunk_size}'. Use (steps,paths) format.")
            sys.exit(1)
    else:
        # Auto-chunking for large tasks (default behavior)
        total_cells = steps * paths
        if total_cells > 10_000_000:  # 10M threshold
            print(f"Large task detected ({total_cells:,} cells). Auto-enabling chunking...")
            try:
                devices = jax.devices()
                mem_gb = 16.0  # Default assumption
                if hasattr(devices[0], 'memory_stats'):
                    stats = devices[0].memory_stats()
                    mem_gb = stats.get('bytes_limit', 16 * 1024**3) / 1024**3
                
                chunk_steps, chunk_paths = estimate_optimal_chunk_sizes(steps, paths, mem_gb)
                print(f"Auto-selected chunk size: ({chunk_steps}, {chunk_paths})")
            except Exception as e:
                print(f"Warning: Auto-chunking failed ({e}). Using conservative defaults.")
                # Conservative defaults based on your validation (100-10000 range)
                chunk_steps = min(5000, steps)
                chunk_paths = min(50000, paths) if paths > 50000 else paths
                if chunk_paths == paths:
                    chunk_paths = None  # No path chunking needed
                print(f"Fallback chunk size: ({chunk_steps}, {chunk_paths or paths})")
        else:
            print("Monolithic execution (auto-chunking not triggered for small tasks)")
    
    # 运行基准测试
    all_results = []
    all_timings = []
    
    # 确定要测试的权重方法
    weight_methods_to_test = []
    if args.weight_method == "both":
        weight_methods_to_test = [False, True]  # False=piecewise, True=continuous
    elif args.weight_method == "continuous":
        weight_methods_to_test = [True]
    else:  # piecewise
        weight_methods_to_test = [False]
    
    # 确定要测试的计算方法
    methods_to_test = []
    if args.method == "both":
        methods_to_test = ["scalar", "vectorized"]
    else:
        methods_to_test = [args.method]
    
    # 遍历所有组合
    for use_continuous in weight_methods_to_test:
        weight_label = "continuous" if use_continuous else "piecewise"
        if len(weight_methods_to_test) > 1:
            print(f"\n{'='*60}")
            print(f"Testing with {weight_label.upper()} weight function...")
            print(f"{'='*60}")
        
        for method_type in methods_to_test:
            print(f"\nTesting {method_type} method with {weight_label} weights...")
            
            result, timing = run_benchmark(
                steps=steps,
                paths=paths,
                warmup_iterations=args.warmup,
                test_iterations=args.repeats,
                use_lcg=args.use_lcg,
                method_type=method_type,
                backend=args.backend,
                use_continuous=use_continuous,
                chunk_steps=chunk_steps,
                chunk_paths=chunk_paths,
                use_jax_profiler=args.jax_profiler,
            )
            all_results.append(result)
            all_timings.append(timing)
    
    # 输出结果
    BenchmarkReporter.print_results(all_results, timings=all_timings if args.profile else None, output_file=args.output)
    
    # Small-scale validation (only for first result)
    if all_results:
        print("\n" + "=" * 80)
        print("Validating against CPU reference (small scale)...")
        
        # Use small scale for validation
        ref_steps, ref_paths = min(steps, 1000), min(paths, 100)
        
        # Import CPU reference implementation
        try:
            from cpu import PathIntegralBenchmark
            cpu_ref = PathIntegralBenchmark.python_implementation(
                steps=ref_steps, 
                paths=ref_paths
            )
            
            # Re-run GPU computation with same parameters for validation
            gpu_result = run_benchmark(
                steps=ref_steps,
                paths=ref_paths,
                warmup_iterations=1,
                test_iterations=1,
                use_lcg=args.use_lcg,
                method_type=all_results[0].method_type,
                backend=args.backend,
                use_continuous=all_results[0].implementation.find("Continuous") >= 0 or 
                              all_results[0].implementation.find("continuous") >= 0,
                chunk_steps=None,
                chunk_paths=None,
            )
            
            gpu_value = gpu_result.result_value
            diff = abs(cpu_ref - gpu_value)
            
            print(f"  CPU Reference: {cpu_ref:.6f}")
            print(f"  GPU Result:    {gpu_value:.6f}")
            print(f"  Difference:    {diff:.2e}")
            
            if diff < 1e-4:
                print("  ✓ Validation passed")
            else:
                print("  ⚠ Validation warning: significant difference")
                
        except ImportError:
            print("  Skipping CPU validation (cpu.py not available)")
        
        print("=" * 80)


if __name__ == "__main__":
    main()
