#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Path-Integral Benchmark - JAX通用后端
==========================================

支持的后端：
- Metal (Apple Silicon)
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- Intel GPUs
- CPU (后备)

支持的两种独立技术途径：
1. 标量循环（Scalar Loops） - 仿效OpenCL，显式编写标量计算过程
2. 向量化计算（Vectorized） - 仿效NumPy，显式编写向量计算过程

所有输出遵循MCPC编码标准（仅使用英文）。
算法与OpenCL版本和CPU版本完全一致，确保结果可比性。

使用示例：
  python gpu_traj_jax.py --method scalar --backend metal
  python gpu_traj_jax.py --method vectorized --backend cuda
  python gpu_traj_jax.py --method both --use-lcg --output results.json
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
    def create_compute_function_chunked(steps: int, paths: int, use_lcg: bool = False, 
                                       use_continuous: bool = False, chunk_size: int = 10000):
        """
        创建分块向量化计算函数
        
        Args:
            steps: 总时间步数
            paths: 路径数
            use_lcg: 是否使用LCG随机数
            use_continuous: 是否使用连续权重函数
            chunk_size: 每个JIT块的时间步数（默认10000，可调整）
        """
        import jax
        import jax.numpy as jnp
        from jax import random
        
        delta = 1.0 / steps
        
        # 预计算多项式系数
        if use_continuous:
            c0, c1, c2, c3, c4, c5, c6 = (
                0.3125, 0.234375, -0.2734375, 
                -0.05859375, 0.13671875, 0.01953125, -0.03125
            )
        
        @jax.jit
        def compute_chunk(x_init, integral_init, key, chunk_start_idx: int, chunk_steps: int):
            """
            计算一个时间块（chunk）
            
            Args:
                x_init: 初始状态 (paths,)
                integral_init: 初始积分 (paths,)
                key: 随机数密钥
                chunk_start_idx: 全局起始步索引（用于计算交替因子奇偶性）
                chunk_steps: 本块实际步数（处理余数时可能小于 chunk_size）
            """
            def body_fn(step_in_chunk, carry):
                x_array, integral_array = carry
                global_step = chunk_start_idx + step_in_chunk
                
                # 随机数生成
                if use_lcg:
                    # LCG实现（简化版，实际应传递状态）
                    step_key = random.fold_in(key, step_in_chunk)
                    subkeys = random.split(step_key, paths)
                    rand_vals = jax.vmap(lambda k: random.uniform(k, dtype=jnp.float32))(subkeys) * 0.1
                else:
                    step_key = random.fold_in(key, step_in_chunk)
                    subkeys = random.split(step_key, paths)
                    rand_vals = jax.vmap(lambda k: random.uniform(k, dtype=jnp.float32))(subkeys) * 0.1
                
                # 权重计算（连续或分段）
                if use_continuous:
                    weight_array = c6
                    weight_array = weight_array * x_array + c5
                    weight_array = weight_array * x_array + c4
                    weight_array = weight_array * x_array + c3
                    weight_array = weight_array * x_array + c2
                    weight_array = weight_array * x_array + c1
                    weight_array = weight_array * x_array + c0
                else:
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
                
                # 交替因子（基于全局步数）
                factor_array = jnp.where(
                    global_step % 2 == 0,
                    1.0 + 0.1 * x_array,
                    1.0 - 0.1 * x_array
                )
                
                # 更新积分
                integral_array = integral_array + weight_array * delta * factor_array
                
                # 更新x
                x_array = jnp.clip(x_array + rand_vals, -2.0, 2.0)
                
                return (x_array, integral_array), None
            
            # 使用 scan 处理 chunk 内步骤（更节省内存）
            (final_x, final_integral), _ = jax.lax.scan(
                body_fn, 
                (x_init, integral_init), 
                jnp.arange(chunk_steps)
            )
            
            return final_x, final_integral
        
        def compute_full_chunked(key):
            """主控函数：分块执行"""
            # 初始化
            x_array = jnp.zeros((paths,), dtype=jnp.float32)
            integral_array = jnp.zeros((paths,), dtype=jnp.float32)
            
            # 计算完整 chunks 数量和余数
            num_full_chunks = steps // chunk_size
            remainder = steps % chunk_size
            
            # 为每个 chunk 生成独立密钥
            chunk_keys = random.split(key, num_full_chunks + (1 if remainder > 0 else 0))
            
            # 顺序处理完整 chunks（马尔可夫链必须顺序）
            for i in range(num_full_chunks):
                start_idx = i * chunk_size
                x_array, integral_array = compute_chunk(
                    x_array, integral_array, chunk_keys[i], 
                    start_idx, chunk_size
                )
                # 每 10 个 chunks 同步一次，防止内存堆积（可选）
                if i % 10 == 9:
                    jax.block_until_ready((x_array, integral_array))
            
            # 处理余数（最后不足 chunk_size 的部分）
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
    chunk_size: int = None,  # None 表示不分块（传统模式）
) -> BenchmarkResult:
    """运行路径积分基准测试（支持分块）"""
    import jax
    import jax.numpy as jnp
    from jax import random
    
    # 设置后端
    if backend != "auto":
        set_jax_backend(backend)
    else:
        # 自动检测并选择第一个可用后端
        available_backends = detect_available_backends()
        if available_backends:
            selected_backend = available_backends[0]
            print(f"Auto-selected backend: {selected_backend}")
            set_jax_backend(selected_backend)
        else:
            print("Warning: No backends available, using default", file=sys.stderr)
    
    # 获取当前设备信息
    devices = jax.devices()
    device = devices[0] if devices else None
    platform = device.platform if device else "unknown"
    device_kind = device.device_kind if device else "unknown"
    
    # 确定是否使用分块
    use_chunking = chunk_size is not None and chunk_size > 0 and chunk_size < steps
    
    if use_chunking:
        print(f"Using chunked execution: chunk_size={chunk_size}, total_steps={steps}")
        print(f"Number of chunks: {(steps + chunk_size - 1) // chunk_size}")
    else:
        print(f"Using monolithic execution (single JIT kernel)")
    
    # 添加编译时间监控
    print(f"Creating {method_type} compute function with steps={steps}, paths={paths}...")
    weight_method_str = "continuous" if use_continuous else "piecewise"
    print(f"Weight method: {weight_method_str}")
    
    import time as time_module
    start_compile = time_module.time()

    # 选择计算方法
    if method_type == "scalar":
        # 标量模式通常 paths 较小，保持原有实现
        compute_func = ScalarOpenCLStyle.create_compute_function(steps, paths, use_lcg, use_continuous)
        method_desc = "Scalar (OpenCL-style loops)"
    elif method_type == "vectorized":
        if use_chunking:
            compute_func = VectorizedNumPyStyle.create_compute_function_chunked(
                steps, paths, use_lcg, use_continuous, chunk_size
            )
            method_desc = f"Vectorized Chunked (chunk_size={chunk_size})"
        else:
            compute_func = VectorizedNumPyStyle.create_compute_function(steps, paths, use_lcg, use_continuous)
            method_desc = "Vectorized (monolithic)"
    else:
        raise ValueError(f"Unknown method_type: {method_type}")

    end_compile = time_module.time()
    print(f"Compilation time: {end_compile - start_compile:.2f}s")
    
    # 准备随机种子
    if use_lcg:
        key = jnp.uint32(42)
        rng_method = "LCG"
    else:
        key = random.PRNGKey(42)
        rng_method = "JAX-RNG"
    
    # 构建实现名称（包含权重方法信息）
    weight_label = "Continuous" if use_continuous else "Piecewise"
    chunk_label = f"Chunk{chunk_size}" if use_chunking else "Mono"
    impl_name = f"JAX {platform.upper()} ({device_kind}, {method_desc}, {rng_method}, {weight_label}, {chunk_label})"
    
    # 预热运行
    print(f"Warming up {method_type} implementation...")
    for _ in range(warmup_iterations):
        if use_lcg:
            test_key = jnp.uint32(_ * 1000)
        else:
            test_key = random.PRNGKey(_ * 1000)
        
        compute_func(test_key).block_until_ready()
    
    # 正式测试运行
    print(f"Running {test_iterations} test iterations...")
    times: List[float] = []
    results_list: List[float] = []
    
    for i in range(test_iterations):
        if use_lcg:
            iter_key = jnp.uint32(12345 + i)
        else:
            iter_key = random.PRNGKey(12345 + i)
        
        start_time = time.perf_counter()
        result = compute_func(iter_key).block_until_ready()
        end_time = time.perf_counter()
        
        times.append(end_time - start_time)
        results_list.append(float(result))
    
    # 计算性能指标
    avg_time = float(np.mean(times)) if times else 0.0
    paths_per_sec = paths / avg_time if avg_time > 0 else 0.0
    total_ops_per_sec = (steps * paths) / avg_time if avg_time > 0 else 0.0
    final_result = results_list[-1] if results_list else 0.0
        
    # 添加更多详细的时间信息
    if len(times) > 1:
        print(f"  First iteration: {times[0]:.4f}s")
        print(f"  Best iteration: {min(times):.4f}s")
        print(f"  Std deviation: {np.std(times):.6f}s")
            
        # 计算稳定后的性能（去掉前2次可能较慢的迭代）
        if len(times) > 3:
            stable_times = times[2:]
            stable_avg = sum(stable_times) / len(stable_times)
            print(f"  Stable average (after warmup): {stable_avg:.4f}s")
        
    # 计算和显示性能指标
    if steps > 0 and paths > 0 and avg_time > 0:
        ops_per_second = (steps * paths) / avg_time
        print(f"  Performance: {ops_per_second:.0f} operations/sec")
        print(f"  Throughput: {paths/avg_time:.0f} paths/sec")
        
    return BenchmarkResult(
        task_name="Trajectory Integral (GPU)",
        implementation=impl_name,
        method_type=method_type,
        backend=platform,
        execution_times=times,
        precision="fp32",
        notes=f"Steps={steps}, Paths={paths}, Method={method_type}, "
              f"RNG={rng_method}, Weight={weight_label}, Backend={platform}, "
              f"Throughput={paths_per_sec:.0f} paths/sec, "
              f"Ops={total_ops_per_sec:.0f} steps·paths/sec",
        result_value=final_result,
    )


# ---------- console / JSON reporter (shared schema) ---------- #
class BenchmarkReporter:
    @staticmethod
    def print_results(results: List[BenchmarkResult], output_file: Optional[str] = None) -> None:
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

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("\n" + "=" * 80)
            print("GPU Path Integral Benchmark Results - JAX通用后端")
            print("=" * 80)
            
            # 按方法类型分组显示
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

    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str) -> None:
        BenchmarkReporter.print_results(results, filename)


# ---------- CLI ---------- #
def main():
    import jax
    
    # 检测可用后端
    available_backends = detect_available_backends()
    
    parser = argparse.ArgumentParser(
        description="GPU Path-Integral Benchmark - JAX通用后端（支持多种计算方法和后端）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                                      # 默认参数
  %(prog)s --size (5000,500)                    # 自定义问题规模
  %(prog)s --method scalar                      # 使用标量循环方法
  %(prog)s --method vectorized                  # 使用向量化方法
  %(prog)s --backend {available_backends[0] if available_backends else 'cpu'}  # 指定后端
  %(prog)s --use-lcg                           # 使用LCG随机数生成器
  %(prog)s --weight-method continuous          # 使用连续权重函数（多项式）
  %(prog)s --weight-method piecewise           # 使用分段权重函数（默认）
  %(prog)s --output benchmark.json              # 保存结果到JSON文件
  %(prog)s --list-backends                     # 列出可用后端

Available backends: {', '.join(available_backends)}

Weight Methods:
  --weight-method piecewise   Use original piecewise weight function (with branches)
  --weight-method continuous  Use 6th-degree polynomial approximation (branch-free)
        """,
    )

    parser.add_argument(
        "--size",
        type=str,
        default="(10000,1000)",
        help='问题规模 (steps,paths) 元组，默认 (10000,1000)',
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="测试迭代次数（默认 10）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="预热迭代次数（默认 3）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["scalar", "vectorized", "both"],
        default="both",
        help="计算方法：标量循环（scalar）、向量化（vectorized）、或两者都测试（both）",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help=f"指定JAX后端（auto, {', '.join(available_backends)}）",
    )
    parser.add_argument(
        "--use-lcg",
        action="store_true",
        help="使用LCG随机数生成器（与OpenCL实现匹配）",
    )
    # 新增：权重方法选择参数
    parser.add_argument(
        "--weight-method",
        type=str,
        choices=["piecewise", "continuous", "both"],
        default="piecewise",
        help="权重计算方法：'piecewise'（默认，分段函数）、'continuous'（连续多项式）、或'both'（对比两者）",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="可选JSON输出文件",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="列出所有可用的JAX后端并退出",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=None,
        help="Chunk size for time dimension (steps). Enable 2D chunking when set smaller than total steps.",
    )
    parser.add_argument(
        "--chunk-paths",
        type=int,
        default=None,
        help="Chunk size for path dimension (paths). Enable path chunking to reduce memory usage.",
    )
    parser.add_argument(
        "--chunk-auto",
        action="store_true",
        help="Automatically determine chunk sizes based on problem scale and available memory.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试模式（使用较小的问题规模）",
    )
    
    args = parser.parse_args()
    
    # 列出后端选项
    if args.list_backends:
        print("Available JAX backends:")
        for i, backend in enumerate(available_backends):
            print(f"  {i+1}. {backend}")
        return
    
    # 解析规模参数
    size_str = args.size.strip()
    if size_str.startswith("(") and size_str.endswith(")"):
        size_str = size_str[1:-1]
    steps, paths = map(int, size_str.split(","))
    
    # 快速测试模式
    if args.quick:
        print("Quick mode enabled - reducing problem size")
        steps, paths = min(steps, 1000), min(paths, 100)
    
    print("Starting GPU Path-Integral Benchmark - JAX通用后端")
    print(f"Configuration: steps={steps}, paths={paths}, repeats={args.repeats}")
    print(f"Method: {args.method}")
    print(f"Backend: {args.backend}")
    print(f"RNG: {'LCG' if args.use_lcg else 'JAX-RNG'}")
    print(f"Weight method: {args.weight_method}")
    if args.quick:
        print("Quick mode: ON")
    
    # 设置后端
    if args.backend != "auto":
        set_jax_backend(args.backend)
    else:
        if available_backends:
            selected_backend = available_backends[0]
            print(f"Auto-selected backend: {selected_backend}")
            set_jax_backend(selected_backend)
    
    # 获取设备信息
    devices = jax.devices()
    if devices:
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Available devices: {[str(d) for d in devices]}")
    
    # Determine chunk sizes
    chunk_steps = args.chunk_steps
    chunk_paths = args.chunk_paths

    if args.chunk_auto:
        try:
            devices = jax.devices()
            # Estimate memory (simplified for Metal/CUDA)
            mem_gb = 16.0  # Default assumption
            if hasattr(devices[0], 'memory_stats'):
                # Try to get actual memory info
                stats = devices[0].memory_stats()
                mem_gb = stats.get('bytes_limit', 16 * 1024**3) / 1024**3
            
            chunk_steps, chunk_paths = estimate_optimal_chunk_sizes(
                steps, paths, mem_gb
            )
            print(f"Auto-selected chunk sizes: steps={chunk_steps}, paths={chunk_paths}")
        except Exception as e:
            print(f"Warning: Auto-chunking failed ({e}), using defaults")
            chunk_steps = min(10000, steps)
            chunk_paths = None
    
    # 运行基准测试
    all_results = []
    
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
            
            result = run_benchmark(
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
            )
            all_results.append(result)
    
    # 输出结果
    BenchmarkReporter.print_results(all_results, output_file=args.output)
    
    # 小规模验证（仅对第一个结果进行验证）
    if all_results:
        print("\n" + "=" * 80)
        print("Validating against CPU reference (small scale)...")
        
        # 使用小规模问题进行验证
        ref_steps, ref_paths = min(steps, 1000), min(paths, 100)
        
        # 导入CPU参考实现
        try:
            from cpu import PathIntegralBenchmark
            cpu_ref = PathIntegralBenchmark.python_implementation(
                steps=ref_steps, 
                paths=ref_paths
            )
            
            # 使用相同的参数重新运行GPU计算进行验证
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
