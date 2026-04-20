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
  
性能分析模式：
  python gpu_traj_jax.py --profile --size "(1000,1000)"
  python gpu_traj_jax.py --profile-micro --backend cpu
  python gpu_traj_jax.py --profile --profile-detail scalar --backend metal
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np

# ---------- Profiling Utilities ---------- #
class TimeProfiler:
    """时间分析器，用于测量不同阶段的时间开销"""
    
    @staticmethod
    def profile_scalar_implementation(steps: int = 1000, paths: int = 1000, 
                                    use_lcg: bool = False, backend: str = "auto") -> Dict[str, float]:
        """分析标量实现的时间分布"""
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # 设置后端
        if backend != "auto":
            set_jax_backend(backend)
        
        profile_results = {}
        
        print("\n" + "="*60)
        print("PROFILING: Scalar Implementation Time Distribution")
        print("="*60)
        
        # 阶段1: 编译时间
        print("1. Measuring compilation time...")
        start_compile = time.perf_counter()
        compute_func = ScalarOpenCLStyle.create_compute_function(steps, paths, use_lcg)
        end_compile = time.perf_counter()
        compile_time = end_compile - start_compile
        profile_results["compilation"] = compile_time
        print(f"   Compilation time: {compile_time:.4f}s")
        
        # 准备key
        if use_lcg:
            key = jnp.uint32(42)
        else:
            key = random.PRNGKey(42)
        
        # 阶段2: JIT预热时间
        print("2. Measuring JIT warmup time...")
        start_warmup = time.perf_counter()
        compute_func(key).block_until_ready()
        end_warmup = time.perf_counter()
        warmup_time = end_warmup - start_warmup
        profile_results["warmup"] = warmup_time
        print(f"   First execution (JIT warmup): {warmup_time:.4f}s")
        
        # 阶段3: 多次执行时间
        print("3. Measuring execution time (10 iterations)...")
        execution_times = []
        for i in range(10):
            if use_lcg:
                iter_key = jnp.uint32(12345 + i)
            else:
                iter_key = random.PRNGKey(12345 + i)
            
            start_exec = time.perf_counter()
            compute_func(iter_key).block_until_ready()
            end_exec = time.perf_counter()
            execution_times.append(end_exec - start_exec)
        
        avg_execution = sum(execution_times) / len(execution_times)
        min_execution = min(execution_times)
        max_execution = max(execution_times)
        profile_results["execution_avg"] = avg_execution
        profile_results["execution_min"] = min_execution
        profile_results["execution_max"] = max_execution
        
        print(f"   Average execution: {avg_execution:.4f}s")
        print(f"   Min execution: {min_execution:.4f}s")
        print(f"   Max execution: {max_execution:.4f}s")
        print(f"   Std deviation: {np.std(execution_times):.6f}s")
        
        # 阶段4: 内存使用估计
        # 基于路径数和步数估计内存使用
        estimated_memory_mb = (paths * 4 * 3) / (1024 * 1024)  # 假设3个float32数组
        profile_results["estimated_memory_mb"] = estimated_memory_mb
        print(f"4. Estimated memory usage: {estimated_memory_mb:.2f} MB")
        
        # 计算百分比
        total_time = compile_time + warmup_time + avg_execution * 10  # 近似总时间
        if total_time > 0:
            profile_results["compile_percent"] = (compile_time / total_time) * 100
            profile_results["warmup_percent"] = (warmup_time / total_time) * 100
            profile_results["execution_percent"] = (avg_execution * 10 / total_time) * 100
            
            print("\nTime Distribution (% of total):")
            print(f"   Compilation: {profile_results['compile_percent']:.1f}%")
            print(f"   Warmup: {profile_results['warmup_percent']:.1f}%")
            print(f"   Execution (10x): {profile_results['execution_percent']:.1f}%")
        
        return profile_results
    
    @staticmethod
    def profile_vectorized_implementation(steps: int = 1000, paths: int = 1000,
                                         use_lcg: bool = False, backend: str = "auto") -> Dict[str, float]:
        """分析向量化实现的时间分布"""
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # 设置后端
        if backend != "auto":
            set_jax_backend(backend)
        
        profile_results = {}
        
        print("\n" + "="*60)
        print("PROFILING: Vectorized Implementation Time Distribution")
        print("="*60)
        
        # 阶段1: 编译时间
        print("1. Measuring compilation time...")
        start_compile = time.perf_counter()
        compute_func = VectorizedNumPyStyle.create_compute_function(steps, paths, use_lcg)
        end_compile = time.perf_counter()
        compile_time = end_compile - start_compile
        profile_results["compilation"] = compile_time
        print(f"   Compilation time: {compile_time:.4f}s")
        
        # 准备key
        if use_lcg:
            key = jnp.uint32(42)
        else:
            key = random.PRNGKey(42)
        
        # 阶段2: JIT预热时间
        print("2. Measuring JIT warmup time...")
        start_warmup = time.perf_counter()
        compute_func(key).block_until_ready()
        end_warmup = time.perf_counter()
        warmup_time = end_warmup - start_warmup
        profile_results["warmup"] = warmup_time
        print(f"   First execution (JIT warmup): {warmup_time:.4f}s")
        
        # 阶段3: 多次执行时间
        print("3. Measuring execution time (10 iterations)...")
        execution_times = []
        for i in range(10):
            if use_lcg:
                iter_key = jnp.uint32(12345 + i)
            else:
                iter_key = random.PRNGKey(12345 + i)
            
            start_exec = time.perf_counter()
            compute_func(iter_key).block_until_ready()
            end_exec = time.perf_counter()
            execution_times.append(end_exec - start_exec)
        
        avg_execution = sum(execution_times) / len(execution_times)
        min_execution = min(execution_times)
        max_execution = max(execution_times)
        profile_results["execution_avg"] = avg_execution
        profile_results["execution_min"] = min_execution
        profile_results["execution_max"] = max_execution
        
        print(f"   Average execution: {avg_execution:.4f}s")
        print(f"   Min execution: {min_execution:.4f}s")
        print(f"   Max execution: {max_execution:.4f}s")
        print(f"   Std deviation: {np.std(execution_times):.6f}s")
        
        # 阶段4: 内存使用估计
        # 基于路径数和步数估计内存使用
        estimated_memory_mb = (paths * 4 * 3 + steps * paths * 4) / (1024 * 1024)  # 状态数组 + 随机数数组
        profile_results["estimated_memory_mb"] = estimated_memory_mb
        print(f"4. Estimated memory usage: {estimated_memory_mb:.2f} MB")
        
        # 阶段5: 分析计算密度
        total_operations = steps * paths * 10  # 每个路径每步大约10次操作
        if avg_execution > 0:
            gflops = (total_operations / 1e9) / avg_execution
            profile_results["estimated_gflops"] = gflops
            print(f"5. Estimated compute performance: {gflops:.2f} GFLOPs")
        
        # 计算百分比
        total_time = compile_time + warmup_time + avg_execution * 10  # 近似总时间
        if total_time > 0:
            profile_results["compile_percent"] = (compile_time / total_time) * 100
            profile_results["warmup_percent"] = (warmup_time / total_time) * 100
            profile_results["execution_percent"] = (avg_execution * 10 / total_time) * 100
            
            print("\nTime Distribution (% of total):")
            print(f"   Compilation: {profile_results['compile_percent']:.1f}%")
            print(f"   Warmup: {profile_results['warmup_percent']:.1f}%")
            print(f"   Execution (10x): {profile_results['execution_percent']:.1f}%")
        
        return profile_results

    @staticmethod
    def profile_micro_operations(steps: int = 100, paths: int = 100,
                                backend: str = "cpu", use_lcg: bool = False) -> Dict[str, float]:
        """分析微操作时间开销（小规模测试）"""
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # 设置后端
        if backend != "auto":
            set_jax_backend(backend)
        
        print("\n" + "="*60)
        print("PROFILING: Micro-Operation Timing (Small Scale)")
        print(f"Steps={steps}, Paths={paths}, Backend={backend}")
        print("="*60)
        
        # 使用小规模问题分析不同操作
        delta = 1.0 / steps
        
        # 1. 随机数生成时间
        print("1. Measuring RNG performance...")
        if use_lcg:
            start = time.perf_counter()
            key = jnp.uint32(42)
            states = key + jnp.arange(paths, dtype=jnp.uint32)
            # LCG操作
            new_states = states * jnp.uint32(1103515245) + jnp.uint32(12345)
            rand_vals = (new_states & jnp.uint32(0x7fffffff)).astype(jnp.float32)
            rand_vals = rand_vals / jnp.float32(2147483647.0) * 0.1
            rand_vals.block_until_ready()
            end = time.perf_counter()
            rng_time = end - start
            print(f"   LCG RNG time for {paths} values: {rng_time:.6f}s")
        else:
            start = time.perf_counter()
            key = random.PRNGKey(42)
            rand_vals = random.uniform(key, (paths,), dtype=jnp.float32) * 0.1
            rand_vals.block_until_ready()
            end = time.perf_counter()
            rng_time = end - start
            print(f"   JAX RNG time for {paths} values: {rng_time:.6f}s")
        
        # 2. 分支逻辑计算时间
        print("2. Measuring branch logic performance...")
        x_array = jnp.zeros((paths,), dtype=jnp.float32)
        start = time.perf_counter()
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
        weight_array.block_until_ready()
        end = time.perf_counter()
        branch_time = end - start
        print(f"   Branch logic time for {paths} values: {branch_time:.6f}s")
        
        # 3. 向量操作时间
        print("3. Measuring vector operations...")
        start = time.perf_counter()
        # 向量加减乘除
        a = jnp.ones((paths,), dtype=jnp.float32) * 1.5
        b = jnp.ones((paths,), dtype=jnp.float32) * 2.5
        c = a + b
        d = a * b
        e = jnp.sqrt(c)
        e.block_until_ready()
        end = time.perf_counter()
        vector_ops_time = end - start
        print(f"   Vector operations time: {vector_ops_time:.6f}s")
        
        # 4. 边界裁剪时间
        print("4. Measuring boundary clipping...")
        test_array = jnp.linspace(-3.0, 3.0, paths)
        start = time.perf_counter()
        clipped = jnp.clip(test_array, -2.0, 2.0)
        clipped.block_until_ready()
        end = time.perf_counter()
        clip_time = end - start
        print(f"   Clip operation time: {clip_time:.6f}s")
        
        # 5. 完整的单步时间估计
        print("5. Estimating per-step time...")
        # 完整执行一个时间步
        x_array = jnp.zeros((paths,), dtype=jnp.float32)
        integral_array = jnp.zeros((paths,), dtype=jnp.float32)
        
        @jax.jit
        def single_step(x, integral, step, key):
            if use_lcg:
                states = key + jnp.arange(paths, dtype=jnp.uint32)
                new_states = states * jnp.uint32(1103515245) + jnp.uint32(12345)
                rand_vals = (new_states & jnp.uint32(0x7fffffff)).astype(jnp.float32)
                rand_vals = rand_vals / jnp.float32(2147483647.0) * 0.1
            else:
                step_key = random.fold_in(key, step)
                subkeys = random.split(step_key, paths)
                rand_vals = jax.vmap(lambda k: random.uniform(k, dtype=jnp.float32))(subkeys) * 0.1
            
            weight_array = jnp.where(
                x < -1.0, 0.1,
                jnp.where(
                    x < 0.0, 0.3 * x + 0.4,
                    jnp.where(
                        x < 1.0, 0.5 * (1.0 - x * x),
                        0.2
                    )
                )
            )
            
            factor_array = jnp.where(step % 2 == 0, 1.0 + 0.1 * x, 1.0 - 0.1 * x)
            integral = integral + weight_array * delta * factor_array
            x = jnp.clip(x + rand_vals, -2.0, 2.0)
            
            return x, integral
        
        if use_lcg:
            test_key = jnp.uint32(42)
        else:
            test_key = random.PRNGKey(42)
        
        start = time.perf_counter()
        x_result, integral_result = single_step(x_array, integral_array, 0, test_key)
        x_result.block_until_ready()
        integral_result.block_until_ready()
        end = time.perf_counter()
        step_time = end - start
        
        estimated_total_time = step_time * steps
        print(f"   Single step time: {step_time:.6f}s")
        print(f"   Estimated total for {steps} steps: {estimated_total_time:.4f}s")
        print(f"   Estimated per-path per-step time: {step_time/paths/steps:.9f}s")
        
        return {
            "rng_time": rng_time,
            "branch_time": branch_time,
            "vector_ops_time": vector_ops_time,
            "clip_time": clip_time,
            "single_step_time": step_time,
            "estimated_total_time": estimated_total_time
        }

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
    def _integrate_single_path_scalar(steps: int, key, use_lcg: bool = False):
        """单个路径的积分计算（标量版本）"""
        import jax
        import jax.numpy as jnp
        
        def body_fn_scalar(carry, step):
            x, integral, rng_state = carry
            
            # 分支逻辑（与OpenCL完全一致）
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
    def create_compute_function(steps: int, paths: int, use_lcg: bool = False):
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
                    body_fn = ScalarOpenCLStyle._integrate_single_path_scalar(steps, None, use_lcg)
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
                    body_fn = ScalarOpenCLStyle._integrate_single_path_scalar(steps, path_key, use_lcg)
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
    """向量化实现，与cpu.py的numpy_scipy_implementation完全一致的计算模式"""
    
    @staticmethod
    def create_compute_function(steps: int, paths: int, use_lcg: bool = False):
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
                    
                    # 与cpu.py完全相同的分支逻辑
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
                    
                    # 与cpu.py完全相同的分支逻辑
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


# ---------- public benchmark routine ---------- #
def run_benchmark(
    steps: int = 10_000,
    paths: int = 1_000,
    warmup_iterations: int = 3,
    test_iterations: int = 10,
    use_lcg: bool = False,
    method_type: str = "scalar",  # "scalar" 或 "vectorized"
    backend: str = "auto",  # "auto" 或指定后端
) -> BenchmarkResult:
    """运行路径积分基准测试"""
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
    
    # 添加编译时间监控
    print(f"Creating {method_type} compute function with steps={steps}, paths={paths}...")
    import time as time_module
    start_compile = time_module.time()

    # 选择计算方法
    if method_type == "scalar":
        compute_func = ScalarOpenCLStyle.create_compute_function(steps, paths, use_lcg)
        method_desc = "Scalar (OpenCL-style loops)"
    elif method_type == "vectorized":
        compute_func = VectorizedNumPyStyle.create_compute_function(steps, paths, use_lcg)
        method_desc = "Vectorized (NumPy-style vector ops)"
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
    
    # 构建实现名称
    impl_name = f"JAX {platform.upper()} ({device_kind}, {method_desc}, {rng_method})"
    
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
              f"RNG={rng_method}, Backend={platform}, "
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
  %(prog)s --output benchmark.json              # 保存结果到JSON文件
  %(prog)s --list-backends                     # 列出可用后端

Available backends: {', '.join(available_backends)}
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
        "--quick",
        action="store_true",
        help="快速测试模式（使用较小的问题规模）",
    )
    
    # 在现有的参数之后添加
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用性能分析模式（详细时间分布分析）",
    )
    
    parser.add_argument(
        "--profile-micro",
        action="store_true",
        help="启用微操作分析模式（分析基本操作时间开销）",
    )
    
    parser.add_argument(
        "--profile-detail",
        type=str,
        choices=["scalar", "vectorized", "both"],
        default="both",
        help="指定要分析的实现类型（仅当--profile启用时有效）",
    )

    args = parser.parse_args()
    
    # 列出后端选项
    if args.list_backends:
        print("Available JAX backends:")
        for i, backend in enumerate(available_backends):
            print(f"  {i+1}. {backend}")
        return
    
    # Profiling模式处理
    if args.profile or args.profile_micro:
        print("=" * 80)
        print("PROFILING MODE ENABLED")
        print("=" * 80)
        
        # 解析规模参数
        size_str = args.size.strip()
        if size_str.startswith("(") and size_str.endswith(")"):
            size_str = size_str[1:-1]
        steps, paths = map(int, size_str.split(","))
        
        # 设置后端
        backend_to_use = args.backend
        if backend_to_use == "auto":
            if available_backends:
                backend_to_use = available_backends[0]
            else:
                backend_to_use = "cpu"
        
        # 微操作分析
        if args.profile_micro:
            print(f"Running micro-operation profiling on backend: {backend_to_use}")
            micro_results = TimeProfiler.profile_micro_operations(
                steps=min(steps, 100),  # 小规模测试
                paths=min(paths, 100),
                backend=backend_to_use,
                use_lcg=args.use_lcg
            )
            
            # 输出总结
            print("\n" + "="*60)
            print("MICRO-PROFILING SUMMARY")
            print("="*60)
            for key, value in micro_results.items():
                print(f"{key:25s}: {value:.6f}s")
            
            return
        
        # 详细性能分析
        if args.profile_detail == "both":
            methods_to_profile = ["scalar", "vectorized"]
        else:
            methods_to_profile = [args.profile_detail]
        
        all_profile_results = {}
        for method in methods_to_profile:
            print(f"\nProfiling {method} implementation...")
            
            if method == "scalar":
                profile_results = TimeProfiler.profile_scalar_implementation(
                    steps=steps,
                    paths=paths,
                    use_lcg=args.use_lcg,
                    backend=backend_to_use
                )
            else:  # vectorized
                profile_results = TimeProfiler.profile_vectorized_implementation(
                    steps=steps,
                    paths=paths,
                    use_lcg=args.use_lcg,
                    backend=backend_to_use
                )
            
            all_profile_results[method] = profile_results
        
        # 输出对比总结
        print("\n" + "="*80)
        print("PROFILING COMPARISON SUMMARY")
        print("="*80)
        
        print(f"{'Metric':<25s} {'Scalar':<15s} {'Vectorized':<15s} {'Difference':<15s}")
        print("-" * 70)
        
        # 对比关键指标
        metrics_to_compare = [
            ("compilation", "Compilation (s)"),
            ("warmup", "Warmup (s)"),
            ("execution_avg", "Execution avg (s)"),
            ("estimated_memory_mb", "Memory est (MB)")
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            scalar_val = all_profile_results.get("scalar", {}).get(metric_key, 0)
            vectorized_val = all_profile_results.get("vectorized", {}).get(metric_key, 0)
            
            if scalar_val and vectorized_val:
                diff = vectorized_val - scalar_val
                diff_pct = (diff / scalar_val * 100) if scalar_val != 0 else 0
                diff_str = f"{diff:+.4f}s ({diff_pct:+.1f}%)"
            else:
                diff_str = "N/A"
            
            print(f"{metric_name:<25s} {scalar_val:<15.4f} {vectorized_val:<15.4f} {diff_str:<15s}")
        
        # 计算性能指标
        if "scalar" in all_profile_results and "vectorized" in all_profile_results:
            scalar_ops = steps * paths / all_profile_results["scalar"].get("execution_avg", 1)
            vectorized_ops = steps * paths / all_profile_results["vectorized"].get("execution_avg", 1)
            
            speedup = vectorized_ops / scalar_ops if scalar_ops > 0 else 0
            
            print("\n" + "-" * 70)
            print(f"{'Performance Metric':<25s} {'Scalar':<15s} {'Vectorized':<15s} {'Speedup':<15s}")
            print("-" * 70)
            print(f"{'Ops/sec (steps*paths/s)':<25s} {scalar_ops:<15.0f} {vectorized_ops:<15.0f} {speedup:<15.2f}x")
            
            if speedup > 1.0:
                print(f"\n✓ Vectorized implementation is {speedup:.2f}x faster than scalar")
            elif speedup < 1.0:
                print(f"\n⚠ Scalar implementation is {1/speedup:.2f}x faster than vectorized")
            else:
                print(f"\nBoth implementations have similar performance")
        
        return  # 在profiling模式下不运行常规基准测试
    
    # 如果不在profiling模式，继续执行常规基准测试
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
    
    # 运行基准测试
    all_results = []
    
    # 确定要测试的方法
    methods_to_test = []
    if args.method == "both":
        methods_to_test = ["scalar", "vectorized"]
    else:
        methods_to_test = [args.method]
    
    for method_type in methods_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {method_type} method...")
        
        result = run_benchmark(
            steps=steps,
            paths=paths,
            warmup_iterations=args.warmup,
            test_iterations=args.repeats,
            use_lcg=args.use_lcg,
            method_type=method_type,
            backend=args.backend,
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
