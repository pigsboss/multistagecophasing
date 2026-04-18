#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metal & JAX Environment Detection and Performance Benchmark
======================================================================
Detects macOS Metal devices, JAX installation, and runs performance tests.
All output is in English per MCPC coding standards.
"""

import sys
import time
import platform
import subprocess
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import dataclasses
from dataclasses import dataclass, field
import warnings
from contextlib import contextmanager

# Set default environment variables for logging
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Default: only show warnings and errors
os.environ.setdefault('JAX_LOG_LEVEL', 'WARNING')   # Default: warning level

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', 
                       category=DeprecationWarning,
                       message='jax.lib.xla_bridge.get_backend is deprecated')
warnings.filterwarnings('ignore',
                       category=UserWarning,
                       message='pkg_resources is deprecated')
# Filter experimental warnings
warnings.filterwarnings('ignore', 
                       message='.*experimental.*')
warnings.filterwarnings('ignore', 
                       message='.*Experimental.*')
warnings.filterwarnings('ignore',
                       message='All log messages before absl::InitializeLog')


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section title."""
    print(f"\n{'=' * width}")
    print(f" {title.upper()}")
    print(f"{'=' * width}")


def print_subsection(title: str) -> None:
    """Print a formatted subsection title."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


@contextmanager
def suppress_stderr():
    """完全抑制 stderr 输出（用于抑制 C++ 库日志）"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


@dataclass
class BenchmarkResult:
    """Benchmark result data class."""
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
    result_value: Optional[float] = None
    
    def __post_init__(self):
        if len(self.execution_times) > 0:
            self.min_time = min(self.execution_times)
            self.max_time = max(self.execution_times)
            self.avg_time = float(np.mean(self.execution_times))
            self.median_time = float(np.median(self.execution_times))
            if len(self.execution_times) > 1:
                self.std_time = float(np.std(self.execution_times))
            else:
                self.std_time = 0.0
        else:
            self.min_time = self.max_time = self.avg_time = self.median_time = self.std_time = 0.0
    
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


def get_system_info() -> Dict[str, str]:
    """Gather system information."""
    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'platform_release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor() or 'Unknown',
        'mac_ver': platform.mac_ver()[0] if platform.system() == 'Darwin' else 'N/A',
    }
    
    # Get memory information if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = f"{mem.total / (1024**3):.2f}"
        info['available_ram_gb'] = f"{mem.available / (1024**3):.2f}"
    except ImportError:
        info['total_ram_gb'] = 'Unknown (install psutil)'
        info['available_ram_gb'] = 'Unknown (install psutil)'
    
    return info


def check_metal_installation() -> Dict[str, Any]:
    """Check Metal and related packages installation."""
    packages = {}
    
    # Check platform first
    if platform.system() != 'Darwin':
        packages['metal_support'] = {
            'available': False,
            'error': 'Metal is only available on macOS'
        }
        return packages
    
    # Check PyObjC Metal bindings
    try:
        import Metal
        packages['metal_pyobjc'] = {
            'available': True,
            'version': 'Unknown (via PyObjC)'
        }
    except ImportError:
        packages['metal_pyobjc'] = {
            'available': False,
            'error': 'Metal PyObjC bindings not installed'
        }
    
    # Check system Metal via command line
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            packages['metal_system'] = {
                'available': True,
                'info': 'Metal system profiler available'
            }
        else:
            packages['metal_system'] = {'available': False, 'error': 'Command failed'}
    except (subprocess.SubprocessError, FileNotFoundError):
        packages['metal_system'] = {'available': False, 'error': 'system_profiler not available'}
    
    # Check JAX
    try:
        import jax
        packages['jax'] = {
            'version': jax.__version__,
            'available': True
        }
    except ImportError:
        packages['jax'] = {'available': False, 'error': 'Not installed'}
    
    # Check JAX Metal backend - use case-insensitive comparison
    try:
        import jax
        
        # Use new API if available, fall back to old API
        try:
            # New API in jax >= 0.5.0
            from jax.extend import backend
            backend_obj = backend.get_backend()
        except (ImportError, AttributeError):
            # Fall back to old API
            backend_obj = jax.lib.xla_bridge.get_backend()
        
        backend_platform = backend_obj.platform
        packages['jax_backend'] = {
            'platform': backend_platform,
            'available': True,
            'is_metal': backend_platform.lower() == 'metal'
        }
    except Exception as e:
        packages['jax_backend'] = {
            'available': False,
            'error': str(e)
        }
    
    # Check JAXLIB
    try:
        import jaxlib
        packages['jaxlib'] = {
            'version': jaxlib.__version__ if hasattr(jaxlib, '__version__') else 'Unknown',
            'available': True
        }
    except ImportError:
        packages['jaxlib'] = {'available': False, 'error': 'Not installed'}
    
    # Check if we have jax-metal package using importlib.metadata
    try:
        # Try importlib.metadata first (Python 3.8+)
        import importlib.metadata
        try:
            jax_metal_version = importlib.metadata.version("jax-metal")
            packages['jax_metal'] = {
                'version': jax_metal_version,
                'available': True
            }
        except importlib.metadata.PackageNotFoundError:
            # Fall back to pkg_resources
            try:
                import pkg_resources
                jax_metal_version = pkg_resources.get_distribution("jax-metal").version
                packages['jax_metal'] = {
                    'version': jax_metal_version,
                    'available': True
                }
            except pkg_resources.DistributionNotFound:
                packages['jax_metal'] = {'available': False, 'error': 'Not installed'}
    except ImportError:
        # Both importlib.metadata and pkg_resources not available
        packages['jax_metal'] = {'available': False, 'error': 'Package metadata not available'}
    
    # Check NumPy
    try:
        import numpy
        packages['numpy'] = {
            'version': numpy.__version__,
            'available': True
        }
    except ImportError:
        packages['numpy'] = {'available': False, 'error': 'Not installed'}
    
    return packages


def get_metal_device_info() -> Dict[str, Any]:
    """Get detailed Metal device information using available methods."""
    info = {}
    
    if platform.system() != 'Darwin':
        info['error'] = 'Metal is only available on macOS'
        return info
    
    try:
        # Try to use Metal via PyObjC
        import Metal
        import MetalPerformanceShaders
        
        # Get default device
        device = Metal.MTLCreateSystemDefaultDevice()
        if device:
            info['default_device'] = {
                'name': str(device.name()),
                'registry_id': device.registryID(),
                'is_low_power': device.isLowPower(),
                'is_headless': device.isHeadless(),
                'is_removable': device.isRemovable(),
                'has_unified_memory': device.hasUnifiedMemory(),
                'recommended_max_working_set_size': device.recommendedMaxWorkingSetSize(),
                'max_transfer_rate': device.maxTransferRate(),
            }
            
            # Get memory info
            mem_info = {}
            try:
                mem_info['recommended_max_working_set_size_gb'] = device.recommendedMaxWorkingSetSize() / (1024**3)
            except:
                pass
            
            # Check feature support
            features = {}
            try:
                features['supports_float32'] = device.supportsFamily(Metal.MTLGPUFamilyApple1)
                features['supports_float16'] = hasattr(Metal, 'MTLGPUFamilyApple1')  # Simplified
            except:
                pass
            
            info['default_device']['memory_info'] = mem_info
            info['default_device']['features'] = features
        else:
            info['default_device'] = {'error': 'No default Metal device found'}
        
        # Try to get all devices
        try:
            devices = Metal.MTLCopyAllDevices()
            device_list = []
            for i, dev in enumerate(devices):
                device_list.append({
                    'index': i,
                    'name': str(dev.name()),
                    'is_low_power': dev.isLowPower(),
                    'has_unified_memory': dev.hasUnifiedMemory(),
                })
            info['all_devices'] = device_list
            info['device_count'] = len(devices)
        except Exception as e:
            info['all_devices_error'] = str(e)
    
    except ImportError as e:
        info['error'] = f'Metal PyObjC not available: {str(e)}'
    except Exception as e:
        info['error'] = f'Error getting Metal info: {str(e)}'
    
    # Try alternative method using system_profiler
    if 'error' in info or 'default_device' not in info:
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['system_profiler_output'] = result.stdout[:2000]  # First 2000 chars
                
                # Parse for GPU info
                gpu_info = []
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chipset Model' in line:
                        gpu_info.append(line.strip())
                    elif 'Metal' in line and 'Supported' in line:
                        gpu_info.append(line.strip())
                    elif 'VRAM' in line:
                        gpu_info.append(line.strip())
                
                info['gpu_summary'] = gpu_info
        except:
            pass
    
    return info


def get_jax_device_info() -> Dict[str, Any]:
    """Get JAX device information."""
    info = {}
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Get backend info - use new API if available
        try:
            # New API in jax >= 0.5.0
            from jax.extend import backend
            backend_obj = backend.get_backend()
        except (ImportError, AttributeError):
            # Fall back to old API
            backend_obj = jax.lib.xla_bridge.get_backend()
        
        info['backend'] = {
            'platform': backend_obj.platform,
            'platform_version': backend_obj.platform_version,
            'device_count': backend_obj.device_count(),
        }
        
        # Get device info
        devices = jax.devices()
        device_list = []
        for i, device in enumerate(devices):
            device_info = {
                'index': i,
                'platform': device.platform,
                'device_kind': device.device_kind,
                'client': str(device.client),
            }
            
            # Try to get more details
            try:
                device_info['local_hardware_id'] = device.local_hardware_id
            except:
                pass
            
            try:
                device_info['device_id'] = device.id
            except:
                pass
            
            device_list.append(device_info)
        
        info['devices'] = device_list
        info['device_count'] = len(devices)
        
        # Check if Metal is being used - case-insensitive comparison
        info['using_metal'] = backend_obj.platform.lower() == 'metal'
        
        # Test device capabilities
        try:
            test_array = jnp.array([1.0, 2.0, 3.0])
            info['jax_functional'] = True
            info['test_computation'] = float(jnp.sum(test_array))
        except Exception as e:
            info['jax_functional'] = False
            info['test_error'] = str(e)
    
    except ImportError as e:
        info['error'] = f'JAX not available: {str(e)}'
    except Exception as e:
        info['error'] = f'Error getting JAX info: {str(e)}'
    
    return info


def benchmark_vector_operations(device_idx: int = 0, test_size: int = 1024 * 1024) -> Dict[str, Any]:
    """Benchmark vector operations using JAX."""
    benchmarks = {}
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        # Set device context
        devices = jax.devices()
        if device_idx >= len(devices):
            benchmarks['error'] = f'Device index {device_idx} out of range'
            return benchmarks
        
        device = devices[device_idx]
        
        print(f"  Testing on device: {device.device_kind}")
        
        # Test sizes for different operations
        sizes = {
            'small': 1024,  # 1K
            'medium': 1024 * 1024,  # 1M
            'large': 4 * 1024 * 1024,  # 4M
        }
        
        # Use medium size for quick test
        n = test_size
        
        # Create test data
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (n,))
        b = jax.random.normal(key, (n,))
        
        # 1. Vector addition
        print("  Testing vector addition...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            c = a + b
            c.block_until_ready()  # Ensure computation completes
            times.append(time.perf_counter() - start)
        
        benchmarks['vector_add'] = {
            'test_size': n,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'throughput_gflops': (n / np.mean(times)) / 1e9 if np.mean(times) > 0 else 0,
            'bandwidth_gb_s': (n * 4 * 3) / (np.mean(times) * 1024**3) if np.mean(times) > 0 else 0,  # 3 arrays * float32
        }
        
        # 2. Vector multiplication
        print("  Testing vector multiplication...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            c = a * b
            c.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['vector_mul'] = {
            'test_size': n,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'throughput_gflops': (n / np.mean(times)) / 1e9 if np.mean(times) > 0 else 0,
        }
        
        # 3. Dot product
        print("  Testing dot product...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            result = jnp.dot(a, b)
            result.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['dot_product'] = {
            'test_size': n,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'result': float(result),
            'throughput_gflops': (2 * n / np.mean(times)) / 1e9 if np.mean(times) > 0 else 0,  # 2n operations
        }
        
        # 4. Reduction operations
        print("  Testing reduction operations...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            result = jnp.sum(a)
            result.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['reduction_sum'] = {
            'test_size': n,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'result': float(result),
        }
    
    except Exception as e:
        benchmarks['error'] = f'Vector benchmark failed: {str(e)}'
    
    return benchmarks


def benchmark_matrix_operations(device_idx: int = 0, matrix_size: int = 512) -> Dict[str, Any]:
    """Benchmark matrix operations using JAX."""
    benchmarks = {}
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        devices = jax.devices()
        if device_idx >= len(devices):
            benchmarks['error'] = f'Device index {device_idx} out of range'
            return benchmarks
        
        device = devices[device_idx]
        
        print(f"  Testing matrix operations on device: {device.device_kind}")
        
        # Create test matrices
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        A = jax.random.normal(key1, (matrix_size, matrix_size))
        B = jax.random.normal(key2, (matrix_size, matrix_size))
        
        # 1. Matrix addition
        print("  Testing matrix addition...")
        times = []
        for i in range(5):
            start = time.perf_counter()
            C = A + B
            C.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['matrix_add'] = {
            'size': matrix_size,
            'shape': (matrix_size, matrix_size),
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'throughput_gflops': (matrix_size**2 / np.mean(times)) / 1e9 if np.mean(times) > 0 else 0,
        }
        
        # 2. Matrix multiplication
        print("  Testing matrix multiplication...")
        times = []
        for i in range(5):
            start = time.perf_counter()
            C = jnp.dot(A, B)
            C.block_until_ready()
            times.append(time.perf_counter() - start)
        
        # Theoretical FLOPs: 2 * n^3 for matrix multiplication
        flops = 2 * matrix_size**3
        gflops = (flops / np.mean(times)) / 1e9 if np.mean(times) > 0 else 0
        
        benchmarks['matrix_mul'] = {
            'size': matrix_size,
            'shape': (matrix_size, matrix_size),
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'gflops': gflops,
            'theoretical_gflops': flops / 1e9,
        }
        
        # 3. Matrix transpose
        print("  Testing matrix transpose...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            C = jnp.transpose(A)
            C.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['matrix_transpose'] = {
            'size': matrix_size,
            'shape': (matrix_size, matrix_size),
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'bandwidth_gb_s': (matrix_size**2 * 4 * 2) / (np.mean(times) * 1024**3) if np.mean(times) > 0 else 0,
        }
        
        # 4. Matrix norm
        print("  Testing matrix norm...")
        times = []
        for i in range(10):
            start = time.perf_counter()
            norm = jnp.linalg.norm(A)
            norm.block_until_ready()
            times.append(time.perf_counter() - start)
        
        benchmarks['matrix_norm'] = {
            'size': matrix_size,
            'shape': (matrix_size, matrix_size),
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'result': float(norm),
        }
    
    except Exception as e:
        benchmarks['error'] = f'Matrix benchmark failed: {str(e)}'
    
    return benchmarks


def benchmark_jax_jit(device_idx: int = 0) -> Dict[str, Any]:
    """Benchmark JAX JIT compilation and execution."""
    benchmarks = {}
    
    try:
        import jax
        import jax.numpy as jnp
        
        devices = jax.devices()
        if device_idx >= len(devices):
            benchmarks['error'] = f'Device index {device_idx} out of range'
            return benchmarks
        
        device = devices[device_idx]
        
        print(f"  Testing JIT on device: {device.device_kind}")
        
        # Create a simple function to JIT
        def simple_function(x, y):
            return jnp.dot(x, y) + jnp.sum(x) * jnp.sum(y)
        
        # JIT compile the function
        print("  Testing JIT compilation time...")
        start = time.perf_counter()
        jitted_function = jax.jit(simple_function)
        # Force compilation with test data
        test_x = jnp.ones((1000,))
        test_y = jnp.ones((1000,))
        _ = jitted_function(test_x, test_y).block_until_ready()
        compile_time = time.perf_counter() - start
        
        # Test execution time after compilation
        print("  Testing JIT execution time...")
        times = []
        for i in range(20):
            x = jnp.ones((1000,)) * i
            y = jnp.ones((1000,)) * (i + 1)
            start = time.perf_counter()
            result = jitted_function(x, y)
            result.block_until_ready()
            times.append(time.perf_counter() - start)
        
        # Test without JIT for comparison
        print("  Testing non-JIT execution time...")
        non_jit_times = []
        for i in range(5):
            x = jnp.ones((1000,)) * i
            y = jnp.ones((1000,)) * (i + 1)
            start = time.perf_counter()
            result = simple_function(x, y)
            result.block_until_ready()
            non_jit_times.append(time.perf_counter() - start)
        
        benchmarks['jit'] = {
            'compile_time_ms': compile_time * 1000,
            'jit_execution_avg_ms': np.mean(times) * 1000,
            'jit_execution_min_ms': np.min(times) * 1000,
            'non_jit_execution_avg_ms': np.mean(non_jit_times) * 1000,
            'speedup': np.mean(non_jit_times) / np.mean(times) if np.mean(times) > 0 else 0,
            'test_function': 'dot(x,y) + sum(x)*sum(y)',
            'input_size': 1000,
        }
    
    except Exception as e:
        benchmarks['error'] = f'JIT benchmark failed: {str(e)}'
    
    return benchmarks


def benchmark_host_device_transfer(device_idx: int = 0) -> Dict[str, Any]:
    """Benchmark host-device data transfer."""
    benchmarks = {}
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        
        devices = jax.devices()
        if device_idx >= len(devices):
            benchmarks['error'] = f'Device index {device_idx} out of range'
            return benchmarks
        
        device = devices[device_idx]
        
        print(f"  Testing host-device transfer on device: {device.device_kind}")
        
        # Test different buffer sizes
        sizes_bytes = [1024, 1024*1024, 4*1024*1024]  # 1KB, 1MB, 4MB
        
        for size_bytes in sizes_bytes:
            element_count = size_bytes // 4  # Using float32
            if element_count == 0:
                element_count = 1
            
            # Create host data
            host_data = np.random.randn(element_count).astype(np.float32)
            
            # Measure transfer to device
            to_device_times = []
            for _ in range(10 if size_bytes <= 1024*1024 else 5):
                start = time.perf_counter()
                device_data = jax.device_put(host_data, device=device)
                device_data.block_until_ready()
                to_device_times.append(time.perf_counter() - start)
            
            # Measure transfer from device
            from_device_times = []
            for _ in range(10 if size_bytes <= 1024*1024 else 5):
                start = time.perf_counter()
                host_result = np.array(device_data)
                from_device_times.append(time.perf_counter() - start)
            
            # Calculate bandwidth
            to_device_bw = size_bytes / (np.mean(to_device_times) * 1024*1024)  # MB/s
            from_device_bw = size_bytes / (np.mean(from_device_times) * 1024*1024)  # MB/s
            
            size_key = f'transfer_{size_bytes}_bytes'
            benchmarks[size_key] = {
                'size_bytes': size_bytes,
                'to_device_avg_ms': np.mean(to_device_times) * 1000,
                'from_device_avg_ms': np.mean(from_device_times) * 1000,
                'to_device_bandwidth_mb_s': to_device_bw,
                'from_device_bandwidth_mb_s': from_device_bw,
                'average_bandwidth_mb_s': (to_device_bw + from_device_bw) / 2,
            }
    
    except Exception as e:
        benchmarks['error'] = f'Transfer benchmark failed: {str(e)}'
    
    return benchmarks


def run_jax_benchmarks(device_idx: int = 0) -> Dict[str, Any]:
    """Run all JAX benchmarks for a specific device."""
    results = {}
    
    print(f"\nRunning JAX benchmarks for device {device_idx}")
    
    # Get JAX device info
    jax_info = get_jax_device_info()
    if 'error' in jax_info:
        results['error'] = f'Cannot run benchmarks: {jax_info["error"]}'
        return results
    
    results['device_info'] = jax_info
    
    # Check if using Metal - use case-insensitive comparison
    using_metal = jax_info.get('using_metal', False)
    if not using_metal:
        print("  Note: JAX is not using Metal backend")
        results['backend_warning'] = 'Not using Metal backend'
    else:
        print("  Note: JAX is using Metal backend")
    
    # Run benchmarks
    print("  Running vector operations benchmark...")
    vector_results = benchmark_vector_operations(device_idx=device_idx)
    results['vector_operations'] = vector_results
    
    print("  Running matrix operations benchmark...")
    matrix_results = benchmark_matrix_operations(device_idx=device_idx, matrix_size=256)
    results['matrix_operations'] = matrix_results
    
    print("  Running JIT benchmark...")
    jit_results = benchmark_jax_jit(device_idx=device_idx)
    results['jit_performance'] = jit_results
    
    print("  Running host-device transfer benchmark...")
    transfer_results = benchmark_host_device_transfer(device_idx=device_idx)
    results['data_transfer'] = transfer_results
    
    return results


def print_summary(system_info: Dict, packages: Dict, 
                  metal_info: Dict, jax_info: Dict,
                  benchmark_results: Dict) -> None:
    """Print a summary of key findings."""
    print_section("SUMMARY OF KEY FINDINGS")
    
    summary_points = []
    
    # System info
    summary_points.append(f"• Platform: {system_info['platform']} {system_info['architecture']}")
    if system_info['platform'] == 'Darwin':
        summary_points.append(f"• macOS Version: {system_info['mac_ver']}")
    summary_points.append(f"• Python: {system_info['python_version']}")
    
    # Metal availability
    if platform.system() == 'Darwin':
        if 'default_device' in metal_info:
            device_name = metal_info['default_device'].get('name', 'Unknown')
            summary_points.append(f"• Metal Device: {device_name}")
            
            # Check unified memory
            if metal_info['default_device'].get('has_unified_memory', False):
                summary_points.append(f"• Memory Architecture: Unified Memory")
            
            # Check low power
            if metal_info['default_device'].get('is_low_power', False):
                summary_points.append(f"• Device Type: Low Power (e.g., integrated GPU)")
            else:
                summary_points.append(f"• Device Type: High Performance")
        else:
            summary_points.append(f"• Metal: Not detected")
    else:
        summary_points.append(f"• Metal: Not available (requires macOS)")
    
    # JAX availability
    if packages.get('jax', {}).get('available', False):
        jax_version = packages['jax'].get('version', 'Unknown')
        summary_points.append(f"• JAX: Installed (v{jax_version})")
        
        # Check backend
        backend_info = packages.get('jax_backend', {})
        if backend_info.get('available', False):
            platform_name = backend_info.get('platform', 'Unknown')
            summary_points.append(f"• JAX Backend: {platform_name}")
            
            # Fix: use case-insensitive comparison
            if backend_info.get('is_metal', False):
                summary_points.append(f"• JAX-Metal: Active")
            else:
                summary_points.append(f"• JAX-Metal: Not active")
    else:
        summary_points.append(f"• JAX: Not installed")
    
    # Performance highlights
    if benchmark_results and 'error' not in benchmark_results:
        # Vector operations
        vec_results = benchmark_results.get('vector_operations', {})
        if 'vector_add' in vec_results:
            throughput = vec_results['vector_add'].get('throughput_gflops', 0)
            if throughput > 0:
                summary_points.append(f"\n• Vector Add Throughput: {throughput:.2f} GFLOPs")
        
        # Matrix operations
        mat_results = benchmark_results.get('matrix_operations', {})
        if 'matrix_mul' in mat_results:
            gflops = mat_results['matrix_mul'].get('gflops', 0)
            if gflops > 0:
                summary_points.append(f"• Matrix Multiplication: {gflops:.2f} GFLOPs")
        
        # Data transfer
        transfer_results = benchmark_results.get('data_transfer', {})
        if transfer_results:
            # Find largest transfer size
            transfer_keys = [k for k in transfer_results.keys() if k.startswith('transfer_')]
            if transfer_keys:
                largest = max(transfer_keys, key=lambda x: int(x.split('_')[1]))
                avg_bw = transfer_results[largest].get('average_bandwidth_mb_s', 0)
                if avg_bw > 0:
                    summary_points.append(f"• Average Transfer Bandwidth: {avg_bw:.2f} MB/s")
        
        # JIT performance
        jit_results = benchmark_results.get('jit_performance', {})
        if 'jit' in jit_results:
            speedup = jit_results['jit'].get('speedup', 0)
            if speedup > 0:
                summary_points.append(f"• JIT Speedup: {speedup:.1f}x")
    
    # Compliance note
    summary_points.append("\n• All output is in English per MCPC coding standards")
    
    for point in summary_points:
        print(point)
    
    # Recommendations
    print("\nRecommendations:")
    if platform.system() == 'Darwin':
        backend_info = packages.get('jax_backend', {})
        # Fix: use case-insensitive comparison
        is_metal = backend_info.get('is_metal', False)
        if is_metal:
            print("1. JAX-Metal is active. Good for macOS GPU acceleration.")
        else:
            print("1. Consider enabling JAX-Metal for better GPU performance on macOS.")
    else:
        print("1. For Metal support, use macOS with Apple Silicon or discrete AMD GPU.")
    
    print("2. For compute-intensive workloads, ensure JAX JIT is used for best performance.")
    print("3. Consider batch size optimization for memory-bound operations.")
    print("4. Monitor memory usage for large matrix operations.")
    print("5. Use appropriate data types (float32 typically best for Metal performance).")


def export_results_to_json(system_info: Dict, packages: Dict, 
                          metal_info: Dict, jax_info: Dict,
                          benchmark_results: Dict, output_file: str) -> None:
    """Export benchmark results to JSON file."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "packages": packages,
        "metal_info": metal_info,
        "jax_info": jax_info,
        "benchmark_results": benchmark_results,
        "command_line_args": sys.argv[1:],  # Add command line arguments for record
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults exported to: {output_file}")
        if os.path.getsize(output_file) > 0:
            print(f"File size: {os.path.getsize(output_file)/1024:.2f} KB")
    except Exception as e:
        print(f"Error exporting results: {e}")


def main():
    """Main function to run all checks and benchmarks."""
    # Add command line argument parsing
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(
        description="Metal & JAX Environment Detection and Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default mode: show important info, filter experimental warnings
  %(prog)s --verbose               # Verbose mode: show all logs and warnings
  %(prog)s --quiet                 # Quiet mode: only show essential output and errors
  %(prog)s --verbose --output results.json  # Verbose mode with output to JSON
        """
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output, show all logs and warnings"
    )
    
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Enable quiet mode, only show essential output and errors"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Check mutually exclusive arguments
    if args.verbose and args.quiet:
        print("Error: Cannot specify both --verbose and --quiet")
        return
    
    # Set logging level based on arguments
    if args.verbose:
        # Verbose mode: show all logs and warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['JAX_LOG_LEVEL'] = 'INFO'
        warnings.resetwarnings()  # Reset all warning filters
        logging.basicConfig(level=logging.INFO)
        print(f"Verbose mode enabled - showing all logs and warnings")
    elif args.quiet:
        # Quiet mode: only show errors
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['JAX_LOG_LEVEL'] = 'ERROR'
        # Enable all warning filters
        warnings.filterwarnings('ignore')
        logging.basicConfig(level=logging.ERROR)
    else:
        # Default mode: filter known harmless warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['JAX_LOG_LEVEL'] = 'WARNING'
        # Only filter experimental warnings, keep other warnings
        logging.basicConfig(level=logging.WARNING)
        print(f"Default mode - filtering experimental warnings only")
    
    print_section("Metal & JAX Environment Detection and Performance Benchmark")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get system information
    print_section("System Information")
    if args.quiet:
        # Quiet mode: 抑制 C++ stderr 日志，但保留 Python print 输出（stdout）
        with suppress_stderr():
            system_info = get_system_info()
    else:
        system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check Metal and JAX installation
    print_section("Metal & JAX Installation Check")
    if args.quiet:
        with suppress_stderr():
            packages = check_metal_installation()
    else:
        packages = check_metal_installation()
    
    for pkg_name, pkg_info in packages.items():
        if pkg_info.get('available', False):
            version = pkg_info.get('version', 'Available')
            if pkg_name == 'jax_backend':
                platform_name = pkg_info.get('platform', 'Unknown')
                metal_status = " (Metal)" if pkg_info.get('is_metal', False) else ""
                print(f"{pkg_name}: {platform_name}{metal_status}")
            else:
                print(f"{pkg_name}: Installed (version: {version})")
        else:
            error_msg = pkg_info.get('error', 'Check installation')
            print(f"{pkg_name}: Not available - {error_msg}")
    
    # Get Metal device information
    print_section("Metal Device Information")
    if args.quiet:
        with suppress_stderr():
            metal_info = get_metal_device_info()
    else:
        metal_info = get_metal_device_info()
    
    if 'error' in metal_info:
        print(f"Error: {metal_info['error']}")
    else:
        if 'default_device' in metal_info:
            device = metal_info['default_device']
            if 'error' not in device:
                print(f"Default Metal Device: {device.get('name', 'Unknown')}")
                print(f"  Unified Memory: {'Yes' if device.get('has_unified_memory', False) else 'No'}")
                print(f"  Low Power: {'Yes' if device.get('is_low_power', False) else 'No'}")
                
                mem_info = device.get('memory_info', {})
                if 'recommended_max_working_set_size_gb' in mem_info:
                    mem_gb = mem_info['recommended_max_working_set_size_gb']
                    print(f"  Recommended Max Working Set: {mem_gb:.2f} GB")
            else:
                print(f"Default Metal Device Error: {device.get('error', 'Unknown error')}")
        
        if 'device_count' in metal_info:
            print(f"Total Metal Devices: {metal_info['device_count']}")
        
        if 'gpu_summary' in metal_info and metal_info['gpu_summary']:
            print("\nGPU Summary from system_profiler:")
            for line in metal_info['gpu_summary']:
                print(f"  {line}")
    
    # Get JAX device information
    print_section("JAX Device Information")
    if args.quiet:
        with suppress_stderr():
            jax_info = get_jax_device_info()
    else:
        jax_info = get_jax_device_info()
    
    if 'error' in jax_info:
        print(f"Error: {jax_info['error']}")
    else:
        backend = jax_info.get('backend', {})
        print(f"Backend Platform: {backend.get('platform', 'Unknown')}")
        print(f"Platform Version: {backend.get('platform_version', 'Unknown')}")
        print(f"Device Count: {backend.get('device_count', 0)}")
        
        devices = jax_info.get('devices', [])
        for device in devices:
            print(f"\nDevice {device['index']}:")
            print(f"  Kind: {device.get('device_kind', 'Unknown')}")
            print(f"  Platform: {device.get('platform', 'Unknown')}")
    
    # Ask user if they want to run benchmarks
    if jax_info and 'error' not in jax_info:
        if jax_info.get('device_count', 0) > 0:
            if not args.quiet:
                response = input(f"\nRun JAX performance benchmarks? [Y/n]: ").strip().lower()
                if response == '' or response == 'y':
                    if args.quiet:
                        with suppress_stderr():
                            benchmark_results = run_jax_benchmarks(device_idx=0)
                    else:
                        benchmark_results = run_jax_benchmarks(device_idx=0)
                else:
                    benchmark_results = {}
            else:
                # In quiet mode, automatically run benchmarks
                with suppress_stderr():
                    benchmark_results = run_jax_benchmarks(device_idx=0)
                
                # Print benchmark summary
                print_section("Benchmark Results")
                
                # Vector operations
                vec_results = benchmark_results.get('vector_operations', {})
                if 'error' not in vec_results:
                    print("\nVector Operations:")
                    for op_name, op_result in vec_results.items():
                        if op_name != 'error':
                            avg_ms = op_result.get('avg_time_ms', 0)
                            throughput = op_result.get('throughput_gflops', 0)
                            if throughput > 0:
                                print(f"  {op_name}: {avg_ms:.2f} ms, {throughput:.2f} GFLOPs")
                
                # Matrix operations
                mat_results = benchmark_results.get('matrix_operations', {})
                if 'error' not in mat_results:
                    print("\nMatrix Operations:")
                    for op_name, op_result in mat_results.items():
                        if op_name != 'error':
                            avg_ms = op_result.get('avg_time_ms', 0)
                            gflops = op_result.get('gflops', 0)
                            if gflops > 0:
                                print(f"  {op_name}: {avg_ms:.2f} ms, {gflops:.2f} GFLOPs")
                
                # JIT performance
                jit_results = benchmark_results.get('jit_performance', {})
                if 'jit' in jit_results:
                    jit_info = jit_results['jit']
                    print(f"\nJIT Performance:")
                    print(f"  Compilation Time: {jit_info.get('compile_time_ms', 0):.2f} ms")
                    print(f"  JIT Execution: {jit_info.get('jit_execution_avg_ms', 0):.2f} ms avg")
                    print(f"  Non-JIT Execution: {jit_info.get('non_jit_execution_avg_ms', 0):.2f} ms avg")
                    print(f"  Speedup: {jit_info.get('speedup', 0):.1f}x")
                
                # Data transfer
                transfer_results = benchmark_results.get('data_transfer', {})
                if 'error' not in transfer_results:
                    print("\nData Transfer Bandwidth:")
                    for key, transfer in transfer_results.items():
                        if key.startswith('transfer_'):
                            size_bytes = transfer['size_bytes']
                            avg_bw = transfer['average_bandwidth_mb_s']
                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_bytes < 1024*1024:
                                size_str = f"{size_bytes/1024:.0f} KB"
                            else:
                                size_str = f"{size_bytes/(1024*1024):.0f} MB"
                            print(f"  {size_str}: {avg_bw:.2f} MB/s")
        else:
            print("No JAX devices found for benchmarking.")
            benchmark_results = {}
    else:
        print("JAX not available for benchmarking.")
        benchmark_results = {}
    
    # Print final summary
    print_summary(system_info, packages, metal_info, jax_info, benchmark_results)
    
    # Handle exporting results
    if benchmark_results:
        # If output file is specified via command line, export directly
        if args.output:
            output_file = args.output
            export_results_to_json(system_info, packages, metal_info, jax_info,
                                 benchmark_results, output_file)
        elif not args.quiet:  # Don't ask in quiet mode
            response = input(f"\nExport results to JSON file? [y/N]: ").strip().lower()
            if response == 'y':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"metal_jax_benchmark_{timestamp}.json"
                export_results_to_json(system_info, packages, metal_info, jax_info,
                                     benchmark_results, output_file)
    
    print_section("Benchmark Complete")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
