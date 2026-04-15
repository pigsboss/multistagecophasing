#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCL Environment Detection and Performance Benchmark
======================================================================
Detects OpenCL installation, available platforms and devices, and runs performance tests.
All output is in English per MCPC coding standards.
"""

import sys
import time
import platform
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

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

def get_system_info() -> Dict[str, str]:
    """Gather system information."""
    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'platform_release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor() or 'Unknown',
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

def check_opencl_installation() -> Dict[str, Any]:
    """Check PyOpenCL and related packages installation."""
    packages = {}
    
    # Check PyOpenCL
    try:
        import pyopencl as cl
        packages['pyopencl'] = {
            'version': cl.VERSION_TEXT if hasattr(cl, 'VERSION_TEXT') else 'Unknown',
            'available': True
        }
    except ImportError:
        packages['pyopencl'] = {'available': False, 'error': 'Not installed'}
    
    # Check clinfo command
    try:
        result = subprocess.run(['clinfo', '--version'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            packages['clinfo'] = {
                'available': True,
                'version': result.stdout.strip() if result.stdout else 'Unknown'
            }
        else:
            packages['clinfo'] = {'available': False, 'error': 'Command failed'}
    except (subprocess.SubprocessError, FileNotFoundError):
        packages['clinfo'] = {'available': False, 'error': 'Not installed or not in PATH'}
    
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

def get_opencl_info() -> Dict[str, Any]:
    """Get detailed OpenCL platform and device information."""
    info = {}
    
    try:
        import pyopencl as cl
        
        # Get all platforms
        platforms = cl.get_platforms()
        info['platform_count'] = len(platforms)
        info['platforms'] = []
        
        for i, platform in enumerate(platforms):
            platform_info = {
                'index': i,
                'name': platform.name.strip(),
                'vendor': platform.vendor.strip(),
                'version': platform.version.strip(),
                'profile': platform.profile.strip(),
            }
            
            # Get devices for this platform
            try:
                devices = platform.get_devices()
                platform_info['device_count'] = len(devices)
                platform_info['devices'] = []
                
                for j, device in enumerate(devices):
                    # 修复：将设备类型转换为可读的字符串
                    device_type_str = "Unknown"
                    if device.type == cl.device_type.CPU:
                        device_type_str = "CPU"
                    elif device.type == cl.device_type.GPU:
                        device_type_str = "GPU"
                    elif device.type == cl.device_type.ACCELERATOR:
                        device_type_str = "Accelerator"
                    elif device.type == cl.device_type.DEFAULT:
                        device_type_str = "Default"
                    elif device.type == cl.device_type.ALL:
                        device_type_str = "All"
                    else:
                        # 处理组合类型
                        type_flags = []
                        if device.type & cl.device_type.CPU:
                            type_flags.append("CPU")
                        if device.type & cl.device_type.GPU:
                            type_flags.append("GPU")
                        if device.type & cl.device_type.ACCELERATOR:
                            type_flags.append("Accelerator")
                        if device.type & cl.device_type.DEFAULT:
                            type_flags.append("Default")
                        if type_flags:
                            device_type_str = " | ".join(type_flags)
                    
                    device_info = {
                        'index': j,
                        'name': device.name.strip(),
                        'vendor': device.vendor.strip(),
                        'version': device.version.strip(),
                        'type': device_type_str,  # 使用可读的字符串
                        'type_code': int(device.type),  # 保留原始类型代码
                        'max_compute_units': device.max_compute_units,
                        'max_work_group_size': device.max_work_group_size,
                        'max_work_item_sizes': device.max_work_item_sizes,
                        'global_mem_size': device.global_mem_size,
                        'local_mem_size': device.local_mem_size,
                        'max_clock_frequency': device.max_clock_frequency if hasattr(device, 'max_clock_frequency') else 0,
                        'address_bits': device.address_bits if hasattr(device, 'address_bits') else 0,
                    }
                    
                    # Check extensions
                    if hasattr(device, 'extensions'):
                        extensions = device.extensions.strip().split()
                        device_info['extensions'] = extensions
                        
                        # Check for fp16 support
                        device_info['fp16_support'] = 'cl_khr_fp16' in extensions or 'cl_amd_fp16' in extensions
                        
                        # Check for fp64 support
                        device_info['fp64_support'] = 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions
                    else:
                        device_info['extensions'] = []
                        device_info['fp16_support'] = False
                        device_info['fp64_support'] = True  # Assume fp32 is always supported
                    
                    platform_info['devices'].append(device_info)
                
            except cl.Error as e:
                platform_info['devices_error'] = str(e)
                platform_info['device_count'] = 0
            
            info['platforms'].append(platform_info)
        
        # If no platforms found
        if not platforms:
            info['error'] = 'No OpenCL platforms found'
    
    except ImportError:
        info['error'] = 'PyOpenCL not installed'
    except cl.Error as e:
        info['error'] = f'OpenCL error: {str(e)}'
    except Exception as e:
        info['error'] = f'Unexpected error: {str(e)}'
    
    return info

def create_context_for_device(platform_idx: int, device_idx: int) -> Any:
    """Create OpenCL context for a specific device."""
    try:
        import pyopencl as cl
        
        platforms = cl.get_platforms()
        if platform_idx >= len(platforms):
            return None
        
        platform = platforms[platform_idx]
        devices = platform.get_devices()
        
        if device_idx >= len(devices):
            return None
        
        device = devices[device_idx]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        return {
            'context': context,
            'queue': queue,
            'device': device,
            'platform': platform
        }
    except Exception as e:
        print(f"Error creating context: {e}")
        return None

def benchmark_fp_operations(context_info: Dict) -> Dict[str, Any]:
    """Benchmark floating-point operations for different precisions."""
    benchmarks = {}
    
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        
        context = context_info['context']
        queue = context_info['queue']
        device = context_info['device']
        
        # Check device extensions for fp support
        extensions = device.extensions.strip().split() if hasattr(device, 'extensions') else []
        fp16_support = 'cl_khr_fp16' in extensions or 'cl_amd_fp16' in extensions
        fp64_support = 'cl_khr_fp64' in extensions or 'cl_amd_fp64' in extensions
        
        # Test sizes
        test_size = 1024 * 1024  # 1M elements
        
        # FP32 test (always supported)
        print("  Testing FP32 operations...")
        
        # Create test data
        host_data = np.random.randn(test_size).astype(np.float32)
        
        # Create kernel
        kernel_code = """
        __kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
            int idx = get_global_id(0);
            c[idx] = a[idx] + b[idx] * 2.0f - a[idx] / (b[idx] + 1.0f);
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        
        # Transfer data to device
        start = time.perf_counter()
        a_dev = cl_array.to_device(queue, host_data)
        b_dev = cl_array.to_device(queue, host_data)
        c_dev = cl_array.empty(queue, test_size, dtype=np.float32)
        transfer_time = time.perf_counter() - start
        
        # Execute kernel
        start = time.perf_counter()
        program.vector_add(queue, (test_size,), None, a_dev.data, b_dev.data, c_dev.data)
        queue.finish()
        kernel_time = time.perf_counter() - start
        
        # Transfer back
        start = time.perf_counter()
        result = c_dev.get()
        transfer_back_time = time.perf_counter() - start
        
        benchmarks['fp32'] = {
            'supported': True,
            'test_size': test_size,
            'transfer_to_device_ms': transfer_time * 1000,
            'kernel_execution_ms': kernel_time * 1000,
            'transfer_from_device_ms': transfer_back_time * 1000,
            'total_ms': (transfer_time + kernel_time + transfer_back_time) * 1000,
            'throughput_gflops': (test_size * 4) / (kernel_time * 1e9) if kernel_time > 0 else 0,  # 4 ops per element
        }
        
        # FP64 test if supported
        if fp64_support:
            print("  Testing FP64 operations...")
            
            host_data_fp64 = np.random.randn(test_size).astype(np.float64)
            
            kernel_code_fp64 = """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void vector_add_fp64(__global const double* a, __global const double* b, __global double* c) {
                int idx = get_global_id(0);
                c[idx] = a[idx] + b[idx] * 2.0 - a[idx] / (b[idx] + 1.0);
            }
            """
            
            try:
                program_fp64 = cl.Program(context, kernel_code_fp64).build()
                
                start = time.perf_counter()
                a_dev_fp64 = cl_array.to_device(queue, host_data_fp64)
                b_dev_fp64 = cl_array.to_device(queue, host_data_fp64)
                c_dev_fp64 = cl_array.empty(queue, test_size, dtype=np.float64)
                transfer_time_fp64 = time.perf_counter() - start
                
                start = time.perf_counter()
                program_fp64.vector_add_fp64(queue, (test_size,), None, 
                                            a_dev_fp64.data, b_dev_fp64.data, c_dev_fp64.data)
                queue.finish()
                kernel_time_fp64 = time.perf_counter() - start
                
                start = time.perf_counter()
                result_fp64 = c_dev_fp64.get()
                transfer_back_time_fp64 = time.perf_counter() - start
                
                benchmarks['fp64'] = {
                    'supported': True,
                    'test_size': test_size,
                    'transfer_to_device_ms': transfer_time_fp64 * 1000,
                    'kernel_execution_ms': kernel_time_fp64 * 1000,
                    'transfer_from_device_ms': transfer_back_time_fp64 * 1000,
                    'total_ms': (transfer_time_fp64 + kernel_time_fp64 + transfer_back_time_fp64) * 1000,
                    'throughput_gflops': (test_size * 4) / (kernel_time_fp64 * 1e9) if kernel_time_fp64 > 0 else 0,
                }
            except cl.Error as e:
                benchmarks['fp64'] = {
                    'supported': False,
                    'error': str(e)
                }
        else:
            benchmarks['fp64'] = {'supported': False, 'error': 'FP64 not supported'}
        
        # FP16 test if supported
        if fp16_support:
            print("  Testing FP16 operations...")
            
            # Note: NumPy doesn't have native float16 support for random, so we use float32 and convert
            host_data_fp32 = np.random.randn(test_size).astype(np.float32)
            host_data_fp16 = host_data_fp32.astype(np.float16)
            
            kernel_code_fp16 = """
            #pragma OPENCL EXTENSION cl_khr_fp16 : enable
            __kernel void vector_add_fp16(__global const half* a, __global const half* b, __global half* c) {
                int idx = get_global_id(0);
                c[idx] = a[idx] + b[idx] * 2.0h - a[idx] / (b[idx] + 1.0h);
            }
            """
            
            try:
                program_fp16 = cl.Program(context, kernel_code_fp16).build()
                
                # Convert to appropriate format for OpenCL
                # PyOpenCL might need special handling for half precision
                start = time.perf_counter()
                # Use float16 if supported, otherwise fall back
                a_dev_fp16 = cl_array.to_device(queue, host_data_fp16.astype(np.float16))
                b_dev_fp16 = cl_array.to_device(queue, host_data_fp16.astype(np.float16))
                c_dev_fp16 = cl_array.empty(queue, test_size, dtype=np.float16)
                transfer_time_fp16 = time.perf_counter() - start
                
                start = time.perf_counter()
                program_fp16.vector_add_fp16(queue, (test_size,), None, 
                                            a_dev_fp16.data, b_dev_fp16.data, c_dev_fp16.data)
                queue.finish()
                kernel_time_fp16 = time.perf_counter() - start
                
                start = time.perf_counter()
                result_fp16 = c_dev_fp16.get()
                transfer_back_time_fp16 = time.perf_counter() - start
                
                benchmarks['fp16'] = {
                    'supported': True,
                    'test_size': test_size,
                    'transfer_to_device_ms': transfer_time_fp16 * 1000,
                    'kernel_execution_ms': kernel_time_fp16 * 1000,
                    'transfer_from_device_ms': transfer_back_time_fp16 * 1000,
                    'total_ms': (transfer_time_fp16 + kernel_time_fp16 + transfer_back_time_fp16) * 1000,
                    'throughput_gflops': (test_size * 4) / (kernel_time_fp16 * 1e9) if kernel_time_fp16 > 0 else 0,
                }
            except (cl.Error, TypeError) as e:
                benchmarks['fp16'] = {
                    'supported': False,
                    'error': f'FP16 test failed: {str(e)}'
                }
        else:
            benchmarks['fp16'] = {'supported': False, 'error': 'FP16 not supported'}
    
    except Exception as e:
        benchmarks['error'] = f'Floating-point benchmark failed: {str(e)}'
    
    return benchmarks

def benchmark_host_device_bandwidth(context_info: Dict) -> Dict[str, Any]:
    """Benchmark host-device transfer bandwidth."""
    benchmarks = {}
    
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        
        context = context_info['context']
        queue = context_info['queue']
        
        # Test different buffer sizes
        sizes_bytes = [1024, 1024*1024, 16*1024*1024]  # 1KB, 1MB, 16MB
        
        for size_bytes in sizes_bytes:
            # Create test data
            element_count = size_bytes // 4  # Using float32 (4 bytes)
            if element_count == 0:
                element_count = 1
            
            host_data = np.random.randn(element_count).astype(np.float32)
            
            # 对小数据量进行多次测试取平均值
            if size_bytes == 1024:
                repeat_count = 100  # 对小数据量重复多次
                to_device_times = []
                from_device_times = []
                
                for _ in range(repeat_count):
                    # Measure transfer to device
                    start = time.perf_counter()
                    dev_buffer = cl_array.to_device(queue, host_data)
                    queue.finish()
                    to_device_times.append(time.perf_counter() - start)
                    
                    # Measure transfer from device
                    start = time.perf_counter()
                    host_result = dev_buffer.get()
                    queue.finish()
                    from_device_times.append(time.perf_counter() - start)
                
                to_device_time = np.median(to_device_times)  # 使用中位数减少异常值影响
                from_device_time = np.median(from_device_times)
            else:
                # Measure transfer to device
                start = time.perf_counter()
                dev_buffer = cl_array.to_device(queue, host_data)
                queue.finish()
                to_device_time = time.perf_counter() - start
                
                # Measure transfer from device
                start = time.perf_counter()
                host_result = dev_buffer.get()
                queue.finish()
                from_device_time = time.perf_counter() - start
            
            # Verify data integrity
            assert np.allclose(host_data, host_result, rtol=1e-5), "Data transfer corrupted"
            
            # Calculate bandwidth
            to_device_bw = size_bytes / (to_device_time * 1024*1024)  # MB/s
            from_device_bw = size_bytes / (from_device_time * 1024*1024)  # MB/s
            
            benchmarks[f'transfer_{size_bytes}_bytes'] = {
                'size_bytes': size_bytes,
                'to_device_ms': to_device_time * 1000,
                'from_device_ms': from_device_time * 1000,
                'to_device_bandwidth_mb_s': to_device_bw,
                'from_device_bandwidth_mb_s': from_device_bw,
                'average_bandwidth_mb_s': (to_device_bw + from_device_bw) / 2,
            }
    
    except Exception as e:
        benchmarks['error'] = f'Bandwidth benchmark failed: {str(e)}'
    
    return benchmarks

def benchmark_device_memory(context_info: Dict) -> Dict[str, Any]:
    """Benchmark device memory read/write speed."""
    benchmarks = {}
    
    try:
        import pyopencl as cl
        
        context = context_info['context']
        queue = context_info['queue']
        device = context_info['device']
        
        # Test sizes
        sizes_bytes = [1024, 1024*1024, 4*1024*1024]  # 1KB, 1MB, 4MB
        
        # 修复：将内核代码定义移到循环外部
        kernel_code = """
        __kernel void memory_bandwidth(__global float* output, __global const float* input, 
                                      const int size, const float scale) {
            int idx = get_global_id(0);
            if (idx < size) {
                // Multiple memory operations
                float val = input[idx];
                output[idx] = val * scale + 1.0f / (val + 1.0f);
            }
        }
        """
        
        # 修复：编译程序一次，然后重复使用
        program = cl.Program(context, kernel_code).build()
        kernel = cl.Kernel(program, 'memory_bandwidth')  # 获取内核实例
        
        for size_bytes in sizes_bytes:
            element_count = size_bytes // 4  # float32
            
            # Create buffers
            host_input = np.random.randn(element_count).astype(np.float32)
            host_output = np.zeros(element_count, dtype=np.float32)
            
            # Create device buffers
            mf = cl.mem_flags
            input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_input)
            output_buffer = cl.Buffer(context, mf.WRITE_ONLY, size_bytes)
            
            # Execute kernel multiple times to get average performance
            iterations = 100 if size_bytes <= 1024*1024 else 10
            
            start = time.perf_counter()
            for _ in range(iterations):
                # 修复：使用已获取的内核实例
                kernel(queue, (element_count,), None, 
                       output_buffer, input_buffer, 
                       np.int32(element_count), np.float32(2.0))
                queue.finish()
            kernel_time = time.perf_counter() - start
            
            # Calculate memory bandwidth
            # Each iteration: 1 read + 1 write per element = 2 * size_bytes
            total_bytes_transferred = iterations * 2 * size_bytes
            bandwidth = total_bytes_transferred / (kernel_time * 1024*1024*1024)  # GB/s
            
            benchmarks[f'memory_{size_bytes}_bytes'] = {
                'size_bytes': size_bytes,
                'iterations': iterations,
                'total_kernel_time_ms': kernel_time * 1000,
                'avg_iteration_time_ms': (kernel_time / iterations) * 1000,
                'memory_bandwidth_gb_s': bandwidth,
                'estimated_memory_latency_ns': (kernel_time / (iterations * element_count)) * 1e9 if element_count > 0 else 0,
            }
    
    except Exception as e:
        benchmarks['error'] = f'Device memory benchmark failed: {str(e)}'
    
    return benchmarks

def run_device_benchmarks(platform_idx: int, device_idx: int, 
                          device_info: Dict) -> Dict[str, Any]:
    """Run all benchmarks for a specific device."""
    print(f"\nRunning benchmarks for device {device_idx}: {device_info['name']}")
    
    results = {}
    
    # Create context
    context_info = create_context_for_device(platform_idx, device_idx)
    if not context_info:
        results['error'] = 'Failed to create OpenCL context'
        return results
    
    results['device_info'] = device_info
    
    # Run floating-point benchmarks
    print("  Running floating-point benchmarks...")
    fp_benchmarks = benchmark_fp_operations(context_info)
    results['floating_point'] = fp_benchmarks
    
    # Run bandwidth benchmarks
    print("  Running host-device bandwidth benchmarks...")
    bandwidth_benchmarks = benchmark_host_device_bandwidth(context_info)
    results['bandwidth'] = bandwidth_benchmarks
    
    # Run device memory benchmarks
    print("  Running device memory benchmarks...")
    memory_benchmarks = benchmark_device_memory(context_info)
    results['device_memory'] = memory_benchmarks
    
    return results

def print_summary(system_info: Dict, packages: Dict, 
                  opencl_info: Dict, benchmark_results: List[Dict]) -> None:
    """Print a summary of key findings."""
    print_section("SUMMARY OF KEY FINDINGS")
    
    summary_points = []
    
    # System info
    summary_points.append(f"• Platform: {system_info['platform']} {system_info['architecture']}")
    summary_points.append(f"• Python: {system_info['python_version']}")
    
    # OpenCL availability
    if packages.get('pyopencl', {}).get('available', False):
        summary_points.append(f"• PyOpenCL: Installed")
    else:
        summary_points.append(f"• PyOpenCL: Not installed")
    
    if packages.get('clinfo', {}).get('available', False):
        summary_points.append(f"• clinfo: Installed")
    else:
        summary_points.append(f"• clinfo: Not installed")
    
    # OpenCL platforms
    if 'error' not in opencl_info:
        platform_count = opencl_info.get('platform_count', 0)
        summary_points.append(f"• OpenCL platforms found: {platform_count}")
        
        total_devices = 0
        for platform in opencl_info.get('platforms', []):
            total_devices += platform.get('device_count', 0)
        summary_points.append(f"• Total OpenCL devices: {total_devices}")
    else:
        summary_points.append(f"• OpenCL platforms: {opencl_info.get('error', 'Unknown error')}")
    
    # Performance highlights
    if benchmark_results:
        for i, result in enumerate(benchmark_results):
            if 'error' in result:
                continue
                
            device_info = result.get('device_info', {})
            device_name = device_info.get('name', f'Device {i}')
            
            summary_points.append(f"\n• Device: {device_name}")
            
            # Floating point performance
            fp_results = result.get('floating_point', {})
            if 'fp32' in fp_results and fp_results['fp32'].get('supported', False):
                gflops = fp_results['fp32'].get('throughput_gflops', 0)
                if gflops > 0:
                    summary_points.append(f"  FP32 throughput: {gflops:.2f} GFLOPs")
            
            # Bandwidth
            bw_results = result.get('bandwidth', {})
            if bw_results and 'error' not in bw_results:
                # Get largest transfer size
                bw_keys = [k for k in bw_results.keys() if k.startswith('transfer_')]
                if bw_keys:
                    largest = max(bw_keys, key=lambda x: int(x.split('_')[1]))
                    avg_bw = bw_results[largest].get('average_bandwidth_mb_s', 0)
                    if avg_bw > 0:
                        summary_points.append(f"  Average transfer bandwidth: {avg_bw:.2f} MB/s")
            
            # Memory bandwidth
            mem_results = result.get('device_memory', {})
            if mem_results and 'error' not in mem_results:
                mem_keys = [k for k in mem_results.keys() if k.startswith('memory_')]
                if mem_keys:
                    largest = max(mem_keys, key=lambda x: int(x.split('_')[1]))
                    mem_bw = mem_results[largest].get('memory_bandwidth_gb_s', 0)
                    if mem_bw > 0:
                        summary_points.append(f"  Device memory bandwidth: {mem_bw:.2f} GB/s")
    
    # Compliance note
    summary_points.append("\n• All output is in English per MCPC coding standards")
    
    for point in summary_points:
        print(point)
    
    # Recommendations
    print("\nRecommendations:")
    print("1. For better OpenCL performance, ensure latest GPU drivers are installed")
    print("2. Consider using device-specific optimizations for compute-intensive workloads")
    print("3. For multi-device systems, profile each device separately")
    print("4. Use appropriate work group sizes for optimal performance")
    print("5. Consider memory coalescing for better memory bandwidth utilization")

def main():
    """Main function to run all checks and benchmarks."""
    # 设置环境变量以控制编译器输出
    import os
    if 'PYOPENCL_COMPILER_OUTPUT' not in os.environ:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'  # 默认为0，不显示编译器输出
    
    print_section("OpenCL Environment Detection and Performance Benchmark")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get system information
    print_section("System Information")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check OpenCL installation
    print_section("OpenCL Installation Check")
    packages = check_opencl_installation()
    
    for pkg_name, pkg_info in packages.items():
        if pkg_info.get('available', False):
            version = pkg_info.get('version', 'Unknown')
            print(f"{pkg_name}: Installed (version: {version})")
        else:
            print(f"{pkg_name}: Not available - {pkg_info.get('error', 'Check installation')}")
    
    # Get OpenCL information
    print_section("OpenCL Platform and Device Information")
    opencl_info = get_opencl_info()
    
    if 'error' in opencl_info:
        print(f"Error: {opencl_info['error']}")
        print("Cannot proceed with benchmarks without OpenCL platforms.")
        return
    
    platform_count = opencl_info.get('platform_count', 0)
    print(f"Found {platform_count} OpenCL platform(s)")
    
    all_benchmark_results = []
    
    for platform_info in opencl_info.get('platforms', []):
        print_subsection(f"Platform {platform_info['index']}: {platform_info['name']}")
        print(f"Vendor: {platform_info['vendor']}")
        print(f"Version: {platform_info['version']}")
        print(f"Profile: {platform_info['profile']}")
        print(f"Devices: {platform_info.get('device_count', 0)}")
        
        # Print device details
        devices = platform_info.get('devices', [])
        for device_info in devices:
            print(f"\n  Device {device_info['index']}: {device_info['name']}")
            print(f"    Type: {device_info['type']}")
            print(f"    Compute Units: {device_info['max_compute_units']}")
            print(f"    Global Memory: {device_info['global_mem_size'] / (1024**3):.2f} GB")
            print(f"    Local Memory: {device_info['local_mem_size'] / 1024:.2f} KB")
            print(f"    Max Work Group Size: {device_info['max_work_group_size']}")
            print(f"    Clock Frequency: {device_info.get('max_clock_frequency', 0)} MHz")
            print(f"    FP16 Support: {'Yes' if device_info.get('fp16_support', False) else 'No'}")
            print(f"    FP64 Support: {'Yes' if device_info.get('fp64_support', False) else 'No'}")
            
            # Ask if we should benchmark this device
            response = input(f"\n  Benchmark device {device_info['index']} ({device_info['name']})? [Y/n]: ").strip().lower()
            if response == '' or response == 'y':
                # Run benchmarks for this device
                benchmark_result = run_device_benchmarks(
                    platform_info['index'],
                    device_info['index'],
                    device_info
                )
                all_benchmark_results.append(benchmark_result)
                
                # Print benchmark summary for this device
                if 'error' not in benchmark_result:
                    print(f"\n  Benchmark results for {device_info['name']}:")
                    
                    # Floating point results
                    fp_results = benchmark_result.get('floating_point', {})
                    if 'error' not in fp_results:
                        print("    Floating Point Performance:")
                        for fp_type in ['fp16', 'fp32', 'fp64']:
                            if fp_type in fp_results:
                                fp_info = fp_results[fp_type]
                                if fp_info.get('supported', False):
                                    gflops = fp_info.get('throughput_gflops', 0)
                                    print(f"      {fp_type.upper()}: {gflops:.2f} GFLOPs")
                                else:
                                    print(f"      {fp_type.upper()}: Not supported")
                    
                    # Bandwidth results
                    bw_results = benchmark_result.get('bandwidth', {})
                    if 'error' not in bw_results:
                        print("    Host-Device Bandwidth:")
                        for key, bw_info in bw_results.items():
                            if key.startswith('transfer_'):
                                size_bytes = bw_info['size_bytes']
                                # 根据大小选择合适的单位
                                if size_bytes < 1024:
                                    size_str = f"{size_bytes} B"
                                elif size_bytes < 1024 * 1024:
                                    size_str = f"{size_bytes / 1024:.2f} KB"
                                else:
                                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                                avg_bw = bw_info['average_bandwidth_mb_s']
                                print(f"      {size_str}: {avg_bw:.2f} MB/s")
                    
                    # Memory results
                    mem_results = benchmark_result.get('device_memory', {})
                    if 'error' not in mem_results:
                        print("    Device Memory Bandwidth:")
                        for key, mem_info in mem_results.items():
                            if key.startswith('memory_'):
                                size_bytes = mem_info['size_bytes']
                                if size_bytes < 1024:
                                    size_str = f"{size_bytes} B"
                                elif size_bytes < 1024 * 1024:
                                    size_str = f"{size_bytes / 1024:.2f} KB"
                                else:
                                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                                bandwidth = mem_info['memory_bandwidth_gb_s']
                                print(f"      {size_str}: {bandwidth:.2f} GB/s")
                else:
                    print(f"  Benchmark failed: {benchmark_result.get('error', 'Unknown error')}")
    
    # Print final summary
    print_summary(system_info, packages, opencl_info, all_benchmark_results)
    
    print_section("Benchmark Complete")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
