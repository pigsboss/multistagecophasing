#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX Compilation System and Hardware Backend Detection
======================================================================
Detects JAX compilation system, XLA runtime, PJRT plugins for various
hardware accelerators (NVIDIA GPU, Intel Arc GPU, AMD GPU, Apple Silicon,
Apple Metal GPU). Displays available compute devices for JIT acceleration.
All output is in English per MCPC coding standards.
"""

import sys
import platform
import subprocess
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Environment variables for JAX/XLA logging control
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('JAX_LOG_LEVEL', 'WARNING')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_enable_fast_math=false')


def print_section(title: str, width: int = 70) -> None:
    """Print a formatted section title."""
    print(f"\n{'=' * width}")
    print(f" {title.upper()}")
    print(f"{'=' * width}")


def print_subsection(title: str) -> None:
    """Print a formatted subsection title."""
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")


@contextmanager
def suppress_jax_warnings():
    """Context manager to suppress JAX initialization warnings."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


@dataclass
class BackendInfo:
    """Information about a JAX backend."""
    name: str
    platform: str
    available: bool
    version: str = "Unknown"
    device_count: int = 0
    devices: List[Dict[str, Any]] = field(default_factory=list)
    pjrt_plugin: Optional[str] = None
    xla_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    priority: int = 0  # Lower number = higher priority for auto-selection


def get_system_info() -> Dict[str, str]:
    """Gather system information."""
    info = {
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'platform_release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor() or 'Unknown',
    }
    
    # OS-specific info
    if platform.system() == 'Darwin':
        info['mac_ver'] = platform.mac_ver()[0]
        if platform.machine() == 'arm64':
            info['apple_silicon'] = 'Yes'
        else:
            info['apple_silicon'] = 'No (Intel)'
    elif platform.system() == 'Linux':
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        info['linux_distro'] = line.split('=', 1)[1].strip().strip('"')
                        break
        except:
            info['linux_distro'] = 'Unknown'
    elif platform.system() == 'Windows':
        info['windows_ver'] = platform.version()
    
    # Get memory information if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_ram_gb'] = f"{mem.total / (1024**3):.2f}"
        info['available_ram_gb'] = f"{mem.available / (1024**3):.2f}"
        info['swap_total_gb'] = f"{mem.swap_total / (1024**3):.2f}" if mem.swap_total > 0 else '0.00'
        
        # CPU info
        info['cpu_cores'] = psutil.cpu_count(logical=False) or 'Unknown'
        info['cpu_threads'] = psutil.cpu_count(logical=True) or 'Unknown'
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info['cpu_freq_mhz'] = f"{cpu_freq.current:.1f}"
    except ImportError:
        info['total_ram_gb'] = 'Unknown (install psutil)'
        info['available_ram_gb'] = 'Unknown (install psutil)'
    
    return info


def check_jax_installation() -> Dict[str, Any]:
    """Check JAX core installation and dependencies."""
    info = {
        'jax_available': False,
        'jax_version': None,
        'jaxlib_version': None,
        'xla_build_info': {},
        'dependencies': {},
    }
    
    try:
        # First try to import without triggering warnings
        with suppress_jax_warnings():
            import jax
            import jaxlib
        
        info['jax_available'] = True
        info['jax_version'] = jax.__version__
        
        # Get jaxlib version
        try:
            info['jaxlib_version'] = jaxlib.__version__
        except:
            info['jaxlib_version'] = 'Unknown'
        
        # Get XLA build info
        try:
            from jax.lib import xla_bridge
            info['xla_build_info'] = {
                'backend': xla_bridge.get_backend().platform,
                'xla_version': getattr(xla_bridge, 'XLA_VERSION', 'Unknown'),
            }
        except:
            pass
        
        # Check for GPU-related packages
        try:
            import tensorflow as tf
            info['dependencies']['tensorflow'] = {
                'version': tf.__version__,
                'available': True
            }
        except ImportError:
            info['dependencies']['tensorflow'] = {'available': False}
        
        # Check for PyTorch (sometimes used with JAX)
        try:
            import torch
            info['dependencies']['pytorch'] = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
                'available': True
            }
        except ImportError:
            info['dependencies']['pytorch'] = {'available': False}
        
        # Check for cuDNN
        try:
            result = subprocess.run(['whereis', 'cudnn'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                info['dependencies']['cudnn'] = {'available': True, 'path': result.stdout.strip()}
            else:
                info['dependencies']['cudnn'] = {'available': False}
        except:
            info['dependencies']['cudnn'] = {'available': False, 'error': 'Check failed'}
        
        # Check for NCCL
        try:
            result = subprocess.run(['whereis', 'nccl'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                info['dependencies']['nccl'] = {'available': True, 'path': result.stdout.strip()}
            else:
                info['dependencies']['nccl'] = {'available': False}
        except:
            info['dependencies']['nccl'] = {'available': False, 'error': 'Check failed'}
        
    except ImportError as e:
        info['error'] = f'JAX not installed: {e}'
    
    return info


def detect_pjrt_plugins() -> Dict[str, BackendInfo]:
    """Detect available PJRT plugins for different hardware."""
    plugins = {}
    
    # Always check CPU first (base backend)
    cpu_info = BackendInfo(
        name='CPU',
        platform='cpu',
        available=False,
        pjrt_plugin='pjrt_c_api_cpu_plugin',
        priority=100
    )
    
    try:
        with suppress_jax_warnings():
            import jax
        
        # Get all devices
        all_devices = jax.devices()
        
        # CPU devices
        cpu_devices = [d for d in all_devices if d.platform == 'cpu']
        if cpu_devices:
            cpu_info.available = True
            cpu_info.device_count = len(cpu_devices)
            cpu_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
                'client': str(d.client)[:100],
            } for d in cpu_devices]
        
        plugins['cpu'] = cpu_info
        
        # Check for CUDA (NVIDIA GPU)
        cuda_info = BackendInfo(
            name='NVIDIA GPU (CUDA)',
            platform='cuda',
            available=False,
            pjrt_plugin='pjrt_c_api_gpu_plugin (CUDA)',
            priority=10
        )
        
        cuda_devices = [d for d in all_devices if d.platform == 'cuda']
        if cuda_devices:
            cuda_info.available = True
            cuda_info.device_count = len(cuda_devices)
            cuda_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
                'memory': getattr(d, 'memory', 'Unknown'),
            } for d in cuda_devices]
            
            # Try to get CUDA version
            try:
                from jax.lib import cuda
                if hasattr(cuda, 'cuda_version'):
                    cuda_info.version = str(cuda.cuda_version())
            except:
                pass
        
        plugins['cuda'] = cuda_info
        
        # Check for ROCm (AMD GPU)
        rocm_info = BackendInfo(
            name='AMD GPU (ROCm)',
            platform='rocm',
            available=False,
            pjrt_plugin='pjrt_c_api_gpu_plugin (ROCm)',
            priority=20
        )
        
        rocm_devices = [d for d in all_devices if d.platform == 'rocm']
        if rocm_devices:
            rocm_info.available = True
            rocm_info.device_count = len(rocm_devices)
            rocm_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
            } for d in rocm_devices]
        
        plugins['rocm'] = rocm_info
        
        # Check for Metal (Apple GPU)
        metal_info = BackendInfo(
            name='Apple Metal GPU',
            platform='metal',
            available=False,
            pjrt_plugin='pjrt_c_api_metal_plugin',
            priority=30
        )
        
        metal_devices = [d for d in all_devices if d.platform == 'metal']
        if metal_devices:
            metal_info.available = True
            metal_info.device_count = len(metal_devices)
            metal_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
            } for d in metal_devices]
        
        plugins['metal'] = metal_info
        
        # Check for TPU
        tpu_info = BackendInfo(
            name='Google Cloud TPU',
            platform='tpu',
            available=False,
            pjrt_plugin='pjrt_c_api_tpu_plugin',
            priority=40
        )
        
        tpu_devices = [d for d in all_devices if d.platform == 'tpu']
        if tpu_devices:
            tpu_info.available = True
            tpu_info.device_count = len(tpu_devices)
            tpu_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
            } for d in tpu_devices]
        
        plugins['tpu'] = tpu_info
        
        # Check for Intel GPU (via oneAPI/SYCL)
        intel_info = BackendInfo(
            name='Intel GPU (Arc/Data Center)',
            platform='intel',
            available=False,
            pjrt_plugin='pjrt_c_api_intel_gpu_plugin',
            priority=50
        )
        
        # Intel devices might appear under 'gpu' or 'sycl' platform
        intel_devices = []
        for d in all_devices:
            if (d.platform in ['gpu', 'sycl', 'opencl'] and 
                any(keyword in d.device_kind.lower() for keyword in ['intel', 'arc', 'xe'])):
                intel_devices.append(d)
        
        if intel_devices:
            intel_info.available = True
            intel_info.device_count = len(intel_devices)
            intel_info.devices = [{
                'id': d.id,
                'kind': d.device_kind,
                'platform': d.platform,
            } for d in intel_devices]
        
        plugins['intel_gpu'] = intel_info
        
    except ImportError as e:
        plugins['error'] = f'JAX not available: {e}'
    
    return plugins


def get_xla_runtime_info() -> Dict[str, Any]:
    """Get XLA runtime information and configuration."""
    info = {
        'xla_flags': [],
        'compilation_cache': {},
        'precision_config': {},
        'optimization_flags': {},
    }
    
    # Get XLA_FLAGS from environment
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if xla_flags:
        info['xla_flags'] = xla_flags.split()
    
    # Check compilation cache
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '')
    if cache_dir:
        info['compilation_cache'] = {
            'enabled': True,
            'directory': cache_dir,
        }
        # Check if directory exists and is writable
        if os.path.exists(cache_dir):
            info['compilation_cache']['exists'] = True
            info['compilation_cache']['writable'] = os.access(cache_dir, os.W_OK)
        else:
            info['compilation_cache']['exists'] = False
    else:
        info['compilation_cache'] = {'enabled': False}
    
    try:
        with suppress_jax_warnings():
            import jax
            
            # Check precision configuration
            info['precision_config'] = {
                'float64_enabled': jax.config.x64_enabled,
                'default_dtype': 'float64' if jax.config.x64_enabled else 'float32',
            }
            
            # Check if running with debug mode
            debug_mode = os.environ.get('JAX_DEBUG', '').lower() in ['1', 'true', 'yes']
            info['debug_mode'] = debug_mode
            
            # Check JAX_TRACEBACK_FILTERING
            traceback_filtering = os.environ.get('JAX_TRACEBACK_FILTERING', 'on')
            info['traceback_filtering'] = traceback_filtering
            
    except ImportError:
        info['error'] = 'JAX not available'
    
    return info


def check_device_capabilities(backend_info: BackendInfo) -> Dict[str, Any]:
    """Check capabilities of devices in a backend."""
    capabilities = {
        'precision_support': {},
        'memory_info': {},
        'compute_capability': {},
    }
    
    if not backend_info.available or not backend_info.devices:
        return capabilities
    
    try:
        with suppress_jax_warnings():
            import jax
            import jax.numpy as jnp
            
            # Test device with a simple computation
            device = jax.devices(backend_info.platform)[0]
            
            # Test different precisions
            for dtype_name, dtype in [('fp32', jnp.float32), ('fp64', jnp.float64), ('fp16', jnp.float16)]:
                try:
                    # Create array with the dtype
                    test_array = jnp.array([1.0, 2.0, 3.0], dtype=dtype)
                    # Try a simple operation
                    result = test_array * 2.0
                    jax.block_until_ready(result)
                    capabilities['precision_support'][dtype_name] = True
                except:
                    capabilities['precision_support'][dtype_name] = False
            
            # Test memory allocation (small test)
            try:
                test_size = 1024 * 1024  # 1MB
                test_array = jnp.ones((test_size,), dtype=jnp.float32)
                jax.block_until_ready(test_array)
                capabilities['memory_info']['allocation_test'] = 'Passed'
            except Exception as e:
                capabilities['memory_info']['allocation_test'] = f'Failed: {e}'
            
            # Test JIT compilation
            try:
                @jax.jit
                def test_func(x):
                    return x * 2.0 + 1.0
                
                test_input = jnp.array([1.0, 2.0, 3.0])
                result = test_func(test_input)
                jax.block_until_ready(result)
                capabilities['jit_compilation'] = 'Working'
            except Exception as e:
                capabilities['jit_compilation'] = f'Failed: {e}'
    
    except Exception as e:
        capabilities['error'] = f'Capability check failed: {e}'
    
    return capabilities


def print_backend_summary(plugins: Dict[str, BackendInfo]) -> None:
    """Print summary of available backends."""
    print_section("Available JAX Backends for JIT Acceleration")
    
    available_backends = [p for p in plugins.values() if p.available]
    unavailable_backends = [p for p in plugins.values() if not p.available and 'error' not in p]
    
    if available_backends:
        print("\n✓ Active Backends (Ready for JIT Compilation):")
        for backend in sorted(available_backends, key=lambda x: x.priority):
            print(f"\n  {backend.name}:")
            print(f"    Platform: {backend.platform}")
            print(f"    Devices: {backend.device_count}")
            if backend.pjrt_plugin:
                print(f"    PJRT Plugin: {backend.pjrt_plugin}")
            
            # List devices
            for i, device in enumerate(backend.devices[:3]):  # Show first 3 devices
                device_str = f"      [{i}] {device.get('kind', 'Unknown')}"
                if 'memory' in device and device['memory'] != 'Unknown':
                    if isinstance(device['memory'], (int, float)):
                        mem_gb = device['memory'] / (1024**3)
                        device_str += f" ({mem_gb:.2f} GB)"
                print(device_str)
            
            if backend.device_count > 3:
                print(f"      ... and {backend.device_count - 3} more devices")
    
    if unavailable_backends:
        print("\n✗ Unavailable Backends (Install Required Plugins):")
        for backend in unavailable_backends:
            print(f"\n  {backend.name}:")
            if backend.error:
                print(f"    Reason: {backend.error}")
            else:
                print(f"    Install plugin: {backend.pjrt_plugin or 'Check installation'}")


def print_installation_instructions(plugins: Dict[str, BackendInfo], 
                                   system_info: Dict[str, str]) -> None:
    """Print installation instructions for missing backends."""
    print_section("Installation Instructions for Missing Backends")
    
    platform_system = system_info.get('platform', '').lower()
    
    instructions = []
    
    # CPU backend (should always be available if JAX is installed)
    if not plugins.get('cpu', BackendInfo('', '', False)).available:
        instructions.append("• CPU Backend: Install JAX core - pip install jax jaxlib")
    
    # CUDA backend
    cuda_backend = plugins.get('cuda')
    if cuda_backend and not cuda_backend.available:
        if platform_system in ['linux', 'windows']:
            instructions.append("• NVIDIA CUDA: Install CUDA-enabled JAX - pip install \"jax[cuda12]\"")
            instructions.append("  Requires: CUDA 11.8/12.x, cuDNN 8.6+, NCCL 2.16+")
            instructions.append("  Verify: nvidia-smi shows compatible GPU")
        else:
            instructions.append("• NVIDIA CUDA: Only available on Linux/Windows with NVIDIA GPU")
    
    # Metal backend
    metal_backend = plugins.get('metal')
    if metal_backend and not metal_backend.available:
        if platform_system == 'darwin' and system_info.get('apple_silicon') == 'Yes':
            instructions.append("• Apple Metal: Install jax-metal - pip install jax-metal")
            instructions.append("  Note: Only for Apple Silicon (M1/M2/M3) GPUs")
        else:
            instructions.append("• Apple Metal: Requires macOS with Apple Silicon GPU")
    
    # ROCm backend
    rocm_backend = plugins.get('rocm')
    if rocm_backend and not rocm_backend.available:
        if platform_system == 'linux':
            instructions.append("• AMD ROCm: Install ROCm-enabled JAX - pip install jax-rocm")
            instructions.append("  Requires: ROCm 5.7+, compatible AMD GPU (RDNA2/RDNA3)")
        else:
            instructions.append("• AMD ROCm: Only available on Linux with AMD GPU")
    
    # Intel GPU backend
    intel_backend = plugins.get('intel_gpu')
    if intel_backend and not intel_backend.available:
        instructions.append("• Intel GPU: Install Intel Extension for JAX")
        instructions.append("  Available for: Intel Arc, Data Center GPU Max/Series")
        instructions.append("  Installation: See https://github.com/intel/intel-extension-for-jax")
    
    # TPU backend
    tpu_backend = plugins.get('tpu')
    if tpu_backend and not tpu_backend.available:
        instructions.append("• Google TPU: Requires Google Cloud TPU access")
        instructions.append("  Installation: pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html")
    
    if instructions:
        for instruction in instructions:
            print(instruction)
    else:
        print("All desired backends are available. No installation needed.")


def export_results_to_json(system_info: Dict, jax_info: Dict, 
                          plugins: Dict[str, BackendInfo], 
                          xla_info: Dict, output_file: str) -> None:
    """Export detection results to JSON file."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "jax_installation": jax_info,
        "backends": {},
        "xla_runtime": xla_info,
    }
    
    for name, plugin in plugins.items():
        export_data["backends"][name] = {
            "name": plugin.name,
            "platform": plugin.platform,
            "available": plugin.available,
            "device_count": plugin.device_count,
            "devices": plugin.devices,
            "pjrt_plugin": plugin.pjrt_plugin,
            "error": plugin.error,
        }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results exported to: {output_file}")
        file_size = os.path.getsize(output_file) / 1024
        print(f"  File size: {file_size:.2f} KB")
    except Exception as e:
        print(f"✗ Error exporting results: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="JAX Compilation System and Hardware Backend Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Default: show all information
  %(prog)s --quiet                   # Quiet mode: essential info only
  %(prog)s --output results.json     # Export to JSON file
  %(prog)s --check-cuda              # Specifically check CUDA capabilities
  %(prog)s --check-metal             # Specifically check Metal capabilities
        """
    )
    
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet mode: show only essential information"
    )
    
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--check-cuda", action="store_true",
        help="Perform detailed CUDA capability check"
    )
    
    parser.add_argument(
        "--check-metal", action="store_true",
        help="Perform detailed Metal capability check"
    )
    
    parser.add_argument(
        "--check-all", action="store_true",
        help="Perform detailed checks on all available backends"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_section("JAX Compilation System and Hardware Backend Detection")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get system information
    if not args.quiet:
        print_section("System Information")
    system_info = get_system_info()
    if not args.quiet:
        for key, value in system_info.items():
            if key not in ['error']:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check JAX installation
    if not args.quiet:
        print_section("JAX Installation Check")
    
    with suppress_jax_warnings():
        jax_info = check_jax_installation()
    
    if jax_info.get('jax_available', False):
        if not args.quiet:
            print(f"✓ JAX Version: {jax_info.get('jax_version', 'Unknown')}")
            print(f"✓ JAXLIB Version: {jax_info.get('jaxlib_version', 'Unknown')}")
            
            xla_build = jax_info.get('xla_build_info', {})
            if xla_build.get('backend'):
                print(f"✓ XLA Backend: {xla_build['backend']}")
            
            # Show dependencies
            deps = jax_info.get('dependencies', {})
            if any(d.get('available', False) for d in deps.values()):
                print("\n  Additional Dependencies:")
                for dep_name, dep_info in deps.items():
                    if dep_info.get('available', False):
                        version = dep_info.get('version', 'Available')
                        print(f"    • {dep_name}: {version}")
    else:
        print(f"✗ JAX not installed: {jax_info.get('error', 'Unknown error')}")
        print("  Install with: pip install jax jaxlib")
        return
    
    # Detect PJRT plugins
    if not args.quiet:
        print_section("PJRT Runtime Plugins Detection")
    
    with suppress_jax_warnings():
        plugins = detect_pjrt_plugins()
    
    # Print backend summary
    print_backend_summary(plugins)
    
    # Get XLA runtime info
    if not args.quiet:
        print_section("XLA Runtime Configuration")
        xla_info = get_xla_runtime_info()
        
        print(f"X64 (Float64) Support: {xla_info.get('precision_config', {}).get('float64_enabled', 'Unknown')}")
        print(f"Default Dtype: {xla_info.get('precision_config', {}).get('default_dtype', 'Unknown')}")
        
        cache_info = xla_info.get('compilation_cache', {})
        if cache_info.get('enabled', False):
            print(f"Compilation Cache: Enabled ({cache_info.get('directory', 'Unknown')})")
            if cache_info.get('exists', False):
                print(f"  Directory exists: Yes, Writable: {cache_info.get('writable', 'Unknown')}")
        else:
            print(f"Compilation Cache: Disabled (set JAX_COMPILATION_CACHE_DIR to enable)")
        
        xla_flags = xla_info.get('xla_flags', [])
        if xla_flags:
            print(f"XLA_FLAGS: {' '.join(xla_flags)}")
    
    # Perform detailed checks if requested
    if args.check_all or args.check_cuda or args.check_metal:
        print_section("Detailed Device Capability Checks")
        
        backends_to_check = []
        if args.check_all:
            backends_to_check = [p for p in plugins.values() if p.available]
        else:
            if args.check_cuda and plugins.get('cuda', BackendInfo('', '', False)).available:
                backends_to_check.append(plugins['cuda'])
            if args.check_metal and plugins.get('metal', BackendInfo('', '', False)).available:
                backends_to_check.append(plugins['metal'])
        
        for backend in backends_to_check:
            print_subsection(f"Checking {backend.name}")
            capabilities = check_device_capabilities(backend)
            
            # Print precision support
            precision = capabilities.get('precision_support', {})
            if precision:
                print("  Precision Support:")
                for dtype, supported in precision.items():
                    status = "✓" if supported else "✗"
                    print(f"    {status} {dtype.upper()}")
            
            # Print other capabilities
            for key, value in capabilities.items():
                if key not in ['precision_support', 'error'] and value:
                    if isinstance(value, dict):
                        print(f"  {key.replace('_', ' ').title()}:")
                        for subkey, subvalue in value.items():
                            print(f"    • {subkey}: {subvalue}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            
            if 'error' in capabilities:
                print(f"  Error: {capabilities['error']}")
    
    # Print installation instructions for missing backends
    if not args.quiet:
        print_installation_instructions(plugins, system_info)
    
    # Export to JSON if requested
    if args.output:
        with suppress_jax_warnings():
            xla_info = get_xla_runtime_info()
        export_results_to_json(system_info, jax_info, plugins, xla_info, args.output)
    
    # Final summary
    if not args.quiet:
        print_section("Summary")
        
        available_count = sum(1 for p in plugins.values() if p.available and p.name != 'CPU')
        print(f"Available GPU/Accelerator Backends: {available_count}")
        
        if available_count > 0:
            print("\nFor optimal performance:")
            print("1. Use jax.jit() to compile computational graphs")
            print("2. Use appropriate data types (float32 for GPU, float64 for CPU if needed)")
            print("3. Enable compilation cache for repeated computations")
            print("4. Consider batch size optimization for memory-bound operations")
        else:
            print("\nNote: Only CPU backend available.")
            print("For GPU acceleration, install appropriate plugins as shown above.")
        
        print("\nAll output is in English per MCPC coding standards.")
        
        print_section("Detection Complete")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
