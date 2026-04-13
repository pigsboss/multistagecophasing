"""
设备管理模块 - 硬件资源检测与管理

提供 CPU 和 GPU 设备的自动检测、初始化和管理功能。
遵循 MCPC 编码标准：UTF-8 编码，运行时输出使用英文。
"""
from __future__ import annotations

import os
import sys
import platform
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .base import DeviceType, DeviceInterface


@dataclass
class LibraryInfo:
    """库信息数据类"""
    name: str
    version: str
    available: bool
    backend: str = ""  # 例如：'openblas', 'mkl', 'cuda', 'rocm'
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class HardwareSpec:
    """硬件规格数据类"""
    name: str
    cores: int
    memory_gb: float
    clock_speed_ghz: float = 0.0
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []


class BaseDevice:
    """设备基类，提供通用功能"""
    
    def __init__(self, device_type: DeviceType, device_name: str = ""):
        self._device_type = device_type
        self._device_name = device_name or f"Unknown {device_type.value}"
        self._initialized = False
        self._available = False
        self._library_info: Dict[str, LibraryInfo] = {}
        self._hardware_spec: Optional[HardwareSpec] = None
        
    @property
    def device_type(self) -> DeviceType:
        return self._device_type
    
    @property
    def device_name(self) -> str:
        return self._device_name
    
    def is_available(self) -> bool:
        return self._available
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "available": self._available,
            "initialized": self._initialized,
            "libraries": {name: {
                "version": lib.version,
                "backend": lib.backend,
                "available": lib.available
            } for name, lib in self._library_info.items()},
            "hardware_spec": self._hardware_spec.__dict__ if self._hardware_spec else None
        }
    
    def initialize(self) -> bool:
        """初始化设备 - 由子类实现"""
        raise NotImplementedError
    
    def cleanup(self) -> None:
        """清理设备资源"""
        self._initialized = False
    
    def execute_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行计算内核 - 由具体设备实现"""
        raise NotImplementedError(f"Kernel execution not implemented for {self.__class__.__name__}")
    
    def _check_library(self, library_name: str, import_name: Optional[str] = None) -> LibraryInfo:
        """
        检查库是否可用
        
        Args:
            library_name: 库的显示名称
            import_name: 导入名称（如果不同）
            
        Returns:
            LibraryInfo: 库信息
        """
        import_name = import_name or library_name.lower()
        
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            
            # 尝试执行简单操作验证功能
            if import_name == 'numpy':
                # 测试基本的BLAS操作
                a = np.random.rand(10, 10)
                b = np.random.rand(10, 10)
                c = a @ b  # 矩阵乘法测试BLAS
                
                # 检测numpy使用的BLAS后端
                backend = self._detect_numpy_blas_backend()
                details = {"blas_backend": backend}
                
                return LibraryInfo(
                    name=library_name,
                    version=version,
                    available=True,
                    backend=backend,
                    details=details
                )
            else:
                return LibraryInfo(
                    name=library_name,
                    version=version,
                    available=True,
                    backend=""
                )
                
        except ImportError:
            return LibraryInfo(
                name=library_name,
                version="not installed",
                available=False,
                backend=""
            )
        except Exception as e:
            return LibraryInfo(
                name=library_name,
                version=f"error: {str(e)}",
                available=False,
                backend=""
            )
    
    def _detect_numpy_blas_backend(self) -> str:
        """检测numpy使用的BLAS后端"""
        try:
            # 方法1: 检查numpy配置
            import numpy as np
            config = np.__config__
            
            if hasattr(config, 'show'):
                import io
                import re
                
                output = io.StringIO()
                config.show(output)
                config_str = output.getvalue()
                
                # 检查常见的BLAS后端
                if 'openblas' in config_str.lower():
                    return 'openblas'
                elif 'mkl' in config_str.lower():
                    return 'mkl'
                elif 'blis' in config_str.lower():
                    return 'blis'
                elif 'atlas' in config_str.lower():
                    return 'atlas'
                    
            # 方法2: 检查numpy的内部属性
            if hasattr(np.core, '_multiarray_umath'):
                # 尝试获取更多信息
                try:
                    from numpy.core._multiarray_umath import __cpu_features__
                    if 'AVX512F' in __cpu_features__:
                        return 'optimized'
                except:
                    pass
                    
        except Exception:
            pass
            
        return 'unknown'


class CPUDevice(BaseDevice, DeviceInterface):
    """CPU设备实现类"""
    
    def __init__(self, device_id: int = 0):
        super().__init__(DeviceType.CPU, f"CPU-{device_id}")
        self.device_id = device_id
        self._detect_hardware()
        
    def _detect_hardware(self) -> None:
        """检测CPU硬件信息"""
        try:
            import multiprocessing
            import psutil
            
            # 获取CPU核心数
            physical_cores = psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
            logical_cores = psutil.cpu_count(logical=True) or multiprocessing.cpu_count()
            
            # 获取CPU型号
            cpu_name = "Unknown CPU"
            if platform.system() == "Windows":
                try:
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                       r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                    winreg.CloseKey(key)
                except:
                    cpu_name = platform.processor()
            elif platform.system() == "Linux":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if 'model name' in line:
                                cpu_name = line.split(':')[1].strip()
                                break
                except:
                    cpu_name = platform.processor()
            elif platform.system() == "Darwin":  # macOS
                try:
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                          capture_output=True, text=True)
                    cpu_name = result.stdout.strip()
                except:
                    cpu_name = platform.processor()
            
            # 获取内存信息
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # 获取CPU频率（如果可用）
            clock_speed = 0.0
            try:
                if hasattr(psutil, 'cpu_freq'):
                    freq = psutil.cpu_freq()
                    if freq:
                        clock_speed = freq.current / 1000  # 转换为GHz
            except:
                pass
            
            # 检测CPU特性
            features = []
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                
                # 添加指令集扩展
                for feature in ['avx', 'avx2', 'avx512', 'fma', 'sse', 'sse2', 'sse3', 'sse4', 'neon']:
                    if info.get(feature, False):
                        features.append(feature.upper())
                        
                # 如果有更详细的品牌信息，使用它
                if 'brand_raw' in info:
                    cpu_name = info['brand_raw']
                    
            except ImportError:
                # cpuinfo不可用，跳过特性检测
                pass
            
            self._hardware_spec = HardwareSpec(
                name=cpu_name,
                cores=physical_cores,
                memory_gb=memory_gb,
                clock_speed_ghz=clock_speed,
                features=features
            )
            
        except Exception as e:
            # 回退到基本检测
            import multiprocessing
            cores = multiprocessing.cpu_count()
            
            self._hardware_spec = HardwareSpec(
                name=platform.processor() or "Unknown CPU",
                cores=cores,
                memory_gb=0.0,  # 未知
                features=[]
            )
    
    def initialize(self) -> bool:
        """初始化CPU设备并检查必要的库"""
        try:
            # 检查NumPy（包含BLAS/LAPACK）
            numpy_info = self._check_library("NumPy")
            self._library_info["numpy"] = numpy_info
            
            # 检查SciPy（包含更高级的数值计算库）
            scipy_info = self._check_library("SciPy", "scipy")
            self._library_info["scipy"] = scipy_info
            
            # 检查Numba（JIT编译器，可选）
            try:
                numba_info = self._check_library("Numba", "numba")
                self._library_info["numba"] = numba_info
            except:
                pass  # Numba是可选的
            
            # 确定设备是否可用
            self._available = numpy_info.available
            
            if self._available:
                self._initialized = True
                print(f"[INFO] CPU device initialized: {self.device_name}")
                print(f"       Cores: {self._hardware_spec.cores}")
                print(f"       BLAS backend: {numpy_info.backend}")
                
                # 如果有SciPy，显示信息
                if scipy_info.available:
                    print(f"       SciPy available: {scipy_info.version}")
                    
            return self._available
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize CPU device: {str(e)}")
            self._available = False
            return False
    
    def execute_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行CPU计算内核"""
        # 这里实现具体的CPU计算逻辑
        # 例如：调用BLAS函数、执行NumPy操作等
        
        if kernel_name == "matrix_multiply":
            # 示例：矩阵乘法
            if len(args) >= 2:
                a, b = args[0], args[1]
                return a @ b
            else:
                raise ValueError("matrix_multiply requires two matrices")
        
        elif kernel_name == "vector_dot":
            # 示例：向量点积
            if len(args) >= 2:
                a, b = args[0], args[1]
                return np.dot(a, b)
            else:
                raise ValueError("vector_dot requires two vectors")
        
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")


class GPUDevice(BaseDevice, DeviceInterface):
    """GPU设备实现类"""
    
    def __init__(self, device_id: int = 0, backend: str = "auto"):
        """
        初始化GPU设备
        
        Args:
            device_id: GPU设备ID
            backend: 计算后端 ('cuda', 'opencl', 'rocm', 'auto')
        """
        super().__init__(DeviceType.GPU, f"GPU-{device_id}")
        self.device_id = device_id
        self.backend = backend
        self._gpu_info: Dict[str, Any] = {}
        self._detect_hardware()
        
    def _detect_hardware(self) -> None:
        """检测GPU硬件信息"""
        self._gpu_info = {
            "name": "Unknown GPU",
            "memory_gb": 0.0,
            "compute_capability": "",
            "driver_version": "",
            "backend": self.backend
        }
        
        # 尝试自动检测GPU
        if self.backend in ["auto", "cuda"]:
            if self._detect_cuda():
                return
        
        if self.backend in ["auto", "rocm"]:
            if self._detect_rocm():
                return
        
        if self.backend in ["auto", "opencl"]:
            if self._detect_opencl():
                return
    
    def _detect_cuda(self) -> bool:
        """检测NVIDIA CUDA GPU"""
        try:
            # 方法1: 使用pycuda
            try:
                import pycuda.driver as cuda
                cuda.init()
                
                device = cuda.Device(self.device_id)
                self._gpu_info.update({
                    "name": device.name(),
                    "memory_gb": device.total_memory() / (1024**3),
                    "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                    "driver_version": cuda.get_version(),
                    "backend": "cuda"
                })
                
                self._device_name = f"NVIDIA {device.name()} (CUDA)"
                return True
                
            except ImportError:
                pass
            
            # 方法2: 使用numba.cuda
            try:
                from numba import cuda
                
                gpu = cuda.get_current_device()
                self._gpu_info.update({
                    "name": gpu.name.decode('utf-8') if isinstance(gpu.name, bytes) else gpu.name,
                    "memory_gb": gpu.get_memory_info().total / (1024**3),
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                    "backend": "cuda"
                })
                
                self._device_name = f"NVIDIA {gpu.name} (CUDA)"
                return True
                
            except ImportError:
                pass
            
            # 方法3: 使用系统命令（Linux）
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                           '--format=csv,noheader'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        info = result.stdout.strip().split(',')
                        if len(info) >= 2:
                            self._gpu_info.update({
                                "name": info[0].strip(),
                                "memory_gb": float(info[1].strip().split()[0]),
                                "driver_version": info[2].strip() if len(info) > 2 else "",
                                "backend": "cuda"
                            })
                            self._device_name = f"NVIDIA {info[0].strip()} (CUDA)"
                            return True
                except:
                    pass
            
            return False
            
        except Exception as e:
            return False
    
    def _detect_opencl(self) -> bool:
        """检测OpenCL GPU"""
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if self.device_id < len(devices):
                    device = devices[self.device_id]
                    
                    self._gpu_info.update({
                        "name": device.name,
                        "memory_gb": device.global_mem_size / (1024**3),
                        "compute_capability": device.version,
                        "driver_version": platform.version,
                        "backend": "opencl"
                    })
                    
                    self._device_name = f"{device.name} (OpenCL)"
                    return True
            
            return False
            
        except ImportError:
            return False
        except Exception as e:
            return False
    
    def _detect_rocm(self) -> bool:
        """检测AMD ROCm GPU"""
        try:
            # ROCm可以通过pyopencl或专门的ROCm库检测
            # 首先尝试通过OpenCL检测AMD GPU
            if self._detect_opencl():
                if "amd" in self._gpu_info["name"].lower() or "radeon" in self._gpu_info["name"].lower():
                    self._gpu_info["backend"] = "rocm"
                    self._device_name = f"{self._gpu_info['name']} (ROCm)"
                    return True
            
            # 检查ROCm系统文件（Linux）
            if platform.system() == "Linux":
                rocm_paths = ['/opt/rocm', '/usr/local/rocm']
                for path in rocm_paths:
                    if os.path.exists(path):
                        self._gpu_info.update({
                            "name": "AMD ROCm GPU",
                            "backend": "rocm"
                        })
                        self._device_name = "AMD ROCm GPU"
                        return True
            
            return False
            
        except Exception as e:
            return False
    
    def initialize(self) -> bool:
        """初始化GPU设备并检查必要的库"""
        try:
            # 根据后端检查相应的库
            if self._gpu_info["backend"] == "cuda":
                # 检查CUDA库
                cuda_available = False
                
                # 检查pycuda
                try:
                    import pycuda
                    cuda_info = LibraryInfo(
                        name="PyCUDA",
                        version=getattr(pycuda, '__version__', 'unknown'),
                        available=True,
                        backend="cuda"
                    )
                    self._library_info["pycuda"] = cuda_info
                    cuda_available = True
                except ImportError:
                    pass
                
                # 检查numba.cuda
                try:
                    from numba import cuda
                    numba_info = LibraryInfo(
                        name="Numba-CUDA",
                        version=getattr(cuda, '__version__', 'unknown'),
                        available=True,
                        backend="cuda"
                    )
                    self._library_info["numba_cuda"] = numba_info
                    cuda_available = True
                except ImportError:
                    pass
                
                if not cuda_available:
                    print(f"[WARNING] CUDA device detected but no CUDA libraries found")
                    self._available = False
                    return False
                    
            elif self._gpu_info["backend"] == "opencl":
                # 检查OpenCL库
                try:
                    import pyopencl
                    opencl_info = LibraryInfo(
                        name="PyOpenCL",
                        version=getattr(pyopencl, '__version__', 'unknown'),
                        available=True,
                        backend="opencl"
                    )
                    self._library_info["pyopencl"] = opencl_info
                except ImportError:
                    print(f"[WARNING] OpenCL device detected but PyOpenCL not found")
                    self._available = False
                    return False
                    
            elif self._gpu_info["backend"] == "rocm":
                # 检查ROCm库
                rocm_available = False
                
                # 尝试通过OpenCL检测
                try:
                    import pyopencl
                    opencl_info = LibraryInfo(
                        name="PyOpenCL",
                        version=getattr(pyopencl, '__version__', 'unknown'),
                        available=True,
                        backend="rocm"
                    )
                    self._library_info["pyopencl"] = opencl_info
                    rocm_available = True
                except ImportError:
                    pass
                
                if not rocm_available:
                    print(f"[WARNING] ROCm device detected but no compatible libraries found")
                    self._available = False
                    return False
            
            else:
                print(f"[WARNING] Unknown GPU backend: {self._gpu_info['backend']}")
                self._available = False
                return False
            
            # 创建硬件规格
            self._hardware_spec = HardwareSpec(
                name=self._gpu_info["name"],
                cores=0,  # GPU核心数不同架构定义不同
                memory_gb=self._gpu_info.get("memory_gb", 0.0),
                features=[self._gpu_info["backend"]]
            )
            
            self._available = True
            self._initialized = True
            
            print(f"[INFO] GPU device initialized: {self.device_name}")
            print(f"       Backend: {self._gpu_info['backend']}")
            print(f"       Memory: {self._gpu_info.get('memory_gb', 0.0):.1f} GB")
            
            if "compute_capability" in self._gpu_info and self._gpu_info["compute_capability"]:
                print(f"       Compute capability: {self._gpu_info['compute_capability']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize GPU device: {str(e)}")
            self._available = False
            return False
    
    def execute_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行GPU计算内核"""
        backend = self._gpu_info.get("backend", "unknown")
        
        if backend == "cuda":
            return self._execute_cuda_kernel(kernel_name, *args, **kwargs)
        elif backend == "opencl" or backend == "rocm":
            return self._execute_opencl_kernel(kernel_name, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported GPU backend: {backend}")
    
    def _execute_cuda_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行CUDA内核"""
        # 这里实现具体的CUDA计算逻辑
        # 注意：这只是一个示例，实际实现需要具体的CUDA代码
        
        if kernel_name == "test_availability":
            return {"status": "available", "backend": "cuda"}
        
        # 实际应用中，这里会调用pycuda或numba.cuda的具体函数
        raise NotImplementedError("CUDA kernel execution not yet implemented")
    
    def _execute_opencl_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """执行OpenCL内核"""
        # 这里实现具体的OpenCL计算逻辑
        
        if kernel_name == "test_availability":
            return {"status": "available", "backend": "opencl"}
        
        # 实际应用中，这里会调用pyopencl的具体函数
        raise NotImplementedError("OpenCL kernel execution not yet implemented")


def detect_available_devices() -> List[DeviceInterface]:
    """
    检测所有可用设备
    
    Returns:
        List[DeviceInterface]: 可用设备列表
    """
    devices: List[DeviceInterface] = []
    
    print("[INFO] Scanning for available devices...")
    
    # 1. 检测CPU设备
    try:
        cpu_device = CPUDevice(device_id=0)
        if cpu_device.initialize():
            devices.append(cpu_device)
            print(f"[INFO] Found CPU: {cpu_device.device_name}")
    except Exception as e:
        print(f"[WARNING] Failed to detect CPU: {str(e)}")
    
    # 2. 检测GPU设备
    gpu_backends = ["cuda", "opencl", "rocm"]
    gpu_found = False
    
    for backend in gpu_backends:
        if gpu_found:
            break
            
        try:
            # 尝试检测第一个GPU
            gpu_device = GPUDevice(device_id=0, backend=backend)
            if gpu_device.initialize():
                devices.append(gpu_device)
                print(f"[INFO] Found GPU: {gpu_device.device_name}")
                gpu_found = True
                
                # 尝试检测更多GPU（如果有）
                for i in range(1, 4):  # 最多检测4个GPU
                    try:
                        additional_gpu = GPUDevice(device_id=i, backend=backend)
                        if additional_gpu.initialize():
                            devices.append(additional_gpu)
                            print(f"[INFO] Found additional GPU: {additional_gpu.device_name}")
                    except:
                        break
                        
        except Exception as e:
            # 这个后端失败，尝试下一个
            continue
    
    if not gpu_found:
        print("[INFO] No GPU devices found")
    
    print(f"[INFO] Device scan completed. Found {len(devices)} device(s)")
    return devices


def get_device_by_type(device_type: DeviceType, device_id: int = 0) -> Optional[DeviceInterface]:
    """
    根据类型获取特定设备
    
    Args:
        device_type: 设备类型
        device_id: 设备ID
        
    Returns:
        Optional[DeviceInterface]: 设备实例，如果找不到则返回None
    """
    devices = detect_available_devices()
    
    for device in devices:
        if device.device_type == device_type:
            # 简单的设备ID匹配逻辑
            if (isinstance(device, CPUDevice) and device.device_id == device_id) or \
               (isinstance(device, GPUDevice) and device.device_id == device_id):
                return device
    
    return None


def print_device_summary(devices: Optional[List[DeviceInterface]] = None) -> None:
    """
    打印设备摘要信息
    
    Args:
        devices: 设备列表，如果为None则自动检测
    """
    if devices is None:
        devices = detect_available_devices()
    
    print("\n" + "="*60)
    print("DEVICE SUMMARY")
    print("="*60)
    
    if not devices:
        print("No devices found")
        return
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i+1}:")
        print(f"  Type: {device.device_type.value}")
        print(f"  Name: {device.device_name}")
        print(f"  Available: {device.is_available()}")
        
        metrics = device.get_performance_metrics()
        if metrics.get("hardware_spec"):
            spec = metrics["hardware_spec"]
            print(f"  Cores: {spec.get('cores', 'N/A')}")
            print(f"  Memory: {spec.get('memory_gb', 0):.1f} GB")
            
            if device.device_type == DeviceType.CPU and spec.get('clock_speed_ghz'):
                print(f"  Clock: {spec['clock_speed_ghz']:.2f} GHz")
            
            if spec.get('features'):
                print(f"  Features: {', '.join(spec['features'])}")
        
        if metrics.get("libraries"):
            print(f"  Libraries:")
            for lib_name, lib_info in metrics["libraries"].items():
                if lib_info["available"]:
                    print(f"    - {lib_name}: {lib_info['version']} ({lib_info.get('backend', 'unknown')})")
    
    print("\n" + "="*60)


# 导出公共接口
__all__ = [
    'CPUDevice',
    'GPUDevice',
    'detect_available_devices',
    'get_device_by_type',
    'print_device_summary',
    'DeviceInterface',
    'DeviceType',
]
