#!/usr/bin/env python3
"""
NumPy配置与性能检测工具
检测NumPy版本、底层BLAS/LAPACK实现，并运行基线性能测试
"""

import numpy as np
import sys
import json
import time
import subprocess
import platform
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import argparse

@dataclass
class NumPyConfig:
    """NumPy配置信息"""
    version: str
    installation_path: str
    blas_info: Dict[str, Any] = field(default_factory=dict)
    lapack_info: Dict[str, Any] = field(default_factory=dict)
    build_config: Dict[str, Any] = field(default_factory=dict)
    runtime_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    implementation: str
    parameters: Dict[str, Any]
    execution_times: List[float]
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    std_time: float
    gflops: Optional[float] = None
    notes: str = ""
    
    def __post_init__(self):
        """计算统计信息"""
        self.min_time = min(self.execution_times)
        self.max_time = max(self.execution_times)
        self.avg_time = sum(self.execution_times) / len(self.execution_times)
        self.execution_times_sorted = sorted(self.execution_times)
        n = len(self.execution_times_sorted)
        if n % 2 == 0:
            self.median_time = (self.execution_times_sorted[n//2 - 1] + 
                               self.execution_times_sorted[n//2]) / 2
        else:
            self.median_time = self.execution_times_sorted[n//2]
        
        if len(self.execution_times) > 1:
            self.std_time = np.std(self.execution_times).item()
        else:
            self.std_time = 0.0

class NumPyDiagnostics:
    """NumPy诊断工具"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.config = NumPyConfig(
            version=np.__version__,
            installation_path=np.__file__
        )
    
    def detect_blas_lapack(self):
        """检测BLAS/LAPACK实现"""
        try:
            from numpy.distutils.system_info import get_info
            
            # 获取BLAS信息
            blas_info = get_info('blas_opt') or {}
            self.config.blas_info = self._sanitize_dict(blas_info)
            
            # 获取LAPACK信息
            lapack_info = get_info('lapack_opt') or {}
            self.config.lapack_info = self._sanitize_dict(lapack_info)
            
        except Exception as e:
            print(f"Warning: Could not get BLAS/LAPACK info: {e}")
    
    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """清理字典，确保可序列化"""
        result = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                result[k] = list(v)
            elif isinstance(v, dict):
                result[k] = self._sanitize_dict(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                result[k] = v
            else:
                result[k] = str(v)
        return result
    
    def detect_library_dependencies(self):
        """检测库依赖（动态/静态链接）"""
        if self.verbose:
            print("检测库依赖...")
        
        numpy_path = np.__file__
        system = platform.system()
        
        if system == "Linux":
            self._detect_linux_dependencies(numpy_path)
        elif system == "Darwin":  # macOS
            self._detect_macos_dependencies(numpy_path)
        elif system == "Windows":
            self._detect_windows_dependencies(numpy_path)
        
        # 运行时信息
        self.config.runtime_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "blas_threads": self._detect_blas_threads(),
            "memory": self._get_memory_info()
        }
    
    def _detect_linux_dependencies(self, numpy_path: str):
        """Linux系统检测依赖"""
        try:
            result = subprocess.run(['ldd', numpy_path], 
                                   capture_output=True, text=True, timeout=5)
            
            libraries = {}
            for line in result.stdout.split('\n'):
                if '=>' in line:
                    parts = line.split('=>')
                    lib_name = parts[0].strip()
                    if len(parts) > 1:
                        lib_path = parts[1].strip().split('(')[0].strip()
                        libraries[lib_name] = lib_path
            
            # 过滤数学相关库
            math_libs = {}
            for name, path in libraries.items():
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in 
                      ['blas', 'lapack', 'mkl', 'openblas', 'atlas', 
                       'gfortran', 'gomp', 'pthread', 'fftw']):
                    math_libs[name] = path
                    # 判断是否静态链接
                    if path and path != 'not found':
                        math_libs[f"{name}_static"] = self._is_static_library(path)
            
            self.config.build_config["dynamic_libraries"] = math_libs
            
        except Exception as e:
            if self.verbose:
                print(f"ldd检测失败: {e}")
    
    def _detect_macos_dependencies(self, numpy_path: str):
        """macOS系统检测依赖"""
        try:
            result = subprocess.run(['otool', '-L', numpy_path], 
                                   capture_output=True, text=True, timeout=5)
            
            libraries = {}
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and not line.startswith(numpy_path):
                    # 格式: /path/libname.dylib (compatibility version ...)
                    if '(' in line:
                        lib_path = line.split('(')[0].strip()
                        lib_name = Path(lib_path).name
                        libraries[lib_name] = lib_path
            
            # 过滤数学相关库
            math_libs = {}
            for name, path in libraries.items():
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in 
                      ['blas', 'lapack', 'mkl', 'openblas', 'accelerate', 
                       'veclib', 'gfortran', 'fftw']):
                    math_libs[name] = path
            
            self.config.build_config["dynamic_libraries"] = math_libs
            
        except Exception as e:
            if self.verbose:
                print(f"otool检测失败: {e}")
    
    def _detect_windows_dependencies(self, numpy_path: str):
        """Windows系统检测依赖"""
        # Windows上检测DLL依赖较复杂，这里简化处理
        numpy_dir = Path(numpy_path).parent
        dlls = {}
        
        for dll_file in numpy_dir.glob("*.dll"):
            dll_name = dll_file.name.lower()
            if any(keyword in dll_name for keyword in 
                  ['blas', 'lapack', 'mkl', 'openblas']):
                dlls[dll_file.name] = str(dll_file)
        
        self.config.build_config["dynamic_libraries"] = dlls
    
    def _is_static_library(self, lib_path: str) -> bool:
        """判断是否为静态库"""
        if not lib_path or lib_path == 'not found':
            return False
        
        # 根据文件扩展名判断
        static_extensions = ['.a', '.lib']
        shared_extensions = ['.so', '.dylib', '.dll']
        
        for ext in static_extensions:
            if lib_path.endswith(ext):
                return True
        
        for ext in shared_extensions:
            if lib_path.endswith(ext):
                return False
        
        return False
    
    def _detect_blas_threads(self) -> Dict[str, Any]:
        """检测BLAS线程配置"""
        threads_info = {}
        
        # 检查环境变量
        env_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                threads_info[var] = value
        
        # 通过性能测试推断线程数
        threads_info["inferred_threads"] = self._infer_blas_threads()
        
        return threads_info
    
    def _infer_blas_threads(self) -> Optional[int]:
        """通过性能测试推断BLAS线程数"""
        try:
            # 运行一个小型矩阵乘法测试
            n = 500
            A = np.random.randn(n, n).astype(np.float64)
            B = np.random.randn(n, n).astype(np.float64)
            
            # 首次运行（可能包含JIT编译时间）
            start = time.perf_counter()
            C = A @ B
            elapsed = time.perf_counter() - start
            
            # 如果速度非常快，可能是多线程
            gflops = 2 * n**3 / elapsed / 1e9
            cpu_count = os.cpu_count() or 1
            
            # 经验法则：单线程通常 < 5 GFLOPS，多线程可能 > 20 GFLOPS
            if gflops > 10 * cpu_count:
                return cpu_count  # 推测使用所有核心
            elif gflops > 5:
                return 2  # 推测使用2-4个线程
            else:
                return 1  # 推测单线程
            
        except:
            return None
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory_info = {}
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_percent": memory.percent
            }
        except ImportError:
            memory_info["psutil_not_available"] = True
        
        return memory_info
    
    def run_benchmarks(self, warmup_iterations: int = 2, 
                      test_iterations: int = 5) -> List[BenchmarkResult]:
        """运行基准测试"""
        results = []
        
        print("\n" + "="*80)
        print("NumPy性能基准测试")
        print("="*80)
        
        # 1. 矩阵乘法 (BLAS Level 3)
        print("\n1. 矩阵乘法测试 (BLAS GEMM)...")
        matmul_results = self._benchmark_matmul(warmup_iterations, test_iterations)
        results.extend(matmul_results)
        
        # 2. 奇异值分解 (LAPACK)
        print("\n2. 奇异值分解测试 (LAPACK SVD)...")
        svd_results = self._benchmark_svd(warmup_iterations, test_iterations)
        results.extend(svd_results)
        
        # 3. FFT测试
        print("\n3. FFT测试...")
        fft_results = self._benchmark_fft(warmup_iterations, test_iterations)
        results.extend(fft_results)
        
        # 4. 向量运算测试
        print("\n4. 向量运算测试...")
        vector_results = self._benchmark_vector_ops(warmup_iterations, test_iterations)
        results.extend(vector_results)
        
        return results
    
    def _benchmark_matmul(self, warmup: int, iterations: int) -> List[BenchmarkResult]:
        """基准测试：矩阵乘法"""
        results = []
        
        # 测试不同规模
        sizes = [500, 1000, 2000]
        
        for n in sizes:
            print(f"  测试 {n}x{n} 矩阵...")
            
            # 准备数据
            A = np.random.randn(n, n).astype(np.float64)
            B = np.random.randn(n, n).astype(np.float64)
            
            # 预热
            for _ in range(warmup):
                C = A @ B
            
            # 正式测试
            execution_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                C = A @ B
                end = time.perf_counter()
                execution_times.append(end - start)
                # 确保结果被使用，避免优化器消除计算
                np.sum(C)  # 使用结果
            
            # 计算GFLOPS (矩阵乘法浮点操作数: 2*n^3)
            avg_time = sum(execution_times) / len(execution_times)
            gflops = 2 * n**3 / avg_time / 1e9
            
            result = BenchmarkResult(
                test_name="Matrix Multiplication",
                implementation="NumPy (BLAS GEMM)",
                parameters={"size": n, "dtype": "float64"},
                execution_times=execution_times,
                min_time=0, max_time=0, avg_time=0, median_time=0, std_time=0,  # 将由__post_init__计算
                gflops=gflops,
                notes=f"GEMM operation, size={n}x{n}"
            )
            
            # 手动触发__post_init__
            result.__post_init__()
            results.append(result)
            
            print(f"    平均时间: {avg_time:.4f}s, GFLOPS: {gflops:.2f}")
        
        return results
    
    def _benchmark_svd(self, warmup: int, iterations: int) -> List[BenchmarkResult]:
        """基准测试：奇异值分解"""
        results = []
        
        sizes = [200, 500, 1000]
        
        for n in sizes:
            print(f"  测试 {n}x{n} 矩阵SVD...")
            
            A = np.random.randn(n, n).astype(np.float64)
            
            # 预热
            for _ in range(warmup):
                U, s, Vh = np.linalg.svd(A, full_matrices=False)
            
            # 正式测试
            execution_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                U, s, Vh = np.linalg.svd(A, full_matrices=False)
                end = time.perf_counter()
                execution_times.append(end - start)
                np.sum(s)  # 使用结果
            
            avg_time = sum(execution_times) / len(execution_times)
            
            result = BenchmarkResult(
                test_name="Singular Value Decomposition",
                implementation="NumPy (LAPACK GESDD)",
                parameters={"size": n, "dtype": "float64", "full_matrices": False},
                execution_times=execution_times,
                min_time=0, max_time=0, avg_time=0, median_time=0, std_time=0,
                notes=f"SVD with reduced form, size={n}x{n}"
            )
            
            result.__post_init__()
            results.append(result)
            
            print(f"    平均时间: {avg_time:.4f}s")
        
        return results
    
    def _benchmark_fft(self, warmup: int, iterations: int) -> List[BenchmarkResult]:
        """基准测试：FFT"""
        results = []
        
        sizes = [100000, 1000000, 10000000]
        
        for n in sizes:
            print(f"  测试 {n:,} 点FFT...")
            
            x = np.random.randn(n).astype(np.complex128)
            
            # 预热
            for _ in range(warmup):
                y = np.fft.fft(x)
            
            # 正式测试
            execution_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                y = np.fft.fft(x)
                end = time.perf_counter()
                execution_times.append(end - start)
                np.sum(y.real)  # 使用结果
            
            avg_time = sum(execution_times) / len(execution_times)
            
            result = BenchmarkResult(
                test_name="Fast Fourier Transform",
                implementation="NumPy FFT",
                parameters={"size": n, "dtype": "complex128"},
                execution_times=execution_times,
                min_time=0, max_time=0, avg_time=0, median_time=0, std_time=0,
                notes=f"Complex FFT, {n:,} points"
            )
            
            result.__post_init__()
            results.append(result)
            
            print(f"    平均时间: {avg_time:.4f}s")
        
        return results
    
    def _benchmark_vector_ops(self, warmup: int, iterations: int) -> List[BenchmarkResult]:
        """基准测试：向量运算"""
        results = []
        
        sizes = [1000000, 10000000]
        
        for n in sizes:
            print(f"  测试 {n:,} 元素向量运算...")
            
            x = np.random.randn(n).astype(np.float64)
            y = np.random.randn(n).astype(np.float64)
            
            # 测试不同的向量运算
            ops = [
                ("Vector Add", lambda a, b: a + b),
                ("Vector Multiply", lambda a, b: a * b),
                ("Vector Dot", lambda a, b: np.dot(a, b)),
                ("Exponential", lambda a, b: np.exp(a)),
                ("Sine", lambda a, b: np.sin(a))
            ]
            
            for op_name, op_func in ops:
                # 预热
                for _ in range(warmup):
                    if op_name == "Vector Dot":
                        result = op_func(x, y)
                    else:
                        result = op_func(x, y) if 'b' in op_func.__code__.co_varnames else op_func(x)
                
                # 正式测试
                execution_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    if op_name == "Vector Dot":
                        result = op_func(x, y)
                    else:
                        result = op_func(x, y) if 'b' in op_func.__code__.co_varnames else op_func(x)
                    end = time.perf_counter()
                    execution_times.append(end - start)
                    float(result) if isinstance(result, (int, float, np.number)) else np.sum(result)
                
                avg_time = sum(execution_times) / len(execution_times)
                
                result_obj = BenchmarkResult(
                    test_name=f"Vector Operation - {op_name}",
                    implementation="NumPy",
                    parameters={"size": n, "dtype": "float64", "operation": op_name},
                    execution_times=execution_times,
                    min_time=0, max_time=0, avg_time=0, median_time=0, std_time=0,
                    notes=f"{op_name} on {n:,} elements"
                )
                
                result_obj.__post_init__()
                results.append(result_obj)
                
                print(f"    {op_name}: {avg_time:.4f}s")
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult], 
                       output_format: str = "both") -> Dict[str, Any]:
        """生成报告"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_config": asdict(self.config),
            "benchmark_results": [],
            "summary": {}
        }
        
        # 添加基准测试结果
        for result in results:
            result_dict = {
                "test_name": result.test_name,
                "implementation": result.implementation,
                "parameters": result.parameters,
                "statistics": {
                    "execution_times": result.execution_times,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "avg_time": result.avg_time,
                    "median_time": result.median_time,
                    "std_time": result.std_time,
                    "gflops": result.gflops
                },
                "notes": result.notes
            }
            report["benchmark_results"].append(result_dict)
        
        # 生成摘要
        summary = self._generate_summary(results)
        report["summary"] = summary
        
        return report
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """生成性能摘要"""
        summary = {
            "total_tests": len(results),
            "performance_by_test": {},
            "blas_implementation": self._infer_blas_implementation(),
            "recommendations": []
        }
        
        # 按测试类型分组
        from collections import defaultdict
        by_test = defaultdict(list)
        
        for result in results:
            by_test[result.test_name].append(result)
        
        # 分析每个测试
        for test_name, test_results in by_test.items():
            if test_results:
                # 找到最佳（最快）结果
                best_result = min(test_results, key=lambda x: x.avg_time)
                
                summary["performance_by_test"][test_name] = {
                    "best_avg_time": best_result.avg_time,
                    "parameters": best_result.parameters,
                    "gflops": best_result.gflops
                }
        
        # 生成建议
        recommendations = self._generate_recommendations()
        summary["recommendations"] = recommendations
        
        return summary
    
    def _infer_blas_implementation(self) -> str:
        """推断BLAS实现"""
        blas_info = self.config.blas_info
        
        # 检查库名
        libraries = blas_info.get("libraries", [])
        for lib in libraries:
            lib_lower = str(lib).lower()
            if "mkl" in lib_lower:
                return "Intel MKL"
            elif "openblas" in lib_lower:
                return "OpenBLAS"
            elif "atlas" in lib_lower:
                return "ATLAS"
            elif "blis" in lib_lower:
                return "BLIS"
            elif "accelerate" in lib_lower:
                return "Apple Accelerate (macOS)"
        
        # 检查额外信息
        extra_info = str(blas_info).lower()
        if "mkl" in extra_info:
            return "Intel MKL"
        elif "openblas" in extra_info:
            return "OpenBLAS"
        
        return "Unknown (可能是Netlib参考实现)"
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # BLAS实现建议
        blas_impl = self._infer_blas_implementation()
        if "MKL" not in blas_impl and "OpenBLAS" not in blas_impl:
            recommendations.append(
                "考虑安装优化的BLAS实现：Intel MKL或OpenBLAS以获得更好的性能"
            )
        
        # 线程配置建议
        env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
        if not any(var in os.environ for var in env_vars):
            cpu_count = os.cpu_count() or 4
            recommendations.append(
                f"建议设置环境变量，例如：export OMP_NUM_THREADS={cpu_count}"
            )
        
        # 内存建议
        if "memory" in self.config.runtime_info:
            memory_info = self.config.runtime_info["memory"]
            if "available_gb" in memory_info and memory_info["available_gb"] < 2.0:
                recommendations.append(
                    "可用内存较低，考虑减少测试规模或关闭其他应用"
                )
        
        return recommendations
    
    def print_human_readable_report(self, report: Dict[str, Any]):
        """打印人类可读的报告"""
        print("\n" + "="*80)
        print("NumPy诊断报告")
        print("="*80)
        
        # 系统信息
        print(f"\n系统信息:")
        print(f"  平台: {report['numpy_config']['runtime_info'].get('platform', 'Unknown')}")
        print(f"  CPU核心数: {report['numpy_config']['runtime_info'].get('cpu_count', 'Unknown')}")
        
        # NumPy信息
        print(f"\nNumPy信息:")
        print(f"  版本: {report['numpy_config']['version']}")
        print(f"  安装路径: {report['numpy_config']['installation_path']}")
        
        # BLAS/LAPACK信息
        print(f"\nBLAS/LAPACK实现:")
        blas_impl = report['summary'].get('blas_implementation', 'Unknown')
        print(f"  推断实现: {blas_impl}")
        
        if report['numpy_config']['blas_info']:
            print(f"  BLAS库: {report['numpy_config']['blas_info'].get('libraries', [])}")
        
        # 动态库信息
        if 'dynamic_libraries' in report['numpy_config']['build_config']:
            libs = report['numpy_config']['build_config']['dynamic_libraries']
            if libs:
                print(f"\n动态库依赖:")
                for name, path in libs.items():
                    if not name.endswith('_static'):
                        print(f"  {name}: {path}")
        
        # 线程配置
        print(f"\n线程配置:")
        threads_info = report['numpy_config']['runtime_info'].get('blas_threads', {})
        for key, value in threads_info.items():
            if key != 'inferred_threads':
                print(f"  {key}: {value}")
        
        if 'inferred_threads' in threads_info and threads_info['inferred_threads']:
            print(f"  推断线程数: {threads_info['inferred_threads']}")
        
        # 性能摘要
        print(f"\n性能摘要:")
        perf_by_test = report['summary'].get('performance_by_test', {})
        for test_name, perf_info in perf_by_test.items():
            print(f"  {test_name}:")
            print(f"    最佳平均时间: {perf_info['best_avg_time']:.4f}s")
            if perf_info.get('gflops'):
                print(f"    GFLOPS: {perf_info['gflops']:.2f}")
        
        # 建议
        recommendations = report['summary'].get('recommendations', [])
        if recommendations:
            print(f"\n优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NumPy配置与性能检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 运行完整检测
  %(prog)s --output report.json     # 保存JSON报告
  %(prog)s --no-benchmark           # 只检测配置，不运行性能测试
  %(prog)s --verbose                # 显示详细调试信息
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        help="输出JSON报告文件的路径"
    )
    
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="跳过性能基准测试"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="预热迭代次数 (默认: 2)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="每个测试的迭代次数 (默认: 5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    print("NumPy诊断工具")
    print("="*80)
    
    # 创建诊断器
    diag = NumPyDiagnostics(verbose=args.verbose)
    
    # 检测配置
    print("检测NumPy配置...")
    diag.detect_blas_lapack()
    diag.detect_library_dependencies()
    
    # 运行基准测试
    results = []
    if not args.no_benchmark:
        results = diag.run_benchmarks(
            warmup_iterations=args.warmup,
            test_iterations=args.iterations
        )
    
    # 生成报告
    report = diag.generate_report(results)
    
    # 打印人类可读报告
    diag.print_human_readable_report(report)
    
    # 保存JSON报告
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON报告已保存到: {output_path.absolute()}")
    
    print("\n完成！")

if __name__ == "__main__":
    main()
