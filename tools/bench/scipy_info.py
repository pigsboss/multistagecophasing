#!/usr/bin/env python3
"""
SciPy Library Information and Baseline Performance Test
Displays underlying math library implementations and runs benchmark tests.

Note: All output is in English per MCPC coding standards.
"""

import sys
import platform
import time
import numpy as np
import scipy
from scipy import linalg, sparse, fft, optimize
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def get_system_info():
    """Collect and display system information."""
    print_section("SYSTEM INFORMATION")
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")
    
    # Get memory information if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("Note: Install 'psutil' for detailed memory information")

def get_scipy_info():
    """Display SciPy configuration and library information."""
    print_section("SCIPY CONFIGURATION")
    
    print(f"SciPy Version: {scipy.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Show build configuration
    print("\nSciPy Build Configuration:")
    print("-" * 40)
    scipy.__config__.show()

def get_library_info():
    """Detect and display underlying math library implementations."""
    print_section("MATH LIBRARY IMPLEMENTATIONS")
    
    # Check BLAS/LAPACK
    try:
        from scipy.linalg import blas
        print("BLAS Information:")
        print(f"  BLAS available: {hasattr(blas, 'dgemm')}")
        
        # Try to detect implementation
        blas_info = scipy.__config__.get_info('blas_opt')
        if blas_info:
            libs = blas_info.get('libraries', ['unknown'])
            print(f"  BLAS libraries: {libs}")
            # Try to identify specific implementations
            for lib in libs:
                lib_lower = lib.lower()
                if 'mkl' in lib_lower:
                    print(f"    -> Using Intel MKL")
                elif 'openblas' in lib_lower:
                    print(f"    -> Using OpenBLAS")
                elif 'blis' in lib_lower:
                    print(f"    -> Using BLIS")
                elif 'atlas' in lib_lower:
                    print(f"    -> Using ATLAS")
            # Print extra information
            macros = blas_info.get('define_macros', [])
            if macros:
                print(f"  BLAS macros: {macros}")
    except Exception as e:
        print(f"  Error checking BLAS: {e}")
    
    # Check LAPACK
    try:
        from scipy.linalg import lapack
        print("\nLAPACK Information:")
        print(f"  LAPACK available: {hasattr(lapack, 'dgesv')}")
        
        lapack_info = scipy.__config__.get_info('lapack_opt')
        if lapack_info:
            libs = lapack_info.get('libraries', ['unknown'])
            print(f"  LAPACK libraries: {libs}")
    except Exception as e:
        print(f"  Error checking LAPACK: {e}")
    
    # Check FFT implementation
    print("\nFFT Implementation:")
    try:
        import pyfftw
        print("  Using FFTW via pyfftw")
        print(f"  pyfftw version: {pyfftw.__version__}")
    except ImportError:
        # Check if using MKL FFT
        try:
            # Check numpy's FFT backend
            if hasattr(np.fft, '__name__'):
                print("  Using default NumPy/SciPy FFT")
            # Additional checks could be added here
        except:
            print("  Using default SciPy FFT")
    
    # Additional library checks
    print("\nAdditional Library Information:")
    # Check for MKL
    try:
        import mkl
        print(f"  Intel MKL version: {mkl.__version__}")
    except ImportError:
        pass
    
    # Check for OpenBLAS
    try:
        import ctypes
        # Try to find OpenBLAS
        libopenblas = ctypes.CDLL('libopenblas.so', ctypes.RTLD_GLOBAL)
        print("  OpenBLAS detected via shared library")
    except:
        pass

def benchmark_matrix_multiplication():
    """Benchmark matrix multiplication operations."""
    print_section("MATRIX MULTIPLICATION BENCHMARK")
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create random matrices
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        
        # Standard NumPy matmul
        start = time.perf_counter()
        C = np.matmul(A, B)
        numpy_time = time.perf_counter() - start
        print(f"  NumPy matmul: {numpy_time:.4f} seconds")
        
        # SciPy BLAS dgemm if available
        try:
            from scipy.linalg.blas import dgemm
            start = time.perf_counter()
            C = dgemm(1.0, A, B)
            blas_time = time.perf_counter() - start
            print(f"  BLAS dgemm:   {blas_time:.4f} seconds (speedup: {numpy_time/blas_time:.2f}x)")
        except ImportError:
            print(f"  BLAS dgemm:   Not available")

def benchmark_linear_algebra():
    """Benchmark linear algebra operations."""
    print_section("LINEAR ALGEBRA BENCHMARK")
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create a symmetric positive definite matrix
        A = np.random.randn(size, size)
        A = A @ A.T + np.eye(size) * 0.1
        
        # 1. Cholesky decomposition
        start = time.perf_counter()
        try:
            L = linalg.cholesky(A, lower=True)
            cholesky_time = time.perf_counter() - start
            print(f"  Cholesky decomposition: {cholesky_time:.4f} seconds")
        except linalg.LinAlgError:
            print(f"  Cholesky: Matrix not positive definite")
        
        # 2. LU decomposition
        start = time.perf_counter()
        P, L, U = linalg.lu(A)
        lu_time = time.perf_counter() - start
        print(f"  LU decomposition:        {lu_time:.4f} seconds")
        
        # 3. Eigenvalue decomposition
        start = time.perf_counter()
        eigenvalues, eigenvectors = linalg.eigh(A)
        eig_time = time.perf_counter() - start
        print(f"  Eigenvalue (eigh):       {eig_time:.4f} seconds")

def benchmark_sparse_operations():
    """Benchmark sparse matrix operations."""
    print_section("SPARSE MATRIX BENCHMARK")
    
    if not hasattr(sparse, 'random'):
        print("Sparse random matrix generation not available")
        return
    
    sizes = [1000, 5000]
    densities = [0.01, 0.001]
    
    for size, density in zip(sizes, densities):
        print(f"\nSparse matrix: {size}x{size} (density: {density})")
        
        # Create sparse matrix
        A_sparse = sparse.random(size, size, density=density, format='csr')
        x = np.random.randn(size)
        
        # Sparse matrix-vector multiplication
        start = time.perf_counter()
        y = A_sparse.dot(x)
        sparse_mv_time = time.perf_counter() - start
        print(f"  Sparse mat-vec: {sparse_mv_time:.6f} seconds")
        
        # Convert to dense for comparison
        if size <= 5000:
            A_dense = A_sparse.toarray()
            start = time.perf_counter()
            y_dense = A_dense @ x
            dense_mv_time = time.perf_counter() - start
            print(f"  Dense mat-vec:  {dense_mv_time:.6f} seconds")
            print(f"  Sparse speedup: {dense_mv_time/sparse_mv_time:.2f}x")

def benchmark_fft():
    """Benchmark FFT operations."""
    print_section("FFT BENCHMARK")
    
    sizes = [1024, 8192, 65536, 262144]
    
    for size in sizes:
        print(f"\nFFT size: {size}")
        
        # Create complex data
        data = np.random.randn(size) + 1j * np.random.randn(size)
        
        # 1D FFT
        start = time.perf_counter()
        result = fft.fft(data)
        fft_time = time.perf_counter() - start
        print(f"  1D FFT: {fft_time:.6f} seconds")
        
        # 1D inverse FFT
        start = time.perf_counter()
        original = fft.ifft(result)
        ifft_time = time.perf_counter() - start
        print(f"  1D IFFT: {ifft_time:.6f} seconds")

def benchmark_optimization():
    """Benchmark optimization routines."""
    print_section("OPTIMIZATION BENCHMARK")
    
    print("Rosenbrock function minimization:")
    
    # Rosenbrock function
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
    # Initial guess
    x0 = np.array([-1.5, 2.0, 1.0, -2.0, 0.5])
    
    # BFGS
    start = time.perf_counter()
    result = optimize.minimize(rosenbrock, x0, method='BFGS', 
                               options={'disp': False})
    bfgs_time = time.perf_counter() - start
    print(f"  BFGS: {bfgs_time:.4f} seconds (fval: {result.fun:.2e})")
    
    # L-BFGS-B
    start = time.perf_counter()
    result = optimize.minimize(rosenbrock, x0, method='L-BFGS-B',
                               options={'disp': False})
    lbfgs_time = time.perf_counter() - start
    print(f"  L-BFGS-B: {lbfgs_time:.4f} seconds (fval: {result.fun:.2e})")

def print_summary():
    """Print a summary of key findings."""
    print_section("SUMMARY OF KEY FINDINGS")
    
    # Collect key information
    summary_points = []
    
    # System info
    summary_points.append(f"• Platform: {platform.system()} {platform.machine()}")
    
    # SciPy and NumPy versions
    summary_points.append(f"• SciPy {scipy.__version__}, NumPy {np.__version__}")
    
    # BLAS/LAPACK detection
    try:
        blas_info = scipy.__config__.get_info('blas_opt')
        if blas_info:
            libs = blas_info.get('libraries', ['unknown'])
            summary_points.append(f"• Primary BLAS: {libs[0] if libs else 'unknown'}")
    except:
        summary_points.append("• BLAS: Detection failed")
    
    # Performance notes
    summary_points.append("• All benchmarks completed successfully")
    summary_points.append("• Output is in English per MCPC coding standards")
    
    for point in summary_points:
        print(point)

def run_all_benchmarks():
    """Run all benchmark tests."""
    print("SciPy Library Information and Performance Benchmark")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Collect information
    get_system_info()
    get_scipy_info()
    get_library_info()
    
    # Run benchmarks
    benchmark_matrix_multiplication()
    benchmark_linear_algebra()
    benchmark_sparse_operations()
    benchmark_fft()
    benchmark_optimization()
    
    # Print summary
    print_summary()
    
    print_section("BENCHMARK COMPLETE")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main entry point for the benchmark script."""
    try:
        run_all_benchmarks()
        return 0
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
