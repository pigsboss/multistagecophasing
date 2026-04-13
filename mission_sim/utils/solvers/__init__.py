"""
求解器框架 - 主接口模块

基于 map-reduce 架构的通用求解器框架，支持多种并行计算方式和算法。
提供统一接口，易于扩展和集成。

版本: 1.0.0
遵循 MCPC 编码标准（UTF-8，运行时输出使用英文）
"""
from __future__ import annotations

from typing import Dict, List, Any

from .base import (
    # 枚举类型
    SolverStatus,
    DeviceType,
    ParallelBackend,
    AlgorithmType,
    
    # 协议接口
    DeviceInterface,
    WorkerInterface,
    SchedulerInterface,
    LoggerInterface,
    
    # 基础类
    SolverBase,
    OptimizationSolver,
    IntegrationSolver,
    DifferentialCorrectionSolver,
    MonteCarloSolver,
)

# 导出主求解器类
__all__ = [
    # 枚举
    'SolverStatus',
    'DeviceType',
    'ParallelBackend',
    'AlgorithmType',
    
    # 协议接口
    'DeviceInterface',
    'WorkerInterface',
    'SchedulerInterface',
    'LoggerInterface',
    
    # 求解器基类
    'SolverBase',
    'OptimizationSolver',
    'IntegrationSolver',
    'DifferentialCorrectionSolver',
    'MonteCarloSolver',
    
    # 工厂函数和工具
    'create_solver',
    'list_available_backends',
    'get_solver_info',
    'register_algorithm',
    'get_registered_algorithms',
]


# 算法注册表（用于动态扩展）
_ALGORITHM_REGISTRY: Dict[str, type] = {}


def create_solver(
    algorithm_type: AlgorithmType,
    solver_class: type = None,
    **kwargs
) -> SolverBase:
    """
    创建求解器实例的工厂函数
    
    Args:
        algorithm_type: 算法类型
        solver_class: 自定义求解器类（可选）
        **kwargs: 求解器初始化参数
        
    Returns:
        SolverBase: 求解器实例
        
    Raises:
        ValueError: 如果指定的求解器类不支持请求的算法类型
        TypeError: 如果求解器类不是 SolverBase 的子类
        
    Examples:
        >>> from mission_sim.utils.solvers import create_solver, AlgorithmType
        >>> solver = create_solver(AlgorithmType.OPTIMIZATION)
        >>> solver = create_solver(AlgorithmType.INTEGRATION, device_type=DeviceType.GPU)
    """
    # 如果提供了自定义类，使用它
    if solver_class is not None:
        if not issubclass(solver_class, SolverBase):
            raise TypeError(f"solver_class must be a subclass of SolverBase, got {type(solver_class)}")
        return solver_class(algorithm_type=algorithm_type, **kwargs)
    
    # 否则根据算法类型选择默认类
    solver_map = {
        AlgorithmType.OPTIMIZATION: OptimizationSolver,
        AlgorithmType.INTEGRATION: IntegrationSolver,
        AlgorithmType.DIFFERENTIAL_CORRECTION: DifferentialCorrectionSolver,
        AlgorithmType.MONTE_CARLO: MonteCarloSolver,
        AlgorithmType.ROOT_FINDING: SolverBase,  # 默认基类
        AlgorithmType.LINEAR_SOLVER: SolverBase,  # 默认基类
    }
    
    solver_class = solver_map.get(algorithm_type, SolverBase)
    return solver_class(algorithm_type=algorithm_type, **kwargs)


def list_available_backends() -> Dict[str, List[str]]:
    """
    列出可用的并行计算后端
    
    Returns:
        Dict[str, List[str]]: 按类别分类的后端列表
        
    Note:
        这里只返回理论上支持的后端，具体可用性取决于系统环境
    """
    import sys
    import importlib
    
    backends = {
        "always_available": ["numpy", "multiprocessing", "threading"],
        "requires_optional_deps": [],
        "gpu_accelerated": [],
    }
    
    # 检查 numba
    try:
        importlib.import_module("numba")
        backends["requires_optional_deps"].append("numba")
    except ImportError:
        pass
    
    # 检查 pyopencl
    try:
        importlib.import_module("pyopencl")
        backends["gpu_accelerated"].append("opencl")
    except ImportError:
        pass
    
    # 检查 cupy (CUDA)
    try:
        importlib.import_module("cupy")
        backends["gpu_accelerated"].append("cuda")
    except ImportError:
        pass
    
    # 检查 mpi4py
    try:
        importlib.import_module("mpi4py")
        backends["requires_optional_deps"].append("mpi")
    except ImportError:
        pass
    
    return backends


def get_solver_info(solver: SolverBase) -> Dict[str, Any]:
    """
    获取求解器详细信息
    
    Args:
        solver: 求解器实例
        
    Returns:
        Dict[str, Any]: 求解器信息字典
    """
    info = {
        "solver_type": solver.__class__.__name__,
        "algorithm_type": solver.algorithm_type.value,
        "device_type": solver.device_type.value,
        "parallel_backend": solver.parallel_backend.value,
        "max_workers": solver.max_workers,
        "timeout": solver.timeout,
        "checkpoint_interval": solver.checkpoint_interval,
        "verbose": solver.verbose,
    }
    
    # 添加状态信息
    info.update(solver.get_status_report())
    
    return info


def register_algorithm(
    algorithm_name: str,
    algorithm_class: type,
    algorithm_type: AlgorithmType
) -> None:
    """
    注册新算法到求解器框架
    
    Args:
        algorithm_name: 算法名称
        algorithm_class: 算法类（必须是SolverBase的子类）
        algorithm_type: 算法类型
        
    Raises:
        TypeError: 如果algorithm_class不是SolverBase的子类
        ValueError: 如果algorithm_name已注册
        
    Examples:
        >>> from mission_sim.utils.solvers import register_algorithm, AlgorithmType
        >>> class MyCustomSolver(SolverBase):
        ...     pass
        >>> register_algorithm("my_custom", MyCustomSolver, AlgorithmType.OPTIMIZATION)
    """
    if not issubclass(algorithm_class, SolverBase):
        raise TypeError(
            f"algorithm_class must be a subclass of SolverBase, got {algorithm_class}"
        )
    
    if algorithm_name in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{algorithm_name}' is already registered")
    
    # 设置算法类型
    algorithm_class.registered_type = algorithm_type
    _ALGORITHM_REGISTRY[algorithm_name] = algorithm_class
    
    # 输出注册信息
    print(f"[INFO] Algorithm '{algorithm_name}' registered successfully")
    print(f"       Type: {algorithm_type.value}")
    print(f"       Class: {algorithm_class.__name__}")


def get_registered_algorithms() -> Dict[str, Dict[str, Any]]:
    """
    获取所有已注册的算法
    
    Returns:
        Dict[str, Dict[str, Any]]: 算法信息字典
    """
    result = {}
    for name, cls in _ALGORITHM_REGISTRY.items():
        result[name] = {
            "class": cls,
            "type": getattr(cls, 'registered_type', AlgorithmType.OPTIMIZATION),
            "module": cls.__module__,
            "doc": cls.__doc__ or "No documentation",
            "name": cls.__name__,
        }
    return result


def create_solver_by_name(
    algorithm_name: str,
    **kwargs
) -> SolverBase:
    """
    通过算法名称创建求解器
    
    Args:
        algorithm_name: 已注册的算法名称
        **kwargs: 求解器初始化参数
        
    Returns:
        SolverBase: 求解器实例
        
    Raises:
        KeyError: 如果算法名称未注册
    """
    if algorithm_name not in _ALGORITHM_REGISTRY:
        available = list(_ALGORITHM_REGISTRY.keys())
        raise KeyError(
            f"Algorithm '{algorithm_name}' is not registered. "
            f"Available algorithms: {available}"
        )
    
    solver_class = _ALGORITHM_REGISTRY[algorithm_name]
    algorithm_type = getattr(solver_class, 'registered_type', AlgorithmType.OPTIMIZATION)
    
    return solver_class(algorithm_type=algorithm_type, **kwargs)


# 版本信息
__version__ = "1.0.0"
__author__ = "MCPC Development Team"
__license__ = "MIT"


# 框架初始化
def _initialize_framework() -> None:
    """初始化求解器框架（内部使用）"""
    import sys
    import platform
    
    # 输出框架信息
    print("=" * 60)
    print(f"MCPC Solver Framework v{__version__}")
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    print("=" * 60)
    print("Features:")
    print("  • Map-Reduce architecture for parallel computation")
    print("  • Multiple parallel backends: numpy, numba, opencl, cuda, etc.")
    print("  • Support for CPU, GPU, NPU, FPGA devices")
    print("  • Checkpoint and resume capability")
    print("  • Extensible algorithm registration system")
    print("=" * 60)
    
    # 检查可用后端
    backends = list_available_backends()
    print("Available parallel backends:")
    for category, backend_list in backends.items():
        if backend_list:
            print(f"  {category}: {', '.join(backend_list)}")
    
    print("=" * 60)


# 自动初始化（仅在首次导入时执行）
if not hasattr(create_solver, '_initialized'):
    _initialize_framework()
    create_solver._initialized = True
