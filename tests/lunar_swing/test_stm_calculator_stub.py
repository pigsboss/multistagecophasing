"""
STMCalculator 模块的桩测试

第一阶段目标：验证接口设计合理性，不测试实际功能实现。
"""
import pytest
import numpy as np
from typing import Callable

# 导入将被测试的接口
try:
    from mission_sim.utils.dynamics.stm_calculator import STMCalculator
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    # 创建桩类用于测试接口设计
    class STMCalculator:
        @staticmethod
        def compute_numerical(dynamics: Callable,
                             initial_state: np.ndarray,
                             t0: float,
                             tf: float,
                             method: str = 'DOP853') -> np.ndarray:
            raise NotImplementedError("桩实现：接口待实现")
        
        @staticmethod
        def compute_analytic(dynamics_jacobian: Callable,
                            initial_state: np.ndarray,
                            t0: float,
                            tf: float) -> np.ndarray:
            raise NotImplementedError("桩实现：接口待实现")


@pytest.mark.skipif(not HAS_MODULE, reason="模块尚未实现，仅测试接口设计")
class TestSTMCalculatorInterface:
    """测试 STMCalculator 接口设计"""
    
    def test_static_methods(self):
        """测试静态方法存在性"""
        # 验证类有预期的静态方法
        assert hasattr(STMCalculator, 'compute_numerical')
        assert hasattr(STMCalculator, 'compute_analytic')
        
        # 验证它们是静态方法
        import inspect
        assert isinstance(inspect.getattr_static(STMCalculator, 'compute_numerical'), staticmethod)
        assert isinstance(inspect.getattr_static(STMCalculator, 'compute_analytic'), staticmethod)
    
    def test_compute_numerical_signature(self):
        """测试 compute_numerical 方法签名"""
        import inspect
        sig = inspect.signature(STMCalculator.compute_numerical)
        params = list(sig.parameters.keys())
        
        assert 'dynamics' in params
        assert 'initial_state' in params
        assert 't0' in params
        assert 'tf' in params
        assert 'method' in params
        
        # 检查 method 参数的默认值（实际实现为 'rk4'）
        assert sig.parameters['method'].default == 'rk4'
    
    def test_compute_analytic_signature(self):
        """测试 compute_analytic 方法签名"""
        import inspect
        sig = inspect.signature(STMCalculator.compute_analytic)
        params = list(sig.parameters.keys())
        
        assert 'dynamics_jacobian' in params
        assert 'initial_state' in params
        assert 't0' in params
        assert 'tf' in params
    
    def test_compute_numerical_usage(self):
        """测试 compute_numerical 使用示例"""
        # 创建模拟的动力学函数
        def mock_dynamics(t: float, state: np.ndarray) -> np.ndarray:
            return np.zeros_like(state)
        
        initial_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        t0 = 0.0
        tf = 10.0
        
        # 测试不同积分方法（已实现，返回单位矩阵作为占位）
        methods = ['rk4', 'dop853']  # 实际支持的方法
        
        for method in methods:
            stm = STMCalculator.compute_numerical(
                dynamics=mock_dynamics,
                initial_state=initial_state,
                t0=t0,
                tf=tf,
                method=method
            )
            assert isinstance(stm, np.ndarray)
            assert stm.shape == (6, 6)
    
    def test_compute_analytic_usage(self):
        """测试 compute_analytic 使用示例"""
        # 创建模拟的雅可比函数
        def mock_jacobian(t: float, state: np.ndarray) -> np.ndarray:
            return np.eye(6)
        
        initial_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        t0 = 0.0
        tf = 10.0
        
        # 已实现，返回单位矩阵作为占位
        stm = STMCalculator.compute_analytic(
            dynamics_jacobian=mock_jacobian,
            initial_state=initial_state,
            t0=t0,
            tf=tf
        )
        assert isinstance(stm, np.ndarray)
        assert stm.shape == (6, 6)
    
    def test_consistency_between_methods(self):
        """测试两种方法接口一致性"""
        # 两种方法应有相同的核心参数
        import inspect
        sig_numerical = inspect.signature(STMCalculator.compute_numerical)
        sig_analytic = inspect.signature(STMCalculator.compute_analytic)
        
        # 检查共同参数
        common_params = ['initial_state', 't0', 'tf']
        for param in common_params:
            assert param in sig_numerical.parameters
            assert param in sig_analytic.parameters
        
        # 检查返回类型（如果指定了注解）
        if sig_numerical.return_annotation != inspect.Signature.empty:
            assert sig_numerical.return_annotation == np.ndarray
        
        if sig_analytic.return_annotation != inspect.Signature.empty:
            assert sig_analytic.return_annotation == np.ndarray


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
