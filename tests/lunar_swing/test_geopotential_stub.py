"""
HighOrderGeopotential 模块的桩测试

第一阶段目标：验证接口设计合理性，不测试实际功能实现。
"""
import pytest
import numpy as np

# 导入将被测试的接口
try:
    from mission_sim.core.physics.models.gravity.high_order_geopotential import HighOrderGeopotential
    from mission_sim.core.physics.environment import IForceModel
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    # 创建桩类用于测试接口设计
    class IForceModel:
        pass
    
    class HighOrderGeopotential(IForceModel):
        def __init__(self, degree: int = 10, order: int = 10, coeff_file: str = None):
            self.degree = degree
            self.order = order
            self.coeff_file = coeff_file
        
        def compute_accel(self, state: np.ndarray, epoch: float) -> np.ndarray:
            raise NotImplementedError("桩实现：接口待实现")
        
        def set_max_degree(self, degree: int):
            raise NotImplementedError("桩实现：接口待实现")


@pytest.mark.skipif(not HAS_MODULE, reason="模块尚未实现，仅测试接口设计")
class TestHighOrderGeopotentialInterface:
    """测试 HighOrderGeopotential 接口设计"""
    
    def test_initialization_default(self):
        """测试默认初始化"""
        model = HighOrderGeopotential()
        assert model.degree == 10
        assert model.order == 10
        assert model.coeff_file is None
    
    def test_initialization_custom(self):
        """测试自定义参数初始化"""
        model = HighOrderGeopotential(
            degree=20,
            order=20,
            coeff_file='data/egm2008.gfc'
        )
        assert model.degree == 20
        assert model.order == 20
        assert model.coeff_file == 'data/egm2008.gfc'
    
    def test_iforcemodel_inheritance(self):
        """测试是否继承自 IForceModel"""
        model = HighOrderGeopotential()
        assert isinstance(model, IForceModel)
    
    def test_method_signatures(self):
        """测试方法签名"""
        model = HighOrderGeopotential()
        
        import inspect
        
        # 测试 compute_accel 签名
        sig = inspect.signature(model.compute_accel)
        params = list(sig.parameters.keys())
        assert 'state' in params
        assert 'epoch' in params
        
        # 测试 set_max_degree 签名
        sig = inspect.signature(model.set_max_degree)
        params = list(sig.parameters.keys())
        assert 'degree' in params
    
    def test_example_usage(self):
        """测试接口使用示例"""
        model = HighOrderGeopotential(degree=15, order=15)
        
        # 测试状态向量
        test_state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        test_epoch = 0.0
        
        # compute_accel 已实现（J2简化版）
        accel = model.compute_accel(test_state, test_epoch)
        assert isinstance(accel, np.ndarray)
        assert accel.shape == (3,)
        
        # set_max_degree 已实现
        model.set_max_degree(30)
        assert model.degree == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
