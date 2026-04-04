"""
通用CRTBP接口桩测试
验证 UniversalCRTBP 接口设计的合理性
"""
import pytest
import numpy as np

# 导入现有MCPC契约
from mission_sim.core.physics.environment import IForceModel
from mission_sim.core.spacetime.ids import CoordinateFrame

# 导入新接口
from mission_sim.core.physics.models.threebody.universal_crtbp import UniversalCRTBP


def test_implements_force_model():
    """验证 UniversalCRTBP 实现了 IForceModel 接口"""
    # 创建实例
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 验证类型
    assert isinstance(crtbp, IForceModel)
    
    # 验证必需的方法存在
    assert hasattr(crtbp, 'compute_accel')
    assert callable(crtbp.compute_accel)


def test_convenience_constructors():
    """验证便捷构造方法"""
    # 测试地月系统
    earth_moon = UniversalCRTBP.earth_moon_system()
    assert earth_moon is not None
    
    # 测试日地系统
    sun_earth = UniversalCRTBP.sun_earth_system()
    assert sun_earth is not None
    
    # 验证它们是不同的实例
    assert earth_moon is not sun_earth


def test_compute_accel_interface():
    """验证加速度计算接口"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 测试状态
    test_state = np.array([1.0e8, 0.0, 0.0, 0.0, 1.0e3, 0.0])
    
    # 调用接口
    accel = crtbp.compute_accel(test_state, epoch=0.0)
    
    # 验证返回
    assert isinstance(accel, np.ndarray)
    assert accel.shape == (3,)  # [ax, ay, az]
    
    # 验证物理意义：在旋转系中，加速度应与位置有关
    # （桩实现可能返回零，但接口测试只关心形状和类型）


def test_jacobi_constant():
    """验证雅可比常数计算接口"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    test_state = np.array([1.0e8, 0.0, 0.0, 0.0, 1.0e3, 0.0])
    C = crtbp.jacobi_constant(test_state)
    
    # 验证返回类型
    assert isinstance(C, float)
    
    # 雅可比常数应为标量
    assert np.isscalar(C)


def test_coordinate_transformations():
    """验证坐标系转换接口"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 测试状态
    inertial_state = np.array([1.5e8, 0.0, 0.0, 0.0, 1.0e3, 0.0])
    
    # 测试转换到旋转系
    rotating_state = crtbp.to_rotating_frame(inertial_state, epoch=0.0)
    assert rotating_state.shape == (6,)
    
    # 测试转换回惯性系
    recovered_state = crtbp.to_inertial_frame(rotating_state, epoch=0.0)
    assert recovered_state.shape == (6,)
    
    # 验证往返转换的一致性（在桩测试中可能不严格，但接口存在即可）


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
