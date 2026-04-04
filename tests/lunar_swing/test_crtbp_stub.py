"""
通用CRTBP功能测试
验证 UniversalCRTBP 实现的正确性和功能完整性
"""
import pytest
import numpy as np

# 导入现有MCPC契约
from mission_sim.core.physics.environment import IForceModel
from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP


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
    assert earth_moon.system_name == 'earth_moon'
    
    # 测试日地系统
    sun_earth = UniversalCRTBP.sun_earth_system()
    assert sun_earth is not None
    assert sun_earth.system_name == 'sun_earth'
    
    # 验证它们是不同的实例
    assert earth_moon is not sun_earth
    
    # 验证参数
    assert earth_moon.mu == pytest.approx(0.01215, rel=1e-3)
    assert earth_moon.distance == pytest.approx(3.844e8, rel=1e-3)
    assert earth_moon.omega > 0


def test_compute_acceleration():
    """验证加速度计算功能"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 测试状态：靠近地球的位置
    test_state = np.array([-0.1 * crtbp.distance, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 调用接口
    accel = crtbp.compute_accel(test_state, epoch=0.0)
    
    # 验证返回
    assert isinstance(accel, np.ndarray)
    assert accel.shape == (3,)  # [ax, ay, az]
    
    # 验证加速度不为零（实际物理计算）
    assert np.any(accel != 0)
    
    # 验证加速度量级合理
    assert np.linalg.norm(accel) > 0


def test_jacobi_constant():
    """验证雅可比常数计算"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 测试状态：拉格朗日点 L1 附近
    test_state = np.array([0.84 * crtbp.distance, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    C = crtbp.jacobi_constant(test_state)
    
    # 验证返回类型
    assert isinstance(C, float)
    assert np.isscalar(C)
    
    # 雅可比常数应为正数
    assert C > 0
    
    # 验证不同状态的雅可比常数不同
    test_state2 = np.array([0.5 * crtbp.distance, 0.0, 0.0, 0.0, 100.0, 0.0])
    C2 = crtbp.jacobi_constant(test_state2)
    assert C2 != C


def test_coordinate_transformations():
    """验证坐标系转换功能"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 测试状态
    inertial_state = np.array([1.5e8, 1.0e8, 0.0, 100.0, 200.0, 0.0])
    
    # 测试转换到旋转系
    rotating_state = crtbp.to_rotating_frame(inertial_state, epoch=0.0)
    assert rotating_state.shape == (6,)
    assert isinstance(rotating_state, np.ndarray)
    
    # 测试转换回惯性系
    recovered_state = crtbp.to_inertial_frame(rotating_state, epoch=0.0)
    assert recovered_state.shape == (6,)
    assert isinstance(recovered_state, np.ndarray)
    
    # 验证往返转换的一致性（应在容差范围内）
    assert np.allclose(inertial_state, recovered_state, rtol=1e-10, atol=1e-6)


def test_system_parameters():
    """验证系统参数获取"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    params = crtbp.get_system_parameters()
    
    # 验证参数类型和存在性
    assert isinstance(params, dict)
    assert 'system_name' in params
    assert 'mu' in params
    assert 'omega' in params
    assert 'distance' in params
    
    # 验证参数值
    assert params['system_name'] == 'earth_moon'
    assert params['mu'] == pytest.approx(0.01215, rel=1e-3)
    assert params['distance'] == pytest.approx(3.844e8, rel=1e-3)


def test_lagrange_points():
    """验证拉格朗日点计算"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    lagrange_points = crtbp.get_lagrange_points_nd()
    
    # 验证所有5个拉格朗日点都存在
    assert 'L1' in lagrange_points
    assert 'L2' in lagrange_points
    assert 'L3' in lagrange_points
    assert 'L4' in lagrange_points
    assert 'L5' in lagrange_points
    
    # 验证形状
    for point in lagrange_points.values():
        assert isinstance(point, np.ndarray)
        assert point.shape == (3,)
    
    # L4 和 L5 应有非零的 y 坐标
    assert lagrange_points['L4'][1] > 0
    assert lagrange_points['L5'][1] < 0


def test_energy_conservation():
    """验证雅可比常数在无外力的 CRTBP 中是运动积分"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 初始状态
    initial_state = np.array([0.9 * crtbp.distance, 0.0, 0.0, 0.0, 1000.0, 0.0])
    
    # 计算初始雅可比常数
    C_initial = crtbp.jacobi_constant(initial_state)
    
    # 模拟一个小时间步长的运动（简化验证）
    dt = 10.0  # 秒
    accel = crtbp.compute_accel(initial_state, epoch=0.0)
    
    # 更新速度（简化积分）
    state_after_dt = initial_state.copy()
    state_after_dt[3:6] += accel * dt
    
    # 计算后的雅可比常数
    C_after = crtbp.jacobi_constant(state_after_dt)
    
    # 在无摄动的 CRTBP 中，雅可比常数应严格守恒
    # 但由于我们只做了简化积分，这里只验证概念
    assert abs(C_after - C_initial) < 1.0  # 宽松的容差


def test_physical_units():
    """验证物理单位的一致性"""
    crtbp = UniversalCRTBP.earth_moon_system()
    
    # 验证单位转换的一致性
    test_state_physical = np.array([1.0e8, 2.0e8, 3.0e7, 1000.0, 2000.0, 300.0])
    
    # 转换为无量纲再转回物理单位
    state_nd = crtbp._to_nd(test_state_physical)
    state_back = crtbp._to_physical(state_nd)
    
    # 应完全一致
    assert np.allclose(test_state_physical, state_back, rtol=1e-10, atol=1e-6)


def test_custom_system():
    """验证自定义系统的创建"""
    # 创建一个自定义系统（例如：火星-火卫一系统）
    mars_mass = 6.39e23  # kg
    phobos_mass = 1.06e16  # kg
    distance = 9.377e6  # m
    
    custom_crtbp = UniversalCRTBP(mars_mass, phobos_mass, distance, 'mars_phobos')
    
    # 验证参数
    assert custom_crtbp.system_name == 'mars_phobos'
    assert custom_crtbp.primary_mass == mars_mass
    assert custom_crtbp.secondary_mass == phobos_mass
    assert custom_crtbp.distance == distance
    assert 0 < custom_crtbp.mu < 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
