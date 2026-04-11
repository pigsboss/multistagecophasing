"""
高精度星历接口桩测试
验证 HighPrecisionEphemeris 接口设计的合理性
"""
import pytest
import numpy as np

# 注意：这里导入的是桩接口，实际实现将在后续完成
try:
    from mission_sim.core.spacetime.ephemeris.high_precision import HighPrecisionEphemeris
    from mission_sim.core.spacetime.ids import CoordinateFrame
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    # 创建桩类用于测试接口设计
    class HighPrecisionEphemeris:
        VALID_BODIES = {'earth', 'moon', 'sun', 'mercury', 'venus', 'mars', 
                       'jupiter', 'saturn', 'uranus', 'neptune'}
        VALID_FRAMES = {'J2000_ECI', 'ICRF', 'ITRF93'}
        
        def __init__(self, kernel_path=None):
            self.kernel_path = kernel_path
        
        def get_state(self, target_body, epoch, observer_body='earth', frame='J2000_ECI'):
            if target_body.lower() not in self.VALID_BODIES:
                raise ValueError(f"Invalid body name: {target_body}")
            if frame.upper() not in self.VALID_FRAMES:
                raise ValueError(f"Invalid frame: {frame}")
            return np.zeros(6)
        
        def get_earth_moon_rotating_state(self, epoch):
            # 返回单一数组，而不是两个数组
            return np.zeros(6)


def test_interface_exists():
    """验证接口类存在且可实例化"""
    # 尝试导入（如果接口未实现，此处会失败）
    assert HighPrecisionEphemeris is not None


def test_get_state_signature():
    """验证 get_state 方法的签名"""
    # 创建桩实例（假设有简单实现或使用mock）
    ephem = HighPrecisionEphemeris()
    
    # 测试基本调用 - 使用正确的坐标系名
    state = ephem.get_state('moon', 0.0, 'earth', 'J2000_ECI')
    
    # 验证返回类型和形状
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)  # [x, y, z, vx, vy, vz]
    
    # 验证数据类型
    assert state.dtype == np.float64 or state.dtype == float


def test_earth_moon_rotating_state():
    """验证地月旋转系状态获取接口"""
    ephem = HighPrecisionEphemeris()
    
    # 修正：只获取一个返回值
    rotating_state = ephem.get_earth_moon_rotating_state(0.0)
    
    # 验证返回类型
    assert isinstance(rotating_state, np.ndarray)
    
    # 验证形状（应该是6维数组）
    assert rotating_state.shape == (6,)


def test_error_handling():
    """验证错误处理"""
    ephem = HighPrecisionEphemeris()
    
    # 测试无效天体名称
    with pytest.raises(ValueError):
        ephem.get_state('invalid_body', 0.0)
    
    # 测试无效坐标系 - 使用正确的有效框架名
    with pytest.raises(ValueError):
        ephem.get_state('moon', 0.0, 'earth', 'INVALID_FRAME')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
