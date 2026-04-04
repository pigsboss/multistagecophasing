"""
Lunar Swing 项目测试配置
提供项目专用的测试夹具和配置
"""
import pytest
import numpy as np

@pytest.fixture
def earth_moon_mu():
    """地月系统质量参数"""
    return 0.01215

@pytest.fixture
def sample_crtbp_state():
    """CRTBP测试状态（无量纲单位）"""
    return np.array([0.8, 0.0, 0.1, 0.0, 0.5, 0.0])

@pytest.fixture
def sample_inertial_state():
    """惯性系测试状态（SI单位）"""
    return np.array([1.5e8, 0.0, 0.0, 0.0, 1.0e3, 0.0])
