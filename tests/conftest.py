import pytest
import numpy as np

@pytest.fixture
def temp_dir(tmp_path):
    """临时目录，用于测试输出文件"""
    return tmp_path

@pytest.fixture
def default_config():
    """默认仿真配置"""
    return {
        "mission_name": "Test Mission",
        "simulation_days": 0.01,
        "time_step": 10.0,
        "Az": 0.05,
        "mu": 3.00348e-6,
        "L": 1.495978707e11,
        "omega": 1.990986e-7,
        "spacecraft_mass": 6200.0,
        "data_dir": "data/test"
    }

@pytest.fixture
def crtbp_params():
    """CRTBP 标准参数"""
    return {
        "mu": 3.00348e-6,
        "L": 1.495978707e11,
        "omega": 1.990986e-7
    }

@pytest.fixture
def halo_initial_state():
    """标准 Halo 轨道初始状态（无量纲）"""
    return np.array([1.01106, 0.0, 0.05, 0.0, 0.0105, 0.0])