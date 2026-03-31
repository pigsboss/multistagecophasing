"""
集成测试：验证 L1 级核心仿真的完整闭环。
"""
import pytest
import os
from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.simulation.twobody.leo import LEOL1Simulation
from mission_sim.simulation.twobody.geo import GEOL1Simulation

@pytest.fixture
def temp_dir(tmp_path):
    """提供临时目录用于存储仿真输出"""
    return tmp_path

@pytest.fixture
def default_config():
    """提供基础的仿真配置参数"""
    return {
        "mission_name": "Test Mission",
        "description": "Short integration test",
        "L": 1.495978707e11,
        "m1": 1.989e30,
        "m2": 5.972e24,
        "mu": 3.003e-6,
        "Ax": 0.00133,
        "Az": 0.05,
        "data_dir": "data/test"
    }

def test_short_simulation_sun_earth_l2(temp_dir, default_config):
    """运行简短仿真（1 天）验证集成（日地 L2 点）"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False
    config["verbose"] = False

    sim = SunEarthL2L1Simulation(config)
    success = sim.run()
    assert success

    # 只验证 HDF5 日志是否正常落盘
    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

def test_leo_simulation(temp_dir, default_config):
    """运行 LEO 仿真（1 天）验证集成（有控）"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 10.0
    config["enable_visualization"] = False
    config["elements"] = [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 1000.0
    config["area_to_mass"] = 0.02
    config["Cd"] = 2.2
    config["enable_atmospheric_drag"] = True
    config["enable_j2"] = True
    config["use_j2_generator"] = False
    config["control_gain_scale"] = 5e-9
    config["verbose"] = False

    sim = LEOL1Simulation(config)
    success = sim.run()
    assert success

    h5_file = sim.h5_file
    assert os.path.exists(h5_file)

def test_geo_simulation(temp_dir, default_config):
    """运行 GEO 仿真（1 天）验证集成"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False
    config["elements"] = [42164000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config["spacecraft_mass"] = 2000.0
    config["enable_atmospheric_drag"] = False
    config["enable_j2"] = True
    config["use_j2_generator"] = False
    config["control_gain_scale"] = 5e-9
    config["verbose"] = False

    sim = GEOL1Simulation(config)
    success = sim.run()
    assert success

    h5_file = sim.h5_file
    assert os.path.exists(h5_file)
