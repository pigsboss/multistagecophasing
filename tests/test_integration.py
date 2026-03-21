import pytest
from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation
from mission_sim.utils.logger import HDF5Logger
import os

def test_short_simulation(temp_dir, default_config):
    """运行简短仿真（1 天）验证集成"""
    config = default_config.copy()
    config["data_dir"] = str(temp_dir)
    config["simulation_days"] = 1
    config["time_step"] = 60.0
    config["enable_visualization"] = False

    sim = SunEarthL2L1Simulation(config)
    success = sim.run()
    assert success

    # 检查输出文件
    h5_file = sim.h5_file
    assert os.path.exists(h5_file)
    # 检查燃料账单
    fuel_bill = os.path.join(temp_dir, f"fuel_bill_{sim.mission_id}.csv")
    assert os.path.exists(fuel_bill)