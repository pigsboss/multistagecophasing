#!/usr/bin/env python3
import numpy as np
from mission_sim.simulation.formation_simulation import FormationSimulation

config = {
    "mission_name": "MyFormation",
    "simulation_days": 0.5,
    "time_step": 5.0,
    "data_dir": "data",
    "verbose": True,
    "chief_initial_state": [7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0],
    "deputy_initial_states": [
        ("DEP1", [7000e3, 100.0, 0.0, 0.0, 7.5e3, 0.0]),
        ("DEP2", [7000e3, -50.0, 50.0, 0.0, 7.5e3, 0.0]),
    ],
    "chief_frame": "J2000_ECI",
    "chief_mass_kg": 2000.0,
    "deputy_mass_kg": 500.0,
    "orbit_angular_rate": 0.001071,  # 根据轨道计算
    "thruster_min_thrust_n": 0.0,
    "thruster_max_thrust_n": 50.0,   # 与闭环设计量级匹配（WP-2.1 接通）
    "control_gain_scale": 1e5,
}

sim = FormationSimulation(config)
success = sim.run()
if success:
    print(f"输出文件: {sim.h5_file}")