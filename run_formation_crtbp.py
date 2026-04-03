#!/usr/bin/env python3
"""
MCPC L2 级多星编队仿真启动脚本 (SI 单位制修正版)
"""

import numpy as np
import os
from mission_sim.simulation.formation_simulation import FormationSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
# 关键修正：lvl -> lvlh
from mission_sim.utils.math_tools import lvlh_to_absolute, get_lqr_gain
from mission_sim.core.cyber.models.threebody.base import CRTBP
from mission_sim.core.cyber.models.crtbp_relative_dynamics import CRTBPRelativeDynamics

# 1. 定义物理环境参数
mu = 3.00348e-6
L = 1.495978707e11
omega = 1.990986e-7
crtbp = CRTBP(mu, L, omega)

# 2. 定义主星初始状态
chief0 = np.array([
    1.50613280e11,
    0.0,
    1.20000000e8,
    0.0,
    1.51320000e2,
    0.0
])

# 3. 定义从星目标构型 (LVLH 坐标, m)
rel_targets = {
    "DEP1": np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
    "DEP2": np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]),
}

# 4. 计算从星初始绝对状态
rel_initial = {
    "DEP1": np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]),
    "DEP2": np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0]),
}
deputy_initial_configs = []
for dep_id, rel_state in rel_initial.items():
    # 修正后的调用
    r_abs, v_abs = lvlh_to_absolute(chief0[:3], chief0[3:], rel_state[:3], rel_state[3:])
    deputy_initial_configs.append((dep_id, np.concatenate([r_abs, v_abs])))

# 5. 预计算 LQR 控制增益 K
temp_dynamics = CRTBPRelativeDynamics(crtbp)
A_cont = temp_dynamics._linearized_matrix(chief0)
B = np.zeros((6, 3))
B[3:6, 0:3] = np.eye(3)

Q = np.diag([1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0]) 
R = np.diag([1e8, 1e8, 1e8])                    
K = get_lqr_gain(A_cont, B, Q, R)

# 6. 构建仿真配置
config = {
    "mission_name": "Halo_Formation_CRTBP",
    "simulation_days": 0.5,
    "time_step": 10.0,
    "data_dir": "data",
    "verbose": True,
    "chief_initial_state": chief0.tolist(),
    "deputy_initial_states": deputy_initial_configs,
    "formation_targets": rel_targets, 
    "mu": mu,
    "L": L,
    "omega": omega,

    "chief_mass_kg": 2000.0,
    "deputy_mass_kg": 500.0,

    # ========== ISL 天线参数 ==========
    "isl_range_noise_std": 0.01,          # 10 cm 测距噪声
    "isl_angle_noise_std": 1e-4,         # 约 20 角秒
    "isl_max_range_m": 500000.0,         # 500 km 最大测距
    "isl_ref_range_m": 1000.0,

    # ========== 推力器参数 ==========
    "thruster_max_thrust_n": 0.5,
    "thruster_min_thrust_n": 0.001,      # 1 mN 死区
    "thruster_noise_std_n": 0.01,
    "thruster_isp_s": 3000.0,

    # ========== 网络路由器参数 ==========
    "router_base_latency_s": 0.1,
    "router_jitter_s": 0.02,
    "router_packet_loss_rate": 0.05,
    "router_seed": 123,

    # ========== 控制器参数 ==========
    "lqr_gain": K,                    # 自动计算（CW 模型）
    "control_gain_scale": 1.0,           # 增益缩放（可选）
    "generation_threshold_pos": 10.0,
    "generation_threshold_vel": 0.5,
    "keeping_threshold_pos": 0.01,
    "keeping_threshold_vel": 1e-4,
}

if __name__ == "__main__":
    if not os.path.exists(config["data_dir"]):
        os.makedirs(config["data_dir"])
    sim = FormationSimulation(config)
    sim.run()
