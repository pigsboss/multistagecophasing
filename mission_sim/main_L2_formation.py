# main_L2_formation.py
import os
import numpy as np
from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.environment import CelestialEnvironment
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.core.ground_station import GroundStation
from mission_sim.core.gnc_subsystem import GNC_Subsystem
from mission_sim.core.formation_manager import FormationManager
from mission_sim.utils.math_tools import get_lqr_gain, get_lvlh_dcm
from mission_sim.utils.loggers import HDF5Logger

class L2FormationSimulation:
    def __init__(self):
        # --- 仿真基础参数 ---
        self.dt = 0.05
        self.sim_time = 1000.0
        self.steps = int(self.sim_time / self.dt)
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        
        # --- 物理域：目标与初始状态定义 ---
        # 1. 定义绝对参考：日地 L2 点
        self.abs_L2_point = np.array([1.511e11, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 2. 定义纯相对目标：从星相对于主星的 LVLH 矢量 (Logic-Consistent)
        # 目标：Y轴（切向）前方 100m
        self.target_rel_deputy = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]) 
        
        # 3. 初始状态注入
        # 主星偏差：偏离 L2 点 X轴 500m, Y轴 100m
        init_chief_state = self.abs_L2_point + np.array([500.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        
        # 从星偏差：基于主星实时位置 + 目标距离 + 50m/20m 的控制扰动
        init_rel_disturbance = np.array([50.0, -20.0, 0.0, 0.0, 0.0, 0.0])
        init_deputy_state = init_chief_state + self.target_rel_deputy + init_rel_disturbance

        # 4. 物理实体实例化
        self.chief_sc = SpacecraftPointMass("Chief", init_chief_state, self.frame, 1000.0)
        self.deputy_sc = SpacecraftPointMass("Deputy_1", init_deputy_state, self.frame, 500.0)
        
        # --- 信息域：GNC 与 编队管理 ---
        self.gs = GroundStation("DSN_Network", self.frame)
        self.formation_net = FormationManager(self.chief_sc)
        self.formation_net.add_deputy("Deputy_1", self.deputy_sc)
        
        # 主星运行在绝对系，从星运行在相对系
        self.gnc_chief = GNC_Subsystem("Chief", self.frame)
        self.gnc_deputy = GNC_Subsystem("Deputy_1", CoordinateFrame.LVLH)
        
        # 求解 LQR 增益 (稍微调强 Q 矩阵以获得更好的刚度)
        self.K_abs = self._compute_lqr_gain(mass=1000.0, Q_val=10.0)
        self.K_rel = self._compute_lqr_gain(mass=500.0, Q_val=100.0) 
        
        # 环境与日志
        self.env = CelestialEnvironment(region="SUN_EARTH_L2")
        os.makedirs("data", exist_ok=True)
        self.logger = HDF5Logger("data/L2_formation_data.h5", flush_interval=500)
        self._setup_metadata()

    def _compute_lqr_gain(self, mass, Q_val):
        gamma_L, omega = 3.9405, 1.991e-7  
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 0], A[4, 1], A[5, 2] = (2*gamma_L+1)*omega**2, (1-gamma_L)*omega**2, -gamma_L*omega**2
        A[3, 4], A[4, 3] = 2*omega, -2*omega
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / mass
        
        # Q 矩阵：位置误差权重 Q_val，速度误差权重 1e4
        Q = np.diag([Q_val, Q_val, Q_val, 1e4, 1e4, 1e4])
        R = np.diag([1.0, 1.0, 1.0]) 
        return get_lqr_gain(A, B, Q, R)

    def _setup_metadata(self):
        self.logger.set_metadata("sim_config", {"dt": self.dt})
        self.logger.set_metadata("targets", {"rel_deputy": self.target_rel_deputy})

    def execute(self):
        print(f"🚀 L2 Formation (Pure Relative Mode) Started...")
        
        # 上行指令：主星追绝对点，从星追相对矢量
        cmd_c = self.gs.generate_telecommand("ORBIT_MAINTENANCE", self.abs_L2_point, self.frame)
        cmd_d = self.gs.generate_telecommand("FORMATION_KEEPING", self.target_rel_deputy, CoordinateFrame.LVLH)
        
        self.gnc_chief.process_telecommand(cmd_c)
        self.gnc_deputy.process_telecommand(cmd_d)
        for step in range(self.steps):
            # 1. 相对/绝对感知
            obs_c, _ = self.gs.track_spacecraft(self.chief_sc.state, self.frame)
            rel_state_lvlh = self.formation_net.get_lvlh_relative_state("Deputy_1")
            
            # 2. GNC 决策 (解耦运行)
            self.gnc_chief.update_navigation(obs_c, self.frame)
            self.gnc_deputy.update_navigation(rel_state_lvlh, CoordinateFrame.LVLH)
            
            f_c_cmd, _ = self.gnc_chief.compute_control_force(self.K_abs)
            f_d_cmd, _ = self.gnc_deputy.compute_control_force(self.K_rel)
            
            # 3. 坐标逆变换 (将从星推力转回物理系)
            dcm_r2l = get_lvlh_dcm(self.chief_sc.position, self.chief_sc.velocity)
            f_d_phys = dcm_r2l.T @ f_d_cmd 
            
            # 4. 物理积分
            self.chief_sc.apply_thrust(f_c_cmd, self.frame)
            self.deputy_sc.apply_thrust(f_d_phys, self.frame)
            
            g_c, _ = self.env.get_gravity_acceleration(self.chief_sc.state, self.frame)
            g_d, _ = self.env.get_gravity_acceleration(self.deputy_sc.state, self.frame)
            
            self.chief_sc.state += self.chief_sc.get_derivative(g_c, self.frame) * self.dt
            self.deputy_sc.state += self.deputy_sc.get_derivative(g_d, self.frame) * self.dt
            
            self.chief_sc.clear_thrust()
            self.deputy_sc.clear_thrust()
            
            # 5. 记录
            self.logger.log("Chief", "state", self.chief_sc.state)
            self.logger.log("Deputy_1", "state", self.deputy_sc.state)
            self.logger.log("Formation", "rel_state_lvlh", rel_state_lvlh)
            self.logger.step()

        self.logger.close()
        print("✅ Simulation Finished. Analyzing data...")

if __name__ == "__main__":
    sim = L2FormationSimulation()
    sim.execute()
    
    # 自动触发多级可视化
    from mission_sim.utils.visualizer_L1 import L1Visualizer
    from mission_sim.utils.visualizer_L2 import L2Visualizer
    
    path = "data/L2_formation_data.h5"
    L1Visualizer(path).plot_absolute_convergence("Chief", "data/L1_chief_abs_error.png")
    L2Visualizer(path).plot_formation_convergence("data/L2_relative_error.png")
