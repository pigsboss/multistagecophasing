# main.py
import os
import numpy as np

# 导入契约与核心领域模型
from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.environment import CelestialEnvironment
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.core.ground_station import GroundStation
from mission_sim.core.gnc_subsystem import GNC_Subsystem

# 导入基础设施
from mission_sim.utils.math_tools import get_lqr_gain
from mission_sim.utils.loggers import HDF5Logger
from mission_sim.utils.visualizer import L1Visualizer

class SimulationRunner:
    """
    MCPC 仿真引擎入口
    负责读取配置、组装 Core 组件、调用 Utils 工具，驱动时间步进主循环。
    """
    def __init__(self):
        # --- 1. 仿真全局配置 ---
        self.dt = 0.1                     # 积分步长 (s)
        self.sim_time = 2000.0            # 总仿真时间 (s)
        self.steps = int(self.sim_time / self.dt)
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        
        # --- 2. 物理域初始化 (Physical Domain) ---
        self.env = CelestialEnvironment(region="SUN_EARTH_L2")
        
        # 目标绝对静止 (位置固定在L2，相对速度为0)
        self.target_state = np.array([1.511e11, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 初始注入偏差：X 偏移 1000m，Y 偏移 200m
        init_state = self.target_state + np.array([1000.0, 200.0, 0.0, 0.0, 0.0, 0.0])
        
        self.sc = SpacecraftPointMass(
            sc_id="Chief_Alpha", 
            initial_state=init_state, 
            frame=self.frame, 
            initial_mass=1000.0
        )
        
        # --- 3. 信息域初始化 (Information Domain) ---
        self.gs = GroundStation(name="DSN_Network", operating_frame=self.frame, pos_noise_std=0.01)
        self.gnc = GNC_Subsystem(sc_id="Chief_Alpha", operating_frame=self.frame)
        
        # --- 4. 最优控制律解算 ---
        self.K = self._compute_optimal_gain()
        
        # --- 5. 数据持久化初始化 ---
        os.makedirs("data", exist_ok=True)
        self.logger = HDF5Logger(filepath="data/L1_simulation_data.h5", flush_interval=500)
        self._setup_logger_metadata()

    def _compute_optimal_gain(self):
        """构造物理量纲下的 A/B 矩阵，求解 LQR 增益"""
        # 日地系统真实物理常数
        gamma_L = 3.9405  
        omega = 1.991e-7  
        omega2 = omega**2 
        
        # 状态空间方程: dx = Ax + Bu
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 0] = (2 * gamma_L + 1) * omega2
        A[4, 1] = (1 - gamma_L) * omega2
        A[5, 2] = -gamma_L * omega2
        A[3, 4] = 2 * omega
        A[4, 3] = -2 * omega
        
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.sc.mass
        
        # 权重配置 (针对真实物理量级的调优)
        Q = np.diag([1.0, 1.0, 1.0, 1e4, 1e4, 1e4])
        R = np.diag([1.0, 1.0, 1.0]) 
        
        K = get_lqr_gain(A, B, Q, R)
        print(f"[Init] LQR Optimal Gain Matrix Synthesized. K[0,0] = {K[0,0]:.4e}")
        return K

    def _setup_logger_metadata(self):
        """记录仿真元数据"""
        meta = {
            "dt": self.dt,
            "sim_time": self.sim_time,
            "target_state": self.target_state,
            "control_method": "LQR_Optimal"
        }
        self.logger.set_metadata("simulation_info", meta)

    def execute_time_loop(self):
        """核心主循环：二维正交解耦的实战演练"""
        print(f"\n🚀 Starting L1 Simulation: {self.steps} steps...")
        
        for step in range(self.steps):
            current_time = step * self.dt
            
            # --- 数据记录 ---
            self.logger.log("Spacecraft", "state", self.sc.state)
            
            # --- 信息域交互 ---
            # 1. 测站观测真实物理状态并加噪
            obs_state, obs_frame = self.gs.track_spacecraft(self.sc.state, self.sc.frame)
            
            # 2. 生成带坐标系契约的上行指令
            cmd = self.gs.generate_telecommand(
                cmd_type="ORBIT_MAINTENANCE", 
                target_state=self.target_state, 
                target_frame=self.frame
            )
            
            # 3. GNC 大脑处理指令与导航
            self.gnc.process_telecommand(cmd)
            self.gnc.update_navigation(obs_state, obs_frame)
            
            # 4. 基于 LQR 计算推力
            force_vector, force_frame = self.gnc.compute_control_force(self.K)
            self.logger.log("GNC", "thrust_cmd", force_vector)
            
            # --- 物理域演化 ---
            # 1. 物理引擎接受外部推力 (内部强制校验 CoordinateFrame)
            self.sc.apply_thrust(force_vector, force_frame)
            
            # 2. 获取宇宙环境的真实加速度
            grav_accel, grav_frame = self.env.get_gravity_acceleration(self.sc.state, self.sc.frame)
            
            # 3. 动力学积分
            deriv = self.sc.get_derivative(grav_accel, grav_frame)
            self.sc.state += deriv * self.dt
            
            # 4. 清理上一步作动器状态
            self.sc.clear_thrust()

            # 进度打印
            if step % 2000 == 0 and step > 0:
                err_norm = np.linalg.norm(self.sc.state - self.target_state)
                print(f"  [{current_time:6.1f}s] Error Norm: {err_norm:8.2f} m")

        # --- 收尾与持久化 ---
        self.logger.flush()
        self.logger.close()
        
        final_err = np.linalg.norm(self.sc.state - self.target_state)
        print(f"✅ Simulation Complete! Final Error: {final_err:.2f} m")

    def run_post_processing(self):
        """调用可视化基础设施"""
        print("\n📊 Generating visualization assets...")
        vis = L1Visualizer(filepath="data/L1_simulation_data.h5", sc_id="Chief_Alpha")
        
        # 1. 轨迹收敛图
        vis.plot_state_history(save_path="data/L1_trajectory.png")
        
        # 2. 控制指令图
        vis.plot_gnc_activity(save_path="data/L1_gnc_thrust.png")
        
        # 3. 3D 动态渲染 (降采样 100 倍，即 20000 帧抽取 200 帧生成 GIF)
        vis.create_animation(save_path="data/L1_animation.gif", downsample=100)
        
        print("🎉 All tasks finished.")

if __name__ == "__main__":
    runner = SimulationRunner()
    runner.execute_time_loop()
    runner.run_post_processing()
