import unittest
import numpy as np
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.core.environment import CelestialEnvironment
from mission_sim.core.ground_station import GroundStation
from mission_sim.core.gnc_subsystem import GNC_Subsystem

class TestL1Integration(unittest.TestCase):
    """
    Level 1 集成测试：验证单星绝对轨道维持的物理/信息全闭环。
    """
    def setUp(self):
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.dt = 1.0
        
        # 初始化物理域
        self.env = CelestialEnvironment(region="SUN_EARTH_L2")
        # 设定目标点 (L2 理论位置)
        self.target_state = np.array([1.511e11, 0, 0, 0, 178.5, 0])
        # 初始偏差：1000m
        initial_state = self.target_state + np.array([1000.0, 0, 0, 0, 0, 0])
        self.sc = SpacecraftPointMass("Alpha", initial_state, self.frame)
        
        # 初始化信息域
        self.gs = GroundStation("DSN", self.frame, pos_noise_std=0.1) # 低噪声便于验证收敛
        self.gnc = GNC_Subsystem("Alpha", self.frame)
        
        # 简单比例增益
        self.K = np.zeros((3, 6))
        self.K[0, 0] = 0.01  # X轴位置增益

    def test_closed_loop_convergence(self):
        """测试闭环控制：验证 100 步后偏差是否减小"""
        initial_error = np.linalg.norm(self.sc.state[0:3] - self.target_state[0:3])
        
        # 运行 100 步微型仿真
        for _ in range(100):
            # 1. 测控与指令
            obs, obs_f = self.gs.track_spacecraft(self.sc.state, self.sc.frame)
            cmd = self.gs.generate_telecommand("ORBIT_MAINTENANCE", self.target_state, self.frame)
            
            # 2. GNC 解算
            self.gnc.process_telecommand(cmd)
            self.gnc.update_navigation(obs, obs_f)
            force, force_f = self.gnc.compute_control_force(self.K)
            
            # 3. 动力学执行
            self.sc.apply_thrust(force, force_f)
            grav, grav_f = self.env.get_gravity_acceleration(self.sc.state, self.sc.frame)
            deriv = self.sc.get_derivative(grav, grav_f)
            
            self.sc.state += deriv * self.dt
            self.sc.clear_thrust()

        final_error = np.linalg.norm(self.sc.state[0:3] - self.target_state[0:3])
        
        # 验证偏差是否收敛
        self.assertLess(final_error, initial_error, "L1 闭环控制未能减小轨道偏差")
        print(f"\n[L1 Integration] Initial Error: {initial_error:.2f}m -> Final Error: {final_error:.2f}m")

if __name__ == '__main__':
    unittest.main()
