import unittest
import numpy as np
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.core.environment import CelestialEnvironment
from mission_sim.core.ground_station import GroundStation
from mission_sim.core.gnc_subsystem import GNC_Subsystem
from mission_sim.utils.math_tools import get_lqr_gain

class TestL1Integration(unittest.TestCase):
    """
    L1 级集成测试 (v2.0): 基于 LQR 最优控制的绝对轨道维持验证。
    验证在强耦合 CRTBP 环境下，三轴是否能同步收敛。
    """
    def setUp(self):
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.dt = 0.1
        self.sc_mass = 1000.0
        
        self.env = CelestialEnvironment(region="SUN_EARTH_L2")
        
        # 【修复核心】：目标点的速度必须严格为零！
        self.target_state = np.array([1.511e11, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 初始偏差 (X偏移1000m，Y偏移200m，X还带有100m/s速度残差)
        initial_state = self.target_state + np.array([1000.0, 200.0, 0.0, 100.0, 0.0, 0.0])
        self.sc = SpacecraftPointMass("Alpha", initial_state, self.frame, initial_mass=self.sc_mass)
        
        self.gs = GroundStation("DSN", self.frame, pos_noise_std=0.01)
        self.gnc = GNC_Subsystem("Alpha", self.frame)
        self.K = self._generate_lqr_k()

    def _generate_lqr_k(self):
        gamma_L = 3.9405
        omega = 1.991e-7
        omega2 = omega**2
        
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 0] = (2 * gamma_L + 1) * omega2
        A[4, 1] = (1 - gamma_L) * omega2
        A[5, 2] = -gamma_L * omega2
        A[3, 4] = 2 * omega
        A[4, 3] = -2 * omega
        
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.sc_mass
        
        # 稍微增大 R，使推力不至于过猛，保障平稳收敛
        Q = np.diag([1.0, 1.0, 1.0, 1e4, 1e4, 1e4])
        R = np.diag([1e4, 1e4, 1e4]) 
        
        return get_lqr_gain(A, B, Q, R)

    def test_L1_optimal_convergence(self):
        """执行 50000 步仿真，验证 LQR 是否实现三轴全收敛"""
        initial_error_norm = np.linalg.norm(self.sc.state - self.target_state)
        history = []
        # 如果 K[0,0] 超过了 10.0，在 dt=0.1s 下非常容易发散
        for step in range(50000):
            # 测控与指令推送
            obs, obs_f = self.gs.track_spacecraft(self.sc.state, self.sc.frame)
            cmd = self.gs.generate_telecommand("ORBIT_MAINTENANCE", self.target_state, self.frame)
            
            # GNC 逻辑
            self.gnc.process_telecommand(cmd)
            self.gnc.update_navigation(obs, obs_f)
            # LQR 控制律应用
            force, force_f = self.gnc.compute_control_force(self.K)
            
            # 动力学步进
            self.sc.apply_thrust(force, force_f)
            grav, grav_f = self.env.get_gravity_acceleration(self.sc.state, self.sc.frame)
            deriv = self.sc.get_derivative(grav, grav_f)
            
            self.sc.state += deriv * self.dt
            self.sc.clear_thrust()
            
            current_error = np.linalg.norm(self.sc.state - self.target_state)
            history.append(current_error)

        final_error_norm = history[-1]
        
        # 断言：最终误差必须远小于初始误差
        print(f"\n[L1 LQR Test] Initial Norm: {initial_error_norm:.2f}m")
        print(f"\n[L1 LQR Test] Final Norm: {final_error_norm:.2f}m")
        
        self.assertLess(final_error_norm, initial_error_norm * 0.1, 
                       "LQR 未能在预定时间内实现 90% 以上的收敛")

if __name__ == '__main__':
    unittest.main()
