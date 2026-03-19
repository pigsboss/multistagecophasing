import unittest
import numpy as np
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.core.types import CoordinateFrame

class TestSpacecraft(unittest.TestCase):
    def setUp(self):
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.sc = SpacecraftPointMass(
            sc_id="TestSat", 
            initial_state=[1.5e11, 0, 0, 0, 0, 0], 
            frame=self.frame
        )

    def test_coordinate_protection(self):
        """测试坐标系强校验机制：非法坐标系推力应被拦截"""
        wrong_frame = CoordinateFrame.J2000_ECI
        force = np.array([1.0, 0, 0])
        
        # 验证是否抛出 ValueError
        with self.assertRaises(ValueError):
            self.sc.apply_thrust(force, wrong_frame)

    def test_mass_consumption(self):
        """测试质量消耗逻辑"""
        initial_mass = self.sc.mass
        self.sc.consume_mass(m_dot=0.1, dt=10.0)
        self.assertEqual(self.sc.mass, initial_mass - 1.0)

    def test_derivative_logic(self):
        """测试状态导数计算"""
        grav_accel = np.array([0.1, 0.2, 0.3])
        # 模拟施加 100N 推力
        self.sc.mass = 100.0
        self.sc.apply_thrust([100.0, 0, 0], self.frame)
        
        deriv = self.sc.get_derivative(grav_accel, self.frame)
        # 验证速度项 [0:3] 对应导数中的速度部分
        # 验证加速度项 [3:6] 应为 引力(0.1) + 推力产生的加速度(1.0) = 1.1
        self.assertEqual(deriv[3], 1.1)

if __name__ == '__main__':
    unittest.main()
