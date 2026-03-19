import unittest
import numpy as np
from mission_sim.core.gnc_subsystem import GNC_Subsystem
from mission_sim.core.types import CoordinateFrame

class TestGNC(unittest.TestCase):
    def setUp(self):
        self.frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.gnc = GNC_Subsystem("Alpha", self.frame)

    def test_telecommand_rejection(self):
        """测试 GNC 是否拒绝坐标系不匹配的地面指令"""
        bad_packet = {
            "header": {"type": "ORBIT_MAINTENANCE"},
            "payload": {
                "target_state": np.zeros(6),
                "frame": CoordinateFrame.J2000_ECI # 错误的系
            }
        }
        with self.assertRaises(ValueError):
            self.gnc.process_telecommand(bad_packet)

    def test_control_output(self):
        """测试简单比例控制输出"""
        self.gnc.target_state = np.zeros(6)
        self.gnc.estimated_state = np.array([100.0, 0, 0, 0, 0, 0]) # 100m 偏差
        
        K = np.zeros((3, 6))
        K[0, 0] = 0.01 # P 增益
        
        force, frame = self.gnc.compute_control_force(K)
        # u = -K * error = -0.01 * 100 = -1.0 N
        self.assertEqual(force[0], -1.0)

if __name__ == '__main__':
    unittest.main()
