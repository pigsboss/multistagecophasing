import unittest
import numpy as np
from mission_sim.core.environment import CelestialEnvironment
from mission_sim.core.types import CoordinateFrame

class TestEnvironment(unittest.TestCase):
    def test_environment_init(self):
        env = CelestialEnvironment(region="SUN_EARTH_L2")
        # 验证日地质量比 mu 是否在合理范围 (~3e-6)
        self.assertLess(env.mu, 4e-6)
        self.assertGreater(env.mu, 2e-6)

    def test_gravity_output(self):
        env = CelestialEnvironment(region="SUN_EARTH_L2")
        state = np.array([1.5e11, 0, 0, 0, 0, 0])
        
        accel, frame = env.get_gravity_acceleration(state, CoordinateFrame.SUN_EARTH_ROTATING)
        
        self.assertEqual(len(accel), 3)
        self.assertEqual(frame, CoordinateFrame.SUN_EARTH_ROTATING)
        # 验证在 L2 附近的加速度不应为无穷大或 NaN
        self.assertTrue(np.all(np.isfinite(accel)))

if __name__ == '__main__':
    unittest.main()
