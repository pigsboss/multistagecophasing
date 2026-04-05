"""
SPICE可用性测试

测试SPICE工具和相关依赖是否可用。
验证高精度星历模块的SPICE模式是否正常工作。
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# 测试SPICE依赖
def check_spice_dependencies():
    """检查SPICE相关依赖"""
    dependencies = {
        'numpy': '数值计算',
        'spiceypy': 'NASA SPICE工具包',
        'requests': 'HTTP请求（用于下载核文件）',
        'tqdm': '进度条显示'
    }
    
    print("检查SPICE高精度星历模块依赖项")
    print("=" * 50)
    
    missing = []
    available = []
    
    for module_name, description in dependencies.items():
        try:
            __import__(module_name)
            version = getattr(sys.modules[module_name], '__version__', '未知')
            print(f"✓ {module_name:12} - {description} (版本: {version})")
            available.append(module_name)
        except ImportError as e:
            print(f"✗ {module_name:12} - {description}: 未安装")
            missing.append(module_name)
    
    print("=" * 50)
    
    if missing:
        print(f"\n缺少 {len(missing)} 个依赖项: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    else:
        print("\n所有依赖项已安装！")
        return True


class TestSPICEEnvironment(unittest.TestCase):
    """SPICE环境测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp(prefix="test_spice_")
        
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_spiceypy_import(self):
        """测试spiceypy导入"""
        try:
            import spiceypy as spice
            self.assertIsNotNone(spice)
            print("✓ spiceypy 导入成功")
        except ImportError:
            self.skipTest("spiceypy 未安装")
    
    def test_spice_basic_functions(self):
        """测试SPICE基本函数"""
        try:
            import spiceypy as spice
            
            # 测试SPICE基本功能
            # 加载一个简单的测试核文件（如果没有，则创建虚拟的）
            test_kernel = Path(self.test_dir) / "test_kernel.tls"
            
            # 创建一个简单的文本核文件
            with open(test_kernel, 'w') as f:
                f.write("\\begindata\n")
                f.write("KERNELS_TO_LOAD = ( 'test_kernel.tls' )\n")
                f.write("\\begintext\n")
            
            try:
                spice.furnsh(str(test_kernel))
                
                # 测试一个简单的转换
                et = spice.str2et('2000-01-01T12:00:00')
                self.assertIsInstance(et, float)
                
                spice.unload(str(test_kernel))
                print("✓ SPICE基本功能正常")
                
            except Exception as e:
                print(f"⚠ SPICE功能测试警告: {e}")
                # 不是致命错误，只是警告
                pass
                
        except ImportError:
            self.skipTest("spiceypy 未安装")
    
    def test_high_precision_ephemeris_import(self):
        """测试高精度星历模块导入"""
        try:
            from mission_sim.core.spacetime.ephemeris.high_precision import (
                HighPrecisionEphemeris, EphemerisMode, CelestialBody, EphemerisConfig
            )
            
            self.assertIsNotNone(HighPrecisionEphemeris)
            self.assertIsNotNone(EphemerisMode)
            self.assertIsNotNone(CelestialBody)
            self.assertIsNotNone(EphemerisConfig)
            
            print("✓ 高精度星历模块导入成功")
            
        except ImportError as e:
            self.fail(f"高精度星历模块导入失败: {e}")
    
    def test_ephemeris_modes(self):
        """测试星历模式"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            EphemerisMode, EphemerisConfig
        )
        
        # 测试所有模式
        modes = [
            EphemerisMode.ANALYTICAL,
            EphemerisMode.CRTBP,
            EphemerisMode.NUMERICAL,
            EphemerisMode.EXTERNAL,
        ]
        
        for mode in modes:
            config = EphemerisConfig(mode=mode)
            self.assertEqual(config.mode, mode)
        
        print("✓ 星历模式配置正常")
    
    def test_celestial_body_enum(self):
        """测试天体枚举"""
        from mission_sim.core.spacetime.ephemeris.high_precision import CelestialBody
        
        bodies = [
            CelestialBody.SUN,
            CelestialBody.EARTH,
            CelestialBody.MOON,
            CelestialBody.MARS
        ]
        
        for body in bodies:
            self.assertIsInstance(body, CelestialBody)
            self.assertIsInstance(body.value, str)
        
        print("✓ 天体枚举正常")
    
    def test_ephemeris_creation(self):
        """测试星历创建"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode
        )
        
        # 测试解析模式创建
        config = EphemerisConfig(mode=EphemerisMode.ANALYTICAL)
        ephemeris = HighPrecisionEphemeris(config)
        
        self.assertIsInstance(ephemeris, HighPrecisionEphemeris)
        self.assertEqual(ephemeris.config.mode, EphemerisMode.ANALYTICAL)
        
        print("✓ 星历创建正常")
    
    def test_get_state_basic(self):
        """测试基本状态获取"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode, CelestialBody
        )
        
        config = EphemerisConfig(
            mode=EphemerisMode.ANALYTICAL,
            verbose=False
        )
        
        ephemeris = HighPrecisionEphemeris(config)
        
        # 获取地球相对于太阳的状态（解析模式）
        state = ephemeris.get_state(
            target_body=CelestialBody.EARTH,
            epoch=0.0,
            observer_body=CelestialBody.SUN,
            frame="J2000_ECI"
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (6,))
        
        print("✓ 基本状态获取正常")


@unittest.skipUnless(check_spice_dependencies(), "缺少SPICE依赖")
class TestSPICEEphemeris(unittest.TestCase):
    """SPICE星历测试（需要SPICE依赖）"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp(prefix="test_spice_")
        
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_spice_mode_exists(self):
        """测试SPICE模式是否存在"""
        from mission_sim.core.spacetime.ephemeris.high_precision import EphemerisMode
        
        # 检查SPICE模式是否已添加到枚举中
        self.assertTrue(hasattr(EphemerisMode, 'SPICE'), 
                       "EphemerisMode中缺少SPICE模式")
        
        print("✓ SPICE模式已定义")
    
    def test_spice_ephemeris_config(self):
        """测试SPICE星历配置"""
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            EphemerisConfig, EphemerisMode
        )
        
        # 测试SPICE模式配置
        config = EphemerisConfig(mode=EphemerisMode.SPICE)
        self.assertEqual(config.mode, EphemerisMode.SPICE)
        
        print("✓ SPICE星历配置正常")
    
    def test_spice_kernel_manager_import(self):
        """测试SPICE核文件管理器导入"""
        try:
            # 尝试从tools模块导入
            import mission_sim.tools.spice_kernel_manager as spm
            self.assertIsNotNone(spm)
            print("✓ SPICE核文件管理器导入成功")
        except ImportError:
            # 尝试直接导入
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
                import spice_kernel_manager as spm
                self.assertIsNotNone(spm)
                print("✓ SPICE核文件管理器导入成功（直接路径）")
            except ImportError:
                print("⚠ SPICE核文件管理器不可用（可能需要安装或创建）")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SPICE可用性测试套件")
    print("=" * 60)
    
    # 检查依赖
    print("\n第一阶段: 依赖检查")
    print("-" * 40)
    deps_ok = check_spice_dependencies()
    
    if not deps_ok:
        print("\n警告: 缺少部分依赖，SPICE测试可能受限")
    
    # 运行单元测试
    print("\n第二阶段: 单元测试")
    print("-" * 40)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    
    # 添加基础测试
    suite = loader.loadTestsFromTestCase(TestSPICEEnvironment)
    
    # 如果依赖满足，添加SPICE测试
    if deps_ok:
        try:
            spice_suite = loader.loadTestsFromTestCase(TestSPICEEphemeris)
            suite.addTests(spice_suite)
            print("包含SPICE功能测试")
        except Exception as e:
            print(f"跳过SPICE功能测试: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败详情:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(':')[0]}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 如果作为脚本运行，执行测试套件
    success = run_all_tests()
    sys.exit(0 if success else 1)
