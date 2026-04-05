"""
SPICE高精度星历使用示例

展示如何使用HighPrecisionEphemeris的SPICE模式进行高精度星历计算。

运行前请确保已安装依赖:
    pip install spiceypy requests tqdm
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("SPICE高精度星历使用示例")
print("=" * 70)
print()


def check_dependencies():
    """检查依赖项"""
    print("检查依赖项...")
    dependencies = ['numpy', 'spiceypy', 'requests', 'tqdm']
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("所有依赖项已安装！\n")
    return True


def example_basic_usage():
    """示例1: 基本使用"""
    print("示例1: 基本使用")
    print("-" * 40)
    
    try:
        # 导入高精度星历模块
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode, CelestialBody,
            create_high_precision_ephemeris
        )
        
        # 创建星历实例（解析模式 - 无需下载核文件）
        print("创建高精度星历实例（解析模式）...")
        config = EphemerisConfig(
            mode=EphemerisMode.ANALYTICAL,
            verbose=True
        )
        
        ephemeris = HighPrecisionEphemeris(config)
        
        # 获取地球相对于太阳的状态
        epoch = 0.0  # J2000历元
        earth_state = ephemeris.get_state(
            target_body=CelestialBody.EARTH,
            epoch=epoch,
            observer_body=CelestialBody.SUN,
            frame="J2000_ECI"
        )
        
        print(f"地球在J2000历元时的状态（相对于太阳）:")
        print(f"  位置: [{earth_state[0]:.3e}, {earth_state[1]:.3e}, {earth_state[2]:.3e}] m")
        print(f"  速度: [{earth_state[3]:.3e}, {earth_state[4]:.3e}, {earth_state[5]:.3e}] m/s")
        print(f"  距离: {np.linalg.norm(earth_state[:3]):.3e} m ≈ 1 AU")
        
        # 获取月球相对于地球的状态
        moon_state = ephemeris.get_state(
            target_body=CelestialBody.MOON,
            epoch=epoch,
            observer_body=CelestialBody.EARTH,
            frame="J2000_ECI"
        )
        
        print(f"\n月球在J2000历元时的状态（相对于地球）:")
        print(f"  位置: [{moon_state[0]:.3e}, {moon_state[1]:.3e}, {moon_state[2]:.3e}] m")
        print(f"  距离: {np.linalg.norm(moon_state[:3]):.3e} m")
        
        return ephemeris
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_time_series():
    """示例2: 时间序列计算"""
    print("\n示例2: 时间序列计算")
    print("-" * 40)
    
    try:
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            create_high_precision_ephemeris, CelestialBody
        )
        
        # 使用工厂函数创建星历
        print("创建高精度星历实例...")
        ephemeris = create_high_precision_ephemeris(mode="analytical")
        
        # 计算一个月内的地月距离变化
        days = 30
        seconds_per_day = 24 * 3600
        num_points = 100
        
        times = np.linspace(0, days * seconds_per_day, num_points)
        distances = []
        
        print(f"计算 {days} 天内的地月距离变化 ({num_points} 个点)...")
        
        for i, t in enumerate(times):
            moon_state = ephemeris.get_state(
                target_body=CelestialBody.MOON,
                epoch=t,
                observer_body=CelestialBody.EARTH,
                frame="J2000_ECI"
            )
            distance = np.linalg.norm(moon_state[:3])
            distances.append(distance)
            
            # 显示进度
            if i % 20 == 0:
                print(f"  进度: {i+1}/{num_points}")
        
        distances_array = np.array(distances)
        
        # 绘制结果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(times / seconds_per_day, distances_array)
        plt.xlabel('时间 (天)')
        plt.ylabel('地月距离 (m)')
        plt.title(f'{days} 天内地月距离变化')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(distances_array, bins=20, edgecolor='black')
        plt.xlabel('地月距离 (m)')
        plt.ylabel('频数')
        plt.title('地月距离分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "earth_moon_distance_analysis.png", dpi=150)
        print(f"\n图形已保存到: {output_dir / 'earth_moon_distance_analysis.png'}")
        
        # 显示统计信息
        print(f"\n地月距离统计 ({days} 天):")
        print(f"  平均值: {np.mean(distances_array):.3e} m")
        print(f"  最小值: {np.min(distances_array):.3e} m (第 {np.argmin(distances_array)} 个点)")
        print(f"  最大值: {np.max(distances_array):.3e} m (第 {np.argmax(distances_array)} 个点)")
        print(f"  变化幅度: {np.max(distances_array) - np.min(distances_array):.3e} m")
        
        return distances_array
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_coordinate_frames():
    """示例3: 坐标系转换"""
    print("\n示例3: 坐标系转换")
    print("-" * 40)
    
    try:
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            create_high_precision_ephemeris, CelestialBody
        )
        from mission_sim.core.spacetime.ids import CoordinateFrame
        
        ephemeris = create_high_precision_ephemeris(mode="analytical")
        
        epoch = 0.0
        
        # 测试不同坐标系
        frames_to_test = ["J2000_ECI", "EME2000", "GCRF"]
        
        print("月球在不同坐标系下的位置 (J2000历元):")
        print(f"{'坐标系':<15} {'X (m)':<20} {'Y (m)':<20} {'Z (m)':<20} {'距离 (m)':<20}")
        print("-" * 95)
        
        for frame_name in frames_to_test:
            try:
                state = ephemeris.get_state(
                    target_body=CelestialBody.MOON,
                    epoch=epoch,
                    observer_body=CelestialBody.EARTH,
                    frame=frame_name
                )
                
                distance = np.linalg.norm(state[:3])
                print(f"{frame_name:<15} {state[0]:<20.3e} {state[1]:<20.3e} {state[2]:<20.3e} {distance:<20.3e}")
            except Exception as e:
                print(f"{frame_name:<15} 错误: {str(e)[:40]}...")
        
        # 获取地月旋转系状态
        print(f"\n地月旋转坐标系状态:")
        rotating_state = ephemeris.get_earth_moon_rotating_state(epoch)
        print(f"  位置: [{rotating_state[0]:.3e}, {rotating_state[1]:.3e}, {rotating_state[2]:.3e}] m")
        print(f"  速度: [{rotating_state[3]:.3e}, {rotating_state[4]:.3e}, {rotating_state[5]:.3e}] m/s")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_celestial_bodies():
    """示例4: 不同天体计算"""
    print("\n示例4: 不同天体计算")
    print("-" * 40)
    
    try:
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            create_high_precision_ephemeris, CelestialBody
        )
        
        ephemeris = create_high_precision_ephemeris(mode="analytical")
        
        epoch = 0.0
        
        # 定义天体对
        body_pairs = [
            ("地球-太阳", CelestialBody.EARTH, CelestialBody.SUN),
            ("月球-地球", CelestialBody.MOON, CelestialBody.EARTH),
            ("火星-太阳", CelestialBody.MARS, CelestialBody.SUN),
        ]
        
        print(f"J2000历元时不同天体的相对位置:")
        print(f"{'天体对':<15} {'距离 (m)':<20} {'距离 (AU)':<15} {'速度 (km/s)':<15}")
        print("-" * 65)
        
        for name, target, observer in body_pairs:
            state = ephemeris.get_state(
                target_body=target,
                epoch=epoch,
                observer_body=observer,
                frame="J2000_ECI"
            )
            
            distance_m = np.linalg.norm(state[:3])
            distance_au = distance_m / 1.496e11  # 转换为AU
            velocity_kms = np.linalg.norm(state[3:]) / 1000  # 转换为km/s
            
            print(f"{name:<15} {distance_m:<20.3e} {distance_au:<15.3f} {velocity_kms:<15.3f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_spice_kernel_manager():
    """示例5: SPICE核文件管理器"""
    print("\n示例5: SPICE核文件管理器")
    print("-" * 40)
    
    try:
        # 尝试导入SPICE核文件管理器
        import mission_sim.tools.spice_kernel_manager as spm
        
        print("创建SPICE核文件管理器...")
        
        # 创建临时目录用于测试
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp(prefix="spice_test_")
        kernel_dir = Path(temp_dir) / "kernels"
        
        config = spm.KernelConfig(
            kernel_dir=kernel_dir,
            auto_download=False,  # 测试时不自动下载
            verbose=True
        )
        
        manager = spm.SPICEKernelManager(config)
        
        # 列出核文件
        kernels = manager.list_kernels()
        print(f"\n可用核文件 ({len(kernels)} 个):")
        for k in kernels[:5]:  # 只显示前5个
            status = "已存在" if k['exists'] else "未下载"
            print(f"  {k['id']:20} {k['name']:25} {status:10}")
        
        if len(kernels) > 5:
            print(f"  ... 还有 {len(kernels)-5} 个")
        
        # 获取统计信息
        stats = manager.get_stats()
        print(f"\n核文件统计:")
        print(f"  总核文件: {stats['total_kernels']}")
        print(f"  已下载: {stats['downloaded']}")
        print(f"  核目录: {stats['kernel_dir']}")
        
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n临时目录已清理: {temp_dir}")
        
        print("\n注意: 此示例仅展示了核文件管理器的基本功能。")
        print("在实际使用中，您需要下载核文件才能使用SPICE模式。")
        print("运行: python mission_sim/tools/spice_kernel_manager.py --mission earth_moon")
        
    except ImportError:
        print("SPICE核文件管理器不可用")
        print("请确保文件 mission_sim/tools/spice_kernel_manager.py 存在")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def example_advanced_features():
    """示例6: 高级功能"""
    print("\n示例6: 高级功能")
    print("-" * 40)
    
    try:
        from mission_sim.core.spacetime.ephemeris.high_precision import (
            HighPrecisionEphemeris, EphemerisConfig, EphemerisMode, CelestialBody
        )
        
        # 测试缓存功能
        print("测试缓存功能...")
        
        config = EphemerisConfig(
            mode=EphemerisMode.ANALYTICAL,
            cache_size=10,
            verbose=False
        )
        
        ephemeris = HighPrecisionEphemeris(config)
        
        # 第一次获取（应该计算）
        import time
        start_time = time.time()
        state1 = ephemeris.get_state(
            target_body=CelestialBody.EARTH,
            epoch=0.0,
            observer_body=CelestialBody.SUN,
            frame="J2000_ECI"
        )
        time1 = time.time() - start_time
        
        # 第二次获取（应该从缓存）
        start_time = time.time()
        state2 = ephemeris.get_state(
            target_body=CelestialBody.EARTH,
            epoch=0.0,
            observer_body=CelestialBody.SUN,
            frame="J2000_ECI"
        )
        time2 = time.time() - start_time
        
        print(f"  第一次计算时间: {time1*1000:.2f} ms")
        print(f"  第二次缓存时间: {time2*1000:.2f} ms")
        print(f"  缓存加速: {time1/time2:.1f}x")
        
        # 测试模式切换
        print("\n测试模式切换...")
        ephemeris.set_mode(EphemerisMode.CRTBP)
        print(f"  当前模式: {ephemeris.config.mode.value}")
        
        # 清除缓存
        ephemeris.clear_cache()
        print("  缓存已清除")
        
        # 获取可用天体
        bodies = ephemeris.get_available_bodies()
        print(f"\n支持的天体 ({len(bodies)} 个): {', '.join(bodies)}")
        
        # 获取天体参数
        earth_params = ephemeris.get_body_parameters(CelestialBody.EARTH)
        print(f"\n地球物理参数:")
        for key, value in earth_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3e}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("MCPC SPICE高精度星历示例程序")
    print()
    
    # 检查依赖
    if not check_dependencies():
        print("\n缺少必要依赖，程序退出。")
        return
    
    # 运行示例
    examples = [
        ("基本使用", example_basic_usage),
        ("时间序列计算", example_time_series),
        ("坐标系转换", example_coordinate_frames),
        ("不同天体计算", example_celestial_bodies),
        ("SPICE核文件管理器", example_spice_kernel_manager),
        ("高级功能", example_advanced_features),
    ]
    
    successful = 0
    
    for name, func in examples:
        try:
            result = func()
            if result is not None or name in ["坐标系转换", "不同天体计算", "SPICE核文件管理器", "高级功能"]:
                successful += 1
            print()
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"{name} 失败: {e}")
            print()
    
    # 总结
    print("=" * 70)
    print(f"示例程序完成: {successful}/{len(examples)} 个示例成功")
    print()
    
    # 使用建议
    print("使用建议:")
    print("1. 运行SPICE核文件管理器下载核文件:")
    print("   python mission_sim/tools/spice_kernel_manager.py --mission earth_moon")
    print()
    print("2. 测试SPICE可用性:")
    print("   python tests/test_spice_availability.py")
    print()
    print("3. 运行完整测试套件:")
    print("   python -m pytest tests/ -v")
    print()
    print("4. 扩展SPICE模式:")
    print("   在 high_precision.py 中添加 EphemerisMode.SPICE 枚举")
    print("   并实现 _compute_spice_state 方法")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
        
        # 显示图形（如果有）
        import matplotlib.pyplot as plt
        plt.show()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
