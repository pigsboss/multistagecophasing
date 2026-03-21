# tests/test_data_integrity.py
"""
数据完整性、坐标系校验与异常恢复测试
使用 pytest 运行：pytest tests/test_data_integrity.py -v
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import h5py

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.physics.environment import CelestialEnvironment, IForceModel
from mission_sim.core.physics.spacecraft import SpacecraftPointMass
from mission_sim.core.gnc.ground_station import GroundStation
from mission_sim.core.gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.utils.logger import HDF5Logger
from mission_sim.main_L1_runner import L1MissionSimulation


class MockForceModel(IForceModel):
    """用于测试的简单力模型"""
    def compute_accel(self, state, epoch):
        return np.zeros(3)


def test_hdf5_logger_file_creation():
    """测试 HDF5Logger 能正确创建文件并初始化结构"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger = HDF5Logger(filepath=tmp_path, buffer_size=100, auto_flush=False)
        logger.initialize_file({"test": "metadata"})

        # 检查文件是否存在
        assert os.path.exists(tmp_path)

        # 检查数据集是否创建
        with h5py.File(tmp_path, 'r') as f:
            expected_datasets = ['epochs', 'nominal_states', 'true_states', 'nav_states',
                                 'tracking_errors', 'control_forces', 'accumulated_dvs']
            for ds in expected_datasets:
                assert ds in f, f"数据集 {ds} 未创建"
            # 检查元数据
            assert f.attrs.get('test') == 'metadata'
        logger.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_hdf5_logger_log_step():
    """测试 log_step 方法能正确记录数据并刷新缓冲区"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger = HDF5Logger(filepath=tmp_path, buffer_size=2, auto_flush=False)
        logger.initialize_file()

        # 记录 3 条数据，触发 flush
        for i in range(3):
            logger.log_step(
                epoch=float(i),
                nominal_state=np.ones(6) * i,
                true_state=np.ones(6) * i,
                nav_state=np.ones(6) * i,
                tracking_error=np.ones(6) * i,
                control_force=np.ones(3) * i,
                accumulated_dv=float(i)
            )
        logger.close()

        # 验证写入的数据
        with h5py.File(tmp_path, 'r') as f:
            epochs = f['epochs'][:]
            assert len(epochs) == 3
            assert np.all(epochs == [0, 1, 2])
            # 检查数据形状
            assert f['nominal_states'].shape == (3, 6)
            assert f['control_forces'].shape == (3, 3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_hdf5_logger_control_force_standardization():
    """测试 control_force 标准化（标量/数组）"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger = HDF5Logger(filepath=tmp_path, buffer_size=10)
        logger.initialize_file()

        # 标量
        logger.log_step(0, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), 5.0, 0.0)
        # 1D 数组
        logger.log_step(1, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.array([1.0, 2.0, 3.0]), 0.0)
        # 超出长度
        logger.log_step(2, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.array([1.0, 2.0, 3.0, 4.0]), 0.0)
        logger.close()

        with h5py.File(tmp_path, 'r') as f:
            forces = f['control_forces'][:]
            # 第一条应为 [5,0,0]
            assert np.allclose(forces[0], [5.0, 0.0, 0.0])
            # 第二条应为 [1,2,3]
            assert np.allclose(forces[1], [1.0, 2.0, 3.0])
            # 第三条应为前三个元素
            assert np.allclose(forces[2], [1.0, 2.0, 3.0])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_coordinate_frame_consistency_environment():
    """测试 CelestialEnvironment 的坐标系一致性校验"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    env = CelestialEnvironment(computation_frame=frame)

    # 注册一个力模型（使用 mock）
    env.register_force(MockForceModel())

    # 正确坐标系
    state = np.zeros(6)
    acc, out_frame = env.get_total_acceleration(state, frame)
    assert out_frame == frame
    assert np.all(acc == 0)

    # 错误坐标系应抛出异常
    wrong_frame = CoordinateFrame.J2000_ECI
    with pytest.raises(ValueError, match="坐标系冲突"):
        env.get_total_acceleration(state, wrong_frame)


def test_coordinate_frame_consistency_spacecraft():
    """测试 SpacecraftPointMass 的坐标系一致性校验"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    sc = SpacecraftPointMass("test", np.zeros(6), frame)

    # 正确推力坐标系
    sc.apply_thrust(np.ones(3), frame)   # 不应报错

    # 错误推力坐标系
    wrong_frame = CoordinateFrame.J2000_ECI
    with pytest.raises(ValueError, match="推力坐标系不匹配"):
        sc.apply_thrust(np.ones(3), wrong_frame)

    # 正确环境加速度坐标系
    env_acc = np.ones(3)
    deriv = sc.get_derivative(env_acc, frame)
    assert len(deriv) == 6

    # 错误环境加速度坐标系
    with pytest.raises(ValueError, match="动力学基准冲突"):
        sc.get_derivative(env_acc, wrong_frame)


def test_gnc_coordinate_consistency():
    """测试 GNC 子系统的坐标系校验"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    gnc = GNC_Subsystem("test", frame)

    # 生成一个简单的星历
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, frame)
    gnc.load_reference_trajectory(eph)   # 应通过

    # 错误坐标系的星历
    eph_wrong = Ephemeris(times, states, CoordinateFrame.J2000_ECI)
    with pytest.raises(ValueError, match="标称星历坐标系不匹配"):
        gnc.load_reference_trajectory(eph_wrong)

    # 导航状态更新
    obs = np.zeros(6)
    gnc.update_navigation(obs, frame)   # 应通过
    with pytest.raises(ValueError, match="导航状态坐标系不匹配"):
        gnc.update_navigation(obs, CoordinateFrame.J2000_ECI)


def test_ground_station_visibility():
    """测试地面站可视弧段逻辑"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    # 定义一个可视窗口 [10, 20] 秒
    windows = [(10.0, 20.0)]
    gs = GroundStation("test", frame, visibility_windows=windows)

    true_state = np.zeros(6)
    # 在窗口内
    obs, out_frame = gs.track_spacecraft(true_state, frame, epoch=15.0)
    assert obs is not None
    # 在窗口外
    obs, out_frame = gs.track_spacecraft(true_state, frame, epoch=5.0)
    assert obs is None
    # 无窗口（默认全天候）
    gs_all = GroundStation("test_all", frame, visibility_windows=[])
    obs, out_frame = gs_all.track_spacecraft(true_state, frame, epoch=5.0)
    assert obs is not None


def test_ground_station_sampling_rate():
    """测试地面站采样率控制"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    gs = GroundStation("test", frame, sampling_rate_hz=1.0)  # 每秒一次

    true_state = np.zeros(6)
    # 第一次采样，应返回
    obs, _ = gs.track_spacecraft(true_state, frame, epoch=0.0)
    assert obs is not None
    # 同一时间点重复调用，不应返回（因为 last_track_time 相同）
    obs, _ = gs.track_spacecraft(true_state, frame, epoch=0.0)
    assert obs is None
    # 过 0.5 秒，不足 1 秒，不应返回
    obs, _ = gs.track_spacecraft(true_state, frame, epoch=0.5)
    assert obs is None
    # 过 1 秒，应返回
    obs, _ = gs.track_spacecraft(true_state, frame, epoch=1.0)
    assert obs is not None


def test_gnc_control_force_standardization():
    """测试 GNC 的 compute_control_force 能标准化控制力输出"""
    frame = CoordinateFrame.SUN_EARTH_ROTATING
    gnc = GNC_Subsystem("test", frame)

    # 创建一个简单星历
    times = np.array([0.0, 1.0])
    states = np.zeros((2, 6))
    eph = Ephemeris(times, states, frame)
    gnc.load_reference_trajectory(eph)
    gnc.update_navigation(np.zeros(6), frame)

    # 模拟 K 矩阵（简单比例控制）
    K = np.eye(3, 6)

    # 应该返回形状为 (3,) 的数组
    force, out_frame = gnc.compute_control_force(epoch=0.0, K_matrix=K)
    assert force.shape == (3,)
    assert out_frame == frame

    # 测试 K 矩阵形状修复
    K_bad = np.ones(6)   # 形状 (6,)
    force, _ = gnc.compute_control_force(epoch=0.0, K_matrix=K_bad)
    assert force.shape == (3,)  # 应正常工作，内部已修复


def test_simulation_runner_exception_recovery():
    """测试仿真运行时异常恢复（如键盘中断）"""
    # 创建一个很短的仿真，并强制抛出异常
    config = {
        "mission_name": "ExceptionTest",
        "simulation_days": 0.01,      # 非常短
        "time_step": 1.0,
        "log_buffer_size": 10,
        "enable_visualization": False,
        "data_dir": "data/test_exception"
    }
    sim = L1MissionSimulation(config)

    # 模拟键盘中断：这里我们直接调用 _emergency_shutdown 作为测试
    try:
        # 我们只测试初始化后是否能正常关闭（不实际运行）
        sim._emergency_shutdown()
        # 检查日志文件是否至少存在（可能因未运行而不存在，但不应崩溃）
        # 实际异常恢复测试较为复杂，此处只验证方法不崩溃
        assert True
    except Exception as e:
        pytest.fail(f"异常恢复失败: {e}")


def test_hdf5_logger_auto_flush():
    """测试自动刷新功能"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger = HDF5Logger(filepath=tmp_path, buffer_size=3, auto_flush=True)
        logger.initialize_file()
        # 记录 2 条，不应自动刷新
        for i in range(2):
            logger.log_step(i, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), 0.0, 0.0)
        assert logger.buffer_count == 2
        # 记录第 3 条，应触发自动刷新
        logger.log_step(2, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), 0.0, 0.0)
        assert logger.buffer_count == 0
        logger.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_hdf5_logger_statistics():
    """测试 get_statistics 方法"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger = HDF5Logger(filepath=tmp_path, buffer_size=10)
        logger.initialize_file({"test": "value"})
        # 记录几条数据
        for i in range(5):
            logger.log_step(i, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), 0.0, 0.0)
        logger.close()

        stats = logger.get_statistics()
        assert stats["filepath"] == tmp_path
        assert stats["total_steps"] == 5
        assert stats["file_exists"] is True
        assert "file_size_mb" in stats
        # 检查数据集记录数
        assert "epochs_records" in stats
        assert stats["epochs_records"] == 5
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])