import numpy as np
import pytest
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.gnc.ground_station import GroundStation

def test_ground_station_visibility():
    """测试可视弧段逻辑"""
    windows = [(10.0, 20.0)]
    gs = GroundStation("test", CoordinateFrame.SUN_EARTH_ROTATING, visibility_windows=windows)
    true_state = np.zeros(6)
    # 在窗口内
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 15.0)
    assert obs is not None
    # 窗口外
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 5.0)
    assert obs is None
    # 无窗口（全天候）
    gs_all = GroundStation("test_all", CoordinateFrame.SUN_EARTH_ROTATING, visibility_windows=[])
    obs, _ = gs_all.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 5.0)
    assert obs is not None

def test_ground_station_sampling_rate():
    """测试采样率控制"""
    gs = GroundStation("test", CoordinateFrame.SUN_EARTH_ROTATING, sampling_rate_hz=1.0)
    true_state = np.zeros(6)
    # 第一次采样应返回
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 0.0)
    assert obs is not None
    # 同一时间点不应返回
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 0.0)
    assert obs is None
    # 0.5 秒后不足 1 秒，不应返回
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 0.5)
    assert obs is None
    # 1 秒后应返回
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 1.0)
    assert obs is not None

def test_ground_station_noise():
    """测试噪声注入"""
    gs = GroundStation("test", CoordinateFrame.SUN_EARTH_ROTATING, pos_noise_std=1.0, vel_noise_std=0.1)
    true_state = np.zeros(6)
    obs, _ = gs.track_spacecraft(true_state, CoordinateFrame.SUN_EARTH_ROTATING, 0.0)
    assert obs is not None
    assert np.linalg.norm(obs) > 0  # 应有噪声