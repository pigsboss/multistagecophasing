import numpy as np
import pytest
from mission_sim.core.types import CoordinateFrame, Telecommand

def test_coordinate_frame_enum():
    """测试坐标系枚举定义"""
    assert CoordinateFrame.J2000_ECI.value == "J2000_Earth_Centered_Inertial"
    assert CoordinateFrame.SUN_EARTH_ROTATING.value == "Sun_Earth_Rotating"
    assert len(CoordinateFrame) == 7

def test_telecommand_creation():
    """测试 Telecommand 创建和校验"""
    target = np.array([1e11, 0, 0, 0, 0, 0])
    cmd = Telecommand("TEST", target, CoordinateFrame.SUN_EARTH_ROTATING, 0.0)
    assert cmd.cmd_type == "TEST"
    assert np.array_equal(cmd.target_state, target)
    assert cmd.frame == CoordinateFrame.SUN_EARTH_ROTATING
    assert cmd.execution_epoch == 0.0

def test_telecommand_type_conversion():
    """测试目标状态自动转换为 numpy 数组"""
    target_list = [1e11, 0, 0, 0, 0, 0]
    cmd = Telecommand("TEST", target_list, CoordinateFrame.SUN_EARTH_ROTATING)
    assert isinstance(cmd.target_state, np.ndarray)

def test_telecommand_invalid_frame():
    """测试无效坐标系类型"""
    target = np.zeros(6)
    with pytest.raises(TypeError, match="必须是 CoordinateFrame 枚举类型"):
        Telecommand("TEST", target, "invalid_frame")  # type: ignore