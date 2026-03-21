import numpy as np
import pytest
from mission_sim.core.types import CoordinateFrame
from mission_sim.core.trajectory.generators import (
    KeplerianGenerator,
    J2KeplerianGenerator,
    HaloDifferentialCorrector,
    create_generator
)

def test_keplerian_generator():
    """测试开普勒轨道生成器"""
    config = {
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dt": 10.0,
        "sim_time": 86400.0
    }
    gen = KeplerianGenerator()
    eph = gen.generate(config)
    assert len(eph.times) == 8641
    assert eph.states.shape == (8641, 6)
    assert eph.frame == CoordinateFrame.J2000_ECI

def test_j2_keplerian_generator():
    """测试 J2 轨道生成器（至少不崩溃）"""
    config = {
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        "dt": 60.0,
        "sim_time": 3600.0
    }
    gen = J2KeplerianGenerator()
    eph = gen.generate(config)
    assert len(eph.times) == 61
    assert eph.states.shape == (61, 6)

def test_halo_generator():
    """测试 Halo 轨道生成器"""
    config = {
        "Az": 0.05,
        "dt": 0.001
    }
    gen = HaloDifferentialCorrector()
    eph = gen.generate(config)
    # 应有数据
    assert len(eph.times) > 0
    assert eph.states.shape[0] == len(eph.times)
    assert eph.frame == CoordinateFrame.SUN_EARTH_ROTATING

def test_create_generator():
    """测试工厂函数"""
    gen1 = create_generator("keplerian")
    assert isinstance(gen1, KeplerianGenerator)
    gen2 = create_generator("j2_keplerian")
    assert isinstance(gen2, J2KeplerianGenerator)
    gen3 = create_generator("halo")
    assert isinstance(gen3, HaloDifferentialCorrector)
    with pytest.raises(ValueError):
        create_generator("invalid")
