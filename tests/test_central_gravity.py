# tests/test_central_gravity.py
# -*- coding: utf-8 -*-
"""
DD-003 T-L1-001 / T-L1-002：中心引力模型正确性 与 LEO 环境合力核对。

WP-1.1 红基线：本文件在修复前预期失败（模块不存在 / 环境缺中心引力）。
修复后应全绿。
"""

import numpy as np

from mission_sim.core.physics.models.j2_gravity import J2Gravity
from mission_sim.simulation.twobody.leo import LEOL1Simulation

MU_EARTH = 3.986004418e14


def _leo_sim_config(**overrides):
    config = {
        "mission_name": "T_L1_002",
        "simulation_days": 0.001,
        "time_step": 10.0,
        "data_dir": "/tmp/mcpc_test_l1",
        "verbose": False,
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# T-L1-001：CentralGravity 单元正确性
# ---------------------------------------------------------------------------
def test_t_l1_001_central_gravity_matches_analytic():
    """100 组随机状态：|a - (-mu*r/|r|^3)| 相对误差 < 1e-12。"""
    from mission_sim.core.physics.models.central_gravity import CentralGravity

    model = CentralGravity(mu=MU_EARTH)
    rng = np.random.default_rng(20261010)
    for _ in range(100):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = direction * rng.uniform(6.6e6, 5.0e7)
        state = np.concatenate([r, rng.normal(0.0, 100.0, 3)])

        acc = np.asarray(model.compute_accel(state, 0.0), dtype=np.float64)
        ref = -MU_EARTH * r / np.linalg.norm(r) ** 3
        rel = np.linalg.norm(acc - ref) / np.linalg.norm(ref)
        assert rel < 1e-12, f"相对误差 {rel:.3e} 超限"


def test_t_l1_001_vectorized_matches_scalar():
    """向量化接口与标量接口一致（绝对 <1e-18）。"""
    from mission_sim.core.physics.models.central_gravity import CentralGravity

    model = CentralGravity(mu=MU_EARTH)
    rng = np.random.default_rng(20261010)
    states = rng.normal(0.0, 5e6, size=(20, 6))
    states[:, 0:3] += 7e6

    batch = np.asarray(model.compute_vectorized_acc(states, 0.0))
    assert batch.shape == (20, 3)
    for i in range(states.shape[0]):
        single = np.asarray(model.compute_accel(states[i], 0.0))
        assert np.allclose(batch[i], single, rtol=1e-14, atol=1e-18)


def test_t_l1_001_r_min_protection():
    """r→0 数值保护：不产生 NaN/Inf。"""
    from mission_sim.core.physics.models.central_gravity import CentralGravity

    model = CentralGravity(mu=MU_EARTH)
    acc = np.asarray(model.compute_accel(np.zeros(6), 0.0))
    assert np.all(np.isfinite(acc))
    assert np.allclose(acc, 0.0)


# ---------------------------------------------------------------------------
# T-L1-002：LEO 场景环境合力 = 中心引力 + J2（+阻力，可忽略）
# ---------------------------------------------------------------------------
def test_t_l1_002_leo_environment_includes_central_gravity():
    """修复后：LEO 注册表含中心引力；环境合力与"中心+J2"理论值相对误差 <1e-6。"""
    sim = LEOL1Simulation(_leo_sim_config())
    assert sim._generate_nominal_orbit()
    sim._initialize_physical_domain()

    state = sim.spacecraft.state
    r0 = state[:3]
    acc, _ = sim.environment.get_total_acceleration(state, sim.spacecraft.frame)

    a_central = -MU_EARTH * r0 / np.linalg.norm(r0) ** 3
    a_j2 = np.asarray(J2Gravity(mu_earth=MU_EARTH).compute_accel(state, 0.0))
    a_ref = a_central + a_j2  # 该高度阻力项量级 ~1e-26 m/s²，可忽略

    rel = np.linalg.norm(np.asarray(acc) - a_ref) / np.linalg.norm(a_ref)
    assert rel < 1e-6, (
        f"LEO 环境合力与理论不符：|a_env|={np.linalg.norm(acc):.6e} m/s², "
        f"|a_ref|={np.linalg.norm(a_ref):.6e} m/s², 相对差={rel:.3e}"
    )
