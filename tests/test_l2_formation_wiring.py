# tests/test_l2_formation_wiring.py
# -*- coding: utf-8 -*-
"""
DD-003 T-L2-001~009：L2 编队接通与验收测试。

WP-2.1 红基线：本文件在接通实现前预期失败（update() 未调用、零引力、坐标系混算等）。
接通后应全绿；门限标注"暂定"者为 M2 校准基线（见 M2 测试报告）。

运行时长提示：T-L2-002（0.3 天 dt=1s）约 20 s，T-L2-004 约 12 s，其余秒级。
"""

import os

import h5py
import numpy as np
import pytest

from mission_sim.simulation.formation_simulation import FormationSimulation
from mission_sim.core.physics.components.actuators.impulse_smoother import ImpulseSmoother
from mission_sim.core.cyber.ids import FormationMode
from mission_sim.core.physics.models.central_gravity import CentralGravity
from mission_sim.utils.math_tools import lvlh_to_absolute

MU_EARTH = 3.986004418e14
R0 = 7000e3
V0 = float(np.sqrt(MU_EARTH / R0))

# —— M2 校准值（原始暂定值见 DD-003 v1.1；校准依据见 DD-008 测试报告）——
RMSE_POS_LIMIT_M = 1.0
RMSE_VEL_LIMIT_MPS = 0.01
ROBUSTNESS_RATIO_LIMIT = 2.0


def _chief_state():
    return np.array([R0, 0.0, 0.0, 0.0, V0, 0.0])


def _dep_state(rho):
    r, v = lvlh_to_absolute(_chief_state()[:3], _chief_state()[3:],
                            np.asarray(rho, float), np.zeros(3))
    return np.concatenate([r, v])


def _cfg(tmp_path, days, **overrides):
    cfg = dict(
        mission_name="L2_Test",
        simulation_days=days,
        time_step=1.0,
        data_dir=str(tmp_path),
        verbose=False,
        chief_initial_state=_chief_state().tolist(),
        deputy_initial_states=[
            ("DEP1", _dep_state([0.0, 1000.0, 0.0]).tolist()),
            ("DEP2", _dep_state([0.0, -2000.0, 0.0]).tolist()),
        ],
        formation_targets={
            "DEP1": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            "DEP2": [0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
        },
        chief_frame="J2000_ECI",
        chief_mass_kg=2000.0,
        deputy_mass_kg=500.0,
        # 执行器与闭环设计量级匹配（LQR 指令在 km 级误差下可达 ~1e2 N）
        thruster_max_thrust_n=50.0,
        isl_seed=20261010,
        router_seed=20261010,
        thruster_seed=20261010,
        generation_threshold_pos=50.0,
        generation_threshold_vel=0.1,
        keeping_threshold_pos=1.0,
        keeping_threshold_vel=0.01,
    )
    cfg.update(overrides)
    return cfg


def _load_formation(path):
    out = {}
    with h5py.File(path, "r") as f:
        for dep in ("DEP1", "DEP2"):
            g = f["formation"][f"deputy_{dep}"]
            out[dep] = {k: g[k][()] for k in
                        ["time", "rel_position", "rel_velocity", "control_force", "mode"]}
    return out


# ---------------------------------------------------------------------------
# T-L2-001：update() 接通与状态机迁移
# ---------------------------------------------------------------------------
def test_t_l2_001_update_wired_and_state_machine(tmp_path):
    sim = FormationSimulation(_cfg(tmp_path, 0.15))
    assert sim.run()

    for dep in sim.deputies:
        assert sim.deputy_controllers[dep.id].update_calls == sim.total_steps, \
            f"{dep.id}: update() 调用次数与步数不符（未接通）"

    data = _load_formation(sim.h5_file)
    for dep in ("DEP1", "DEP2"):
        modes = np.unique(data[dep]["mode"])
        assert FormationMode.GENERATION.value - 1 in modes  # 0
        assert 1 in modes, f"{dep} 未迁移到 KEEPING：modes={modes}"

    assert sim.deputy_controllers["DEP1"].mode == FormationMode.KEEPING
    assert sim.deputy_controllers["DEP2"].mode == FormationMode.KEEPING


# ---------------------------------------------------------------------------
# T-L2-002：三星编队闭环门限（CW 场景）
# ---------------------------------------------------------------------------
def test_t_l2_002_cw_formation_convergence(tmp_path):
    sim = FormationSimulation(_cfg(tmp_path, 0.3))
    assert sim.run()
    data = _load_formation(sim.h5_file)

    targets = {"DEP1": np.array([0.0, 100.0, 0.0]),
               "DEP2": np.array([0.0, -100.0, 0.0])}
    for dep, tgt in targets.items():
        t = data[dep]["time"]
        err = data[dep]["rel_position"] - tgt
        v = data[dep]["rel_velocity"]
        window = t > (t[-1] - 5830.0)  # 末一个轨道周期（稳态窗）

        rmse_pos = float(np.sqrt(np.mean(np.sum(err[window] ** 2, axis=1))))
        rmse_vel = float(np.sqrt(np.mean(np.sum(v[window] ** 2, axis=1))))

        assert rmse_pos <= RMSE_POS_LIMIT_M, f"{dep} 稳态位置 RMSE={rmse_pos:.3f} m 超限"
        assert rmse_vel <= RMSE_VEL_LIMIT_MPS, f"{dep} 稳态速度 RMSE={rmse_vel:.5f} m/s 超限"

        # 模式：末期应为 KEEPING
        assert data[dep]["mode"][-1] == 1, f"{dep} 末期未处于 KEEPING"


# ---------------------------------------------------------------------------
# T-L2-003：LVLH 一致性（解析比对）
# ---------------------------------------------------------------------------
def test_t_l2_003_lvlh_control_consistency(tmp_path):
    from mission_sim.core.cyber.platform_gnc.formation_controller import FormationController
    from mission_sim.core.cyber.models.cw_dynamics import CWDynamics
    from mission_sim.core.cyber.ids import ISLNetworkFrame
    from mission_sim.core.physics.ids import MicrowaveISLMeasurement

    n = float(np.sqrt(MU_EARTH / R0 ** 3))
    ctrl = FormationController("DEP1", "CHIEF", CWDynamics(n))
    ctrl.target_state = np.zeros(6)

    # 已知 LVLH 误差：径向 +50 m（球坐标 az=0, el=0）
    meas = MicrowaveISLMeasurement(
        phys_timestamp=0.0, range_m=50.0, azimuth_rad=0.0, elevation_rad=0.0,
        signal_strength=1.0)
    frame = ISLNetworkFrame(payload=meas, source_id="CHIEF", dest_id="DEP1",
                            tx_time=0.0, rx_time=0.0)
    cmd = ctrl.update(0.0, [frame], 1.0)

    est = 0.7 * np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 互补滤波（首发）
    expected = -ctrl.K @ est
    assert np.allclose(cmd.force_vector, expected, rtol=1e-9, atol=1e-12), \
        "控制器输出与 -K·e 解析值不一致（LVLH 口径错误）"
    assert cmd.force_vector[0] < 0.0, "径向正向偏差应产生向内的控制（方向错误）"


# ---------------------------------------------------------------------------
# T-L2-004：时延/丢包鲁棒性
# ---------------------------------------------------------------------------
def test_t_l2_004_delay_loss_robustness(tmp_path):
    def run(tag, **over):
        sim = FormationSimulation(_cfg(tmp_path / tag, 0.1, **over))
        assert sim.run()
        data = _load_formation(sim.h5_file)
        out = {}
        for dep, tgt in (("DEP1", np.array([0.0, 100.0, 0.0])),
                         ("DEP2", np.array([0.0, -100.0, 0.0]))):
            t = data[dep]["time"]
            err = data[dep]["rel_position"] - tgt
            w = t > (t[-1] - 5830.0)
            out[dep] = float(np.sqrt(np.mean(np.sum(err[w] ** 2, axis=1))))
        return out

    base = run("base", router_base_latency_s=0.0, router_jitter_s=0.0,
               router_packet_loss_rate=0.0)
    pert = run("pert", router_base_latency_s=0.05, router_jitter_s=0.01,
               router_packet_loss_rate=0.02)

    # M2 校准：基准 RMSE 处于估计器噪声底（~0.01 m），固定 2× 比值判据不适用；
    # 改为"绝对劣化 ≤0.1 m 或 2× 基准（取大者）"，并仍受 ≤1 m 绝对门限约束（见 DD-008）。
    for dep in ("DEP1", "DEP2"):
        allowed = max(ROBUSTNESS_RATIO_LIMIT * base[dep], 0.1)
        assert pert[dep] <= allowed, \
            f"{dep}: 时延/丢包下 RMSE={pert[dep]:.4f} m 超过校准包络 {allowed:.4f} m（base={base[dep]:.4f}）"
        assert pert[dep] <= RMSE_POS_LIMIT_M


# ---------------------------------------------------------------------------
# T-L2-005：ISL 链路帧守恒
# ---------------------------------------------------------------------------
def test_t_l2_005_isl_frame_conservation(tmp_path):
    sim = FormationSimulation(_cfg(tmp_path, 0.01))
    assert sim.run()

    in_flight = sim._frames_sent - sim._frames_dropped - sim._frames_delivered
    assert 0 <= in_flight <= 2, f"帧守恒被破坏：发送{sim._frames_sent} 丢弃{sim._frames_dropped} 交付{sim._frames_delivered}"

    p = sim.config.get("router_packet_loss_rate", 0.02)
    n = sim._frames_sent
    sigma = float(np.sqrt(p * (1 - p) / max(n, 1)))
    drop_rate = sim._frames_dropped / max(n, 1)
    assert abs(drop_rate - p) <= 3 * sigma + 1e-9, \
        f"丢包率 {drop_rate:.4f} 偏离配置值 {p} 超过 3σ（σ={sigma:.4f}）"


# ---------------------------------------------------------------------------
# T-L2-006：冲量平滑/MIB 等效一致性
# ---------------------------------------------------------------------------
def test_t_l2_006_impulse_smoother_conservation():
    mib = 0.01
    sm = ImpulseSmoother(mib_ns=mib, max_thrust_n=100.0)
    F_cmd = np.array([0.5, -0.25, 0.1])
    dt = 1.0
    n_steps = 1000

    emitted_impulse = np.zeros(3)
    for _ in range(n_steps):
        out = sm.smooth(F_cmd, dt)
        emitted_impulse += out * dt
        # 量化检查：每步输出冲量为 MIB 的整数倍（允许浮点）
        mag = float(np.linalg.norm(out)) * dt
        if mag > 0.0:
            k = mag / mib
            assert abs(k - round(k)) < 1e-6, f"输出冲量 {mag} 非 MIB 整数倍"

    desired = F_cmd * dt * n_steps
    rel = float(np.linalg.norm(emitted_impulse - desired) / np.linalg.norm(desired))
    assert rel < 0.01, f"冲量守恒误差 {rel:.2%} 超过 1%"


# ---------------------------------------------------------------------------
# T-L2-007：燃料账单一致性
# ---------------------------------------------------------------------------
def test_t_l2_007_fuel_accounting_consistency(tmp_path):
    sim = FormationSimulation(_cfg(tmp_path, 0.01, log_interval=1))
    assert sim.run()
    data = _load_formation(sim.h5_file)

    for dep, node in (("DEP1", sim.deputies[0]), ("DEP2", sim.deputies[1])):
        forces = data[dep]["control_force"]
        dv_integral = float(np.sum(np.linalg.norm(forces, axis=1)) * 1.0 / node.mass)
        dv_report = float(node.accumulated_dv)
        assert dv_report > 0.0
        rel = abs(dv_integral - dv_report) / dv_report
        assert rel < 0.01, f"{dep}: 燃料账单不一致（报表 {dv_report:.6f} vs 力积分 {dv_integral:.6f}, rel={rel:.3%}）"


# ---------------------------------------------------------------------------
# T-L2-008：力模型注册防呆
# ---------------------------------------------------------------------------
def test_t_l2_008_forces_registered(tmp_path):
    sim = FormationSimulation(_cfg(tmp_path, 0.001))
    sim._generate_nominal_orbit()
    sim._initialize_physical_domain()

    registry = sim.environment._force_registry
    assert len(registry) >= 1, "编队环境注册表为空（零引力）"
    assert any(isinstance(m, CentralGravity) for m in registry), "未注册中心引力"

    # 初始点环境加速度应为中心+J2 量级（~8 m/s²，而非 1e-2 摄动量级）
    state = sim.chief.state
    acc, _ = sim.environment.get_total_acceleration(state, sim.chief.frame)
    assert np.linalg.norm(np.asarray(acc)) > 1.0, \
        f"环境合力 {np.linalg.norm(np.asarray(acc)):.3e} m/s² 异常（疑似缺中心引力）"


# ---------------------------------------------------------------------------
# T-L2-009：重构态（第三态）与重收敛
# ---------------------------------------------------------------------------
def test_t_l2_009_reconfiguration_state_and_reconvergence(tmp_path):
    t_event = 0.05 * 86400.0
    new_target = [0.0, 200.0, 0.0, 0.0, 0.0, 0.0]
    sim = FormationSimulation(_cfg(
        tmp_path, 0.15,
        reconfiguration_events=[(t_event, "DEP1", new_target)]))
    assert sim.run()

    data = _load_formation(sim.h5_file)
    t = data["DEP1"]["time"]
    mode = data["DEP1"]["mode"]

    rec_idx = np.where(mode == 2)[0]  # HDF5 模式映射：0=GEN, 1=KEEPING, 2=RECONFIGURATION
    assert len(rec_idx) > 0, "未观测到 RECONFIGURATION 态（第三态未触发）"
    t_rec = t[rec_idx[0]]
    assert t_rec >= t_event - 2.0, f"重构触发时刻异常：{t_rec}"

    after = mode[rec_idx[0]:]
    assert 1 in after, "重构后未回到 KEEPING"
    assert mode[-1] == 1, "末期未处于 KEEPING"

    # 重收敛：误差相对新目标
    tgt = np.array(new_target[:3])
    w = t > (t[-1] - 5830.0)
    err = data["DEP1"]["rel_position"] - tgt
    rmse = float(np.sqrt(np.mean(np.sum(err[w] ** 2, axis=1))))
    assert rmse <= RMSE_POS_LIMIT_M, f"重构后稳态 RMSE={rmse:.3f} m 超限"


# ---------------------------------------------------------------------------
# T-L2-002②：CRTBP（日地 L2）编队闭环门限（M2 校准判据）
# ---------------------------------------------------------------------------
def test_t_l2_002b_crtbp_formation_convergence(tmp_path):
    """日地 L2 Halo 编队 0.5 天：末 6 h 稳态 RMSE（M2 校准：pos ≤10 m / vel ≤10 mm/s，见 DD-008 §4）。"""
    from mission_sim.core.cyber.models.threebody.base import CRTBP
    from mission_sim.core.cyber.models.crtbp_relative_dynamics import CRTBPRelativeDynamics
    from mission_sim.utils.math_tools import get_lqr_gain

    mu, L, omega = 3.00348e-6, 1.495978707e11, 1.990986e-7
    crtbp = CRTBP(mu, L, omega)
    chief0 = np.array([1.50613280e11, 0.0, 1.20000000e8, 0.0, 1.51320000e2, 0.0])
    rel_targets = {"DEP1": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
                   "DEP2": [0.0, -100.0, 0.0, 0.0, 0.0, 0.0]}
    rel_initial = {"DEP1": np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]),
                   "DEP2": np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])}
    deps = []
    for dep_id, rel_state in rel_initial.items():
        r_abs, v_abs = lvlh_to_absolute(chief0[:3], chief0[3:], rel_state[:3], rel_state[3:])
        deps.append((dep_id, np.concatenate([r_abs, v_abs]).tolist()))

    dyn = CRTBPRelativeDynamics(crtbp)
    A = dyn._linearized_matrix(chief0)
    B = np.zeros((6, 3)); B[3:6, 0:3] = np.eye(3)
    K = get_lqr_gain(A, B, np.diag([1e-4] * 3 + [1.0] * 3), np.diag([1e8] * 3))

    cfg = dict(
        mission_name="L2_CRTBP", simulation_days=0.5, time_step=10.0,
        data_dir=str(tmp_path), verbose=False,
        chief_initial_state=chief0.tolist(),
        deputy_initial_states=deps,
        formation_targets=rel_targets,
        chief_frame="SUN_EARTH_ROTATING",
        chief_mass_kg=2000.0, deputy_mass_kg=500.0,
        enable_crtbp=True, mu=mu, L=L, omega=omega,
        lqr_gain=K,
        isl_range_noise_std=0.01, isl_angle_noise_std=1e-4, isl_max_range_m=500000.0,
        thruster_max_thrust_n=0.5, thruster_min_thrust_n=0.001, thruster_noise_std_n=0.01,
        router_base_latency_s=0.1, router_jitter_s=0.02, router_packet_loss_rate=0.05,
        router_seed=123, isl_seed=20261010, thruster_seed=20261010,
        generation_threshold_pos=10.0, generation_threshold_vel=0.5,
        keeping_threshold_pos=0.01, keeping_threshold_vel=1e-4,
    )
    sim = FormationSimulation(cfg)
    assert sim.run()
    data = _load_formation(sim.h5_file)

    crtbp_rmse_pos_limit = 1.0   # M4 收紧：Q_pos 1e-6→1e-4；实测 0.23/0.21 m，裕度 4.3×（原校准 10 m 系保守增益所致）
    crtbp_rmse_vel_limit = 0.01
    for dep, tgt in (("DEP1", np.array([0.0, 100.0, 0.0])),
                     ("DEP2", np.array([0.0, -100.0, 0.0]))):
        t = data[dep]["time"]
        rp = data[dep]["rel_position"]
        rv = data[dep]["rel_velocity"]
        w = t > (t[-1] - 21600.0)  # 末 6 小时
        rmse_p = float(np.sqrt(np.mean(np.sum((rp[w] - tgt) ** 2, axis=1))))
        rmse_v = float(np.sqrt(np.mean(np.sum(rv[w] ** 2, axis=1))))
        assert rmse_p <= crtbp_rmse_pos_limit, f"{dep}: CRTBP 稳态 RMSE={rmse_p:.3f} m 超校准门限"
        assert rmse_v <= crtbp_rmse_vel_limit, f"{dep}: CRTBP 稳态速度 RMSE={rmse_v:.5f} m/s 超限"
        assert data[dep]["mode"][-1] == 1, f"{dep}: 末期未处于 KEEPING"
