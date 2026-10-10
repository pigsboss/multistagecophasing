# tests/test_l4_engineering.py
# -*- coding: utf-8 -*-
"""
DD-003 T-L4-001~007：L4 工程化原型测试（6-DOF 姿轨耦合、硬件非线性、噪声谱、蒙卡、OPD 极值、HITL、姿态包处置）。

WP-4.1 红基线（O-9 强制）：本文件先于实现落地并跑红留档；实现后应全绿。
阈值标注"（校准）"者为 M4 校准基线（见 DD-013 测试报告）。
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MU_EARTH = 3.986004418e14

# —— 判据（暂定，M4 校准）——
SIXDOF_CONSERVATION_LIMIT = 1e-10
POINTING_P95_LIMIT_RAD = 3.0e-8          # 指向残差 P95（暂定，M4 校准）
OPD_P95_LIMIT_M = 1.5e-7                 # OPD P95（暂定，M4 校准）
MC_PASS_RATE_MIN = 0.95                  # 蒙卡通过率（DD 原值）
MC_RMS_LIMIT_M = 5.0e-8                  # 单次通过门限（λ/10）
ALLAN_SLOPE_TOL = 0.15                   # 斜率带（±）


# ---------------------------------------------------------------------------
# T-L4-007：姿态包处置与依赖体检（R-L4-06）
# ---------------------------------------------------------------------------
def test_t_l4_007_attitude_package_repaired():
    import importlib
    pkg = importlib.import_module("mission_sim.core.spacetime.attitude")
    assert hasattr(pkg, "AttitudeModel"), "AttitudeModel 未导出"

    # 静态扫描：不得再有对不存在子模块的引用
    pattern_candidates = ["attitude.config", "attitude.models", "attitude.interpolated"]
    bad = []
    for p in (REPO_ROOT / "mission_sim").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if any(s in text for s in pattern_candidates):
            bad.append(str(p.relative_to(REPO_ROOT)))
    assert not bad, f"存在对不存在子模块的引用：{bad}"

    # L1–L3 依赖体检（子进程导入关键模块）
    code = (
        "import mission_sim.simulation.threebody.sun_earth_l2, "
        "mission_sim.simulation.twobody.leo, "
        "mission_sim.simulation.formation_simulation, "
        "mission_sim.core.physics.payload.phasing_loop"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"L1–L3 依赖体检失败：{r.stderr[-400:]}"

    # AttitudeState 契约就绪
    from mission_sim.core.physics.ids import AttitudeState
    from mission_sim.core.spacetime.ids import CoordinateFrame
    att = AttitudeState(quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                        angular_velocity=np.zeros(3),
                        frame=CoordinateFrame.J2000_ECI)
    d = att.as_dict()
    assert d["frame"] == "J2000_ECI" and len(d["quaternion"]) == 4


# ---------------------------------------------------------------------------
# T-L4-001：6-DOF 姿轨耦合守恒
# ---------------------------------------------------------------------------
def test_t_l4_001_sixdof_conservation():
    from mission_sim.core.physics.attitude_dynamics import RigidBody6DOF

    body = RigidBody6DOF(mass_kg=500.0, inertia_diag_kgm2=(10.0, 12.0, 15.0), mu=MU_EARTH)
    body.set_state(
        r=np.array([7.0e6, 0.0, 0.0]),
        v=np.array([0.0, 7545.9, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.05, 0.03, -0.02]),
    )
    L0 = body.angular_momentum_inertial()
    E0 = body.mechanical_energy()

    dt = 0.01
    for _ in range(10000):  # 100 s 无控自由演化
        body.step(np.zeros(3), dt)

    L1 = body.angular_momentum_inertial()
    E1 = body.mechanical_energy()
    dL = float(np.linalg.norm(L1 - L0) / np.linalg.norm(L0))
    dE = float(abs(E1 - E0) / abs(E0))
    assert dL < SIXDOF_CONSERVATION_LIMIT, f"角动量漂移 {dL:.3e} 超限"
    assert dE < SIXDOF_CONSERVATION_LIMIT, f"能量漂移 {dE:.3e} 超限"
    assert abs(float(np.linalg.norm(body.quaternion)) - 1.0) < 1e-12, "四元数未保持归一"


# ---------------------------------------------------------------------------
# T-L4-002：执行器非线性边界值（死区 / 饱和 / MIB / 延迟）
# ---------------------------------------------------------------------------
def test_t_l4_002_actuator_nonlinearity_boundaries():
    from mission_sim.core.physics.components.actuators.thruster import Thruster
    from mission_sim.core.physics.components.actuators.impulse_smoother import ImpulseSmoother

    # 死区边界（噪声关闭，确定性）
    thr = Thruster(max_thrust_n=1.0, min_thrust_n=0.01, noise_std_n=0.0)
    f_below, _ = thr.execute(np.array([0.009, 0.0, 0.0]))
    assert np.allclose(f_below, 0.0), "死区边界内应输出零"
    f_at, _ = thr.execute(np.array([0.01, 0.0, 0.0]))
    assert abs(np.linalg.norm(f_at) - 0.01) < 1e-12, "死区边界上应正常通过"

    # 饱和边界
    f_sat, _ = thr.execute(np.array([1.5, 0.0, 0.0]))
    assert abs(np.linalg.norm(f_sat) - 1.0) < 1e-12, "饱和应限幅到上限"

    # MIB 边界（冲量平滑器：累计到恰好 MIB 时开火）
    sm = ImpulseSmoother(mib_ns=0.01, max_thrust_n=100.0)
    out1 = sm.smooth(np.array([0.0095, 0.0, 0.0]), dt=1.0)
    assert np.allclose(out1, 0.0), "未达 MIB 不应输出"
    out2 = sm.smooth(np.array([0.0006, 0.0, 0.0]), dt=1.0)
    assert abs(np.linalg.norm(out2) * 1.0 - 0.01) < 1e-12, "恰好达到 MIB 应输出一个脉冲"

    # 命令延迟边界（纯传输延迟，按步长量化）
    thr_d = Thruster(max_thrust_n=1.0, min_thrust_n=0.0, noise_std_n=0.0, command_delay_s=0.05)
    outs = []
    for k in range(8):
        cmd = np.array([1.0, 0.0, 0.0]) if k == 0 else np.zeros(3)
        f, _ = thr_d.execute(cmd, dt=0.01)
        outs.append(float(np.linalg.norm(f)))
    assert all(o == 0.0 for o in outs[:5]), f"延迟期内不应输出：{outs}"
    assert outs[5] > 0.0, f"延迟 5 步后应输出：{outs}"


# ---------------------------------------------------------------------------
# T-L4-003：传感器噪声谱（Allan 偏差斜率）
# ---------------------------------------------------------------------------
def test_t_l4_003_allan_spectrum_slopes():
    from mission_sim.core.physics.components.sensors.noise_spectrum import AllanNoiseGenerator
    from mission_sim.core.physics.payload.opd_link import allan_deviation

    dt = 0.1
    n = 16384
    taus = np.array([dt * 2.0 ** k for k in range(1, 9)])
    for kind, target in (("white", -0.5), ("flicker", 0.0), ("random_walk", +0.5)):
        gen = AllanNoiseGenerator(kind=kind, sigma=1.0, seed=20261010)
        x = gen.generate(n=n, dt=dt)
        ad = allan_deviation(x, dt=dt, taus=taus)
        assert np.all(np.isfinite(ad)) and np.all(ad > 0)
        slope = float(np.polyfit(np.log10(taus), np.log10(ad), 1)[0])
        assert abs(slope - target) <= ALLAN_SLOPE_TOL, \
            f"{kind}: Allan 斜率 {slope:.3f} 偏离目标 {target}（带 ±{ALLAN_SLOPE_TOL}）"


# ---------------------------------------------------------------------------
# T-L4-004：蒙特卡洛鲁棒边界（N=50，通过率 ≥95%）
# ---------------------------------------------------------------------------
def test_t_l4_004_monte_carlo_robustness():
    from mission_sim.core.physics.payload.phasing_loop import PhasingLoop
    from mission_sim.core.cyber.payload_ctrl.hierarchical_controller import HierarchicalPhasingController

    rng = np.random.default_rng(20261010)
    n_runs = 50
    passed = 0
    for i in range(n_runs):
        scale = float(rng.uniform(0.8, 1.2))      # 扰动幅值散布 ±20%
        wn_mul = float(rng.uniform(0.95, 1.05))   # FSM 带宽散布 ±5%
        gain_mul = float(rng.uniform(0.9, 1.1))   # 控制增益散布 ±10%
        ctrl = HierarchicalPhasingController(
            dt=1e-4, kp_fsm=2.5 * gain_mul, ki_fsm=800.0 * gain_mul, kd_fsm=2.0e-4)
        loop = PhasingLoop(lever_arm_m=2.5, seed=1000 + i,
                           disturbance_scale=scale, fsm_wn_hz=500.0 * wn_mul,
                           controller=ctrl)
        res = loop.run(duration_s=0.6)
        rms = float(np.sqrt(np.mean(res["opd_residual"] ** 2)))
        if rms <= MC_RMS_LIMIT_M:
            passed += 1
    rate = passed / n_runs
    assert rate >= MC_PASS_RATE_MIN, f"蒙卡通过率 {rate:.0%}（{passed}/{n_runs}）低于 {MC_PASS_RATE_MIN:.0%}"


# ---------------------------------------------------------------------------
# T-L4-005：OPD/指向极值评估（P95）
# ---------------------------------------------------------------------------
def test_t_l4_005_opd_pointing_p95():
    from mission_sim.core.physics.payload.phasing_loop import PhasingLoop
    from mission_sim.core.physics.ids import OPDMeasurement

    loop = PhasingLoop(lever_arm_m=2.5, seed=20261010)
    res = loop.run(duration_s=2.0)
    theta_res = res["theta_platform"] - res["theta_stewart"] - res["theta_fsm"]
    theta_norm = np.linalg.norm(theta_res, axis=1)
    opd = np.asarray(res["opd_residual"])

    theta_p95 = float(np.percentile(theta_norm, 95))
    opd_p95 = float(np.percentile(opd, 95))
    assert theta_p95 <= POINTING_P95_LIMIT_RAD, f"指向残差 P95={theta_p95:.3e} rad 超限"
    assert opd_p95 <= OPD_P95_LIMIT_M, f"OPD P95={opd_p95:.3e} m 超限"

    # 契约使用：把 P95 评估结果登记为 OPDMeasurement
    meas = OPDMeasurement(opd_m=opd_p95, epoch=2.0, quality="P95-assessment")
    assert meas.opd_m == opd_p95 and meas.quality == "P95-assessment"


# ---------------------------------------------------------------------------
# T-L4-006：HITL 接口契约（mock）
# ---------------------------------------------------------------------------
def test_t_l4_006_hitl_contract():
    from mission_sim.simulation.hitl import (
        HITLInterface, HITLMessage, MockEchoTransport,
        HITLHandshakeError, HITLTimeoutError, PROTOCOL_VERSION)
    from mission_sim.core.physics.ids import AttitudeState
    from mission_sim.core.spacetime.ids import CoordinateFrame

    # 握手成功
    transport = MockEchoTransport()
    hitl = HITLInterface(transport=transport, dt=0.01, timeout_steps=5)
    info = hitl.handshake()
    assert info["protocol_version"] == PROTOCOL_VERSION

    # 版本不匹配 → 握手失败
    bad = MockEchoTransport(reply_version=PROTOCOL_VERSION + 1)
    with pytest.raises(HITLHandshakeError):
        HITLInterface(transport=bad, dt=0.01).handshake()

    # 链路失效 → 超时
    dead = MockEchoTransport(dead=True)
    with pytest.raises(HITLTimeoutError):
        HITLInterface(transport=dead, dt=0.01, timeout_steps=3).handshake()

    # 消息契约：序列化往返
    msg = HITLMessage(msg_type="state", sim_time=1.25, payload={"a": 1})
    assert HITLMessage.from_dict(msg.to_dict()).to_dict() == msg.to_dict()

    # 时间步握手：state → actuator_cmd
    att = AttitudeState(quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                        angular_velocity=np.zeros(3),
                        frame=CoordinateFrame.J2000_ECI)
    cmd = hitl.step(epoch=0.01, attitude=att)
    assert isinstance(cmd, np.ndarray) and cmd.shape == (3,)
    assert transport.sent[-1].msg_type == "state"
    assert transport.sent[-1].payload["attitude"]["frame"] == "J2000_ECI"
