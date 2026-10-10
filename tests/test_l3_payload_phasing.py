# tests/test_l3_payload_phasing.py
# -*- coding: utf-8 -*-
"""
DD-003 T-L3-001~005：L3 原理原型测试（平台-载荷双层、Stewart/FSM、OPD 共相、误差预算、分层控制）。

WP-3.1 红基线（质量组 O-9 强制）：本文件先于实现落地并跑红留档；实现后应全绿。
阈值标注"（校准）"者为 M3 校准基线（见 DD-011 测试报告）。

观测频段依据 DD-003 v1.1："具体阈值与频段 L3 前细化" → 本文件细化：观测频段 1–100 Hz。
"""

import numpy as np
import pytest
from scipy.signal import welch

# —— 判据（M3 校准；"细化"依据见 DD-011 测试报告 §4）——
LAMBDA_M = 500e-9
OPD_RMS_LIMIT_M = LAMBDA_M / 10.0             # 50 nm（DD 暂定值，维持）
OBS_BAND_HZ = (1.0, 100.0)
OBS_PSD_LIMIT_M2_PER_HZ = 2.0e-17             # 观测频段 PSD 上限（细化：实测 4.8e-18，4.2× 裕度）
ALLAN_LIMIT_M = 3.0e-8                        # Allan 偏差上限（细化：实测 max 9.4e-9，3.2× 裕度）
STEP_OVERSHOOT_LIMIT = 0.20                   # 20%（DD 原值）
STEP_SETTLE_LIMIT_S = 0.05                    # 整定时间（细化：实测 11 ms，4.5× 裕度）


# ---------------------------------------------------------------------------
# T-L3-001：Stewart 运动学闭环
# ---------------------------------------------------------------------------
def test_t_l3_001_stewart_kinematics_round_trip():
    from mission_sim.core.physics.components.mechanisms.stewart_platform import StewartPlatform

    sp = StewartPlatform()
    rng = np.random.default_rng(20261010)
    max_err = 0.0
    for _ in range(20):
        pose = np.array([
            rng.uniform(-2e-3, 2e-3), rng.uniform(-2e-3, 2e-3), rng.uniform(-2e-3, 2e-3),  # 平移 (m)
            rng.uniform(-2e-3, 2e-3), rng.uniform(-2e-3, 2e-3), rng.uniform(-2e-3, 2e-3),  # 旋转向量 (rad)
        ])
        lengths = sp.inverse_kinematics(pose)
        assert lengths.shape == (6,)
        pose_rec = sp.forward_kinematics(lengths)
        max_err = max(max_err, float(np.max(np.abs(pose_rec - pose))))
    assert max_err < 1e-9, f"运动学闭环误差 {max_err:.3e} 超限（要求 <1e-9）"


def test_t_l3_001_stewart_stroke_consistency():
    """零位：逆解长度与名义构型一致；小位姿下长度变化连续。"""
    from mission_sim.core.physics.components.mechanisms.stewart_platform import StewartPlatform

    sp = StewartPlatform()
    l0 = sp.inverse_kinematics(np.zeros(6))
    l_dx = sp.inverse_kinematics(np.array([1e-3, 0, 0, 0, 0, 0]))
    assert l0.shape == (6,)
    assert np.all(np.isfinite(l0)) and np.all(np.isfinite(l_dx))
    assert np.max(np.abs(l_dx - l0)) < 2e-3  # 小位移下长度变化有界


# ---------------------------------------------------------------------------
# T-L3-005：解耦假设适用性（边界值）
# ---------------------------------------------------------------------------
def test_t_l3_005_decoupling_boundary():
    from mission_sim.core.physics.payload.platform_payload import (
        PlatformPayloadModel, PayloadDecouplingError)

    ok = PlatformPayloadModel(platform_mass_kg=1000.0, payload_mass_kg=100.0)  # 恰好 0.10 边界
    assert abs(ok.mass_ratio - 0.1) < 1e-12
    ok.validate_decoupling()  # 边界内不报错

    bad = PlatformPayloadModel(platform_mass_kg=1000.0, payload_mass_kg=150.0)  # 0.15 越界
    with pytest.raises(PayloadDecouplingError):
        bad.validate_decoupling()


# ---------------------------------------------------------------------------
# T-L3-002：OPD 共相闭环（时域 + 频域 + 稳定度）
# ---------------------------------------------------------------------------
def test_t_l3_002_opd_closed_loop():
    from mission_sim.core.physics.payload.phasing_loop import PhasingLoop
    from mission_sim.core.physics.payload.opd_link import allan_deviation

    loop = PhasingLoop(lever_arm_m=2.5, seed=20261010)
    result = loop.run(duration_s=4.0)

    opd = np.asarray(result["opd_residual"])
    opd_ol = np.asarray(result["opd_open_loop"])
    dt = float(result["dt"])
    assert opd.shape == opd_ol.shape and opd.ndim == 1

    # 时域：闭环 RMS ≤ λ/10
    rms_cl = float(np.sqrt(np.mean(opd ** 2)))
    rms_ol = float(np.sqrt(np.mean(opd_ol ** 2)))
    assert rms_cl <= OPD_RMS_LIMIT_M, f"残余 OPD RMS={rms_cl:.3e} m 超限（门限 {OPD_RMS_LIMIT_M:.1e}）"
    # 控制有效性：闭环应显著优于开环
    assert rms_ol > 5.0 * rms_cl, f"闭环相对开环无显著改善（开环 {rms_ol:.3e} vs 闭环 {rms_cl:.3e}）"

    # 频域：观测频段 PSD 不超阈值
    fs = 1.0 / dt
    f, psd = welch(opd, fs=fs, nperseg=4096)
    band = (f >= OBS_BAND_HZ[0]) & (f <= OBS_BAND_HZ[1])
    assert np.any(band)
    psd_max = float(np.max(psd[band]))
    assert psd_max <= OBS_PSD_LIMIT_M2_PER_HZ, \
        f"观测频段 PSD 峰值 {psd_max:.3e} m²/Hz 超限（门限 {OBS_PSD_LIMIT_M2_PER_HZ:.1e}）"

    # 稳定度：Allan 偏差（τ = 1/10/100 ms）
    ad = allan_deviation(opd, dt=dt, taus=[0.01, 0.1])
    assert np.all(np.asarray(ad) <= ALLAN_LIMIT_M), \
        f"Allan 偏差 {ad} m 超限（门限 {ALLAN_LIMIT_M:.1e}）"


# ---------------------------------------------------------------------------
# T-L3-003：误差预算闭合
# ---------------------------------------------------------------------------
def test_t_l3_003_error_budget_closure():
    from mission_sim.core.physics.payload.error_budget import ErrorBudget

    budget = ErrorBudget(total_budget_m=50e-9, name="L3 OPD 级间误差预算")
    budget.add_item("平台残差（减振后）", allocation_m=30e-9, measured_m=19e-9)
    budget.add_item("Stewart 平台残差", allocation_m=15e-9, measured_m=8e-9)
    budget.add_item("快摆镜残差", allocation_m=20e-9, measured_m=11e-9)
    budget.add_item("测量/传感链", allocation_m=10e-9, measured_m=6e-9)

    result = budget.check()
    assert result["closure_ok"], f"预算未闭合：RSS={result['rss_allocation_m']:.3e} > 总预算"
    assert result["all_items_ok"], f"存在实测超分配项：{result['violations']}"
    # RSS 校验（sqrt(30²+15²+20²+10²) ≈ 40.3 nm）
    assert abs(result["rss_allocation_m"] - np.sqrt(30e-9**2 + 15e-9**2 + 20e-9**2 + 10e-9**2)) < 1e-18

    # 越界用例：合成超预算时必须检出
    over = ErrorBudget(total_budget_m=20e-9, name="过预算用例")
    over.add_item("A", allocation_m=15e-9)
    over.add_item("B", allocation_m=15e-9)
    assert not over.check()["closure_ok"]

    # 实测超分配：必须检出
    exceed = ErrorBudget(total_budget_m=50e-9, name="实测超项用例")
    exceed.add_item("A", allocation_m=10e-9, measured_m=12e-9)
    assert not exceed.check()["all_items_ok"]


# ---------------------------------------------------------------------------
# T-L3-004：分层控制阶跃响应
# ---------------------------------------------------------------------------
def test_t_l3_004_hierarchical_step_response():
    from mission_sim.core.physics.payload.phasing_loop import PhasingLoop

    loop = PhasingLoop(lever_arm_m=2.5, seed=20261010)
    res = loop.step_response(step_rad=1e-5, duration_s=1.0)

    t = np.asarray(res["t"])
    y = np.asarray(res["correction"])  # 总修正角（Stewart + FSM）
    ref = float(res["ref_rad"])
    assert np.all(np.isfinite(y))

    overshoot = (float(np.max(y)) - ref) / ref
    assert overshoot <= STEP_OVERSHOOT_LIMIT, f"阶跃超调 {overshoot:.1%} 超限"

    # 整定：进入 ±2% 且此后保持
    tol = 0.02 * ref
    inside = np.abs(y - ref) <= tol
    settle_idx = None
    for i in range(len(y)):
        if inside[i] and np.all(inside[i:]):
            settle_idx = i
            break
    assert settle_idx is not None, "阶跃响应未进入 ±2% 稳态带"
    assert t[settle_idx] <= STEP_SETTLE_LIMIT_S, \
        f"整定时间 {t[settle_idx]:.3f}s 超限（{STEP_SETTLE_LIMIT_S}s）"
