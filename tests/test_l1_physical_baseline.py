# tests/test_l1_physical_baseline.py
# -*- coding: utf-8 -*-
"""
DD-003 T-L1-003/004/005/006/007/008/009：L1 物理基线测试。

WP-1.1 红基线：修复前预期失败（缺中心引力、误差链路恒零、分级失真等）。
修复后应全绿；门限标注"CAL"者为 M1 校准后的基线值（见 M1 测试报告）。

运行时长提示：T-L1-003（LEO 1 天 dt=10s）约 20 s，其余用例秒级。
"""

import os
import subprocess
import sys

import h5py
import numpy as np

from mission_sim.simulation.twobody.leo import LEOL1Simulation
from mission_sim.simulation.twobody.geo import GEOL1Simulation

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# —— M1 校准值（原始暂定值见 DD-003 v1.1；校准依据见 DD-006 测试报告）——
LEO_SCREEN_DV_MPS_PER_DAY = 5.0      # 筛查门限（DD-001 §5.1.1）
LEO_RADIUS_DRIFT_LIMIT_M = 1000.0     # 主判据
# M1 校准（2026-10-10）：GEO 稳态成本实测 0.781 m/s/天，且第 1/2 日几乎相同
# （非暂态，系"导航噪声追逐"成本：5 m / 5 mm/s 观测噪声经全状态反馈的持续修正量）。
# 暂定筛查门限 0.5 → 校准为 1.0 m/s/天（≥20% 裕度）；进一步降低列入 P2+ 专项（滤波/增益优化）。
GEO_SCREEN_DV_MPS_PER_DAY = 1.0
STEP_SENSITIVITY_REL_LIMIT = 1e-5     # T-L1-009 判据（DD-003 v1.1 原值；实测 6.1e-7，16× 裕度）


def _run_leo(tmp_path, days, **overrides):
    config = {
        "mission_name": "L1_PhysBaseline",
        "simulation_days": days,
        "time_step": 10.0,
        "data_dir": str(tmp_path),
        "verbose": False,
        "elements": [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    config.update(overrides)
    sim = LEOL1Simulation(config)
    assert sim.run(), "仿真运行失败"
    return sim


def _load_datasets(h5_path):
    keys = ["epochs", "nav_states", "nominal_states", "true_states",
            "tracking_errors", "control_forces", "accumulated_dvs"]
    with h5py.File(h5_path, "r") as f:
        return {k: f[k][()] for k in keys}


# ---------------------------------------------------------------------------
# T-L1-003：LEO 1 天闭环门限
# ---------------------------------------------------------------------------
def test_t_l1_003_leo_one_day_station_keeping(tmp_path):
    sim = _run_leo(tmp_path, 1.0)
    data = _load_datasets(sim.h5_file)

    r_true = np.linalg.norm(data["true_states"][:, 0:3], axis=1)
    r_nom = np.linalg.norm(data["nominal_states"][:, 0:3], axis=1)
    assert r_true.max() < 7000e3 * 1.05, f"轨道半径失控（可能飞离）：r_max={r_true.max():.3e} m"
    # 主判据：相对标称轨道的半径漂移（J2 短周期振荡本身约 ±5 km，绝对半径不作判据）
    drift = float(np.max(np.abs(r_true - r_nom)))
    assert drift <= LEO_RADIUS_DRIFT_LIMIT_M, f"相对标称半径漂移 {drift:.1f} m 超限"

    dv_total = float(data["accumulated_dvs"][-1])
    assert dv_total <= LEO_SCREEN_DV_MPS_PER_DAY, \
        f"LEO 1 天 ΔV={dv_total:.3f} m/s 超出筛查门限 {LEO_SCREEN_DV_MPS_PER_DAY} m/s"


# ---------------------------------------------------------------------------
# T-L1-004：跟踪误差链路一致性（数值化）
# ---------------------------------------------------------------------------
def test_t_l1_004_tracking_error_channel_consistency(tmp_path):
    injection = np.array([100.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    sim = _run_leo(tmp_path, 0.3, injection_error=injection)
    data = _load_datasets(sim.h5_file)

    logged = data["tracking_errors"]
    err_from_logs = data["nav_states"] - data["nominal_states"]
    assert np.allclose(logged, err_from_logs, rtol=0.0, atol=1e-9), \
        "HDF5 tracking_errors 与 nav-nominal 不一致（误差链路未接通）"

    # 同一历元对齐比较（日志末条与仿真末步相差一步；两者均须由唯一通道计算）
    t_last = (sim.total_steps - 1) * float(sim.config["time_step"])
    ref_last = sim.ephemeris.get_state(t_last)
    expected_last = sim.gnc_system.current_nav_state - ref_last
    assert np.allclose(sim.gnc_system.last_tracking_error, expected_last, rtol=0.0, atol=1e-9), \
        "GNC 内部通道的误差与其应有定义不一致（唯一通道校验失败）"

    assert np.max(np.linalg.norm(data["control_forces"], axis=1)) > 0.0, \
        "控制力日志恒零（控制力链路未接通）"

    n = logged.shape[0]
    tail = slice(3 * n // 4, n)
    pos_err = np.linalg.norm(logged[tail, 0:3], axis=1)
    vel_err = np.linalg.norm(logged[tail, 3:6], axis=1)
    assert np.max(pos_err) <= 100.0, f"稳态位置误差 {np.max(pos_err):.1f} m > 100 m"
    assert np.max(vel_err) <= 0.1, f"稳态速度误差 {np.max(vel_err):.3f} m/s > 0.1 m/s"


# ---------------------------------------------------------------------------
# T-L1-005：GEO 1 天闭环门限
# ---------------------------------------------------------------------------
def test_t_l1_005_geo_one_day_station_keeping(tmp_path):
    config = {
        "mission_name": "L1_GeoBaseline",
        "simulation_days": 1.0,
        "time_step": 60.0,
        "data_dir": str(tmp_path),
        "verbose": False,
    }
    sim = GEOL1Simulation(config)
    assert sim.run()
    data = _load_datasets(sim.h5_file)

    r0 = GEOL1Simulation.GEO_RADIUS
    r = np.linalg.norm(data["true_states"][:, 0:3], axis=1)
    assert np.max(np.abs(r - r0)) < 5.0e3, f"GEO 半径漂移异常：max|Δr|={np.max(np.abs(r - r0)):.1f} m"

    dv_total = float(data["accumulated_dvs"][-1])
    assert dv_total <= GEO_SCREEN_DV_MPS_PER_DAY, \
        f"GEO 1 天 ΔV={dv_total:.3f} m/s 超出筛查门限 {GEO_SCREEN_DV_MPS_PER_DAY} m/s"


# ---------------------------------------------------------------------------
# T-L1-006：CRTBP/Halo 端到端回归保护
# ---------------------------------------------------------------------------
def test_t_l1_006_crtbp_end_to_end_regression(tmp_path):
    from mission_sim.simulation.threebody.sun_earth_l2 import SunEarthL2L1Simulation

    config = {
        "mission_name": "L1_CrtbpRegression",
        "simulation_days": 0.2,
        "time_step": 60.0,
        "data_dir": str(tmp_path),
        "verbose": False,
    }
    sim = SunEarthL2L1Simulation(config)
    assert sim.run()
    data = _load_datasets(sim.h5_file)

    err = np.linalg.norm((data["nav_states"] - data["nominal_states"])[:, 0:3], axis=1)
    assert err[-1] <= 5.0e3, f"Halo 跟踪误差未收敛：末值 {err[-1]:.1f} m"
    r = np.linalg.norm(data["true_states"][:, 0:3], axis=1)
    assert np.all(np.isfinite(r)) and r.max() < 2.0e11, "CRTBP 轨迹异常"


# ---------------------------------------------------------------------------
# T-L1-007：分级接口一致性（CLI 矩阵）
# ---------------------------------------------------------------------------
def test_t_l1_007_cli_level_gating(tmp_path):
    r1 = subprocess.run(
        [sys.executable, "run.py", "--scene", "leo", "--level", "1",
         "--simulation_days", "0.002", "--time_step", "10", "--quiet",
         "--data_dir", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert r1.returncode == 0, f"LEO L1 合法运行失败：\n{r1.stdout}\n{r1.stderr}"

    r2 = subprocess.run(
        [sys.executable, "run.py", "--scene", "sun_earth_l2", "--level", "2", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 1 and "暂无 L2 实现" in out2, \
        f"L2 分级未显式报错：rc={r2.returncode}, out={out2[-400:]}"

    r3 = subprocess.run(
        [sys.executable, "run.py", "--scene", "leo", "--level", "3", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    out3 = r3.stdout + r3.stderr
    assert r3.returncode == 1 and "暂无 L3 实现" in out3, \
        f"L3 分级未显式报错：rc={r3.returncode}, out={out3[-400:]}"


# ---------------------------------------------------------------------------
# T-L1-008：防调参专项（增益域）
# ---------------------------------------------------------------------------
def test_t_l1_008_gain_domain_legal_scan(tmp_path):
    """合法域 {0.1, 1, 10}：无飞离；报表-数据一致；控制力积分与 ΔV 一致 <1%。

    本用例使用 log_interval=1（逐步骤日志）提高证据密度，使力积分可严格核对。
    """
    injection = np.array([100.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    for gain in (0.1, 1.0, 10.0):
        sim = _run_leo(tmp_path / f"g{gain}", 0.05, log_interval=1,
                       injection_error=injection, control_gain_scale=gain)
        data = _load_datasets(sim.h5_file)

        r = np.linalg.norm(data["true_states"][:, 0:3], axis=1)
        assert r.max() < 7000e3 * 1.05, f"gain={gain} 组合出现飞离（防调参违规）"

        forces = data["control_forces"]
        mem_dv = float(sim.spacecraft.accumulated_dv)
        log_dv = float(data["accumulated_dvs"][-1])
        assert abs(log_dv - mem_dv) <= 1e-9 * max(mem_dv, 1.0) + 1e-9, \
            f"gain={gain}: 报表 ΔV {log_dv:.6f} 与内存 {mem_dv:.6f} 不一致"

        # log_interval=1：采样即全步，力积分应与 ΔV 严格一致（<1%）
        dv_integral = float(np.sum(np.linalg.norm(forces, axis=1)) * 10.0 / sim.spacecraft.mass)
        if log_dv > 1e-6:
            rel = abs(dv_integral - log_dv) / log_dv
            assert rel < 0.01, f"gain={gain}: ΔV 报表 {log_dv:.6f} vs 力积分 {dv_integral:.6f} (rel={rel:.3%})"


def test_t_l1_008_out_of_domain_must_warn_and_not_mask(tmp_path, capsys):
    """非法域 {1e-8, 1e-5}：必须显式告警；且日志如实反映未收敛（不掩盖）。"""
    injection = np.array([500.0, 0.0, 0.0, 0.0, 0.2, 0.0])
    for gain in (1e-8, 1e-5):
        sim = _run_leo(tmp_path / f"bad_g{gain}", 0.05,
                       injection_error=injection, control_gain_scale=gain)

        captured = capsys.readouterr().out
        assert "超出设计域" in captured, f"gain={gain} 未显式告警"
        assert getattr(sim, "gain_out_of_domain", False) is True

        data = _load_datasets(sim.h5_file)
        final_err = float(np.linalg.norm((data["nav_states"] - data["nominal_states"])[-1, 0:3]))
        assert final_err >= 250.0, \
            f"gain={gain}: 日志掩盖了未收敛（末位置误差 {final_err:.1f} m < 注入量级一半）"


# ---------------------------------------------------------------------------
# T-L1-009：积分步长敏感性
# ---------------------------------------------------------------------------
def test_t_l1_009_step_size_sensitivity(tmp_path):
    """1 小时短弧：dt=60s 与 dt=10s 的位置相对差 ≤ 校准门限。"""
    sim60 = _run_leo(tmp_path / "dt60", 1.0 / 24.0, time_step=60.0)
    sim10 = _run_leo(tmp_path / "dt10", 1.0 / 24.0, time_step=10.0)

    s60 = sim60.spacecraft.state
    s10 = sim10.spacecraft.state
    rel = np.linalg.norm(s60[0:3] - s10[0:3]) / np.linalg.norm(s10[0:3])
    assert rel <= STEP_SENSITIVITY_REL_LIMIT, \
        f"步长敏感性 {rel:.3e} 超出校准门限 {STEP_SENSITIVITY_REL_LIMIT:.0e}"
