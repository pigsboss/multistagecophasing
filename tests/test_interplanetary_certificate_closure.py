# tests/test_interplanetary_certificate_closure.py
# -*- coding: utf-8 -*-
"""小推力证书＋返程闭合＋火星借力链＋全链对账（Ceres R5／R6 能力沉淀件）契约测试。

对拍口径
--------
* 源实现：谷神星探测专题 自由探索 R5／R6 仿真线 ``工具/s4_lt3_r5.py``、``工具/s4_lt4_r6.py``
  （连同其只读复用的 ``工具/s4_lt2_r1.py`` v0.2）。
* 对拍锚：其产物 ``任务/拓展_R1_返程能力.json``、``任务/拓展_R5_{时间延后,kdeg提升,
  地球借力EFB,化学离场脉冲,vinf前沿}.json``、``任务/拓展_R6_{去程火星借力,返程2099复核,
  全链对账}.json`` 的逐字子集，固化在 ``tests/fixtures/interplanetary_r5r6_anchors.json``，
  使测试可离线独立复算。
* 容差：星历／螺旋积分（RK4）／多圈 Lambert／枚举类量要求**逐位相等**（``==``，
  机实测 diff 0）；Nelder-Mead（``min_time_spiral``）与 least_squares/SLSQP（SF）的绝对
  读数随 scipy 版本漂移 ⇒ 只锚关系或按 ``rtol`` 声明容差（本批实测 ``min_radius_at_T``
  在 scipy 1.18.1 下与源件**逐位一致**，故仍按 ``==`` 锚）。
* 时间盒：本件含 3 项慢算例（``min_radius_at_T`` ×2 ≈ 190 s、``opt3_efb`` ≈ 21 s、
  ``run_vinf`` ≈ 38 s）；合计 < 6 min，在 15 min 内。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mission_sim.analysis.interplanetary import chain_accounting as CA
from mission_sim.analysis.interplanetary import closure_options as CO
from mission_sim.analysis.interplanetary import flyby_mars as FM
from mission_sim.analysis.interplanetary import low_thrust as LT
from mission_sim.analysis.interplanetary import sf_design as D
from mission_sim.analysis.interplanetary import thrust_certificate as TC

ANCH = json.loads(
    (Path(__file__).parent / "fixtures" / "interplanetary_r5r6_anchors.json").read_text(
        encoding="utf-8"
    )
)
R1C = ANCH["r1_capability"]
R5D = ANCH["r5_time_delay"]
R5K = ANCH["r5_kdeg"]
R5E = ANCH["r5_efb"]
R5C = ANCH["r5_chem"]
R5V = ANCH["r5_vinf"]
R6M = ANCH["r6_mars"]
R6B = ANCH["r6_return2099"]
R6C = ANCH["r6_chain"]

TGT = R1C["target_radius_au"]
RTOL_GRAD = 1e-9  # lstsq 梯度（BLAS 求和次序差异，尾数级）


def _rows_equal(got, want, keys):
    return all(got[k] == want[k] for k in keys)


# ====================================================================== 螺旋核与弹道界
class TestSpiralCoreAndBallistic:
    """固定律螺旋积分、SEP 加速度一致性、弹道几何界（逐位锚）。"""

    def test_ephemeris_derived_anchors_bitwise(self):
        dep_r, dep_v = LT.state_off("ceres", TC.T_DEPART)
        arr_r, earth_v = LT.state_off("earth", TC.T_BACK)
        assert len(dep_r) == 3 and len(dep_v) == 3  # 三维状态
        assert LT.vnorm(dep_r) / LT.AU_KM == R1C["depart_r_au"]
        assert LT.vnorm(dep_v) == R1C["depart_v_kms"]
        assert LT.vnorm(earth_v) == R1C["earth_arr_v_kms"]
        assert LT.vnorm(arr_r) / LT.AU_KM == TGT
        assert R1C["tof_avail_d"] == 1719.0
        assert LT.ymd(TC.T_DEPART) == R1C["t_depart"] == "2038-10-30"
        assert LT.ymd(TC.T_BACK) == R1C["t_back"] == "2043-07-15"

    def test_ballistic_bounds_bitwise(self):
        r_dep = LT.vnorm(LT.state_off("ceres", TC.T_DEPART)[0]) / LT.AU_KM
        r_arr = LT.vnorm(LT.state_off("earth", TC.T_BACK)[0]) / LT.AU_KM
        v_e = LT.vnorm(LT.state_off("earth", TC.T_BACK)[1])
        bb = TC.ballistic_bounds(r_dep, r_arr, v_e)
        want = R1C["ballistic"]
        for k, v in want.items():
            assert bb[k] == v, k
        # 解析关系：a_max = (r_dep+r_arr)/2；a_min = r_dep/2
        assert bb["a_max_au"] == (r_dep + r_arr) / 2.0
        assert bb["a_min_au"] == r_dep / 2.0
        assert bb["vinf_at_amax"] == bb["v_arr_at_amax"] - v_e
        assert bb["vinf_hard_lower"] == bb["v_arr_at_amin"] - v_e
        assert bb["vinf_hard_lower"] < bb["vinf_at_amax"]

    def test_a_sep_matches_sep_model_bitwise(self):
        m = LT.SEPModel()
        for r_au in (1.0, 2.7655, 2.979):
            for mass in (TC.M0, 4000.0, 3500.0):
                r_km = r_au * LT.AU_KM
                assert TC.a_sep(r_km, mass) == m.a_km_s2(r_km, mass)
        assert TC.a_sep(LT.AU_KM, -5.0) == m.a_km_s2(LT.AU_KM, 1.0)  # 质量下限守卫

    def test_tangential_tmin_and_r_curve_bitwise(self):
        sp = CO.tangential_tmin(k_deg=1.0)
        got = R5D["tangential_tmin"]
        assert sp["reached"] is True
        assert sp["t_d"] == got["t_min_d"]
        assert sp["dv"] == got["dv_kms"]
        assert sp["state"][6] == got["m_final_kg"]
        assert sp["speed"] == got["speed_kms"]
        assert sp["vinf_lower_kms"] == got["vinf_lower_kms"]
        assert sp["arrival"] == got["arrival"]
        curve = R5D["r_curve"]
        assert len(sp["hist"]) == len(curve)
        for h, w in zip(sp["hist"], curve):
            assert h["t_d"] == w["t_d"]
            assert h["r_au"] == w["r_au"]

    def test_tmin_monotone_and_gap_in_kdeg(self):
        """k_deg 增大 ⇒ t_min 减小；k_deg=1.0/0.9262 档均**不足** 1719 d（边界档）。"""
        t09262 = CO.tangential_tmin(k_deg=0.9262)
        t100 = CO.tangential_tmin(k_deg=1.0)
        assert t09262["t_d"] > t100["t_d"] > 1719.0
        row1 = next(r for r in R5K["rows"] if r["k_deg"] == 1.0)
        assert t100["t_d"] == row1["t_min_d"]

    def test_thrust_mult_for_tof_cross_checks_kstar(self):
        """bisection（固定律）与行表插值（opt2）两条独立路径给同一 k*。"""
        kstar = TC.thrust_mult_for_tof(1719.0, target_au=TGT, lo=1.0, hi=1.35)
        assert kstar is not None
        assert kstar == pytest.approx(R5K["k_star_for_1719d"], rel=3e-3)
        assert (
            TC.thrust_mult_for_tof(600.0, target_au=TGT, lo=1.0, hi=1.35) is None
        )  # 太短

    def test_spiral_run_wall_and_target_crossing(self):
        r0, v0 = LT.state_off("ceres", TC.T_DEPART)
        wall = TC.spiral_run(
            r0,
            v0,
            TC.M0,
            lambda tt: math.pi,
            k_deg=1.0,
            tmax_d=200.0,
            dt_d=0.5,
            target_au=None,
            sample_d=1e9,
        )
        assert wall["reached"] is False
        assert abs(wall["t_d"] - 200.0) < 1e-9  # 跑满 tmax
        assert wall["dv"] > 0.0
        cross = TC.spiral_run(
            r0,
            v0,
            TC.M0,
            lambda tt: math.pi,
            k_deg=1.0,
            tmax_d=2400.0,
            dt_d=0.5,
            target_au=TGT,
            sample_d=1e9,
        )
        assert cross["reached"] is True
        assert cross["t_d"] < 2400.0

    def test_self_test_relations(self):
        st = TC.selftest(verbose=False)
        assert st["two_runs_identical"] is True
        assert st["a_sep_max_rel"] < 1e-12
        assert st["ballistic_bounds"]["a_max_au"] == R1C["ballistic"]["a_max_au"]
        vc = abs(
            math.sqrt(LT.MU_SUN / (TGT * LT.AU_KM))
            - math.sqrt(LT.MU_SUN / (R1C["depart_r_au"] * LT.AU_KM))
        )
        assert st["edelbaum_one_way"] == vc


# ====================================================================== 证书
class TestCertificate:
    """半径-时间不可行证书（固定 T 最小可达半径）与最短时螺旋。"""

    def test_min_radius_at_1719_bitwise_infeasible(self):
        """r_min(1719 d) > 地球再入半径 ⇒ **不可达（不可行证书）**（逐位锚）。"""
        want = next(r for r in R5D["min_radius_at_T"] if r["T_d"] == 1719.0)
        mr = TC.min_radius_at_T(1719.0, k_deg=1.0, nsec=10, dt_d=0.5, nstart=2)
        assert mr["r_min_au"] == want["r_min_au"]
        assert mr["dv_kms"] == want["dv_kms"]
        assert mr["speed_kms"] == want["speed_kms"]
        assert mr["vinf_lower_kms"] == want["vinf_lower_kms"]
        assert mr["r_min_au"] > TGT  # 不可行
        assert want["closed"] is False
        assert R5D["T_smallest_closed_d"] > 1719.0

    def test_min_radius_at_1800_closes(self):
        want = next(r for r in R5D["min_radius_at_T"] if r["T_d"] == 1800.0)
        mr = TC.min_radius_at_T(1800.0, k_deg=1.0, nsec=10, dt_d=0.5, nstart=2)
        assert mr["r_min_au"] == want["r_min_au"]
        assert mr["r_min_au"] <= TGT  # 闭合
        assert want["closed"] is True

    def test_min_radius_at_2099_r6_recheck(self):
        """R6-2 的 003 图口径复核：r_min(2099.4 d) 余量大（逐位锚）。"""
        want = R6B["certificate"]
        mr = TC.min_radius_at_T(2099.4, k_deg=1.0, nsec=10, dt_d=0.5, nstart=2)
        assert mr["r_min_au"] == want["r_min_au"]
        assert mr["dv_kms"] == want["dv_kms"]
        assert TGT - mr["r_min_au"] == want["margin_au"]
        assert mr["r_min_au"] <= TGT
        assert want["closed"] is True

    def test_min_radius_monotone_in_T_reduced(self):
        """r_min(T) 随 T 单调不增（缩减档：nsec=6、nstart=1、maxiter=600、dt=1 d）。"""
        vals = [
            TC.min_radius_at_T(T, k_deg=1.0, nsec=6, dt_d=1.0, nstart=1, maxiter=600)[
                "r_min_au"
            ]
            for T in (1400.0, 1600.0, 1800.0)
        ]
        assert vals[0] > vals[1] > vals[2]
        assert all(v > 0.0 for v in vals)

    def test_min_radius_payload_shape(self):
        mr = TC.min_radius_at_T(
            900.0, k_deg=1.0, nsec=4, dt_d=1.0, nstart=1, maxiter=200
        )
        assert mr["T_d"] == 900.0 and mr["k_deg"] == 1.0
        assert mr["dv_kms"] > 0.0 and mr["m_final_kg"] < TC.M0
        assert len(mr["law"]) == 4
        assert "Nelder-Mead" in mr["note"]

    def test_min_time_spiral_below_fixed_law(self):
        """η 律优化解的最短时 ≤ 纯切向（固定律）解（交叉验证；**缩减档** dt=1 d、maxiter=150）。"""
        r0, v0 = LT.state_off("ceres", TC.T_DEPART)
        fixed = TC.spiral_run(
            r0,
            v0,
            TC.M0,
            lambda tt: math.pi,
            k_deg=0.9262,
            tmax_d=2600.0,
            dt_d=1.0,
            target_au=TGT,
            sample_d=1e9,
        )
        opt = TC.min_time_spiral(k_deg=0.9262, target_au=TGT, dt_d=1.0, maxiter=150)
        assert opt["reached"] is True
        assert opt["t_d"] <= fixed["t_d"] + 1e-6
        assert opt["t_min_d"] == opt["t_d"]
        assert opt["m_final_kg"] == opt["state"][6]
        assert 8.0 < opt["dv"] < 20.0
        assert len(opt["law"]) == 3


# ====================================================================== 脉冲 v∞ 前沿
class TestVinfFrontier:
    """脉冲（多圈 Lambert）v∞ 扫描／Pareto 前沿／门限表。"""

    def test_vinf_scan_r6_2099_bitwise(self):
        t0_list = [TC.T_DEPART, LT.dfe(2038, 9, 4)]
        tof_grid = list(np.arange(1800.0, 2200.0 + 1e-9, 25.0))
        tof_grid.append(2099.4)
        tof_grid.sort()
        rows = TC.vinf_scan(t0_list, tof_grid, nmax=3, nscan=150)
        want = R6B["vinf_scan"]
        assert len(rows) == want["n_rows"]
        assert want["n_tof"] == len(tof_grid)
        assert want["departures"] == [LT.ymd(t) for t in t0_list]
        fr = TC.frontier(rows)
        assert fr[0] == want["min_vinf_overall"]
        assert (
            TC.threshold_table(fr, thresholds=(6.0, 6.3, 6.6, 7.0))
            == want["thresholds"]
        )
        near = [r for r in rows if abs(r["tof_d"] - 2099.4) < 1.0]
        assert len(near) == want["near_2099"]["n"]
        assert min(near, key=lambda r: r["vinf_kms"]) == want["near_2099"]["min_vinf"]
        # 全部门限不可达 ⇒ 003 的 v∞=6.00 在本岛口径下无脉冲解
        assert all(not t["feasible"] for t in want["thresholds"])
        assert min(r["vinf_kms"] for r in near) > 6.0

    def test_run_vinf_r5_window_bitwise(self):
        out = CO.run_vinf(verbose=False)
        assert out["n_rows"] == R5V["n_rows"]
        assert out["n_t0"] == R5V["n_t0"]
        assert out["n_tof"] == R5V["n_tof"]
        assert out["t0_range"] == R5V["t0_range"]
        assert out["tof_range"] == R5V["tof_range"]
        assert out["min_vinf"] == R5V["min_vinf"]
        assert out["thresholds"] == R5V["thresholds"]
        for k, v in R5V["ballistic"].items():
            assert out["ballistic"][k] == v, k
        assert out["ballistic"]["vinf_at_amax"] == R1C["ballistic"]["vinf_at_amax"]
        assert out["ballistic"]["note"] == R1C["ballistic_note"]
        # 可达性：≤6.0／6.3 无解，≥6.6 有解 ⇒ "v∞=6.00 不可达"证书
        th = {t["vinf_max"]: t["feasible"] for t in out["thresholds"]}
        assert th[6.0] is False and th[6.3] is False and th[6.6] is True

    def test_frontier_is_pareto_ascending_vinf(self):
        rows = [
            dict(vinf_kms=7.0, dv1_kms=5.0),
            dict(vinf_kms=6.5, dv1_kms=6.0),
            dict(vinf_kms=8.0, dv1_kms=4.5),
            dict(vinf_kms=6.5, dv1_kms=7.5),  # 被 (6.5, 6.0) 支配
        ]
        fr = TC.frontier(rows)
        assert fr == [
            dict(vinf_kms=6.5, dv1_kms=6.0),
            dict(vinf_kms=7.0, dv1_kms=5.0),
            dict(vinf_kms=8.0, dv1_kms=4.5),
        ]
        assert fr[0]["vinf_kms"] == min(r["vinf_kms"] for r in rows)
        assert all(
            fr[i]["vinf_kms"] > fr[i - 1]["vinf_kms"]
            and fr[i]["dv1_kms"] < fr[i - 1]["dv1_kms"]
            for i in range(1, len(fr))
        )
        # 前沿随 v∞ 单调下探：再加一条更省 Δv 的更高 v∞ 解 ⇒ 被纳入（不被支配）
        fr2 = TC.frontier(rows + [dict(vinf_kms=9.0, dv1_kms=3.0)])
        assert fr2[-1] == dict(vinf_kms=9.0, dv1_kms=3.0)

    def test_empty_scan_and_infeasible_thresholds(self):
        assert TC.vinf_scan([TC.T_DEPART], [], nmax=2, nscan=50) == []
        assert TC.frontier([]) == []
        rows = [dict(vinf_kms=100.0, dv1_kms=1.0)]
        tab = TC.threshold_table(rows, thresholds=(6.0, 6.3))
        assert tab == [
            dict(vinf_max=6.0, feasible=False),
            dict(vinf_max=6.3, feasible=False),
        ]


# ====================================================================== 闭合矩阵（R5 四案）
class TestClosureMatrix:
    """① 时间延后／② k_deg／③ EFB／④ 化学离场（逐位锚）。"""

    def test_opt2_kdeg_rows_bitwise(self):
        out = CO.opt2()
        assert len(out["rows"]) == len(R5K["rows"])
        for got, want in zip(out["rows"], R5K["rows"]):
            assert _rows_equal(
                got,
                want,
                (
                    "k_deg",
                    "t_min_d",
                    "dv_kms",
                    "m_final_kg",
                    "speed_kms",
                    "vinf_lower_kms",
                    "arrival",
                ),
            )
        assert out["k_star_for_1719d"] == R5K["k_star_for_1719d"]
        assert out["gradient_d_per_kdeg"] == pytest.approx(
            R5K["gradient_d_per_kdeg"], rel=RTOL_GRAD
        )
        assert out["gradient_lstsq_rms_d"] == pytest.approx(
            R5K["gradient_lstsq_rms_d"], rel=RTOL_GRAD
        )
        assert out["local_gradients"] == R5K["local_gradients"]
        assert out["gradient_r1_ref_d"] == -2630.0
        # 梯度线性度判据（源件：RMS<25 d ⇒ 近线性）
        assert out["gradient_lstsq_rms_d"] < 25.0
        assert abs(out["gradient_d_per_kdeg"] - (-2630.0)) < 100.0

    def test_opt4_chem_rows_bitwise(self):
        out = CO.opt4_chem()
        assert len(out["rows"]) == len(R5C["rows"])
        for got, want in zip(out["rows"], R5C["rows"]):
            assert _rows_equal(
                got,
                want,
                (
                    "dv_chem_kms",
                    "t_min_d",
                    "dv_sep_kms",
                    "m_final_kg",
                    "speed_kms",
                    "vinf_lower_kms",
                    "arrival",
                    "prop_mono_kg",
                    "prop_bipro_kg",
                    "closed",
                ),
            )
        assert out["dt_per_dvchem_d_per_kms"] == pytest.approx(
            R5C["dt_per_dvchem_d_per_kms"], rel=1e-12
        )
        assert out["dv_chem_for_1719d_kms"] == R5C["dv_chem_for_1719d_kms"]
        # 分岔：Δv_c=0.30 **不闭合**，Δv_c=0.37 **闭合**（源件"救得了"的最小脉冲）
        closed = {r["dv_chem_kms"]: r["closed"] for r in out["rows"]}
        assert closed[0.0] is False and closed[0.30] is False
        assert closed[0.37] is True and closed[2.00] is True
        # 推进剂账：双组元 < 单组元（同 Δv_c）
        assert all(
            r["prop_bipro_kg"] < r["prop_mono_kg"]
            for r in out["rows"]
            if r["dv_chem_kms"]
        )

    def test_opt3_efb_bitwise_and_negative_certificate(self):
        out = CO.opt3_efb()
        assert out["arc2_count"] == R5E["arc2_count"]
        assert out["n_t1"] == R5E["n_t1"]
        assert out["arc2_mneed_max_kms"] == R5E["arc2_mneed_max_kms"]
        assert out["arc1_vinf_in_min_kms"] == R5E["arc1_vinf_in_min_kms"]
        assert out["n_match_free"] == R5E["n_match_free"] == 0
        assert out["matches_free"] == R5E["matches_free"] == []
        assert out["status"] == R5E["status"] == "free_infeasible"
        assert out["rp_km"] == R5E["rp_km"]
        assert out["powered_best"] == R5E["powered_best"]
        # **自由飞掠不可行证书**：两区间无交集（飞掠保 |v∞|，无法把 v∞ 降下来）
        assert out["arc2_mneed_max_kms"] < out["arc1_vinf_in_min_kms"]

    def test_opt3_no_arc2_boundary(self):
        out = CO.opt3_efb(
            t1_grid=[LT.dfe(2041, 1, 1)], tof2_grid=[700.0, 900.0], vinf2_max=0.0
        )
        assert out["arc2_count"] == 0
        assert out["status"] == "no_arc2"
        assert "arc1_vinf_in_min_kms" not in out

    def test_efb_dvmax_formula(self):
        assert CO.flyby_dvmax(0.0, 6578.137) == math.pi  # e=1 ⇒ 无约束
        d = CO.flyby_dvmax(6.0, 6578.137)
        e = 1.0 + 6578.137 * 36.0 / CO.MU_EARTH
        assert d == 2.0 * math.asin(1.0 / e)
        assert CO.flyby_dvmax(6.0, 6578.137) < CO.flyby_dvmax(
            3.0, 6578.137
        )  # v∞ 小 ⇒ 转角大

    def test_tangential_impulse_directions(self):
        retro = CO.tangential_tmin(impulse_kms=0.5, impulse="retro")
        pro = CO.tangential_tmin(impulse_kms=0.5, impulse="pro")
        rad = CO.tangential_tmin(impulse_kms=0.5, impulse="rad_in")
        assert retro["t_d"] < pro["t_d"]  # 逆行减速最省时
        assert retro["t_d"] < rad["t_d"]
        assert all(r["reached"] for r in (retro, pro, rad))

    def test_sf_solve2_local_smoke_single_leg(self):
        p = TC.prob_ret(tof_d=1867.0, nseg=8, vinf_max=10.0)
        x0 = CO.init_spiral_local(p)
        assert x0.shape == (p.nv,)
        assert np.any(x0 != 0.0)
        r = CO.sf_solve2_local(p, x0=x0, feas_iter=300, polish_iter=60)
        assert len(r["x"]) == p.nv
        assert math.isfinite(r["resid_km"]) and math.isfinite(r["total"])
        assert r["total"] > 0.0
        assert r["counters"]["enc"][-1]["body"] == "earth"
        assert "init_spiral_local" in r["method"]

    def test_init_spiral_single_leg_defect_and_local_fix(self):
        """边界：源件 ``init_spiral`` 对单腿问题 IndexError；本件 ``init_spiral_local`` 正常。"""
        p = TC.prob_ret(tof_d=1719.0, nseg=4, vinf_max=10.0)
        with pytest.raises(IndexError):
            TC.init_spiral(p)
        x = CO.init_spiral_local(p)
        assert x.shape == (p.nv,) and float(np.abs(x).max()) > 0.0

    def test_cli_opt2_writes_json(self, tmp_path):
        code = CO.main(["--opt2", "--out-dir", str(tmp_path)])
        assert code == 0
        payload = json.loads(
            (tmp_path / "拓展_R5_kdeg提升.json").read_text(encoding="utf-8")
        )
        assert len(payload["rows"]) == len(R5K["rows"])
        assert payload["k_star_for_1719d"] == R5K["k_star_for_1719d"]

    def test_cli_no_args_returns_help(self, capsys):
        assert CO.main([]) == 0
        assert "闭合方案矩阵" in capsys.readouterr().out


# ====================================================================== 火星借力链（R6-1／R6-2）
class TestMarsFlyby:
    """火星星历／飞掠几何／GA 可行性扫描／链式 SF／R6-2 复核。"""

    def test_mars_spec_bitwise(self):
        spec = FM.mars_spec(2463963.5)
        want = R6M["mars_spec"]
        for k, v in want.items():
            assert spec[k] == v, k
        assert spec["name"] == "mars(JPL近似根数@2034-01-01)"

    def test_register_mars_adds_body(self):
        spec = FM.register_mars()
        assert LT.is_body("mars")
        assert spec == R6M["mars_spec"]
        r, v = LT.state_off("mars", LT.dfe(2031, 8, 13))
        assert 1.38 < LT.vnorm(r) / LT.AU_KM < 1.67  # 火星日心距带
        r2, v2 = LT.kep(r, v, 0.0)
        assert tuple(r2) == tuple(r) and tuple(v2) == tuple(v)

    def test_dmax_deg_anchor_and_monotone(self):
        alt300 = R6M["chains"][2]["flyby"]["dmax_deg_alt300"]
        vinf = R6M["chains"][2]["flyby"]["vinf_in_kms"]
        assert FM.dmax_deg(vinf, 300.0) == alt300
        assert FM.dmax_deg(vinf, 500.0) == R6M["chains"][2]["flyby"]["dmax_deg_alt500"]
        assert FM.dmax_deg(4.85, 300.0) > FM.dmax_deg(
            vinf, 300.0
        )  # 4.85 < 4.88 ⇒ 转角略大
        # 源件 dmax_deg 返回**度**且无 e≤1 守卫 ⇒ v∞→0 给 180°（而非 π）；
        # 对比 closure_options.flyby_dvmax（返回**弧度**且 e≤1 ⇒ π）。两处口径差异**原样继承**。
        assert FM.dmax_deg(0.0, 300.0) == 180.0
        assert CO.flyby_dvmax(0.0, 6578.137) == math.pi

    def test_angle_and_helio_arc(self):
        FM.register_mars()
        assert FM.angle_deg([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]) == 90.0
        assert math.isnan(FM.angle_deg([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        arc = FM.helio_arc_deg(
            ("earth", "mars"), LT.dfe(2030, 12, 15), LT.dfe(2031, 8, 13)
        )
        assert 0.0 < arc <= 180.0

    def test_ga_scan_full_grid_bitwise(self):
        FM.register_mars()
        got = FM.ga_scan(verbose=False)
        want = R6M["ga_scan"]
        for k in (
            "kind",
            "nmax",
            "nscan",
            "alt_km",
            "vinf_window",
            "mag_tol",
            "t1_range",
            "n_lambert",
            "n_magnitude_matched",
            "n_feasible",
            "min_turn",
            "best_feasible",
            "vout_min_by_rev",
        ):
            assert got[k] == want[k], k
        assert want["n_feasible"] == 0 and want["best_feasible"] is None
        assert (
            got["min_turn"]["turn_deg"] > got["min_turn"]["dmax_deg"]
        )  # 最小转角仍超界

    def test_ga_scan_wide_window_bitwise(self):
        FM.register_mars()
        got = FM.ga_scan(vinf_lo=0.5, vinf_hi=8.0, mag_tol=1.0, verbose=False)
        want = R6M["ga_scan_wide"]
        for k in (
            "nmax",
            "nscan",
            "vinf_window",
            "mag_tol",
            "n_lambert",
            "n_magnitude_matched",
            "n_feasible",
            "min_turn",
        ):
            assert got[k] == want[k], k
        assert want["n_feasible"] == 0
        assert want["min_turn"]["turn_deg"] > want["min_turn"]["dmax_deg"]

    def test_ga_scan_reduced_consistency(self):
        FM.register_mars()
        t1 = LT.dfe(2031, 8, 13)
        got = FM.ga_scan(
            t1_grid=[t1 - 8.0, t1, t1 + 8.0],
            dt0_grid=[200.0, 220.0, 240.0],
            dt2_grid=[1500.0, 1600.0, 1700.0, 1800.0],
            vinf_lo=0.5,
            vinf_hi=8.0,
            mag_tol=1.0,
            verbose=False,
        )
        assert got["n_lambert"] > 0
        for r in got["sample_top"]:
            assert r["feasible"] == (
                abs(r["vinf_in"] - r["vinf_out"]) <= 1.0
                and r["turn_deg"] <= r["dmax_deg"]
            )
        assert set(got["vout_min_by_rev"]) == {
            "rev0pro",
            "rev0ret",
            "rev1pro",
            "rev1ret",
        }
        assert got["sample_top"] == sorted(
            got["sample_top"],
            key=lambda r: (abs(r["vinf_in"] - r["vinf_out"]) > 1.0, r["turn_deg"]),
        )

    def test_chain_sf_geometry_flags_relation(self):
        """链式 SF 的**飞掠几何判定**（绝对 Δv 依赖 scipy 版本，不锚定）。"""
        FM.register_mars()
        c = FM.chain_sf(
            LT.dfe(2030, 12, 16),
            LT.dfe(2031, 8, 13),
            LT.dfe(2037, 1, 20),
            n1=8,
            n2=8,
            feas_iter=600,
            polish_iter=120,
            label="test",
        )
        assert c["dv_propulsive_kms"] == c["dv_total_kms"] - c["flyby_impulse_kms"]
        assert c["flyby_impulse_kms"] >= 0.0
        assert len(c["legs_dv"]) == 2
        fb = c["flyby"]
        assert fb["magnitude_ok"] == (
            abs(fb["vinf_in_kms"] - fb["vinf_out_kms"]) <= 0.5
        )
        assert fb["turn_ok"] == (fb["turn_deg"] <= fb["dmax_deg_alt300"])
        assert fb["free_flyby_ok"] == (fb["magnitude_ok"] and fb["turn_ok"])
        assert 3.0 < fb["vinf_in_kms"] < 8.0  # 与 003 的 4.85 同量级（松带）
        assert c["tof1_d"] == 240.0 and c["tof2_d"] == 1987.0

    def test_chain_sf_matches_island_geometry_within_band(self):
        """同日期链（候选B）的飞掠 v∞ 与岛内锚同带（scipy 漂移 ⇒ 只做松带判据）。"""
        want = R6M["chains"][2]["flyby"]
        FM.register_mars()
        c = FM.chain_sf(
            LT.dfe(2030, 12, 16),
            LT.dfe(2031, 8, 13),
            LT.dfe(2037, 1, 20),
            n1=8,
            n2=8,
            feas_iter=600,
            polish_iter=120,
            label="band",
        )
        assert c["flyby"]["vinf_in_kms"] == pytest.approx(want["vinf_in_kms"], rel=0.35)
        assert c["flyby"]["vinf_out_kms"] == pytest.approx(
            want["vinf_out_kms"], rel=0.35
        )
        # 转角判定：本链（无论 scipy 档）在 300 km 高度下均为自由飞掠可行（岛内同一判定）
        assert c["flyby"]["turn_ok"] is True and c["flyby"]["free_flyby_ok"] is True

    def test_direct_sf_relation(self):
        FM.register_mars()
        d = FM.direct_sf(
            LT.dfe(2030, 12, 15), LT.dfe(2037, 6, 17), nseg=8, maxiter=40, multistart=1
        )
        assert d["tof_d"] == 2376.0
        assert math.isfinite(d["dv_total_kms"]) and d["dv_total_kms"] > 0.0
        assert math.isfinite(d["resid_km"])

    def test_opt2_r6_return_recheck_structure(self):
        """R6-2 复核（缩减证书档 nsec=4；vinf 扫描逐位锚）。"""
        out = FM.opt2(
            tof=2099.4,
            tofs_sf=(),
            nsec=4,
            tdep=TC.T_DEPART,
            t0_alt=LT.dfe(2038, 9, 4),
            verbose=False,
        )
        assert out["reentry"] == R6B["reentry"] == "2044-07-29"
        assert out["f003"] == R6B["f003"]
        cert = out["certificate"]
        assert cert["T_d"] == 2099.4 and cert["closed"] is True
        assert cert["margin_au"] > 0.0
        assert cert["note"] == "r_min 是「最小半径解」（会把半径压过头），非 Δv 最优"
        assert out["vinf_scan"]["n_rows"] == R6B["vinf_scan"]["n_rows"]
        assert (
            out["vinf_scan"]["min_vinf_overall"] == R6B["vinf_scan"]["min_vinf_overall"]
        )
        assert out["vinf_scan"]["departures"] == R6B["vinf_scan"]["departures"]
        assert (
            out["vinf_scan"]["near_2099"]["min_vinf"]
            == R6B["vinf_scan"]["near_2099"]["min_vinf"]
        )
        assert all(not t["feasible"] for t in out["vinf_scan"]["thresholds"])
        assert out["sf_checks"] == []
        assert out["text_vs_figure"]["gap_d"] == R6B["text_vs_figure"]["gap_d"]

    def test_flyby_chain_runs_table(self):
        assert [s["label"][:3] for s in FM.FLYBY_CHAIN_RUNS] == [
            "003",
            "候选A",
            "候选B",
        ]
        assert FM.DIRECT_SAME_DATES == ((2030, 12, 16), (2037, 3, 21))
        assert FM.F003["vinf_mars"] == 4.85 and FM.F003["tof_ret"] == 2099.4
        assert FM.DV_NOFLYBY_KMS == R6M["dv_no_flyby_ref_kms"] == 17.075
        assert FM.XE_BUDGET_KG == 1320.0

    def test_cli_no_args_returns_help(self, capsys):
        assert FM.main([]) == 0
        assert "火星借力链" in capsys.readouterr().out


# ====================================================================== 全链对账（R6-3）
class TestChainAccounting:
    """全链 Δv 对账与时间线建议（以岛内 opt1／opt2 结果为输入复算）。"""

    def _inputs(self):
        return (
            dict(chains=[R6C["outbound"]]),
            dict(sf_checks=[R6C["return"]], certificate=R6B["certificate"]),
        )

    def test_chain_reconcile_bitwise(self):
        o1, o2 = self._inputs()
        out = CA.chain_reconcile(o1, o2, verbose=False)
        assert out["reconcile"] == R6C["reconcile"]
        assert out["outbound"] == R6C["outbound"]
        assert out["return"] == R6C["return"]
        assert out["rings"] == R6C["rings"]
        assert out["timeline_options"] == R6C["timeline_options"]
        r = out["reconcile"]
        assert r["ratio_vs_003"] == r["dv_sum_sf_kms"] / 14.0
        assert r["diff_kms"] == r["dv_sum_sf_kms"] - 14.0
        assert r["prop_sum_kg"] == CA.prop_kg(r["dv_sum_sf_kms"])
        assert r["dv_at_xe_budget_kms"] == LT.dv_from_prop(CA.M0, 1320.0, CA.ISP)
        assert abs(r["dv_at_xe_budget_kms"] - 13.034337120377815) < 1e-12

    def test_chain_picks_min_propulsive_chain(self):
        o2 = dict(sf_checks=[R6C["return"]], certificate=R6B["certificate"])
        worse = dict(R6C["outbound"])
        worse["dv_propulsive_kms"] = R6C["outbound"]["dv_propulsive_kms"] + 5.0
        out = CA.chain_reconcile(
            dict(chains=[worse, R6C["outbound"]]), o2, verbose=False
        )
        assert out["outbound"] == R6C["outbound"]
        assert out["reconcile"] == R6C["reconcile"]

    def test_chain_ignores_unconverged_chain(self):
        o2 = dict(sf_checks=[R6C["return"]], certificate=R6B["certificate"])
        bad = dict(R6C["outbound"])
        bad["resid_km"] = 1.0e6  # 未收敛 ⇒ 源件按 resid<100 km 过滤
        out = CA.chain_reconcile(dict(chains=[bad, R6C["outbound"]]), o2, verbose=False)
        assert out["outbound"] == R6C["outbound"]

    def test_chain_empty_inputs_boundary(self):
        out = CA.chain_reconcile(
            dict(chains=[]),
            dict(sf_checks=[], certificate=R6B["certificate"]),
            verbose=False,
        )
        assert "reconcile" not in out
        assert out["outbound"] is None and out["return"] is None
        assert "n/a" in out["rings"][1]["island"]

    def test_prop_kg_matches_rocket_equation(self):
        mf, prop = LT.rocket_prop(CA.M0, 5.0, CA.ISP)
        assert CA.prop_kg(5.0) == pytest.approx(prop, rel=1e-15)
        assert CA.prop_kg(0.0) == 0.0

    def test_cli_chain_roundtrip(self, tmp_path):
        o1, o2 = self._inputs()
        p1 = tmp_path / "o1.json"
        p2 = tmp_path / "o2.json"
        p1.write_text(json.dumps(o1), encoding="utf-8")
        p2.write_text(json.dumps(o2), encoding="utf-8")
        outp = tmp_path / "chain.json"
        code = CA.main(
            [
                "--chain",
                "--opt1-json",
                str(p1),
                "--opt2-json",
                str(p2),
                "--out-json",
                str(outp),
            ]
        )
        assert code == 0
        payload = json.loads(outp.read_text(encoding="utf-8"))
        assert payload["reconcile"] == R6C["reconcile"]
        assert CA.main(["--chain"]) == 0  # 缺输入 ⇒ 只打印提示


# ====================================================================== 问题工厂与边界
class TestProblemFactoryAndBoundaries:
    """返程腿问题工厂、证书小 T、极值档。"""

    def test_prob_ret_bookkeeping(self):
        p = TC.prob_ret(tof_d=1719.0, nseg=16)
        assert p.bodies == ["ceres", "earth"]
        assert p.kinds == ["start_body", "arrive_earth"]
        assert p.bstart is False and p.c3 is None and p.vinf_max == 6.0
        assert p.N == 16 and p.nv == 3 * 15
        assert p.nodes == [TC.T_DEPART, TC.T_DEPART + 1719.0]
        assert "返程" in p.name

    def test_prob_ret_custom_vinf_and_t0(self):
        t0 = LT.dfe(2038, 9, 4)
        p = TC.prob_ret(tof_d=2099.4, nseg=8, vinf_max=10.0, t0=t0)
        assert p.nodes == [t0, t0 + 2099.4] and p.vinf_max == 10.0

    def test_min_radius_small_T_finite(self):
        mr = TC.min_radius_at_T(120.0, nsec=3, dt_d=1.0, nstart=1, maxiter=150)
        assert math.isfinite(mr["r_min_au"]) and mr["r_min_au"] > 0.0
        assert mr["dv_kms"] > 0.0

    def test_sep_accel_inverse_square_and_kdeg_linear(self):
        a1 = TC.a_sep(LT.AU_KM, TC.M0, k_deg=1.0)
        assert TC.a_sep(2.0 * LT.AU_KM, TC.M0, k_deg=1.0) == pytest.approx(
            a1 / 4.0, rel=1e-15
        )
        assert TC.a_sep(LT.AU_KM, TC.M0, k_deg=0.9262) == pytest.approx(
            0.9262 * a1, rel=1e-15
        )

    def test_thrust_dir_orthogonal_and_radial(self):
        rv, vv = LT.state_off("ceres", TC.T_DEPART)
        t_dir = TC._thrust_dir(rv, vv, 0.0)
        out_dir = TC._thrust_dir(rv, vv, math.pi / 2)
        n = LT.vnorm(rv)
        radial = [q / n for q in rv]
        assert LT.vdot(t_dir, radial) == pytest.approx(0.0, abs=1e-12)
        assert LT.vdot(out_dir, radial) == pytest.approx(1.0, rel=1e-12)
        assert LT.vnorm(t_dir) == pytest.approx(1.0, rel=1e-15)

    def test_spiral_nodes_count_and_monotone_mass(self):
        r0, v0 = LT.state_off("ceres", TC.T_DEPART)
        nodes = TC.spiral_nodes(
            r0, v0, TC.M0, [0.0, 50.0, 100.0], lambda tt: math.pi, dt_d=1.0
        )
        assert len(nodes) == 3
        masses = [n[2] for n in nodes]
        assert masses[0] == TC.M0
        assert masses[0] > masses[1] > masses[2]
        assert len(nodes[0][3]) == 3
