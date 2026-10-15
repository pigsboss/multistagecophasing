# tests/test_interplanetary_low_thrust.py
# -*- coding: utf-8 -*-
"""SEP 低推力设计器＋顺访链（Ceres E1 能力沉淀件）契约测试。

对拍口径
--------
* 源实现：谷神星探测专题 E1 仿真线 ``工具/s4_lt_common.py``、``工具/s4_lt_design.py``、
  ``工具/s4_lt_variants.py``（日心二体 + 分段冲量 Sims-Flanagan；a(r,m)=2ηP0·k_deg/(r_AU²·v_e·m)）。
* 对拍锚：其产物 ``任务/拓展_SF_基线.json``、``任务/拓展_SF_顺访_V1_out_lutetia.json``、
  ``任务/拓展_SF_顺访_V2_out_vesta.json``。锚（含**设计向量 x**）以常量固化在
  ``tests/fixtures/interplanetary_low_thrust_anchors.json``（岛内机读结果的逐字子集），
  使测试可离线独立复算。
* 容差：解向量重放与脉冲参照要求**逐位相等**（``==``，机实测 diff 0）；纯解析/数值
  量按各自 ``rtol`` 声明。优化器输出的绝对值**不锚定**（依赖 scipy 版本），只锚关系。
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mission_sim.analysis.interplanetary import flyby_chain as FC
from mission_sim.analysis.interplanetary import low_thrust as LT
from mission_sim.analysis.interplanetary import sf_design as D
from mission_sim.analysis.interplanetary import two_body as TB

ANCHORS = json.loads(
    (
        Path(__file__).parent / "fixtures" / "interplanetary_low_thrust_anchors.json"
    ).read_text(encoding="utf-8")
)
SR1 = ANCHORS["sr1_baseline"]
V1 = ANCHORS["v1_out_lutetia"]
V2 = ANCHORS["v2_out_vesta"]


def _model(model_dict):
    """把锚里的模型字典还原为 SEPModel。"""
    return LT.SEPModel(
        P0_W=model_dict["P0_W"],
        isp_s=model_dict["isp_s"],
        eta=model_dict["eta"],
        k_deg=model_dict["k_deg"],
    )


class TestConstantsAndEphemeris:
    """常量内联与星历口径（与 two_body 双路一致）。"""

    def test_constants_audit_bitwise(self):
        rows = LT.constants_audit()
        assert rows and all(r["rel"] == 0.0 for r in rows)

    def test_physical_constants_match_source(self):
        assert LT.AU_KM == 149597870.7
        assert LT.MU_SUN == 1.32712440018e11
        assert LT.JD0 == TB.JD_J2000 == 2451545.0
        assert LT.G0 == 9.80665

    def test_state_matches_two_body(self):
        """low_thrust.state_off 与 two_body.Ephemeris 逐位一致（同一星历内核）。"""
        for key in ("earth", "ceres", "lutetia", "vesta"):
            spec = TB.EARTH_J2000 if key == "earth" else LT.BODIES[key]
            ref = TB.Ephemeris(spec)
            for jd in (2451545.0, 2462850.5, 2465226.5, 2461200.5):
                r_mine, v_mine = LT.state(key, jd)
                r_ref, v_ref = ref.state(jd)
                assert tuple(r_mine) == tuple(r_ref)
                assert tuple(v_mine) == tuple(v_ref)

    def test_body_registry_includes_pool_targets(self):
        for key in ("earth", "ceres", "lutetia", "venetia487", "may348", "sidonia579"):
            assert LT.is_body(key)
        assert LT.POOL_TARGETS["venetia487"]["a_au"] == 2.670731541585185
        assert LT.POOL_TARGETS["may348"]["epoch_jd"] == 2461200.5
        assert {v["mid"] for v in FC.VARIANTS} <= set(LT.BODIES)

    def test_register_body_adds_ephemeris(self):
        spec = dict(LT.BODIES["vesta"])
        spec["name"] = "test-body"
        LT.register_body("__test_body__", spec)
        try:
            r, v = LT.state_off("__test_body__", LT.dfe(2033, 2, 14))
            r2, v2 = LT.state_off("vesta", LT.dfe(2033, 2, 14))
            assert tuple(r) == tuple(r2) and tuple(v) == tuple(v2)
        finally:
            LT.BODIES.pop("__test_body__", None)
            LT.EPH.pop("__test_body__", None)

    def test_dfe_and_ymd_round_trip(self):
        assert LT.ymd(LT.dfe(2030, 12, 15)) == "2030-12-15"
        assert LT.dfe(2030, 12, 15) == 11305.5
        assert LT.dfe(2038, 10, 30) == 14181.5


class TestKepler:
    """通用变量 Kepler 传播器。"""

    def test_zero_dt_is_exact_identity(self):
        r0, v0 = LT.state_off("ceres", 13681.5)
        r, v = LT.kep(r0, v0, 0.0)
        assert tuple(r) == tuple(r0) and tuple(v) == tuple(v0)

    def test_vis_viva_and_energy(self):
        """地球平根数态满足活力公式，且传播后比能/半长轴守恒。"""
        r0, v0 = LT.state_off("earth", 11305.5)
        a_src = TB.EARTH_J2000["a_au"] * LT.AU_KM
        assert_allclose(
            LT.vdot(v0, v0), LT.MU_SUN * (2.0 / LT.vnorm(r0) - 1.0 / a_src), rtol=1e-12
        )
        r, v = LT.kep(r0, v0, 800.0)
        assert_allclose(
            LT.specific_energy(r, v), LT.specific_energy(r0, v0), rtol=1e-12
        )
        a0 = 1.0 / (2.0 / LT.vnorm(r0) - LT.vdot(v0, v0) / LT.MU_SUN)
        a1 = 1.0 / (2.0 / LT.vnorm(r) - LT.vdot(v, v) / LT.MU_SUN)
        assert_allclose(a1, a0, rtol=1e-12)

    @pytest.mark.parametrize("dt_d", [100.0, 400.0, 1200.0])
    def test_vs_rk4(self, dt_d):
        r0, v0 = LT.state_off("ceres", 13681.5)
        r1, _ = LT.kep(r0, v0, dt_d)
        rk, _ = LT.rk4(r0, v0, dt_d, n=max(4000, int(dt_d * 20)))
        assert LT.vnorm(LT.vsub(r1, rk)) < 1e-3

    def test_kep_series_applies_node_impulse(self):
        r0, v0 = LT.state_off("earth", 11305.5)
        dv = [0.05, -0.02, 0.01]
        rs, vs = LT.kep_series(r0, v0, [100.0, 100.0], dv_nodes=[dv])
        rs_b, vs_b = LT.kep_series(r0, v0, [100.0, 100.0])
        assert rs[0] == list(r0) and vs[0] == list(v0)
        # 源件语义：冲量在"下一段"之前施加，故节点 1 记录为未施冲量的弹道点
        assert vs[1] == vs_b[1]
        assert vs[2] != vs_b[2]
        r_imp, v_imp = LT.kep(rs_b[1], [vs_b[1][i] + dv[i] for i in range(3)], 100.0)
        assert_allclose(vs[2], v_imp, rtol=1e-14, atol=1e-12)
        assert_allclose(rs[2], r_imp, rtol=1e-14, atol=1e-6)

    def test_overflow_returns_input(self):
        """双曲支溢出（s>350）时约定返回入参本身（源件行为，不静默给错值）。"""
        r0, v0 = [1.0e12, 0.0, 0.0], [0.0, 300.0, 0.0]
        r, v = LT.kep(r0, v0, 3650.0)
        assert list(r) == list(r0)


class TestLambert:
    """Lambert 双路与多圈。"""

    @pytest.mark.parametrize("tof_d", [300.0, 800.0, 1500.0, 2376.0])
    def test_dual_path_and_endpoint_closure(self, tof_d):
        r1, _ = LT.state_off("earth", LT.dfe(2030, 12, 15))
        r2, _ = LT.state_off("ceres", LT.dfe(2030, 12, 15) + tof_d)
        sol = LT.lambert(r1, r2, tof_d)
        assert sol is not None
        rr, _ = LT.kep(r1, sol[0], tof_d)
        assert LT.vnorm(LT.vsub(rr, r2)) < 1.0
        v1_ref, _ = TB.lambert(tuple(r1), tuple(r2), tof_d * LT.DAY)
        assert LT.vnorm(LT.vsub(sol[0], [v1_ref[0], v1_ref[1], v1_ref[2]])) < 1e-6

    def test_lambert2_two_branches(self):
        r1, _ = LT.state_off("earth", 11305.5)
        r2, _ = LT.state_off("ceres", 13681.5)
        sols = LT.lambert2(r1, r2, 2376.0)
        assert 1 <= len(sols) <= 2
        assert {s["branch"] for s in sols} <= {"pro", "ret"}

    def test_lambert_multi_multirev(self):
        """2376 d 的固定日期腿需多圈解（rev≥1）才合理。"""
        r1, _ = LT.state_off("earth", 11305.5)
        r2, _ = LT.state_off("ceres", 13681.5)
        sols = LT.lambert_multi(r1, r2, 2376.0, nmax=3, nscan=300)
        assert sols and max(s["rev"] for s in sols) >= 1

    @pytest.mark.parametrize("tof_d", [0.0, -10.0])
    def test_non_positive_tof_returns_none(self, tof_d):
        r1, _ = LT.state_off("earth", 11305.5)
        r2, _ = LT.state_off("ceres", 13681.5)
        assert LT.lambert(r1, r2, tof_d) is None


class TestSEPModelAndMass:
    """功率-推力模型与质量链。"""

    def test_power_thrust_accel_relations(self):
        m = LT.SEPModel()
        assert m.ve == pytest.approx(4000.0 * 9.80665 / 1000.0, rel=0)
        assert m.P_W(LT.AU_KM) == 27500.0
        assert m.T_N(LT.AU_KM) == pytest.approx(
            2 * 0.70 * 27500.0 / (m.ve * 1000.0), rel=1e-15
        )
        assert m.a_km_s2(LT.AU_KM, 4669.0) == pytest.approx(
            2.1021136324507516e-07, rel=1e-12
        )

    def test_inverse_square_law(self):
        m = LT.SEPModel()
        assert m.P_W(2.0 * LT.AU_KM) == pytest.approx(m.P_W(LT.AU_KM) / 4.0, rel=1e-15)
        assert m.a_km_s2(2.0 * LT.AU_KM, 4669.0) == pytest.approx(
            m.a_km_s2(LT.AU_KM, 4669.0) / 4.0, rel=1e-15
        )

    @pytest.mark.parametrize("k_deg", [0.0, 1e-6, 0.9262, 1.0, 5.0])
    def test_kdeg_linear_scaling(self, k_deg):
        """k_deg 对加速度是严格线性因子（极值档不引入额外分支）。"""
        base = LT.SEPModel(k_deg=1.0)
        scaled = LT.SEPModel(k_deg=k_deg)
        assert scaled.a_km_s2(2.0 * LT.AU_KM, 4669.0) == pytest.approx(
            k_deg * base.a_km_s2(2.0 * LT.AU_KM, 4669.0), rel=1e-15, abs=0.0
        )

    def test_kdeg_zero_gives_zero_accel(self):
        assert LT.SEPModel(k_deg=0.0).a_km_s2(1.5 * LT.AU_KM, 4669.0) == 0.0

    def test_const_a_model(self):
        c = LT.ConstAModel()
        assert c.a_km_s2(1.0 * LT.AU_KM, 4669.0) == 1e-7
        assert c.a_km_s2(3.0 * LT.AU_KM, 100.0) == 1e-7
        assert c.as_dict()["kind"] == "const"

    def test_mass_chain_closes(self):
        mf, prop = LT.rocket_prop(4669.0, 13.0343)
        assert mf + prop == pytest.approx(4669.0, rel=1e-15)
        assert LT.dv_from_prop(4669.0, 1320.0) == pytest.approx(
            13.034337120377815, rel=1e-12
        )
        assert LT.dv_from_prop(4669.0, prop) == pytest.approx(13.0343, rel=1e-12)
        # 与锚（mission.yaml 记"实达 13.03"）一致
        assert abs(LT.dv_from_prop(4669.0, 1320.0) - 13.0343) < 1e-3

    def test_model_mass_matches_rocket(self):
        m = LT.SEPModel()
        assert m.mass(4669.0, 5.0) == LT.rocket_prop(4669.0, 5.0)[0]


class TestPulseBoundAndSpiralScale:
    """脉冲（Lambert）参照与解析螺旋标尺（逐位锚）。"""

    def test_pulse_bound_sr1_bitwise(self):
        pb_out = D.pulse_bound(D.prob_sr1_out(nseg=4))
        pb_ret = D.pulse_bound(D.prob_sr1_ret(nseg=4))
        assert pb_out["ok"] and pb_ret["ok"]
        assert pb_out["total"] == SR1["pulse_bound"]["out"]["total"]
        assert pb_ret["total"] == SR1["pulse_bound"]["ret"]["total"]
        assert pb_out["parts"][0]["dep"] == SR1["pulse_bound"]["out"]["parts"][0]["dep"]
        assert pb_out["parts"][0]["arr"] == SR1["pulse_bound"]["out"]["parts"][0]["arr"]
        assert pb_out["parts"][0]["rev"] == SR1["pulse_bound"]["out"]["parts"][0]["rev"]

    def test_spiral_scale_anchors(self):
        sp = D.spiral_scale()
        assert sp["spiral_round_trip_minus_c3"] == pytest.approx(
            20.088241223109485, rel=1e-12
        )
        assert sp["hohmann_round_trip"] == pytest.approx(18.68127154156226, rel=1e-12)
        assert sp["spiral_one_way"] * 2 == pytest.approx(
            sp["spiral_round_trip"], rel=1e-15
        )

    def test_pulse_bound_freetime_empty_window(self):
        assert D.pulse_bound_freetime(11305.5, "earth", "ceres", []) is None

    def test_pulse_bound_freetime_shapes(self):
        best = D.pulse_bound_freetime(
            11305.5, "earth", "ceres", [2376.0], c3=13.4, nmax=3, nscan=300
        )
        assert best is not None
        assert set(best) == {"total", "dep", "arr", "tof_d", "t1", "rev", "branch"}
        assert best["total"] >= 0.0


class TestSolutionReplay:
    """对拍核心：把岛内解向量重放进落位件，逐位复现 obj/残差/记账。"""

    def test_sr1_out_replay_bitwise(self):
        p = D.prob_sr1_out(nseg=SR1["leg_out"]["N"])
        x = np.array(SR1["leg_out"]["x"], float)
        assert p.obj(x) == SR1["leg_out"]["obj"]
        assert p.resid_km(x) == SR1["leg_out"]["resid_km"]
        c = p.counters(x)
        assert c["total"] == SR1["leg_out"]["counters"]["total"]
        assert c["util"] == SR1["leg_out"]["counters"]["util"]
        assert c["mass_final"] == SR1["leg_out"]["counters"]["mass_final"]
        assert c["n_binding"] == SR1["leg_out"]["counters"]["n_binding"]
        assert c["dv_sum_cap"] == SR1["leg_out"]["counters"]["dv_sum_cap"]
        assert c["enc"][0]["r_miss_km"] == SR1["leg_out"]["enc"][0]["r_miss_km"]
        assert c["enc"][0]["vinf_kms"] == SR1["leg_out"]["enc"][0]["vinf_kms"]

    def test_sr1_ret_replay_bitwise(self):
        p = D.prob_sr1_ret(nseg=SR1["leg_ret"]["N"])
        x = np.array(SR1["leg_ret"]["x"], float)
        assert p.obj(x) == SR1["leg_ret"]["obj"]
        assert p.resid_km(x) == SR1["leg_ret"]["resid_km"]
        assert p.counters(x)["total"] == SR1["leg_ret"]["counters"]["total"]
        assert p.counters(x)["n_binding"] == SR1["leg_ret"]["counters"]["n_binding"]

    def test_sr1_nodes_and_dates(self):
        assert D.prob_sr1_out(nseg=16).nodes == SR1["leg_out"]["nodes"]
        assert D.prob_sr1_out(nseg=16).fmt().count("2037-06-17") == 1
        assert D.prob_sr1_ret(nseg=14).nodes == SR1["leg_ret"]["nodes"]
        for want in ("earth", "ceres"):
            assert want in D.prob_sr1_out(nseg=16).bodies

    def test_sr1_mass_chain_anchor(self):
        dv_total = SR1["dv_total_km_s"]
        _, prop = LT.rocket_prop(SR1["m0_kg"], dv_total, SR1["isp_s"])
        assert prop == SR1["propellant_kg"]
        assert SR1["leg_out"]["obj"] + SR1["leg_ret"]["obj"] == SR1["dv_total_km_s"]
        assert (
            SR1["dv_total_km_s"]
            / (SR1["pulse_bound"]["out"]["total"] + SR1["pulse_bound"]["ret"]["total"])
            == SR1["k_vs_pulse"]
        )

    def test_v1_chain_replay_bitwise(self):
        a = V1["sf"]
        p = FC.make_prob(
            "out",
            V1["mid"],
            a["nodes"][1],
            a["segs"][0],
            a["segs"][1],
            _model(a["model"]),
        )
        x = np.array(a["x"], float)
        assert p.obj(x) == a["obj"]
        assert p.resid_km(x) == a["resid_km"]
        c = p.counters(x)
        assert c["total"] == a["counters"]["total"]
        assert c["util"] == a["counters"]["util"]
        assert c["mass_final"] == a["counters"]["mass_final"]
        assert c["n_binding"] == a["counters"]["n_binding"]
        assert [e["vinf_kms"] for e in c["enc"]] == [e["vinf_kms"] for e in a["enc"]]
        assert [l["dv"] for l in c["legs"]] == a["legs_dv"]

    def test_v2_chain_replay_bitwise(self):
        a = V2["sf"]
        p = FC.make_prob(
            "out",
            V2["mid"],
            a["nodes"][1],
            a["segs"][0],
            a["segs"][1],
            _model(a["model"]),
        )
        x = np.array(a["x"], float)
        assert p.obj(x) == a["obj"]
        assert p.resid_km(x) == a["resid_km"]
        assert p.counters(x)["total"] == a["counters"]["total"]


class TestFlybyChain:
    """顺访链：注册表、问题工厂、飞越日预扫描。"""

    def test_variants_registry(self):
        keys = [v["key"] for v in FC.VARIANTS]
        assert keys == [
            "V1_out_lutetia",
            "V2_out_vesta",
            "V3_ret_vesta",
            "V4_out_venetia487",
            "V5_ret_may348",
        ]
        assert set(ANCHORS["summary_variant_keys"]) <= set(keys)
        # 池内件的键必须在登记表内（否则建链会 KeyError）
        for v in FC.VARIANTS:
            assert LT.is_body(v["mid"])

    def test_make_prob_out_bookkeeping(self):
        a = V1["sf"]
        p = FC.make_prob("out", "lutetia", a["nodes"][1], 16, 16, D.sep_model())
        assert p.bodies == ["earth", "lutetia", "ceres"]
        assert p.kinds == ["depart_earth", "flyby", "rendezvous"]
        assert p.bstart is True and p.c3 == D.SR1_C3 and p.vinf_max is None
        assert p.nodes == a["nodes"]
        assert p.N == 32 and p.nv == 2 + 3 * 31
        assert LT.ymd(p.nodes[1]) == V1["flyby_date"]

    def test_make_prob_ret_bookkeeping(self):
        p = FC.make_prob("ret", "vesta", 14181.5 + 500.0, 8, 8, D.sep_model())
        assert p.bodies == ["ceres", "vesta", "earth"]
        assert p.kinds == ["start_body", "flyby", "arrive_earth"]
        assert p.bstart is False and p.c3 is None and p.vinf_max == 6.0
        assert p.nv == 3 * (p.N - 1)

    def test_scan_flyby_top_date_matches_anchor(self):
        rows, (t0, t2) = FC.scan_flyby("out", "lutetia", D.sep_model())
        assert rows, "预扫描应有可行飞越日"
        assert rows[0]["date"] == V1["flyby_date"]
        assert rows[0]["dv"] == V1["flyby_scan_top"][0]["dv"]
        assert (t0, t2) == (LT.dfe(2030, 12, 15), LT.dfe(2037, 6, 17))
        assert [r["dv"] for r in rows] == sorted(r["dv"] for r in rows)
        assert [r["date"] for r in rows[:3]] == [
            r["date"] for r in V1["flyby_scan_top"]
        ]

    def test_scan_flyby_vesta_top_date_matches_anchor(self):
        rows, _ = FC.scan_flyby("out", "vesta", D.sep_model())
        assert rows[0]["date"] == V2["flyby_date"]
        assert rows[0]["dv"] == V2["flyby_scan_top"][0]["dv"]

    def test_pulse_bound_fixed_matches_anchor(self):
        a = V1["sf"]
        p = FC.make_prob("out", "lutetia", a["nodes"][1], 16, 16, D.sep_model())
        pb = D.pulse_bound(p)
        assert pb["total"] == V1["pulse_bound_fixed_total"]
        assert [q["dv"] for q in pb["parts"]] == [
            q["dv"] for q in V1["pulse_bound_fixed_parts"]
        ]


class TestBoundaries:
    """零 TOF / 空窗口 / 极值 k_deg。"""

    def test_zero_tof_propagation_is_inert(self):
        p = D.Prob(
            [11305.5, 11305.5],
            [2],
            ["earth", "ceres"],
            ["depart_earth", "rendezvous"],
            D.sep_model(),
            c3=13.4,
        )
        assert p.dt == [0.0, 0.0]
        tr = p.propagate(np.zeros(p.nv))
        assert tr["total"] == 0.0
        assert list(tr["caps"]) == [0.0]
        # 位置约束仍成立（短段内不传播），残差有限
        assert math.isfinite(p.resid_km(np.zeros(p.nv)))

    def test_zero_tof_bounds_are_finite(self):
        p = D.Prob(
            [11305.5, 11305.5],
            [2],
            ["earth", "ceres"],
            ["depart_earth", "rendezvous"],
            D.sep_model(),
            c3=13.4,
        )
        bnds = D._bounds(p, np.zeros(p.nv))
        assert all(lo <= hi for lo, hi in bnds)
        assert bnds[-1][1] == pytest.approx(0.05)  # cap=0 → 下界守卫 0.05

    def test_inconsistent_seg_lengths_raise(self):
        with pytest.raises(AssertionError):
            D.Prob(
                [0.0, 100.0],
                [1, 1],
                ["earth", "ceres"],
                ["depart_earth", "rendezvous"],
                D.sep_model(),
            )

    def test_empty_variant_filter_yields_nothing(self, tmp_path):
        outs = FC.run_variants(
            keys=["__no_such_key__"],
            model=D.sep_model(),
            out_prefix=str(tmp_path / "sf"),
        )
        assert outs == []
        summary = json.loads((tmp_path / "sf_汇总.json").read_text(encoding="utf-8"))
        assert summary["variants"] == []

    def test_extreme_kdeg_changes_caps_monotonically(self):
        def cap_sum(k_deg):
            p = D.prob_sr1_out(D.sep_model(k_deg=k_deg), nseg=6)
            tr = p.propagate(np.zeros(p.nv))
            return float(np.sum(tr["caps"]))

        assert cap_sum(0.0) == 0.0
        assert cap_sum(5.0) > cap_sum(1.0) > cap_sum(0.9262) > 0.0

    def test_kdeg_scales_available_dv_linearly(self):
        p1 = D.prob_sr1_out(D.sep_model(k_deg=1.0), nseg=6)
        p2 = D.prob_sr1_out(D.sep_model(k_deg=2.0), nseg=6)
        s1 = float(np.sum(p1.propagate(np.zeros(p1.nv))["caps"]))
        s2 = float(np.sum(p2.propagate(np.zeros(p2.nv))["caps"]))
        assert s2 == pytest.approx(2.0 * s1, rel=1e-15)


class TestSelfTestAndCLI:
    """源件自检入口与 CLI 契约。"""

    def test_low_thrust_self_test(self, capsys):
        assert LT.run_self_test() == 0
        assert LT.main(["--self-test"]) == 0
        assert "通过" in capsys.readouterr().out

    def test_sf_design_self_test_relations(self):
        """源件四项自检（关系判据；绝对值依赖 scipy 版本故不锚定）。"""
        res = D.run_self_test()
        assert res["two_runs_identical"] is True
        assert res["lt_vs_pulse"]["k"] > 1.0
        assert res["spiral_scale"]["spiral_round_trip_minus_c3"] == pytest.approx(
            20.088241223109485, rel=1e-12
        )
        assert res["pulse_bound"]["out"]["total"] == SR1["pulse_bound"]["out"]["total"]
        assert abs(res["mass_chain"]["dv_at_1320kg"] - 13.0343) < 1e-3
        assert res["ok"] is True

    def test_sf_design_cli_variants_delegates(self, tmp_path, monkeypatch):
        """--variants 委托 flyby_chain（此处用不存在键验证委托链与退出码）。"""
        calls = {}

        def fake_run_variants(**kw):
            calls.update(kw)
            return []

        monkeypatch.setattr(FC, "run_variants", fake_run_variants)
        assert D.main(["--variants", "--kdeg", "0.95"]) == 0
        assert calls["model"].k_deg == 0.95

    def test_flyby_chain_cli_writes_summary(self, tmp_path):
        code = FC.main(["--keys", "__none__", "--out-prefix", str(tmp_path / "out")])
        assert code == 0
        payload = json.loads((tmp_path / "out_汇总.json").read_text(encoding="utf-8"))
        assert payload["model"]["k_deg"] == 1.0
        assert payload["variants"] == []

    def test_dump_json_round_trip(self, tmp_path):
        path = LT.dump_json(str(tmp_path / "a" / "b.json"), {"x": 1.5})
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {"x": 1.5}
