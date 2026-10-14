# tests/test_interplanetary_window_scan.py
# -*- coding: utf-8 -*-
"""行星际多目标窗口扫描器（Ceres S4 能力沉淀件）契约测试。

对拍口径
--------
* 源实现：谷神星探测专题 S4/CML5 仿真线 ``工具/s4_target_scan.py`` + ``工具/ceres_transfer.py``
  （二体平根数星历 + 日心二体 Lambert 通用变量法、单圈、脉冲）。
* 对拍锚：其产物 ``任务/S4_仿真线_目标扫描.json`` 与 ``任务/ceres_transfer_direct_wide.json``
  （Ceres 基线）。锚值以常量固化在本文件内，使测试可离线独立复算。
* 容差：整数/日期字段要求逐位相等；浮点锚容差 rtol=1e-12（记录机实测最大差 0，逐位一致）。

复算命令（与岛内同网格）::

    python3 -m mission_sim.analysis.interplanetary.target_scan \\
        --t0 2030-01-01 --t1 2045-12-31 --step 15 \\
        --tof-min 600 --tof-max 1600 --tof-step 10 --top-per-year 3 \\
        --txt /tmp/scan.txt --json /tmp/scan.json
"""

import json
import math

import pytest
from numpy.testing import assert_allclose

from mission_sim.analysis.interplanetary.target_scan import (
    DEFAULT_ORDER,
    SBDB_SNAPSHOT_EPOCH,
    TARGETS,
    hohmann_lower_bound,
    main,
    render_txt,
    run_self_test,
    scan_target,
    spec_of,
)
from mission_sim.analysis.interplanetary.two_body import (
    AU_KM,
    EARTH_J2000,
    JD_J2000,
    MU_SUN,
    SEC_PER_DAY,
    Ephemeris,
    direct_leg,
    iso_from_jd,
    iter_julian_days,
    jd_from_iso,
    jd_from_ymd,
    lambert,
    lambert_on_180,
    return_leg,
    solve_kepler,
    tof_grid,
)

# ---------------------------------------------------------------- 对拍锚（岛内参考结果）
# 发射儒略日, TOF[d], C3[km²/s²], v∞dep[km/s], v∞arr[km/s] —— 全网格全局最优
ANCHOR_BEST = {
    "ceres": (
        2466477.5,
        610.0,
        42.65766712770853,
        6.531283727392995,
        7.274809656642736,
    ),
    "vesta": (2468307.5, 620.0, 34.24041209578728, 5.851530748085263, 9.28307785110542),
    "pallas": (
        2464542.5,
        660.0,
        39.55280852399066,
        6.2891023623400075,
        16.009754452818505,
    ),
    "hygiea": (2465397.5, 690.0, 45.63230038606927, 6.755168420259355, 8.0738469802544),
    "psyche": (
        2463627.5,
        620.0,
        40.12137250484574,
        6.334143391560199,
        6.128699093196049,
    ),
    "lutetia": (
        2467557.5,
        600.0,
        34.29130857015848,
        5.855878121183746,
        5.478047638688319,
    ),
}
ANCHOR_ISO = {
    "ceres": ("2040-11-19", "2042-07-22"),
    "vesta": ("2045-11-23", "2047-08-05"),
    "pallas": ("2035-08-03", "2037-05-24"),
    "hygiea": ("2037-12-05", "2039-10-26"),
    "psyche": ("2033-01-30", "2034-10-12"),
    "lutetia": ("2043-11-04", "2045-06-26"),
}
# 共面圆轨道 Hohmann 解析下界：C3, v∞dep, v∞arr, TOF[d]
ANCHOR_HOHMANN = {
    "ceres": (
        39.859508193735614,
        6.313438698026268,
        4.8574975945271746,
        471.8093190216691,
    ),
    "vesta": (
        30.469892367961386,
        5.519954018645571,
        4.431632361312854,
        397.9212528842316,
    ),
    "pallas": (
        39.94713379019088,
        6.320374497622026,
        4.860916209561962,
        472.5625020891053,
    ),
    "hygiea": (
        47.809155788696245,
        6.914416518311306,
        5.132269517190357,
        546.0701821684789,
    ),
    "psyche": (
        43.279791610946496,
        6.578737843306001,
        4.984243768442276,
        502.22982056388184,
    ),
    "lutetia": (
        32.24686271649725,
        5.678632116671871,
        4.522094229747324,
        410.965705608824,
    ),
}


def _earth() -> Ephemeris:
    return Ephemeris(EARTH_J2000)


class TestJulianDates:
    """儒略日 ↔ 日期换算（扫描网格的时间轴）。"""

    def test_known_epoch(self):
        assert jd_from_ymd(2030, 1, 1) == 2462502.5
        assert jd_from_iso("2030-01-01") == 2462502.5
        assert jd_from_iso("2040-11-19") == ANCHOR_BEST["ceres"][0]

    def test_round_trip(self):
        for iso in ("1900-01-01", "2000-01-01", "2030-01-01", "2045-12-31"):
            assert iso_from_jd(jd_from_iso(iso)) == iso

    def test_anchor_rows_are_day_boundaries(self):
        """锚点发射/到达儒略日均落在 0.5（00:00）半日边界上。"""
        for _key, (launch_jd, tof_d, *_rest) in ANCHOR_BEST.items():
            assert launch_jd % 1.0 == 0.5
            assert (launch_jd + tof_d) % 1.0 == 0.5


class TestTwoBodyEphemeris:
    """二体平根数星历的解析自洽性。"""

    def test_earth_state_scale_and_vis_viva(self):
        earth = _earth()
        r, v = earth.state(JD_J2000)
        dist = sum(c * c for c in r) ** 0.5
        assert 0.98 * AU_KM < dist < 1.02 * AU_KM  # J2000 地球近日点附近
        speed2 = sum(c * c for c in v)
        assert_allclose(speed2, MU_SUN * (2.0 / dist - 1.0 / earth.a), rtol=1e-10)

    @pytest.mark.parametrize("key", DEFAULT_ORDER)
    def test_target_vis_viva(self, key):
        """任意时刻都应满足活力公式 |v|² = μ(2/r − 1/a)。"""
        obj = Ephemeris(spec_of(key))
        for jd in (SBDB_SNAPSHOT_EPOCH, 2462502.5, 2466477.5, 2468307.5):
            r, v = obj.state(jd)
            dist = sum(c * c for c in r) ** 0.5
            speed2 = sum(c * c for c in v)
            assert_allclose(speed2, MU_SUN * (2.0 / dist - 1.0 / obj.a), rtol=1e-10)

    def test_solve_kepler_residual(self):
        for e in (0.0, 0.0167, 0.2307, 0.9):
            for m in (-3.0, -1.0, 0.0, 0.5, 2.0, 6.0):
                ecc_anom = solve_kepler(m, e)
                assert abs(ecc_anom - e * math.sin(ecc_anom) - m) < 1e-12

    def test_epoch_mean_anomaly_matches_ma(self):
        """快照历元处 r = a(1 − e·cosE) 应可由 ma_deg 直接复算。"""
        for key in DEFAULT_ORDER:
            spec = spec_of(key)
            obj = Ephemeris(spec)
            ecc_anom = solve_kepler(spec["ma_deg"] * math.pi / 180.0, spec["e"])
            dist = sum(c * c for c in obj.state(SBDB_SNAPSHOT_EPOCH)[0]) ** 0.5
            assert_allclose(
                dist, obj.a * (1.0 - obj.e * math.cos(ecc_anom)), rtol=1e-12
            )


class TestLambert:
    """Lambert 求解器：解析对拍、不变量与边界。"""

    @pytest.mark.parametrize("key", DEFAULT_ORDER)
    def test_hohmann_analytic_anchor(self, key):
        h = hohmann_lower_bound(TARGETS[key]["a_au"])
        c3, vinf_dep, vinf_arr, tof_d = ANCHOR_HOHMANN[key]
        assert_allclose(
            [h["c3"], h["vinf_dep"], h["vinf_arr"], h["tof_d"]],
            [c3, vinf_dep, vinf_arr, tof_d],
            rtol=1e-12,
        )

    @pytest.mark.parametrize("key", DEFAULT_ORDER)
    def test_apse_branch_reproduces_hohmann(self, key):
        """Δν=180° 拱点稳健分支的 Lambert 复现 Hohmann 解析下界（判据：误差 < 0.01%）。"""
        a_au = TARGETS[key]["a_au"]
        h = hohmann_lower_bound(a_au)
        _vinf_dep, c3, _vinf_arr = lambert_on_180(
            1.0, a_au, h["tof_d"] * SEC_PER_DAY, 0.0
        )
        assert abs(c3 - h["c3"]) / h["c3"] < 1e-4

    def test_vis_viva_invariant_of_solution(self):
        """Lambert 解的两端必须落在同一条二体轨道上（半长轴一致）。"""
        r1 = (1.0 * AU_KM, 0.0, 0.0)
        r2 = (1.5 * AU_KM * math.cos(1.2), 1.5 * AU_KM * math.sin(1.2), 0.0)
        v1, v2 = lambert(r1, r2, 400.0 * SEC_PER_DAY)

        def sma(r, v):
            dist = sum(c * c for c in r) ** 0.5
            speed2 = sum(c * c for c in v)
            return 1.0 / (2.0 / dist - speed2 / MU_SUN)

        assert_allclose(sma(r1, v1), sma(r2, v2), rtol=1e-9)

    def test_zero_tof_raises(self):
        """零飞行时间无解：应为显式 ValueError（不静默返回）。"""
        r1 = (1.0 * AU_KM, 0.0, 0.0)
        r2 = (1.5 * AU_KM * math.cos(1.2), 1.5 * AU_KM * math.sin(1.2), 0.0)
        with pytest.raises(ValueError):
            lambert(r1, r2, 0.0)

    @pytest.mark.parametrize("key", DEFAULT_ORDER)
    def test_anchor_rows_bitwise(self, key):
        """在全网格最优的 (发射日, TOF) 处复算，逐位（rtol=1e-12）复现锚点。"""
        launch_jd, tof_d, c3, vinf_dep, vinf_arr = ANCHOR_BEST[key]
        row = direct_leg(launch_jd, tof_d, _earth(), Ephemeris(spec_of(key)))
        assert row["launch_jd"] == launch_jd
        assert row["arrival_jd"] == launch_jd + tof_d
        assert row["tof_d"] == tof_d
        assert_allclose(
            [row["c3"], row["vinf_dep"], row["vinf_arr"]],
            [c3, vinf_dep, vinf_arr],
            rtol=1e-12,
        )
        assert iso_from_jd(row["launch_jd"]) == ANCHOR_ISO[key][0]
        assert iso_from_jd(row["arrival_jd"]) == ANCHOR_ISO[key][1]

    def test_direct_leg_derived_fields(self):
        row = direct_leg(
            ANCHOR_BEST["ceres"][0], 610.0, _earth(), Ephemeris(spec_of("ceres"))
        )
        assert row["c3"] >= 0.0
        assert_allclose(row["vinf_dep"], row["c3"] ** 0.5, rtol=1e-12)

    def test_return_leg_shape(self):
        row = return_leg(2467087.5, 500.0, _earth(), Ephemeris(spec_of("ceres")))
        assert set(row) == {"depart_jd", "arrival_jd", "tof_d", "vinf_dep", "vinf_arr"}
        assert row["arrival_jd"] == row["depart_jd"] + row["tof_d"]
        assert row["vinf_dep"] > 0.0 and row["vinf_arr"] > 0.0


class TestGridHelpers:
    """网格小工具的边界语义。"""

    def test_tof_grid_inclusive(self):
        assert tof_grid(600.0, 620.0, 10.0) == [600.0, 610.0, 620.0]
        assert tof_grid(600.0, 600.0, 10.0) == [600.0]

    def test_iter_julian_days_inclusive_and_empty(self):
        days = list(iter_julian_days(2462502.5, 2462532.5, 15.0))
        assert days == [2462502.5, 2462517.5, 2462532.5]
        assert list(iter_julian_days(2462532.5, 2462502.5, 15.0)) == []


class TestScanTarget:
    """扫描器记账、排序与空网格边界。"""

    def test_small_grid_bookkeeping(self):
        t0 = ANCHOR_BEST["ceres"][0]
        tofs = tof_grid(600.0, 620.0, 10.0)
        res = scan_target("ceres", t0, t0, 15.0, tofs, 3, earth=_earth())
        assert res["key"] == "ceres"
        assert res["grid_points"] == 3
        assert res["n_years"] == 1
        assert list(res["per_year_top"]) == [2040]
        assert len(res["per_year_top"][2040]) == 3
        best = res["global_best"]
        assert min(r["c3"] for r in res["per_year_top"][2040]) == best["c3"]

    def test_per_year_top_is_sorted_and_truncated(self):
        t0 = ANCHOR_BEST["ceres"][0]
        res = scan_target(
            "ceres", t0, t0, 15.0, tof_grid(600.0, 620.0, 10.0), 2, earth=_earth()
        )
        rows = res["per_year_top"][2040]
        assert len(rows) == 2
        assert rows[0]["c3"] <= rows[1]["c3"]

    def test_empty_grid(self):
        """给定区间为空时不产生任何网格点（记账应为 0，不抛异常）。"""
        res = scan_target(
            "ceres",
            ANCHOR_BEST["ceres"][0],
            ANCHOR_BEST["ceres"][0] - 10.0,
            15.0,
            tof_grid(600.0, 620.0, 10.0),
            3,
            earth=_earth(),
        )
        assert res["grid_points"] == 0
        assert res["global_best"] is None
        assert res["n_years"] == 0

    def test_registry_and_snapshot_epoch(self):
        assert DEFAULT_ORDER == [
            "ceres",
            "vesta",
            "pallas",
            "hygiea",
            "psyche",
            "lutetia",
        ]
        assert set(DEFAULT_ORDER) == set(TARGETS)
        for key in DEFAULT_ORDER:
            spec = spec_of(key)
            assert spec["epoch_jd"] == SBDB_SNAPSHOT_EPOCH == 2461200.5
            assert spec["n_deg_day"] > 0.0 and spec["e"] < 1.0
        # 与源件快照逐位一致（Ceres 基线）
        assert TARGETS["ceres"]["a_au"] == 2.765552595034094
        assert TARGETS["pallas"]["i_deg"] == 34.93279321851542

    def test_render_txt_markers(self):
        t0 = ANCHOR_BEST["ceres"][0]
        res = scan_target(
            "ceres", t0, t0, 15.0, tof_grid(600.0, 620.0, 10.0), 3, earth=_earth()
        )
        params = {
            "t0": iso_from_jd(t0),
            "t1": iso_from_jd(t0),
            "step": 15.0,
            "tof_min": 600.0,
            "tof_max": 620.0,
            "tof_step": 10.0,
        }
        text = render_txt([res], params, "2026-10-15 00:00:00 CST")
        for marker in ("【A】", "【B】", "【ceres】", "Hohmann", "2040-11-19"):
            assert marker in text


class TestCLI:
    """可独立调用的 CLI 接口。"""

    def test_self_test_exit_zero(self, capsys):
        assert run_self_test() == 0
        assert main(["--self-test"]) == 0
        assert "通过" in capsys.readouterr().out

    def test_unknown_target_returns_2(self):
        assert main(["--targets", "eros"]) == 2

    def test_writes_txt_and_json(self, tmp_path):
        txt = tmp_path / "scan.txt"
        js = tmp_path / "scan.json"
        code = main(
            [
                "--targets",
                "ceres",
                "--t0",
                "2040-11-19",
                "--t1",
                "2040-11-19",
                "--tof-min",
                "600",
                "--tof-max",
                "620",
                "--tof-step",
                "10",
                "--txt",
                str(txt),
                "--json",
                str(js),
            ]
        )
        assert code == 0
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert payload["targets"]["ceres"]["grid_points"] == 3
        best = payload["targets"]["ceres"]["global_best"]
        assert_allclose(best["c3"], ANCHOR_BEST["ceres"][2], rtol=1e-9)
        assert "2040-11-19" in txt.read_text(encoding="utf-8")

    def test_no_output_paths_is_quiet(self, monkeypatch, tmp_path):
        """未给输出路径时不落盘（纯计算入口）。"""
        monkeypatch.chdir(tmp_path)
        code = main(
            [
                "--targets",
                "ceres",
                "--t0",
                "2040-11-19",
                "--t1",
                "2040-11-19",
                "--tof-min",
                "610",
                "--tof-max",
                "610",
                "--tof-step",
                "10",
            ]
        )
        assert code == 0
        assert not list(tmp_path.iterdir())
