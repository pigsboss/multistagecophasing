# tests/test_polar_illumination.py
# -*- coding: utf-8 -*-
"""极区光照 / PSR 几何核（Ceres S4 能力沉淀件）契约测试。

对拍口径
--------
* 源实现：谷神星探测专题 S4/CML5 仿真线 ``工具/s4_polar_illum.py``（理想光滑球体 +
  单一有效遮挡角 θ_h + ε=4.028°，上界口径）。
* 对拍锚：其产物 ``任务/S4_仿真线_极区光照_近似.json``（年光照分数 / 年平均入射因子 /
  d/D 阈值 / 冬至日长表，6 位与 4~5 位小数）与真实 DEM 口径的「纯球体冬至日长」列
  （M1/M2/M3）；锚值固化在本文件内，测试可离线独立复算。
* 容差：JSON 锚按其写入位数取 atol（6 位→1e-6、5 位→1e-5、4 位→1e-4）；
  判据类检查按源件容差（光照分数 2%、冬至日长 0.05 h）。

复算命令::

    python3 -m mission_sim.analysis.polar_illumination --self-test
    python3 -m mission_sim.analysis.polar_illumination \\
        --txt /tmp/polar.txt --json /tmp/polar.json --png /tmp/polar.png
"""

import json
import math

import pytest
from numpy.testing import assert_allclose

from mission_sim.analysis.polar_illumination import (
    EPS,
    LATS,
    ROT_H,
    SOL_CONST_WM2,
    SOL_LAT_WINTER,
    THS,
    annual_frac,
    annual_insolation,
    build_payload,
    c1b_lat_limit,
    day_length_h,
    delta_pdf_quad,
    diurnal_frac,
    main,
    peak_elev,
    psr_dD_threshold,
    render_png,
    render_txt,
    run_self_test,
    winter_day_length_h,
)

# ---------------------------------------------------------------- 对拍锚（岛内参考结果）
# (纬度N, θ_h[deg], 年光照分数) —— 6 位小数
ANCHOR_F = [
    (90, 0.0, 0.5),
    (90, 1.0, 0.42),
    (90, 2.0, 0.3345),
    (90, 2.5, 0.287),
    (90, 5.0, 0.0),
    (90, 8.0, 0.0),
    (90, 10.0, 0.0),
    (90, 15.0, 0.0),
    (89, 1.0, 0.418736),
    (89, 2.0, 0.330571),
    (89, 2.5, 0.279621),
    (89, 5.0, 0.002224),
    (85, 1.0, 0.41053),
    (85, 2.0, 0.329643),
    (85, 2.5, 0.296852),
    (85, 5.0, 0.162726),
    (85, 8.0, 0.037421),
    (85, 10.0, 0.0),
    (84, 1.0, 0.437594),
    (84, 2.0, 0.36313),
    (84, 2.5, 0.324968),
    (84, 5.0, 0.190507),
    (84, 8.0, 0.069172),
    (84, 10.0, 0.000902),
    (60, 0.0, 0.5),
    (65, 0.0, 0.5),
    (70, 0.0, 0.5),
    (80, 0.0, 0.5),
]
# (纬度N, θ_h[deg], 年平均入射因子) —— 6 位小数
ANCHOR_I = [
    (90, 0.0, 0.022365),
    (90, 1.0, 0.021662),
    (90, 2.0, 0.019409),
    (90, 2.5, 0.01754),
    (90, 5.0, 0.0),
    (85, 0.0, 0.032411),
    (85, 2.5, 0.028231),
    (85, 5.0, 0.0196),
    (85, 8.0, 0.005537),
]
# 峰值太阳高度 elev_max = 90 − |φ − ε| —— 4 位小数
ANCHOR_PEAK = [(60, 34.028), (80, 14.028), (85, 9.028), (90, 4.028)]
# 极区 PSR 的 d/D 阈值 = tan(elev_max) —— 5 位小数
ANCHOR_PSR = [
    (60, 0.67522),
    (75, 0.34487),
    (80, 0.24985),
    (84, 0.17683),
    (85, 0.15889),
    (89, 0.08798),
    (90, 0.07042),
]
# 纯球体冬至（δ=−ε）日长 [h] —— 4 位小数
ANCHOR_WINTER_H = [
    (60, 4.1838),
    (65, 4.0992),
    (70, 3.9746),
    (75, 3.7689),
    (80, 3.3504),
    (82, 3.0212),
    (84, 2.4164),
    (85, 1.835),
    (86, 0.0),
    (87, 0.0),
    (88, 0.0),
    (89, 0.0),
    (90, 0.0),
]
# 真实 DEM 口径的「纯球体冬至日长」列（name, lat, 冬至日长[h]）；判据容差 0.05 h
ANCHOR_M_SITES = [
    ("M1", 83.3247, 2.67),
    ("M2", 84.3100, 2.27),
    ("M3", 81.8560, 3.05),
]
ANCHOR_C1B_LAT_LIMIT_DEG = 84.773  # 冬至日长 = 2 h 的纬度上限（JSON，3 位小数）
ANCHOR_SOLAR_CONSTANT_WM2 = 178.41  # 谷神星处日射常数 1365/2.766²


class TestDiurnalFraction:
    """一次自转内的日光照分数（几何核的最内层）。"""

    def test_equinox_equator_is_half(self):
        assert_allclose(diurnal_frac(0.0, 0.0, 0.0), 0.5, atol=1e-12)

    def test_polar_day_and_night(self):
        """极点：δ>0 全天日照、δ<0 全天黑夜（分母→0 的极限分支）。"""
        assert diurnal_frac(90.0, 3.0, 0.0) == 1.0
        assert diurnal_frac(90.0, -3.0, 0.0) == 0.0
        assert diurnal_frac(90.0, 0.0, 0.0) == 0.0

    def test_theta_h_screen(self):
        """θ_h ≥ 峰值太阳高度时全年无照。"""
        assert annual_frac(90.0, 5.0) == 0.0
        assert annual_frac(90.0, EPS) == 0.0

    def test_monotone_in_tilt(self):
        """固定纬度与遮挡角，δ 越大日照分数不减。"""
        seq = [diurnal_frac(80.0, d, 0.0) for d in (-EPS, -2.0, 0.0, 2.0, EPS)]
        assert all(seq[i] <= seq[i + 1] + 1e-15 for i in range(len(seq) - 1))


class TestAnnualFraction:
    """年光照分数 F(φ, θ_h)：解析上界口径的判据与不变性。"""

    def test_equator_and_pole_half(self):
        """源件自检判据①②：赤道与北极在 θ_h=0 时 F≈0.5（容差 2%）。"""
        assert abs(annual_frac(0.0, 0.0) - 0.5) < 0.02
        assert abs(annual_frac(90.0, 0.0) - 0.5) < 0.02

    @pytest.mark.parametrize("phi", [0.0, 20.0, 45.0, 60.0, 75.0, 85.0, 89.0, 90.0])
    def test_half_is_structural_upper_bound(self, phi):
        """无地形上界 0.5 是结构性的：δ 反正弦分布对称 ⇒ F(φ, 0) ≡ 0.5。"""
        assert_allclose(annual_frac(phi, 0.0), 0.5, atol=1e-12)

    def test_monotone_in_theta_h(self):
        """源件自检判据③：F(85N) 随 θ_h 单调不增。"""
        seq = [annual_frac(85.0, t) for t in THS]
        assert all(seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1))
        assert seq[0] > seq[-1]

    def test_monotone_decreasing_in_latitude(self):
        """θ_h>0 时纬度越高年光照越少。"""
        seq = [annual_frac(phi, 5.0) for phi in (60.0, 70.0, 80.0, 85.0, 88.0, 90.0)]
        assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))

    @pytest.mark.parametrize("phi,theta,expect", ANCHOR_F)
    def test_anchor_table(self, phi, theta, expect):
        assert_allclose(annual_frac(float(phi), theta), expect, atol=1e-6)

    def test_quadrature_properties(self):
        quad = delta_pdf_quad()
        assert_allclose(sum(w for _d, w in quad), 1.0, rtol=1e-12)
        assert max(abs(d) for d, _w in quad) <= EPS + 1e-12
        assert abs(sum(d * w for d, w in quad)) < 1e-12


class TestAnnualInsolation:
    """年平均入射因子 I(φ, θ_h) = ⟨max(0, sin elev)⟩。"""

    @pytest.mark.parametrize("phi,theta,expect", ANCHOR_I)
    def test_anchor_table(self, phi, theta, expect):
        assert_allclose(annual_insolation(float(phi), theta), expect, atol=1e-6)

    def test_bounds(self):
        for phi in (0.0, 60.0, 85.0, 90.0):
            for theta in THS:
                val = annual_insolation(phi, theta)
                assert 0.0 <= val <= 1.0

    def test_monotone_in_theta_h(self):
        seq = [annual_insolation(85.0, t) for t in THS]
        assert all(seq[i] >= seq[i + 1] - 1e-15 for i in range(len(seq) - 1))

    def test_solar_constant_and_flux_order(self):
        """× 日射常数后与 PGDA 官方年入射通量（同纬 3–24 W/m²）同档。"""
        assert_allclose(SOL_CONST_WM2, ANCHOR_SOLAR_CONSTANT_WM2, atol=0.005)
        flux_90 = annual_insolation(90.0, 0.0) * SOL_CONST_WM2
        flux_82 = annual_insolation(82.0, 0.0) * SOL_CONST_WM2
        assert 3.0 <= flux_90 <= 24.0
        assert flux_82 >= flux_90


class TestPeakElevationAndPSR:
    """峰值太阳高度与极区 PSR 的 d/D 阈值。"""

    def test_pole_peak_equals_tilt(self):
        """源件自检判据④：北极 elev_max = ε。"""
        assert_allclose(peak_elev(90.0), EPS, atol=1e-9)
        assert peak_elev(0.0) == 90.0 - EPS

    def test_monotone_in_latitude(self):
        seq = [peak_elev(phi) for phi in LATS]
        assert all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))

    @pytest.mark.parametrize("phi,expect", ANCHOR_PEAK)
    def test_peak_anchor_table(self, phi, expect):
        assert_allclose(peak_elev(float(phi)), expect, atol=1e-4)

    def test_psr_threshold_is_tan_peak(self):
        """源件自检判据⑤：北极阈值 = tan(ε) = 0.0704。"""
        assert_allclose(
            psr_dD_threshold(90.0), math.tan(EPS * math.pi / 180.0), atol=1e-12
        )
        assert_allclose(psr_dD_threshold(90.0), 0.0704, atol=5e-5)

    @pytest.mark.parametrize("phi,expect", ANCHOR_PSR)
    def test_psr_anchor_table(self, phi, expect):
        assert_allclose(psr_dD_threshold(float(phi)), expect, atol=1e-5)

    def test_psr_monotone_in_latitude(self):
        """纬度越高峰高越低 ⇒ 越浅的坑也能成为 PSR（阈值单调下降）。"""
        seq = [psr_dD_threshold(phi) for phi in LATS]
        assert seq[0] > seq[-1]
        assert all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))


class TestDayLength:
    """日长：等分日、冬至复现与 C1b 纬度上限。"""

    def test_equinox_day_length(self):
        for phi in (0.0, 30.0, 60.0, 85.0):
            assert_allclose(day_length_h(phi, 0.0, 0.0), 0.5 * ROT_H, atol=1e-12)

    @pytest.mark.parametrize("phi,expect", ANCHOR_WINTER_H)
    def test_winter_day_length_anchor(self, phi, expect):
        assert_allclose(winter_day_length_h(float(phi)), expect, atol=1e-4)

    def test_winter_uses_winter_solatitude(self):
        assert SOL_LAT_WINTER == -EPS
        assert_allclose(
            winter_day_length_h(84.0), day_length_h(84.0, -EPS, 0.0), rtol=1e-12
        )

    @pytest.mark.parametrize("name,lat,expect", ANCHOR_M_SITES)
    def test_reproduces_dem_baseline_winter_day_length(self, name, lat, expect):
        """源件自检判据⑥：与真实 DEM 口径的「纯球体冬至日长」列逐站相符（≤0.05 h）。"""
        assert abs(winter_day_length_h(lat, 0.0) - expect) <= 0.05

    def test_c1b_latitude_limit(self):
        """C1b 门限（冬至日长 ≥ 2 h）的纬度上限解析复算 ≈84.8°N。"""
        limit = c1b_lat_limit(2.0)
        assert_allclose(limit, ANCHOR_C1B_LAT_LIMIT_DEG, atol=5e-4)
        assert abs(limit - 84.8) < 0.05
        assert winter_day_length_h(limit) >= 2.0 - 1e-9
        assert winter_day_length_h(limit + 0.05) < 2.0


class TestRendering:
    """文本 / JSON / 图形产物。"""

    def test_render_txt_markers(self):
        text = render_txt("2026-10-15 00:00:00 CST")
        for marker in ("【A】", "【B】", "【C】", "【D】", "【E】", "0.0704", "84.8"):
            assert marker in text
        assert "0.500" in text or "0.5000" in text

    def test_build_payload_structure(self):
        payload = build_payload("2026-10-15 00:00:00 CST")
        assert payload["epsilon_deg"] == EPS
        assert payload["lats"] == LATS
        assert payload["theta_h_deg"] == THS
        assert payload["psr_d_over_D_threshold"]["90"] == pytest.approx(
            0.07042, abs=1e-5
        )
        assert payload["winter_day_length_h"]["86"] == 0.0
        assert payload["c1b_lat_limit_deg"] == pytest.approx(
            ANCHOR_C1B_LAT_LIMIT_DEG, abs=5e-4
        )
        assert payload["solar_constant_wm2"] == ANCHOR_SOLAR_CONSTANT_WM2
        assert len(payload["annual_illumination_fraction"]["90"]) == len(THS)

    def test_render_png_deterministic(self, tmp_path):
        pytest.importorskip("PIL")
        path = tmp_path / "polar.png"
        render_png(str(path), "2026-10-15 00:00:00 CST")
        assert path.exists() and path.stat().st_size > 0

    def test_txt_is_deterministic(self):
        assert render_txt("t") == render_txt("t")


class TestCLI:
    """可独立调用的 CLI 接口。"""

    def test_self_test_exit_zero(self, capsys):
        assert run_self_test() == 0
        assert main(["--self-test"]) == 0
        assert "通过" in capsys.readouterr().out

    def test_writes_txt_and_json(self, tmp_path):
        txt = tmp_path / "polar.txt"
        js = tmp_path / "polar.json"
        code = main(["--txt", str(txt), "--json", str(js), "--no-png"])
        assert code == 0
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert payload["tool"] == "polar_illumination.py"
        assert "【E】" in txt.read_text(encoding="utf-8")

    def test_no_output_paths_is_quiet(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert main([]) == 0
        assert not list(tmp_path.iterdir())
