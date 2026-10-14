# mission_sim/analysis/polar_illumination.py
# -*- coding: utf-8 -*-
"""极区光照 / PSR 几何核（Ceres S4 能力沉淀件）。

来源：谷神星探测专题 S4/CML5 仿真线的 ``工具/s4_polar_illum.py``（零数据依赖的解析
近似，全标【近似】）。用途有二：
  ① 为真实 DEM 结果提供**上界 / 一致性核对**（同纬度、同有效遮挡角下的理想球体光照分数）；
  ② 给出不依赖地形资料即可用的判据量（极区 PSR 的 d/D 阈值、纬度—峰值太阳高度关系）。

模型与假设（全部【近似】，逐条列出以便替换）
--------------------------------------------
* A1 **理想光滑球体**：本地地平线＝切平面（无地形遮挡）。真实地形只会**降低**光照分数
  ⇒ 本模型是真实 DEM 结果的上界。
* A2 **单一有效遮挡角 θ_h**（标量）：把"朝阳方位上的地平线角"折算为一个全年等效遮挡角；
  真实 θ_h 随方位变化，故对一组 θ_h 扫描并给出双口径对照。
* A3 **轴倾角 ε = 4.028°**；亚太阳纬度 δ 全年按 δ = ε·sin ψ 正弦摆动 ⇒ δ 服从反正弦分布。
* A4 **自转远快于公转** ⇒ 年内先按自转求日光照分数，再对 δ 求年平均；忽略自转轴进动与光行差。
* 未含：地形遮挡、多次散射、反照率、热惯量、太阳视面（本件只出**几何可见性**与**入射因子**）。

判据量
------
1. 年光照分数 F(φ, θ_h)：太阳在地平线以上（elev > θ_h）的时间占全年比例。
2. 峰值太阳高度 elev_max(φ) = 90° − |φ − ε|（全年最高）。
3. 极区 PSR 的**深度/直径阈值**：平坦坑底全年无照 ⇔ tan(elev_max) < d/D ⇒ d/D > tan(elev_max)。
4. 年平均入射因子 I(φ, θ_h) = ⟨max(0, sin elev)⟩（相对法向太阳常数）。

用法
----
  python3 -m mission_sim.analysis.polar_illumination --self-test
  python3 -m mission_sim.analysis.polar_illumination --txt out.txt --json out.json --no-png
  # --png 需要可选的 Pillow（未安装时跳过并提示）
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEG = math.pi / 180.0
EPS = 4.028  # 轴倾角 [deg]（Ceres 当前轴倾角，Platz et al. 2016 口径）【文据】
ROT_H = 9.074  # 自转周期 [h]（Platz et al. 2016 口径）【文据】
SOL_CONST_WM2 = 1365.0 / 2.766**2  # 谷神星处日射常数 ≈178.4 W/m²【文据】
SOL_LAT_SUMMER = +EPS  # 夏至亚太阳纬度
SOL_LAT_WINTER = -EPS  # 冬至亚太阳纬度
LATS = [60, 65, 70, 75, 80, 82, 84, 85, 86, 87, 88, 89, 90]  # 北半球纬度带 [deg]
THS = [0.0, 1.0, 2.0, 2.5, 5.0, 8.0, 10.0, 15.0]  # 有效遮挡角 [deg]

# --- 真实 DEM（Dawn 形状模型 + PGDA 官方产品）锚点【文据】，用于上界/一致性核对
DEM_SITES: List[Tuple[str, float, float, Optional[float], Optional[float]]] = [
    # name, lat, 年光照分数（真实 DEM）, θ_h(正南)[deg], θ_h(全年日照方位最大)[deg]
    ("Bilwis NP4", 86.09, 0.8218, 0.95, 0.95),
    ("Zatik NP5", 69.76, 0.6405, None, None),
    ("NP7 (#3)", 77.51, 0.5916, None, 13.48),
    ("Damia NP19", 80.92, 0.7125, None, 3.09),
    ("Enkimdu NP26", 78.51, 0.5509, None, None),
    ("NP #6 (64.0N)", 63.57, 0.5028, None, None),
    ("NP #9 (83.7N)", 83.77, 0.4814, None, None),
]
DEM_BAND_NOTE = (
    "83–88N 原生网格 θ_h(正南) 中位 2.9°[5,95%]=[-2.94,12.94]；"
    "θ_h(全年日照方位最大) 中位 11.63°[5,95%]=[5.18,23.34]；"
    "非 PSR 年均光照率 0.3718（全 80–90N 抽样）"
)
DEM_BOUND_NOTE = (
    "真实 DEM 口径 §4：'无地形上界 = 0.500'（θ_h=0 口径与此一致）；"
    "该上界仅对**水平面且有效遮挡角 ≥0** 成立；"
    "真实 DEM 口径 §5.4 L10 指出 ≥86°N 处 θ_h 可为**负**（朝阳坡可见径向地平线以下），"
    "此时光照分数**可超 0.5**（如 Bilwis 0.82）——属上界例外。"
)
# 三站点【文据】：name, lat, lon, 自算年分数, 纯球体冬至日长[h]
DEM_M_SITES: List[Tuple[str, float, float, float, float]] = [
    ("M1 (83.3247N/128.05E)", 83.3247, 128.0501, 0.241, 2.67),
    ("M2 (84.3100N/220.89E)", 84.3100, 220.8930, 0.302, 2.27),
    ("M3 (81.8560N/318.51E)", 81.8560, 318.5063, 0.308, 3.05),
]


def diurnal_frac(phi_deg: float, delta_deg: float, theta_h_deg: float) -> float:
    """一次自转内太阳高于 (地平线+θ_h) 的时间分数。【近似】

    条件 sin(elev) > sin θ_h，sin elev = sinφ sinδ + cosφ cosδ cosH。
    ⇒ cosH > (sinθ_h − sinφ sinδ)/(cosφ cosδ)；极点附近分母→0 时按极限处理。
    返回 [0,1]；若该 δ 下太阳终日不升/不落则分别为 0/1。
    """
    phi, delta, th = phi_deg * DEG, delta_deg * DEG, theta_h_deg * DEG
    num = math.sin(th) - math.sin(phi) * math.sin(delta)
    den = math.cos(phi) * math.cos(delta)
    if abs(den) < 1e-12:
        return 1.0 if num < 0.0 else 0.0
    x = num / den
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return 1.0
    return math.acos(x) / math.pi


def delta_pdf_quad(n: int = 4000) -> List[Tuple[float, float]]:
    """亚太阳纬度 δ 的反正弦分布抽样（ψ 均匀 ⇒ δ=ε sinψ）。返回 [(δ, w)]，Σw=1。【近似】"""
    out = []
    for k in range(n):
        psi = (k + 0.5) / n * 2.0 * math.pi
        out.append((EPS * math.sin(psi), 1.0 / n))
    return out


def annual_frac(
    phi_deg: float,
    theta_h_deg: float,
    quad: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """年光照分数 F(φ, θ_h)：对 δ 反正弦分布求日光照分数的年平均。【近似】"""
    quad = quad or delta_pdf_quad()
    return sum(w * diurnal_frac(phi_deg, d, theta_h_deg) for d, w in quad)


def annual_insolation(
    phi_deg: float,
    theta_h_deg: float,
    quad: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """年平均入射因子 ⟨max(0, sin elev)⟩（相对法向太阳常数；含 θ_h 遮挡）。【近似】

    解析：固定 (φ,δ) 时，太阳高过 θ_h 的方位段 H∈[−H_max,H_max] 内 sin elev ≥ sin θ_h ≥ 0，
    故该段内 max(0,sin elev)=sin elev，一次自转均值 =
    (H_max·sinφ·sinδ + cosφ·cosδ·sin H_max)/π。
    """
    quad = quad or delta_pdf_quad()
    phi, th = phi_deg * DEG, theta_h_deg * DEG

    def mean_sin_over_rot(delta_deg: float) -> float:
        delta = delta_deg * DEG
        num = math.sin(th) - math.sin(phi) * math.sin(delta)
        den = math.cos(phi) * math.cos(delta)
        if abs(den) < 1e-12:
            h_max = math.pi if num < 0 else 0.0
        else:
            h_max = math.acos(max(-1.0, min(1.0, num / den)))
        if h_max <= 0:
            return 0.0
        return (
            h_max * math.sin(phi) * math.sin(delta)
            + math.cos(phi) * math.cos(delta) * math.sin(h_max)
        ) / math.pi

    return sum(w * mean_sin_over_rot(d) for d, w in quad)


def peak_elev(phi_deg: float) -> float:
    """全年最高太阳高度 elev_max(φ) = 90 − |φ − ε|。【近似】"""
    return 90.0 - abs(phi_deg - EPS)


def day_length_h(phi_deg: float, delta_deg: float, theta_h_deg: float = 0.0) -> float:
    """给定亚太阳纬度 δ 的白昼时长 [h]（理想水平面 + 有效遮挡角 θ_h）。【近似】

    白昼时长 = 日光照分数 × 自转周期。
    """
    return diurnal_frac(phi_deg, delta_deg, theta_h_deg) * ROT_H


def winter_day_length_h(phi_deg: float, theta_h_deg: float = 0.0) -> float:
    """冬至（δ=−ε）白昼时长 [h]。【近似】"""
    return day_length_h(phi_deg, SOL_LAT_WINTER, theta_h_deg)


def c1b_lat_limit(hours: float = 2.0) -> float:
    """纯球体冬至日长 = hours 的纬度上限 [°N]（二分）。【近似】"""
    lo, hi = 80.0, 90.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if winter_day_length_h(mid, 0.0) >= hours:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def psr_dD_threshold(phi_deg: float) -> float:
    """平坦坑底全年无照所需的 深度/直径 比 d/D > tan(elev_max)。【近似】"""
    return math.tan(peak_elev(phi_deg) * DEG)


# ------------------------------------------------------------------ 自检
def run_self_test() -> int:
    """解析自检：赤道/北极 F≈0.5、θ_h 单调性、峰值高度、d/D 阈值、冬至日长对拍。【近似】

    判据：①② 容差 2%；③ 逐点单调不增；④⑤ 与解析式一致；⑥ 与真实 DEM 口径的
    "纯球体冬至日长"逐站差 ≤ 0.05 h。
    """
    quad = delta_pdf_quad()
    checks: List[Tuple[str, Any, bool]] = []
    # ① 赤道 θ_h=0：昼夜各半 ⇒ 0.5（容差 2%）
    f_eq = annual_frac(0.0, 0.0, quad)
    checks.append(("赤道 F(φ=0,θ=0)≈0.5", f_eq, abs(f_eq - 0.5) < 0.02))
    # ② 北极 θ_h=0：δ 正弦摆动，太阳升半程 ⇒ 0.5
    f_np = annual_frac(90.0, 0.0, quad)
    checks.append(("北极 F(φ=90,θ=0)≈0.5", f_np, abs(f_np - 0.5) < 0.02))
    # ③ 单调性：固定 φ=85，F 随 θ_h 单调不增
    seq = [annual_frac(85.0, t, quad) for t in THS]
    mono = all(seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1))
    checks.append(("F(85N) 随 θ_h 单调不增", tuple(round(v, 3) for v in seq), mono))
    # ④ 峰值高度：北极 elev_max = ε
    checks.append(
        ("北极 elev_max=ε=4.028", peak_elev(90.0), abs(peak_elev(90.0) - EPS) < 1e-9)
    )
    # ⑤ 阈值：北极 d/D 阈 = tan(ε)
    checks.append(
        (
            "北极 d/D 阈=tan(ε)=0.0704",
            psr_dD_threshold(90.0),
            abs(psr_dD_threshold(90.0) - math.tan(EPS * DEG)) < 1e-12,
        )
    )
    # ⑥ 对拍「纯球体冬至日长」列（M1/M2/M3；容差 0.05 h）
    ok6 = True
    vals = []
    for name, lat, _lon, _frac, dl in DEM_M_SITES:
        mine = winter_day_length_h(lat, 0.0)
        vals.append(f"{name.split()[0]}:{mine:.2f}/{dl}h")
        ok6 = ok6 and abs(mine - dl) <= 0.05
    checks.append((f"复现纯球体冬至日长[本件/文据]: {' '.join(vals)}", "见左", ok6))
    print("=" * 78)
    print("polar_illumination · self-test")
    print("=" * 78)
    ok = True
    for name, val, passed in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} = {val}")
    print("-" * 78)
    print("自检", "通过" if ok else "**未通过**")
    return 0 if ok else 1


# ------------------------------------------------------------------ 输出
def _table(
    rows: Sequence[Sequence[float]], lats: Sequence[float], ths: Sequence[float]
) -> List[str]:
    out = ["纬度N " + "".join(f"θ_h={t:>4.1f}°" for t in ths)]
    for i, phi in enumerate(lats):
        row = f"{phi:>4.0f}° "
        for j, _t in enumerate(ths):
            row += f"{rows[i][j]:>10.4f}"
        out.append(row)
    return out


def render_txt(generated: str) -> str:
    """算表 → 制表文本（与源件 s4_polar_illum 的版式一致）。"""
    quad = delta_pdf_quad()
    F = [[annual_frac(phi, t, quad) for t in THS] for phi in LATS]
    I = [[annual_insolation(phi, t, quad) for t in THS] for phi in LATS]
    L: List[str] = []
    L.append("=" * 92)
    L.append("S4/CML5 仿真线 · 极区光照 / PSR 近似计算（全标【近似】）")
    L.append("=" * 92)
    L.append(
        f"生成：{generated}（系统 date）｜工具：mission_sim/analysis/polar_illumination.py"
        "｜上界口径：理想光滑球体"
    )
    L.append(
        f"模型假设：A1 无地形遮挡(上界)｜A2 单一有效遮挡角 θ_h｜A3 ε={EPS}°｜A4 自转远快于公转"
    )
    L.append(
        "性质：本表 F(θ_h) 为真实 DEM 年光照分数的**上界**（对水平面、有效遮挡角 ≥0 的坡面；"
    )
    L.append("      朝阳坡面可例外超过 0.5，见【D】）。")
    L.append("-" * 92)
    L.append("【A】年光照分数 F(纬度, θ_h)【近似，上界】")
    L += _table(F, LATS, THS)
    L.append("")
    L.append(
        "【B】年平均入射因子 I(纬度, θ_h) = ⟨max(0, sin elev)⟩（相对法向太阳常数）【近似】"
    )
    L += _table(I, LATS, THS)
    L.append(
        f"  换算：× 日射常数 S0 = {SOL_CONST_WM2:.1f} W/m²（1365/2.766²）【文据】"
        "⇒ 年平均入射通量。"
    )
    L.append(
        "  量级核对：82°N、θ_h=0 得 {:.2f} W/m²；90°N 得 {:.2f} W/m²，"
        "与 PGDA 官方年入射通量（同纬 3–24 W/m²）同档【文据】。".format(
            I[LATS.index(82)][THS.index(0.0)] * SOL_CONST_WM2,
            I[LATS.index(90)][THS.index(0.0)] * SOL_CONST_WM2,
        )
    )
    L.append("")
    L.append(
        "【C】峰值太阳高度与极区 PSR 的 d/D 阈值（平坦坑底全年无照 ⇔ d/D > tan(elev_max)）【近似】"
    )
    L.append(
        f"{'纬度N':>6}{'elev_max(°)':>13}{'tan(elev_max)=d/D阈值':>24}{'说明':>28}"
    )
    for phi in LATS:
        em, dD = peak_elev(phi), psr_dD_threshold(phi)
        note = "北极阈值" if phi == 90 else ("次极区" if phi >= 85 else "高纬")
        L.append(f"{phi:>5.0f}°{em:>13.3f}{dD:>24.4f}{note:>28}")
    L.append(
        "  判读：d/D 越小的坑越难成永久阴影；北极阈值 d/D>0.0704 意味着**极浅坑亦可成 PSR**——"
    )
    L.append(
        "        与实测 83–88N PSR 面积 ~315 km²（PGDA 500 m/px）方向一致【文据】。"
    )
    L.append("")
    L.append("【D】与真实 DEM 的对照（同纬度）【文据 vs 本件近似】")
    L.append(
        f"{'站点':<18}{'纬度N':>8}{'DEM真实分数':>13}{'本件F(θ=0)':>12}{'=0.5上界?':>11}"
        f"{'本件F(θ=8)':>12}{'本件F(θ=10)':>12}"
    )
    for name, lat, frac, _th_s, _th_mx in DEM_SITES:
        f0 = annual_frac(lat, 0.0, quad)
        f8 = annual_frac(lat, 8.0, quad)
        f10 = annual_frac(lat, 10.0, quad)
        tag = ">0.5 朝阳坡" if frac > 0.5 else "≤0.5 ✓"
        L.append(
            f"{name:<18}{lat:>8.2f}{frac:>13.4f}{f0:>12.4f}{tag:>11}{f8:>12.4f}{f10:>12.4f}"
        )
    L.append(f"  锚注（文据）：{DEM_BAND_NOTE}")
    L.append(f"  上界口径（文据）：{DEM_BOUND_NOTE}")
    L.append(
        "  【待核】列中 69–81°N 各站 DEM 分数亦 >0.5（如 Zatik 0.64、Damia 0.71）；DEM 口径仅对 ≥86°N"
    )
    L.append(
        "         明示了负 θ_h 机制。中低纬站点的 >0.5 成因（坡面朝向 vs 采样/归一化"
    )
    L.append("         口径）本件未辨，标【待核】，不据此改动 DEM 结论。")
    L.append("")
    L.append("【E】对拍「纯球体冬至日长」(C1b ≥ 2 h) —— 本件解析复现【近似 vs 文据】")
    L.append(
        f"{'站点':<24}{'纬度N':>8}{'本件冬至日长(h)':>16}{'文据(h)':>14}{'差(h)':>9}"
    )
    for name, lat, _lon, _frac, dl in DEM_M_SITES:
        mine = winter_day_length_h(lat, 0.0)
        L.append(f"{name:<24}{lat:>8.4f}{mine:>16.3f}{dl:>14.2f}{mine - dl:>+9.3f}")
    L.append(
        "  判读：本件解析与 DEM 口径自算的「纯球体冬至日长」逐站相符（差 ≤0.05 h）⇒ 光照几何口径对拍一致；"
    )
    L.append(
        "        C1b 门限（冬至日长 ≥ 2 h）对应纬度上限 φ ≲ 84.8°N，本件可解析复算该上限。"
    )
    L.append(
        f"  解析复算：纯球体冬至日长=2.0 h 的纬度上限 = {c1b_lat_limit():.2f}°N（文据 84.8°N）"
    )
    L.append("-" * 92)
    L.append(
        "口径提醒：本件是**几何可见性**近似，不含反照率/热惯量/散射/地形；用于站点条件**分档**与"
    )
    L.append("        真实 DEM 结果的上下界核对，**不替代** DEM 结论。")
    return "\n".join(L) + "\n"


def render_png(path: str, generated: str) -> None:
    """确定性数据绘图（Pillow，无生成式成分）：F(纬度) 曲线 + 真实 DEM 站点散点。

    Raises:
        ImportError: 未安装 Pillow（可选依赖）时。
    """
    from PIL import Image, ImageDraw, ImageFont

    quad = delta_pdf_quad()
    W, H = 1000, 640
    ml, mr, mt, mb = 90, 330, 60, 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        fsm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
        fsm = font
    x0, x1 = ml, W - mr
    y0, y1 = H - mb, mt
    lo, hi = 60.0, 90.0
    colors = {
        0.0: (200, 30, 30),
        2.0: (230, 140, 20),
        5.0: (30, 140, 60),
        10.0: (30, 70, 200),
        15.0: (120, 40, 160),
    }

    def X(lat: float) -> float:
        return x0 + (lat - lo) / (hi - lo) * (x1 - x0)

    def Y(v: float) -> float:
        return y0 + (v - 0.0) / (1.0 - 0.0) * (y1 - y0)

    # 网格
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        d.line([(x0, Y(v)), (x1, Y(v))], fill=(220, 220, 220))
        d.text((x0 - 50, Y(v) - 8), f"{v:.1f}", fill="black", font=fsm)
    for lat in range(60, 91, 5):
        d.line([(X(lat), y0), (X(lat), y1)], fill=(235, 235, 235))
        d.text((X(lat) - 14, y0 + 8), f"{lat}", fill="black", font=fsm)
    d.text(((x0 + x1) / 2 - 90, y0 + 34), "latitude N (deg)", fill="black", font=fsm)
    d.text((x0 - 78, mt - 34), "annual illuminated fraction F", fill="black", font=fsm)
    d.text(
        (ml, 16),
        f"Ceres polar illumination (ideal smooth sphere, eps={EPS} deg)  [approx: upper bound]",
        fill=(20, 20, 20),
        font=font,
    )

    # 曲线
    lats = [lo + i * (hi - lo) / 300.0 for i in range(301)]
    for th, col in colors.items():
        pts = [(X(la), Y(min(1.0, max(0.0, annual_frac(la, th, quad))))) for la in lats]
        d.line(pts, fill=col, width=3)
    # 真实 DEM 站点散点
    for name, lat, frac, _a, _b in DEM_SITES:
        cx, cy = X(lat), Y(frac)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 0, 0), outline="white")
    # 图例
    lx, ly = x1 + 24, mt + 6
    d.text((lx, ly - 26), "theta_h (effective horizon):", fill="black", font=fsm)
    for th, col in colors.items():
        d.line([(lx, ly), (lx + 26, ly)], fill=col, width=3)
        d.text((lx + 32, ly - 8), f"{th:g} deg", fill="black", font=fsm)
        ly += 26
    ly += 14
    d.text((lx, ly - 20), "black dots = real-DEM sites", fill="black", font=fsm)
    for name, lat, frac, _a, _b in DEM_SITES:
        d.text(
            (lx, ly),
            f"{name}: lat {lat:.1f}, F {frac:.3f}",
            fill=(60, 60, 60),
            font=fsm,
        )
        ly += 20
    d.text(
        (ml, H - 26),
        f"generated {generated} | mcpc polar_illumination.py | deterministic data plot",
        fill=(90, 90, 90),
        font=fsm,
    )
    img.save(path)


def build_payload(generated: str) -> Dict[str, Any]:
    """算表 → JSON 载荷（结构与源件一致，数值取 6 位小数）。"""
    quad = delta_pdf_quad()
    F = {
        str(phi): {str(t): round(annual_frac(phi, t, quad), 6) for t in THS}
        for phi in LATS
    }
    I = {
        str(phi): {str(t): round(annual_insolation(phi, t, quad), 6) for t in THS}
        for phi in LATS
    }
    return {
        "tool": "polar_illumination.py",
        "generated": generated,
        "model": f"理想光滑球体 + 单一有效遮挡角 θ_h + ε={EPS}°（上界口径）",
        "tags": "【近似】",
        "assumptions": [
            "A1 无地形遮挡（上界）",
            "A2 单一有效遮挡角 θ_h",
            f"A3 ε={EPS}°",
            "A4 自转远快于公转",
        ],
        "epsilon_deg": EPS,
        "lats": LATS,
        "theta_h_deg": THS,
        "annual_illumination_fraction": F,
        "annual_insolation_factor": I,
        "peak_elevation_deg": {str(phi): round(peak_elev(phi), 4) for phi in LATS},
        "psr_d_over_D_threshold": {
            str(phi): round(psr_dD_threshold(phi), 5) for phi in LATS
        },
        "winter_day_length_h": {
            str(phi): round(winter_day_length_h(phi, 0.0), 4) for phi in LATS
        },
        "c1b_lat_limit_deg": round(c1b_lat_limit(2.0), 3),
        "solar_constant_wm2": round(SOL_CONST_WM2, 2),
        "dem_anchor_sites": [
            {"name": n, "lat": la, "dem_annual_frac": f}
            for n, la, f, _a, _b in DEM_SITES
        ],
        "dem_band_note": DEM_BAND_NOTE,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="极区光照/PSR 几何近似计算（全标【近似】）")
    p.add_argument("--txt", default="", help="制表文本输出路径（空＝不写）")
    p.add_argument("--json", default="", help="JSON 输出路径（空＝不写）")
    p.add_argument("--png", default="", help="曲线图输出路径（需 Pillow）")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()

    if args.self_test:
        return run_self_test()

    if args.txt:
        with open(args.txt, "w", encoding="utf-8") as fh:
            fh.write(render_txt(generated))
        print(f"已写入：{args.txt}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(build_payload(generated), fh, ensure_ascii=False, indent=2)
        print(f"已写入：{args.json}")

    if args.png and not args.no_png:
        try:
            render_png(args.png, generated)
        except ImportError:
            print(f"跳过 PNG（未安装可选依赖 Pillow）：{args.png}")
        else:
            print(f"已写入：{args.png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
