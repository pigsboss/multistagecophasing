#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LUNAR-SWING 研究总结报告：插图生成（可复现，数据均来自实验 H5）。

用法: python3 make_figures.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

rcParams["font.family"] = ["Songti SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 9

FIG = Path(__file__).resolve().parent / "sources" / "图"
FIG.mkdir(parents=True, exist_ok=True)
LS = ROOT / "docs/excursion/lunar_swing/results"
AS = LS / "as_lowperigee"


def load_alt(path):
    with h5py.File(path, "r") as f:
        t = f["epochs"][:] / 86400.0
        alt = f["altitude_m"][:] / 1e3  # km
    return t, alt


def fig_flow():
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.axis("off")
    items = [
        (0.015, "问题牵引\n阻尼–月补概念"),
        (0.214, "LS 专题\n严格周期解 + 30 天演化"),
        (0.422, "AS 专题\n低近地点族域勘测"),
        (0.630, "AS 专题\n能量账本实验"),
        (0.838, "能力沉淀\nLS 6 项 + AS 7 项"),
    ]
    w = 0.16
    for x, label in items:
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 0.34, boxstyle="round,pad=0.012",
                                    linewidth=1.0, edgecolor="#31506e",
                                    facecolor="#eef3f8"))
        ax.text(x + w / 2, 0.52, label, ha="center", va="center", fontsize=8.2)
    for k in range(len(items) - 1):
        x0 = items[k][0] + w
        x1 = items[k + 1][0]
        ax.add_patch(FancyArrowPatch((x0 + 0.004, 0.52), (x1 - 0.004, 0.52),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color="#31506e"))
    ax.text(0.5, 0.10, "全程：测试驱动开发 · 三角色机制（设计组/专家组/质量组）· 证据链留档",
            ha="center", va="center", fontsize=8)
    fig.savefig(FIG / "flow.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_ls_evolution():
    t_a, a_a = load_alt(LS / "high_fidelity/LS_A_j2only.h5")
    t_b, a_b = load_alt(LS / "high_fidelity/LS_B_plus_moon.h5")
    t_c, a_c = load_alt(LS / "high_fidelity_circular/LS_B_plus_moon.h5")
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    ax.plot(t_a, a_a / 1e4, color="#7f7f7f", label="A：J2 仅（无月球）")
    ax.plot(t_c, a_c / 1e4, color="#2b6cb0", linestyle="--", label="B（理想）：+理想圆轨道月球（对照）")
    ax.plot(t_b, a_b / 1e4, color="#c53030", label="B（真实）：+真实月球星历（SPICE）")
    ax.set_xlabel("时间（天）")
    ax.set_ylabel("地心高度（$10^4$ km）")
    ax.set_xlim(0, 30)
    ax.legend(loc="lower right", fontsize=8)
    ax.annotate("近月交会 1.77×10⁴ km\n@ 29.54 d", xy=(29.54, a_b[np.argmin(np.abs(t_b-29.54))]/1e4),
                xytext=(21.5, 30.0), fontsize=7.6,
                arrowprops=dict(arrowstyle="->", linewidth=0.8))
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.savefig(FIG / "ls_evolution.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_as_evolution():
    t1, a1 = load_alt(AS / "AS_G1_nodrag.h5")
    t2, a2 = load_alt(AS / "AS_G2_drag1.0.h5")
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6),
                             gridspec_kw=dict(height_ratios=[2.1, 1.25], hspace=0.42))
    ax = axes[0]
    ax.plot(t1, a1 / 1e4, color="#7f7f7f", label="G1：无阻力")
    ax.plot(t2, a2 / 1e4, color="#c53030", label="G2：含阻力（$\\rho\\times1.0$）")
    for td, lab in [(0.0, "197 km"), (14.33, "879 km"), (28.72, "208 km")]:
        i = int(np.argmin(np.abs(t2 - td)))
        ax.plot([t2[i]], [a2[i] / 1e4], "o", ms=3.5, color="#c53030")
        ax.annotate(lab, xy=(t2[i], a2[i] / 1e4), xytext=(t2[i] + 0.6, a2[i] / 1e4 + 4.0),
                    fontsize=7.6, arrowprops=dict(arrowstyle="->", linewidth=0.7))
    ax.set_xlabel("时间（天）")
    ax.set_ylabel("地心高度（$10^4$ km）")
    ax.set_xlim(0, 30)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)

    # 下panel：逐次近地点高度（三档密度敏感度）
    from mission_sim.utils.dynamics import events as ev
    ax2 = axes[1]
    for name, color, lab in [("AS_G3_drag0.5.h5", "#2b6cb0", "$\\rho\\times0.5$"),
                             ("AS_G2_drag1.0.h5", "#c53030", "$\\rho\\times1.0$"),
                             ("AS_G4_drag1.5.h5", "#2f855a", "$\\rho\\times1.5$")]:
        with h5py.File(AS / name, "r") as f:
            t = f["epochs"][:]
            st = f["states"][:]
        per = ev.perigee_events(t, st, r_earth=6378137.0)
        alts = [p["altitude"] / 1e3 for p in per]
        ax2.plot(range(1, len(alts) + 1), alts, "o-", ms=3.5, linewidth=1.0,
                 color=color, label=lab + "：近地点高度")
    ax2.set_xlabel("近地点序号")
    ax2.set_ylabel("近地点高度（km）")
    ax2.set_xticks([1, 2, 3])
    ax2.legend(loc="center right", fontsize=7.6)
    ax2.grid(alpha=0.25, linewidth=0.5)
    fig.savefig(FIG / "as_evolution.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_ledger():
    # 数据取自实验日志《AS_exp_lowperigee_v2》（J/kg，周期 1）
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    labels = ["月球做功\n$W_{\\rm moon}$", "J2–J4 做功\n$W_{\\rm J2J4}$",
              "阻力做功\n$W_{\\rm drag}(\\rho\\times1.0)$"]
    values = [10750.0, 27554.0, -116.85]
    colors = ["#2b6cb0", "#2f855a", "#c53030"]
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.set_yscale("symlog", linthresh=100.0)
    ax.set_ylabel("周期功（J/kg，symlog）")
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.errorbar([2], [-116.85], yerr=[[58.43], [58.43]], fmt="none",
                ecolor="#7b341e", capsize=4, linewidth=1.2)
    for b, v in zip(bars, values):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:+,.0f}",
                    ha="center", fontsize=8, va="bottom")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v - 130.0, f"{v:+,.0f}",
                    ha="center", fontsize=8, va="top")
    ax.set_ylim(-1400.0, 3.2e5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.savefig(FIG / "ledger.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _crtbp():
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    return UniversalCRTBP.earth_moon_system()


def fig_concept():
    """图：“大气阻尼—月球借力”机制示意（主图 + 近地点放大的双联图）。"""
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(6.6, 3.5),
                                  gridspec_kw=dict(width_ratios=[2.3, 1.0], wspace=0.05))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "--", color="#b8b8b8", lw=0.9)
    ax.plot([0], [0], "o", color="#2b6cb0", ms=7)
    ax.text(-0.02, -0.17, "地球", fontsize=9, ha="center")
    ax.plot([np.cos(2.2)], [np.sin(2.2)], "o", color="#8a8a8a", ms=6)
    ax.text(np.cos(2.2) + 0.06, np.sin(2.2) + 0.02, "月球", fontsize=9)
    ax.text(-0.06, 1.30, "月球轨道（平均 38.44 万 km）", fontsize=8.2, ha="center",
            color="#666666")
    rp, ra = 0.16, 1.16
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    nu = np.linspace(0.0, 2 * np.pi, 800)
    r = a * (1 - e * e) / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    ax.plot(x, y, "-", color="#c53030", lw=1.4)
    imax = int(np.argmax(r))
    ax.annotate("远地点：接近月球轨道\n借力补充能量", xy=(x[imax], y[imax]),
                xytext=(-0.40, 1.66), fontsize=8.2, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.annotate("近地点：穿越大气\n损失能量（右图放大）", xy=(0.19, -0.06),
                xytext=(-0.28, -1.58), fontsize=8.2, ha="left", color="#c53030",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#c53030"))
    ax.set_xlim(-1.70, 1.62)
    ax.set_ylim(-1.76, 1.78)
    ax.set_aspect("equal")
    ax.axis("off")
    axz.add_patch(plt.Circle((0, 0), 0.34, color="#cfe8ff", zorder=0))
    axz.add_patch(plt.Circle((0, 0), 0.295, color="#2b6cb0", zorder=1))
    xs = np.linspace(-0.46, 0.46, 200)
    ys = 0.352 + 0.60 * xs ** 2
    axz.plot(xs, ys, "-", color="#c53030", lw=1.3)
    axz.text(0.0, -0.43, "大气层（示意）", fontsize=7.4, ha="center", color="#2b6cb0")
    axz.text(0.0, 0.68, "近地点放大：约 200 km", fontsize=7.6, ha="center")
    axz.set_xlim(-0.58, 0.58)
    axz.set_ylim(-0.60, 0.80)
    axz.set_aspect("equal")
    axz.axis("off")
    fig.text(0.5, 0.015, "共振 2 : 1 —— 航天器每绕两圈、月球绕一圈；两处能量收支相抵，即为“自持”概念",
             ha="center", fontsize=8.6)
    fig.savefig(FIG / "concept.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_symmetry():
    """图：对称判据（半周期垂直穿越 ⇒ 全周期闭合）示意。"""
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    ax.axis("off")
    ax.plot([-0.38, 1.34], [0, 0], color="#666666", lw=0.8)
    ax.plot([0], [0], "o", color="#2b6cb0", ms=6)
    ax.text(0.0, -0.12, "地球", fontsize=8.5, ha="center")
    ax.plot([1.04], [0], "o", color="#8a8a8a", ms=5)
    ax.text(1.04, -0.12, "月球", fontsize=8.5, ha="center")
    ts = np.linspace(0, 2 * np.pi, 400)
    xs = 0.62 - 0.40 * np.cos(ts)
    ys = 0.40 * np.sin(ts)
    ax.plot(xs, ys, "-", color="#c53030", lw=1.3)
    ax.annotate("", xy=(0.22, 0.33), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", lw=1.1))
    ax.annotate("", xy=(1.02, -0.33), xytext=(1.02, -0.08),
                arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", lw=1.1))
    ax.text(0.20, -0.15, "t = 0：在 x 轴上、垂直穿越（近地点）",
            fontsize=8, ha="center", va="top")
    ax.text(1.06, 0.10, "t = T/2：再次垂直穿越\n（由对称性，全周期闭合）",
            fontsize=8, ha="center", va="bottom")
    ax.text(0.60, 0.66, "“严格解”的检验条件：半周期截面残差  [y, ẋ] → 0",
            fontsize=9, ha="center")
    ax.set_xlim(-0.46, 1.46)
    ax.set_ylim(-0.78, 0.90)
    ax.set_aspect("equal")
    fig.savefig(FIG / "symmetry.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_model_ladder():
    """图：四层模型阶梯（从理想 CRTBP 到含大气的真实模型）。"""
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.axis("off")
    boxes = [
        ("① 理想模型 CRTBP", "地球点质量\n月球圆轨道"),
        ("② + 地球扁率 J2–J4", "带谐重力场\n轨道面缓慢进动"),
        ("③ + 真实月球星历", "SPICE：真实距离、\n偏心与倾角"),
        ("④ + 简化大气", "指数密度：\n近地点阻尼"),
    ]
    w, gap, x = 0.205, 0.030, 0.015
    for i, (t, s) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 0.36), w, 0.40, boxstyle="round,pad=0.012",
                                    linewidth=1.0, edgecolor="#31506e",
                                    facecolor="#eef3f8"))
        ax.text(x + w / 2, 0.635, t, ha="center", fontsize=8.4)
        ax.text(x + w / 2, 0.46, s, ha="center", fontsize=7.2, color="#333333")
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + w + 0.002, 0.56), (x + w + gap - 0.002, 0.56),
                                         arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                         color="#31506e"))
        x += w + gap
    ax.text(0.5, 0.13, "从理想走向真实：每加一层，“能否保持”的检验就难一层（实验以 A/B/C 组与“理想月球对照”逐层对照）",
            ha="center", fontsize=8.0)
    fig.savefig(FIG / "model_ladder.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _trace_rot(x0, t_shot):
    from scipy.integrate import solve_ivp
    from mission_sim.utils.dynamics import compiled_kernels as ck
    c = _crtbp()
    mu = c.mu

    def rhs(t, y):
        return ck.rhs_crtbp(y, mu)

    sol = solve_ivp(rhs, (0.0, t_shot), x0, rtol=1e-10, atol=1e-12,
                    dense_output=True, method="DOP853")
    ts = np.linspace(0.0, t_shot, 6000)
    return sol.sol(ts)


def fig_ls_orbit():
    """图：LS 严格解的轨道形状（旋转系）。"""
    c = _crtbp()
    mu = c.mu
    x_ls = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
    a = _trace_rot(x_ls, 13.6608305 / 4.342)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "--", color="#c0c0c0", lw=0.8)
    ax.text(0.79, 0.86, "月球轨道半径参考圈", fontsize=8, color="#808080")
    ax.plot([-mu], [0], "o", color="#2b6cb0", ms=7)
    ax.text(-mu - 0.02, -0.10, "地球", fontsize=9, ha="center")
    ax.plot([1 - mu], [0], "o", color="#8a8a8a", ms=6)
    ax.text(1 - mu + 0.03, 0.06, "月球（旋转系中静止）", fontsize=8.5)
    ax.plot(a[0], a[1], "-", color="#2b6cb0", lw=1.5)
    ax.plot([x_ls[0]], [0.0], "o", ms=4, color="#c53030")
    ax.annotate("t = 0 近地点（31.6 万 km）", xy=(0.82, 0.02), xytext=(0.16, 0.62),
                fontsize=8.2, arrowprops=dict(arrowstyle="->", lw=0.8, color="#c53030"))
    iapo = int(np.argmax(np.linalg.norm(a[:3], axis=0)))
    ax.annotate("远地点（45.7 万 km）", xy=(a[0, iapo], a[1, iapo]),
                xytext=(0.28, -0.55), fontsize=8.2,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.text(0.62, 1.34, "每 13.6608 天精确回到原处（全状态闭合）", fontsize=9, ha="center")
    ax.set_xlim(-0.30, 1.42)
    ax.set_ylim(-0.92, 1.46)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(FIG / "ls_orbit.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_as_orbit():
    """图：AS 近失解轨道形状（与 LS 严格解对比）。"""
    c = _crtbp()
    mu = c.mu
    x_ls = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
    x_as = np.array([0.0049602039, 0.0, 0.0, 0.0, 10.6562923, 0.0])
    a = _trace_rot(x_ls, 13.6608305 / 4.342)
    b = _trace_rot(x_as, 27.321661 / 4.342)
    xmin = min(float(b[0].min()), -0.45) - 0.12
    xmax = max(float(b[0].max()), 1.30) + 0.12
    ymin = float(b[1].min()) - 0.12
    ymax = float(b[1].max()) + 0.60
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "--", color="#c0c0c0", lw=0.8)
    ax.plot([-mu], [0], "o", color="#2b6cb0", ms=7)
    ax.text(-mu - 0.02, -0.14, "地球", fontsize=9, ha="center")
    ax.plot([1 - mu], [0], "o", color="#8a8a8a", ms=6)
    ax.text(1 - mu + 0.03, 0.08, "月球（旋转系中静止）", fontsize=8.5)
    ax.plot(a[0], a[1], "--", color="#2b6cb0", lw=1.2,
            label="LS 严格解（高近地点，单圈 13.66 天）")
    ax.plot(b[0], b[1], "-", color="#c53030", lw=1.3,
            label="AS 近失解（低近地点，两圈 /27.32 天）")
    ia = int(np.argmax(np.linalg.norm(b[:3], axis=0)))
    ax.annotate("远地点约 48.6 万 km\n（越过月球轨道）", xy=(b[0, ia], b[1, ia]),
                xytext=(0.45, ymax - 0.50), fontsize=8.2,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.annotate("近地点 197 km\n（贴近地球，图幅不可见）", xy=(0.006, 0.0),
                xytext=(xmax - 1.28, ymin + 0.36), fontsize=8.2, color="#c53030",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#c53030"))
    ax.text((xmin + xmax) / 2.0, ymin + 0.05,
            "虚线圆 = 月球轨道半径参考；半周期残差 2.5×10⁻⁴ —— 未精确闭合",
            fontsize=8.5, ha="center")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(FIG / "as_orbit.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fig_residual_map():
    """图：低近地点搜索区的半周期残差地图（“无零点”的强敏感区）。"""
    from mission_sim.utils.dynamics import compiled_kernels as ck
    c = _crtbp()
    mu = c.mu
    t_half = 13.6608305 / 4.342
    xs = np.linspace(0.0046, 0.0090, 90)
    vs = np.linspace(10.20, 10.75, 110)
    seeds = np.array([[x0, 0.0, 0.0, 0.0, v0, 0.0] for x0 in xs for v0 in vs])
    res = ck.batch_half_period_residual(seeds, t_half, 6000, mu)
    rn = np.hypot(res[:, 0], res[:, 1]).reshape(len(xs), len(vs))
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    im = ax.imshow(np.log10(rn.T), origin="lower", aspect="auto",
                   extent=[xs[0], xs[-1], vs[0], vs[-1]], cmap="viridis")
    cb = fig.colorbar(im)
    cb.set_label("log$_{10}$ 半周期残差（无量纲）", fontsize=8.5)
    ax.plot([0.0049602039], [10.6562923], "*", color="#c53030", ms=13)
    ax.annotate("最优近失：残差 $2.5\\times10^{-4}$\n（20 万步精修后）",
                xy=(0.00496, 10.656), xytext=(0.00530, 10.435), fontsize=8,
                color="#c53030",
                arrowprops=dict(arrowstyle="->", color="#c53030", lw=0.8))
    ax.set_xlabel("$x_0$（无量纲；近地点附近的位置）", fontsize=9)
    ax.set_ylabel("$v_{y0}$（无量纲）", fontsize=9)
    ax.text(0.00628, 10.728, "网格分辨率下全图无“零点”：最深处也停在 $10^{-3}$ 量级——",
            fontsize=8.0)
    ax.text(0.00628, 10.692, "强敏感/近切向结构使常规修正器普遍停滞",
            fontsize=8.0)
    fig.savefig(FIG / "residual_map.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_flow()
    fig_concept()
    fig_symmetry()
    fig_model_ladder()
    fig_ls_evolution()
    fig_ls_orbit()
    fig_as_evolution()
    fig_as_orbit()
    fig_residual_map()
    fig_ledger()
    print("插图已生成:", FIG)
