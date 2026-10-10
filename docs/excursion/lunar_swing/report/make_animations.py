#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LUNAR-SWING 研究总结报告：电子附件动画生成（可复现）。

生成 4 段 MP4（不嵌入 PDF）：
  A1_mechanism.mp4        概念机制示意（大气损耗 / 月球借力 / 能量平衡）
  A2_ls_dual_view.mp4     LS 严格解：旋转系（一圈合） vs 惯性系（两圈合）
  A3_ls_30d_spice.mp4     LS 组 30 天真实星历演化（含 1.77 万 km 近月交会）
  A4_as_30d_spice.mp4     AS 近失解 30 天演化（近地点 197→879→208 km）

用法: python3 make_animations.py [--only A1,A2]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

rcParams["font.family"] = ["Songti SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 9

OUT = Path(__file__).resolve().parent / "animations"
OUT.mkdir(parents=True, exist_ok=True)
LS_RES = ROOT / "docs/excursion/lunar_swing/results"
AS_RES = LS_RES / "as_lowperigee"
FPS = 25
DPI = 150

X_LS = np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
X_AS = np.array([0.0049602039, 0.0, 0.0, 0.0, 10.6562923, 0.0])


def _mu():
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    return UniversalCRTBP.earth_moon_system().mu


def _save(anim, name):
    path = OUT / name
    anim.save(str(path), writer=FFMpegWriter(fps=FPS, bitrate=3200), dpi=DPI)
    plt.close("all")
    print("✔", path, flush=True)


# ---------------------------------------------------------------- A1 机制示意
def make_a1():
    mu = 0.0
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "--", color="#b8b8b8", lw=0.9)
    ax.text(-0.06, 1.28, "月球轨道（平均 38.44 万 km）", fontsize=8.4, ha="center",
            color="#666666")
    ax.add_patch(Circle((0, 0), 0.052, color="#2b6cb0", zorder=5))
    ax.text(-0.02, -0.17, "地球", fontsize=9, ha="center")
    ax.add_patch(Circle((0, 0), 0.085, facecolor="none", edgecolor="#9fd3ff",
                        lw=2.0, linestyle="-", zorder=3))
    ax.text(0.115, 0.10, "大气层（放大示意）", fontsize=7.6, color="#2b6cb0")
    ax.plot([np.cos(2.2)], [np.sin(2.2)], "o", color="#8a8a8a", ms=7, zorder=5)
    ax.text(np.cos(2.2) + 0.06, np.sin(2.2) + 0.02, "月球", fontsize=9)
    rp, ra = 0.16, 1.16
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    nu = np.linspace(0.0, 2 * np.pi, 800)
    r = a * (1 - e * e) / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    ax.plot(x, y, "-", color="#c53030", lw=1.2, alpha=0.4)
    trail, = ax.plot([], [], "-", color="#c53030", lw=1.6)
    dot, = ax.plot([], [], "o", color="#c53030", ms=6, zorder=6)
    flash_peri, = ax.plot([0.115], [0.0], "o", ms=16, color="#e53e3e", alpha=0.0, zorder=7)
    flash_apo, = ax.plot([x[np.argmax(r)]], [y[np.argmax(r)]], "o", ms=20,
                         color="#ecc94b", alpha=0.0, zorder=7)
    txt = ax.text(0.0, 1.48, "", fontsize=9.5, ha="center")
    evt = ax.text(0.0, -1.52, "", fontsize=9.5, ha="center", color="#c53030")
    ax.set_xlim(-1.72, 1.62)
    ax.set_ylim(-1.72, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")
    T = 300

    def frame(i):
        t = i / T  # 0→1 一圈（13.66 天）
        k = int(t * (len(nu) - 1))
        trail.set_data(x[:k + 1], y[:k + 1])
        dot.set_data([x[k]], [y[k]])
        txt.set_text("一圈 = 13.6608 天（航天器）；本视角随月球公转旋转")
        # 事件闪光：近地点 t≈0，远地点 t≈0.5
        f1 = np.exp(-((t - 0.01) / 0.05) ** 2) + np.exp(-((t - 0.99) / 0.05) ** 2)
        f2 = np.exp(-((t - 0.5) / 0.05) ** 2)
        flash_peri.set_alpha(0.85 * f1)
        flash_apo.set_alpha(0.85 * f2)
        if f1 > 0.5:
            evt.set_text("近地点：穿越大气 → 能量 −")
            evt.set_color("#c53030")
        elif f2 > 0.5:
            evt.set_text("远地点：月球借力 → 能量 +")
            evt.set_color("#b7791f")
        else:
            evt.set_text("")
        return trail, dot, flash_peri, flash_apo, txt, evt

    anim = FuncAnimation(fig, frame, frames=T, blit=False, interval=1000 / FPS)
    _save(anim, "A1_mechanism.mp4")


# ---------------------------------------------------------------- A2 双视角
def make_a2():
    from scipy.integrate import solve_ivp
    from mission_sim.utils.dynamics import compiled_kernels as ck
    mu = _mu()
    t_2p = 2 * 13.6608305 / 4.342  # 两个旋转系周期（TU）

    def rhs(t, y):
        return ck.rhs_crtbp(y, mu)

    sol = solve_ivp(rhs, (0.0, t_2p), X_LS, rtol=1e-10, atol=1e-12,
                    dense_output=True, method="DOP853")
    T = 660

    def pos_rot(t):
        s = sol.sol(t)
        return s[0] + mu, s[1]

    def pos_geo_in(t):
        s = sol.sol(t)
        xr, yr = s[0] + mu, s[1]
        c, sn = np.cos(t), np.sin(t)
        return c * xr - sn * yr, sn * xr + c * yr

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.9))
    for ax, title in [(axl, "随月球一同旋转的视角（月球静止）"),
                      (axr, "地心惯性视角")]:
        ax.set_title(title, fontsize=10)
        ax.add_patch(Circle((0, 0), 0.035, color="#2b6cb0", zorder=5))
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(th), np.sin(th), "--", color="#c8c8c8", lw=0.7, zorder=1)
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)
        ax.set_aspect("equal")
        ax.axis("off")
    axl.text(-1.36, 1.24, "地球", fontsize=9)
    axl.text(1.04, 0.06, "月球（静止）", fontsize=9)
    axr.text(-1.36, 1.24, "地球", fontsize=9)
    moon_l, = axl.plot([1 - mu], [0.0], "o", color="#8a8a8a", ms=7, zorder=5)
    moon_r, = axr.plot([], [], "o", color="#8a8a8a", ms=7, zorder=5)
    trail_l, = axl.plot([], [], "-", color="#2b6cb0", lw=1.3)
    trail_r, = axr.plot([], [], "-", color="#c53030", lw=1.3)
    dot_l, = axl.plot([], [], "o", color="#2b6cb0", ms=4, zorder=6)
    dot_r, = axr.plot([], [], "o", color="#c53030", ms=4, zorder=6)
    time_txt = axl.text(-1.36, -1.30, "", fontsize=9)
    end_txt_l = axl.text(0.0, 1.26, "", fontsize=9, ha="center", color="#2b6cb0")
    end_txt_r = axr.text(0.0, -1.30, "", fontsize=9, ha="center", color="#c53030")

    def frame(i):
        t = (i + 1) / T * t_2p
        xs, ys = [], []
        for k in range(0, 120):
            tt = t * k / 119.0
            xr, yr = pos_rot(tt)
            xs.append(xr)
            ys.append(yr)
        trail_l.set_data(xs, ys)
        x0, y0 = pos_rot(t)
        dot_l.set_data([x0], [y0])
        xs, ys = [], []
        for k in range(0, 160):
            tt = t * k / 159.0
            xi, yi = pos_geo_in(tt)
            xs.append(xi)
            ys.append(yi)
        trail_r.set_data(xs, ys)
        x1, y1 = pos_geo_in(t)
        dot_r.set_data([x1], [y1])
        c, sn = np.cos(t), np.sin(t)
        moon_r.set_data([c], [sn])
        time_txt.set_text("t = %.2f 天（一个周期 13.6608 天）" % (t * 4.342))
        days = t * 4.342
        if days > 27.0:
            end_txt_l.set_text("")
            end_txt_r.set_text("惯性系：两圈（27.32 天）之后才回到原处")
        elif days > 13.6:
            end_txt_l.set_text("旋转系：一圈（13.66 天）已经闭合")
            end_txt_r.set_text("")
        else:
            end_txt_l.set_text("")
            end_txt_r.set_text("")
        return trail_l, trail_r, dot_l, dot_r, moon_r, time_txt, end_txt_l, end_txt_r

    anim = FuncAnimation(fig, frame, frames=T, blit=False, interval=1000 / FPS)
    _save(anim, "A2_ls_dual_view.mp4")


# ---------------------------------------------------------------- 数据载入
def _spice_provider(epoch0=0.0):
    from mission_sim.core.spacetime.ephemeris.spice_interface import SPICEInterface
    from mission_sim.core.spacetime.ids import CoordinateFrame
    spice = SPICEInterface()
    assert spice.initialize("earth_moon")

    def get(t):
        return np.asarray(spice.get_state("moon", epoch0 + float(t), observer="earth",
                                          frame=CoordinateFrame.J2000_ECI),
                          dtype=np.float64)
    return get


def _basis_from_moon(moon0):
    d = moon0[:3] / np.linalg.norm(moon0[:3])
    h = np.cross(moon0[:3], moon0[3:6])
    h = h / np.linalg.norm(h)
    y = np.cross(h, d)
    return d, y


def _load_traj(path):
    with h5py.File(path, "r") as f:
        t = f["epochs"][:]
        st = f["states"][:]
    return t, st


# ---------------------------------------------------------------- A3 LS 30d
def make_a3():
    t, st = _load_traj(LS_RES / "high_fidelity/LS_B_plus_moon.h5")
    get_moon = _spice_provider(0.0)
    d, yv = _basis_from_moon(get_moon(0.0))
    rr = st[:, :3]
    px = rr @ d
    py = rr @ yv
    T = 600
    tframes = np.linspace(t[0], t[-1], T)
    fx = np.interp(tframes, t, px)
    fy = np.interp(tframes, t, py)
    days = tframes / 86400.0
    mp = np.array([get_moon(x) for x in tframes])
    mx = mp[:, :3] @ d
    my = mp[:, :3] @ yv
    md = np.linalg.norm(mp[:, :3] - st[np.searchsorted(t, tframes), :3], axis=1) / 1e3
    scale = 1.0e-6  # m → 10^3 km 的显示刻度
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.03, 0.06, 0.62, 0.88])
    axin = fig.add_axes([0.68, 0.10, 0.30, 0.34])
    lim = 5.6e8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 6e6, color="#2b6cb0", zorder=5))
    ax.text(-1.5e7, -4.2e7, "地球", fontsize=9, ha="center")
    moon_trail, = ax.plot([], [], "--", color="#909090", lw=0.8)
    trail, = ax.plot([], [], "-", color="#c53030", lw=1.4)
    dot, = ax.plot([], [], "o", color="#c53030", ms=4, zorder=6)
    moon_dot, = ax.plot([], [], "o", color="#8a8a8a", ms=6, zorder=6)
    ring, = ax.plot([], [], "o", ms=26, mfc="none", mec="#ecc94b", mew=2.0, zorder=7)
    ann = ax.text(-lim * 0.97, lim * 0.88, "", fontsize=10)
    axin.set_xlim(0, 30)
    axin.set_ylim(0, 4.2e5)
    axin.plot(days, md, "-", color="#2b6cb0", lw=1.0)
    axin.set_xlabel("时间（天）", fontsize=8)
    axin.set_ylabel("与月球距离（10³ km）", fontsize=8)
    axin.tick_params(labelsize=7.5)
    cursor, = axin.plot([], [], "o", color="#c53030", ms=4)
    flyby_i = int(np.argmin(md))

    def frame(i):
        k = i + 1
        trail.set_data(px[:int(k / T * len(t))], py[:int(k / T * len(t))])
        dot.set_data([fx[i]], [fy[i]])
        moon_trail.set_data(mx[:i + 1], my[:i + 1])
        moon_dot.set_data([mx[i]], [my[i]])
        cursor.set_data([days[i]], [md[i]])
        ann.set_text("t = %.1f 天（共 30 天，真实月球星历）" % days[i])
        if abs(i - flyby_i) <= 12:
            ring.set_data([mx[i]], [my[i]])
            ann.set_text("t = %.2f 天：近月交会，最近 1.77×10⁴ km" % days[i])
        else:
            ring.set_data([], [])
        return trail, dot, moon_trail, moon_dot, ring, ann, cursor

    anim = FuncAnimation(fig, frame, frames=T, blit=False, interval=1000 / FPS)
    _save(anim, "A3_ls_30d_spice.mp4")


# ---------------------------------------------------------------- A4 AS 30d
def make_a4():
    t, st = _load_traj(AS_RES / "AS_G2_drag1.0.h5")
    get_moon = _spice_provider(0.0)
    d, yv = _basis_from_moon(get_moon(0.0))
    rr = st[:, :3]
    px = rr @ d
    py = rr @ yv
    T = 600
    tframes = np.linspace(t[0], t[-1], T)
    fx = np.interp(tframes, t, px)
    fy = np.interp(tframes, t, py)
    days = tframes / 86400.0
    alt = np.interp(tframes, t, np.linalg.norm(st[:, :3], axis=1) - 6378137.0)
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.03, 0.06, 0.62, 0.88])
    axin = fig.add_axes([0.68, 0.10, 0.30, 0.34])
    lim = 5.2e8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 6e6, color="#2b6cb0", zorder=5))
    ax.add_patch(Circle((0, 0), 1.2e7, facecolor="none", edgecolor="#9fd3ff",
                        lw=1.6, linestyle="--"))
    ax.text(3.6e7, -6e7, "地球 + 大气（大幅放大示意）", fontsize=8.5)
    trail, = ax.plot([], [], "-", color="#c53030", lw=1.2)
    dot, = ax.plot([], [], "o", color="#c53030", ms=4, zorder=6)
    peri_marks = [
        (0.0, "197 km"), (14.33, "879 km"), (28.72, "208 km")]
    ax.text(-lim * 0.97, lim * 0.90,
            "近地点：197 → 879 → 208 km（三次穿越大气）", fontsize=10)
    ann = ax.text(-lim * 0.97, lim * 0.80, "", fontsize=9.5, color="#c53030")
    axin.set_xlim(0, 30)
    axin.set_yscale("log")
    axin.set_ylim(1.0e5, 6.5e8)
    axin.plot(days, alt + 6378137.0, "-", color="#c53030", lw=1.0)
    for dw, lab in peri_marks:
        i = int(np.argmin(np.abs(days - dw)))
        axin.plot([days[i]], [alt[i] + 6378137.0], "o", color="#2b6cb0", ms=3.5)
        axin.annotate(lab, (days[i], alt[i] + 6378137.0), fontsize=6.5,
                      xytext=(days[i] + 0.8, (alt[i] + 6378137.0) * 1.9))
    axin.set_xlabel("时间（天）", fontsize=8)
    axin.set_ylabel("地心距（m，对数）", fontsize=8)
    axin.tick_params(labelsize=7.5)
    cursor, = axin.plot([], [], "o", color="#2b6cb0", ms=4)

    def frame(i):
        k = int((i + 1) / T * len(t))
        trail.set_data(px[:k], py[:k])
        dot.set_data([fx[i]], [fy[i]])
        cursor.set_data([days[i]], [alt[i] + 6378137.0])
        for dw, lab in peri_marks:
            if abs(days[i] - dw) <= 0.8:
                ann.set_text("t = %.1f 天：穿越大气（能量 −）" % days[i])
                break
        else:
            ann.set_text("")
        return trail, dot, cursor, ann

    anim = FuncAnimation(fig, frame, frames=T, blit=False, interval=1000 / FPS)
    _save(anim, "A4_as_30d_spice.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="A1,A2,A3,A4")
    args = ap.parse_args()
    todo = {s.strip().upper() for s in args.only.split(",")}
    if "A1" in todo:
        make_a1()
    if "A2" in todo:
        make_a2()
    if "A3" in todo:
        make_a3()
    if "A4" in todo:
        make_a4()
    print("动画输出目录:", OUT)
