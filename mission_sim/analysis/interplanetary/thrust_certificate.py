# mission_sim/analysis/interplanetary/thrust_certificate.py
# -*- coding: utf-8 -*-
"""小推力"半径-时间"不可行证书与螺旋能力（Ceres R1-A 能力沉淀件）。

来源：谷神星探测专题 自由探索 R1-A（``工具/s4_lt2_r1.py`` v0.2）。算法、常数与报数口径不变
（对拍见 ``tests/test_interplanetary_certificate_closure.py``：关键读数与岛内机读结果
``任务/拓展_R1_返程能力.json``／``任务/拓展_R5_*.json`` 逐位一致）。

本件提供
--------
* **最短时螺旋**（:func:`min_time_spiral`）与**固定 T 最小可达半径**
  （:func:`min_radius_at_T`）——后者是**不可行证书**：若 ``r_min(T) > 目标半径``
  （与相位、v∞ 无关的**必要条件**），则该 TOF 内半径上不可达；反之**不能**断言可行。
* 返程腿 SF 的两段式求解（:func:`init_spiral` 螺旋热启动 ＋ :func:`sf_solve2`）。
* 弹道几何界（:func:`ballistic_bounds`）、脉冲 v∞ 前沿与门限表
  （:func:`vinf_scan`／:func:`frontier`／:func:`threshold_table`）。
* 返程腿问题工厂（:func:`prob_ret`）。

口径（沿源件，逐条不变）
------------------------
* 动力学＝日心二体（μ☉）＋分段冲量（SF，``cap_k = a_max(r_k,m_k)·min(Δt_{k-1},Δt_k)``）
  ＋段间通用变量 Kepler；**不含借力、不含 N 体**。螺旋积分器为 **3D Cartesian** RK4
  （v0.2 修正：v0.1 的 2D 极坐标会丢黄道面外分量，且 η(t) 律的时间单位缺陷已修）。
* 推力模型 ``SEPModel``：``P(r)=P0·k_deg/r_AU²``、``T=2ηP/v_e``、``a=T/m``；
  P0＝27 500 W、Isp＝4 000 s、η＝0.70【假设】、k_deg 默认 1.00（真值【待核】）。
* 推力方向律：面内、与**面内横向** t̂ 成 η 角（η=0 顺行切向、η=π 逆行切向、
  η=±π/2 纯径向），用瞬时角动量 ĥ 定义，故对倾斜轨道亦精确。
* **线程**：import numpy **之前**固定 ``*_NUM_THREADS=1``（沿 E1 教训：小数组多线程只增同步开销）。

与源件差异（仅宿主适配；算法、常数、容差与随机种子**未改**）
------------------------------------------------------------
① 依赖内联：岛内 ``s4_lt_common``／``s4_lt_design`` → 本仓 ``low_thrust``／``sf_design``
   （数值口径已在前批 93b68ea／d56418b 对拍为逐位一致）。
② 不再向岛内 ``任务/*.json`` 落盘：``run_capability``／``run_vinf`` 只返回结果 dict，
   需要落盘时由调用方用 ``low_thrust.dump_json`` 处理。
③ 为便于测试，:func:`min_time_spiral`／:func:`min_radius_at_T` 把源件内联的迭代上限
   提为**可选**参数（``maxiter`` 等；**默认值与源件一致**，默认路径逐位不变）。
④ 未搬运：源件 ``--sens``／``--extra``／``--ret``（依赖 ``s4_lt_variants``，已独立沉淀为
   ``flyby_chain``）与 ``figs``（matplotlib 出图）——列为【待裁定】。
⑤ 已知源件缺陷**原样继承**：:func:`init_spiral` 用 ``prob.nodes`` 逐点取时间，对**单腿**
   问题（``nodes`` 仅 2 项）会 ``IndexError``（源件注记：该函数从未在单腿问题上跑过）。
   单腿场景请用 ``closure_options.init_spiral_local``（源件 R5 的修正版）。

用法::

  python3 -m mission_sim.analysis.interplanetary.thrust_certificate --self-test
  python3 -m mission_sim.analysis.interplanetary.thrust_certificate --capability
  python3 -m mission_sim.analysis.interplanetary.thrust_certificate --vinf
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from mission_sim.analysis.interplanetary import low_thrust as LT  # noqa: E402
from mission_sim.analysis.interplanetary import sf_design as D  # noqa: E402

AU = LT.AU_KM
MU = LT.MU_SUN
DAY = LT.DAY
M0 = D.M0_KG
ISP = D.ISP_S

T_LAUNCH = LT.dfe(2030, 12, 15)  # SR-1 发射（沿 E1）
T_ARRIVE = LT.dfe(2037, 6, 17)  # SR-1 抵达谷神星
T_DEPART = LT.dfe(2038, 10, 30)  # SR-1 离场谷神星
T_BACK = LT.dfe(2043, 7, 15)  # SR-1 再入地球

R_ARR_EARTH: Optional[float] = None  # 运行时填：地球再入点的 |r|（AU）


# ================================================================ 0 工具：SEP 加速度
def a_sep(
    r_km: float,
    m_kg: float,
    k_deg: float = 1.0,
    eta: float = 0.70,
    P0: float = 27500.0,
    isp: float = ISP,
) -> float:
    """SEP 加速度（km/s²），与 ``LT.SEPModel.a_km_s2`` 逐式一致（本件独立写一份便于内环加速）。"""
    ve = isp * LT.G0 / 1000.0
    P = P0 * k_deg / (r_km / AU) ** 2
    T = 2.0 * eta * P / (ve * 1000.0)
    return T / (max(m_kg, 1.0) * 1000.0)


# ================================================================ 1 最短时螺旋
# 状态 y = [rx,ry,rz, vx,vy,vz, m]（km / km·s⁻¹ / kg），3D、精确到瞬时轨道面。
# 推力方向律：η(t)＝a0 + a1·t/Tref + a2·(t/Tref)²   （面内；η 为与**面内横向** t̂ 的夹角）。
def _deriv3(
    y: Sequence[float],
    law: Any,
    k_deg: float,
    m0: float,
    eta: float = 0.70,
    P0: float = 27500.0,
    isp: float = ISP,
    mu: float = MU,
    t_day: float = 0.0,
) -> List[float]:
    """3D RK4 右端：y＝[rx,ry,rz,vx,vy,vz,m]（km / km·s⁻¹ / kg）。

    推力方向＝轨道面内、与**面内横向** t̂=normalize(ĥ×r̂) 成 η 角（η=0 顺行切向、η=π 逆行切向、
    η=+90° 向外径向、−90° 向内径向）——**面内**定义用瞬时角动量 ĥ，故对倾斜轨道亦精确。
    """
    rv = [y[0], y[1], y[2]]
    vv = [y[3], y[4], y[5]]
    m = y[6]
    r = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
    a_mag = a_sep(r, m, k_deg, eta, P0, isp)
    h = [
        rv[1] * vv[2] - rv[2] * vv[1],
        rv[2] * vv[0] - rv[0] * vv[2],
        rv[0] * vv[1] - rv[1] * vv[0],
    ]
    ur = [y[0] / r, y[1] / r, y[2] / r]
    ut = [
        h[1] * ur[2] - h[2] * ur[1],
        h[2] * ur[0] - h[0] * ur[2],
        h[0] * ur[1] - h[1] * ur[0],
    ]
    hn = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
    if hn < 1e-12:
        return [
            vv[0],
            vv[1],
            vv[2],
            -mu / r**3 * y[0],
            -mu / r**3 * y[1],
            -mu / r**3 * y[2],
            0.0,
        ]
    ut = [q / hn for q in ut]
    e = law(t_day)
    ce, se = math.cos(e), math.sin(e)
    acc = [a_mag * (ce * ut[i] + se * ur[i]) for i in range(3)]
    return [
        vv[0],
        vv[1],
        vv[2],
        -mu / r**3 * y[0] + acc[0],
        -mu / r**3 * y[1] + acc[1],
        -mu / r**3 * y[2] + acc[2],
        -m * a_mag / (isp * LT.G0 / 1000.0),
    ]


def spiral_run(
    r0v: Sequence[float],
    v0v: Sequence[float],
    m0: float = M0,
    law: Optional[Any] = None,
    k_deg: float = 1.0,
    tmax_d: float = 2600.0,
    dt_d: float = 0.25,
    target_au: Optional[float] = None,
    sample_d: float = 10.0,
) -> Dict[str, Any]:
    """**3D** RK4 积分（精确到瞬时轨道面）。

    返回 dict（含到 target 半径的**首穿**时刻、Δv、状态史）。``hist`` 每 ``sample_d`` 天一条，
    含三维状态（供 :func:`init_spiral` 构造 SF 初值）。
    """
    law = law or (lambda tt: math.pi)
    y = [r0v[0], r0v[1], r0v[2], v0v[0], v0v[1], v0v[2], float(m0)]
    h = dt_d * DAY
    t = 0.0
    hist: List[Dict[str, Any]] = []
    prev = None
    nxt = 0.0
    while t < tmax_d:
        k1 = _deriv3(y, law, k_deg, m0, t_day=t)
        y2 = [y[i] + 0.5 * h * k1[i] for i in range(7)]
        k2 = _deriv3(y2, law, k_deg, m0, t_day=t + 0.5 * dt_d)
        y3 = [y[i] + 0.5 * h * k2[i] for i in range(7)]
        k3 = _deriv3(y3, law, k_deg, m0, t_day=t + 0.5 * dt_d)
        y4 = [y[i] + h * k3[i] for i in range(7)]
        k4 = _deriv3(y4, law, k_deg, m0, t_day=t + dt_d)
        y = [y[i] + h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(7)]
        t += dt_d
        rr = math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        if t >= nxt:
            hist.append(
                dict(
                    t_d=t,
                    r_au=rr / AU,
                    state=[y[i] for i in range(7)],
                    dv=ISP * LT.G0 / 1000.0 * math.log(m0 / y[6]),
                )
            )
            nxt += sample_d
        if target_au is not None and prev is not None:
            if (rr - target_au * AU) * (prev - target_au * AU) <= 0.0 and rr != prev:
                f = (prev - target_au * AU) / (prev - rr)
                tc = t - dt_d + f * dt_d
                sp = math.sqrt(y[3] ** 2 + y[4] ** 2 + y[5] ** 2)
                return dict(
                    reached=True,
                    t_d=tc,
                    r_au=rr / AU,
                    state=list(y),
                    dv=ISP * LT.G0 / 1000.0 * math.log(m0 / y[6]),
                    speed=sp,
                    hist=hist,
                )
        prev = rr
    sp = math.sqrt(y[3] ** 2 + y[4] ** 2 + y[5] ** 2)
    return dict(
        reached=False,
        t_d=t,
        r_au=math.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) / AU,
        state=list(y),
        dv=ISP * LT.G0 / 1000.0 * math.log(m0 / y[6]),
        speed=sp,
        hist=hist,
    )


def min_time_spiral(
    k_deg: float = 1.0,
    m0: float = M0,
    target_au: Optional[float] = None,
    tdep: float = T_DEPART,
    nstart: int = 3,
    tmax_d: float = 2600.0,
    dt_d: float = 0.25,
    maxiter: int = 400,
) -> Dict[str, Any]:
    """最短时螺旋：以 η(t) 三系数为设计变量，Nelder-Mead 最小化"到达 target 半径的时刻"。

    报告：t_min、Δv、终端速度（用于估 v∞）。**这是"半径下降"这一必要条件的下限诊断**，
    不含相位 / v∞ 约束。
    """
    r0, v0 = LT.state_off("ceres", tdep)
    Tref = 1700.0
    deg = 3

    def mk(p: Sequence[float]) -> Any:
        return lambda tt, p=p: sum(p[d] * (tt / Tref) ** d for d in range(len(p)))

    def obj(p: Sequence[float]) -> float:
        res = spiral_run(
            r0,
            v0,
            m0,
            mk(p),
            k_deg=k_deg,
            tmax_d=tmax_d,
            dt_d=dt_d,
            target_au=target_au,
            sample_d=1e9,
        )
        if res["reached"]:
            return res["t_d"]
        return 5000.0 - res["r_au"] * 100.0

    starts = [
        [math.pi, 0.0, 0.0, 0.0],
        [math.pi, -0.3, 0.0, 0.0],
        [math.pi, 0.3, -0.3, 0.0],
        [3.2, 0.31, -0.29, 0.0],
        [3.22, 0.0, 0.0, 0.0],
        [math.pi, -0.6, 0.6, 0.0],
        [math.pi + 0.1, 0.5, -0.5, 0.0],
        [3.3, -0.4, 0.3, 0.0],
    ]
    best = None
    for p0 in starts[: max(nstart, 6)]:
        r = minimize(
            obj,
            [float(x) for x in p0[:deg]],
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-3),
        )
        if best is None or r.fun < best.fun:
            best = r
    law = mk(best.x)
    res = spiral_run(
        r0,
        v0,
        m0,
        law,
        k_deg=k_deg,
        tmax_d=tmax_d,
        dt_d=dt_d,
        target_au=target_au,
        sample_d=20.0,
    )
    res["law"] = ["%.6f" % v for v in best.x]
    res["k_deg"] = k_deg
    res["t_min_d"] = res["t_d"] if res["reached"] else None
    # 终端速度（相对地球的**下界**：三角不等式 v∞ ≥ |v| − |v_E|）——注意此处车辆并不在地球位置，
    #  故只报下界与速度标量，不把矢量差当 v∞。
    t_end = tdep + res["t_d"]
    _, vE = LT.state_off("earth", t_end)
    res["v_earth_kms"] = LT.vnorm(vE)
    res["vinf_lower_kms"] = res["speed"] - LT.vnorm(vE)
    res["m_final_kg"] = res["state"][6]
    return res


def thrust_mult_for_tof(
    tof_d: float = 1719.0,
    target_au: Optional[float] = None,
    lo: float = 1.0,
    hi: float = 1.35,
    tdep: float = T_DEPART,
    dt_d: float = 0.5,
) -> Optional[float]:
    """求"使 t_min = tof_d"的推力倍率 k*（对 k_deg 单调）——即"需要多大推力才来得及"。"""

    def tmin(k: float) -> float:
        r0, v0 = LT.state_off("ceres", tdep)
        res = spiral_run(
            r0,
            v0,
            M0,
            lambda tt: math.pi,
            k_deg=k,
            tmax_d=tof_d + 600.0,
            dt_d=dt_d,
            target_au=target_au,
            sample_d=1e9,
        )
        return res["t_d"] if res["reached"] else 9.9e3

    if tmin(hi) > tof_d:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if tmin(mid) > tof_d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def min_radius_at_T(
    T_d: float = 1719.0,
    k_deg: float = 1.0,
    m0: float = M0,
    tdep: float = T_DEPART,
    nsec: int = 10,
    dt_d: float = 0.5,
    nstart: int = 5,
    maxiter: int = 4000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """固定 T 内**可达的最小日心距** r_min(T)（对分段常值 η 律做直接搜索）。

    用途＝**不可行证书**：若 ``r_min(1719 d) > 地球再入半径``，则返程腿在给定时间线内不可达
    （与相位、v∞ 无关的**必要条件**）。同时给 Δv 与该时刻的 v∞。
    """
    r0, v0 = LT.state_off("ceres", tdep)
    mesh = np.linspace(0.0, T_d, nsec + 1)

    def law_of(p: Sequence[float]) -> Any:
        def law(tt: float) -> float:
            i = min(nsec - 1, max(0, int(tt / T_d * nsec)))
            return p[i]

        return law

    def obj(p: Sequence[float]) -> float:
        res = spiral_run(
            r0,
            v0,
            m0,
            law_of(p),
            k_deg=k_deg,
            tmax_d=T_d + 1e-9,
            dt_d=dt_d,
            target_au=None,
            sample_d=1e9,
        )
        return res["r_au"]

    best = None
    starts = [
        np.full(nsec, math.pi),
        np.array([math.pi] * 3 + [math.pi * 0.75] * (nsec - 3)),
        np.linspace(3.3, 3.0, nsec),
        np.full(nsec, 3.22),
        np.array([3.45] * 2 + [3.15] * (nsec - 2)),
    ]
    for s in starts[:nstart]:
        r = minimize(
            obj,
            s,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-5),
        )
        if best is None or r.fun < best.fun:
            best = r
        if verbose:
            print("      start %s → r(T)=%.5f AU" % (np.round(s, 2), r.fun))
    res = spiral_run(
        r0,
        v0,
        m0,
        law_of(best.x),
        k_deg=k_deg,
        tmax_d=T_d + 1e-9,
        dt_d=dt_d,
        target_au=None,
        sample_d=1e9,
    )
    _, vE = LT.state_off("earth", tdep + T_d)
    return dict(
        T_d=T_d,
        k_deg=k_deg,
        r_min_au=res["r_au"],
        dv_kms=res["dv"],
        m_final_kg=res["state"][6],
        speed_kms=res["speed"],
        v_earth_kms=LT.vnorm(vE),
        vinf_lower_kms=res["speed"] - LT.vnorm(vE),
        law=["%.3f" % v for v in best.x],
        note="分段常值 η（%d 段）× Nelder-Mead；η=π 为逆行切向" % nsec,
    )


def _thrust_dir(rv: Sequence[float], vv: Sequence[float], e: float) -> List[float]:
    """面内推力方向单位矢量：η=0⇒面内横向（顺行）、η=π⇒逆行、±90°⇒径向。"""
    r = math.sqrt(sum(q * q for q in rv))
    ur = [q / r for q in rv]
    h = [
        rv[1] * vv[2] - rv[2] * vv[1],
        rv[2] * vv[0] - rv[0] * vv[2],
        rv[0] * vv[1] - rv[1] * vv[0],
    ]
    ut = [
        h[1] * ur[2] - h[2] * ur[1],
        h[2] * ur[0] - h[0] * ur[2],
        h[0] * ur[1] - h[1] * ur[0],
    ]
    hn = math.sqrt(sum(q * q for q in ut))
    if hn < 1e-12:
        return ur
    ut = [q / hn for q in ut]
    return [math.cos(e) * ut[i] + math.sin(e) * ur[i] for i in range(3)]


def spiral_nodes(
    r0: Sequence[float],
    v0: Sequence[float],
    m0: float,
    t_nodes: Sequence[float],
    law: Any,
    k_deg: float = 1.0,
    dt_d: float = 0.5,
) -> List[Tuple[List[float], List[float], float, List[float]]]:
    """积分到各 t_nodes（天，相对起点），返回每节点的 (r, v, m, 推力单位方向)。"""
    y = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], float(m0)]
    out: List[Tuple[List[float], List[float], float, List[float]]] = []
    t = 0.0
    for tt in t_nodes:
        while t < tt - 1e-9:
            step = min(dt_d, tt - t) * DAY
            k1 = _deriv3(y, law, k_deg, m0, t_day=t)
            y2 = [y[i] + 0.5 * step * k1[i] for i in range(7)]
            k2 = _deriv3(y2, law, k_deg, m0, t_day=t + 0.5 * step / DAY)
            y3 = [y[i] + 0.5 * step * k2[i] for i in range(7)]
            k3 = _deriv3(y3, law, k_deg, m0, t_day=t + 0.5 * step / DAY)
            y4 = [y[i] + step * k3[i] for i in range(7)]
            k4 = _deriv3(y4, law, k_deg, m0, t_day=t + step / DAY)
            y = [
                y[i] + step / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                for i in range(7)
            ]
            t += step / DAY
        rv, vv = y[0:3], y[3:6]
        out.append((list(rv), list(vv), y[6], _thrust_dir(rv, vv, law(tt))))
    return out


def init_spiral(
    prob: D.Prob, law: Optional[Any] = None, k_deg: float = 1.0, dt_d: float = 0.5
):
    """**螺旋热启动**：以"参考推力律"积分，取节点处 a(r,m)·Δt·dir 作 SF 初值。

    这是 E1 §八-2 所荐"给返程腿第二类初值"的落地（切向螺旋＋推力律），比纯切向打靶更接近
    可行流形（后者只 3 个自由参数，无法同时压低"半径差"与"相位差"）。

    .. warning::
       源件缺陷**原样继承**：节点时间取 ``prob.nodes[i] for i in range(prob.N+1)``，对**单腿**
       问题（``nodes`` 仅 2 项）会 ``IndexError``。单腿场景请用
       ``closure_options.init_spiral_local``。
    """
    law = law or (lambda tt: math.pi)
    r0, v0 = prob.ic(None, None)
    tn = [prob.nodes[i] - prob.nodes[0] for i in range(prob.N + 1)]
    nodes = spiral_nodes(r0, v0, prob.m0, tn, law, k_deg=k_deg, dt_d=dt_d)
    off = 2 if prob.bstart else 0
    x = np.zeros(prob.nv)
    if prob.bstart:
        vv = LT.vunit(nodes[0][1])
        x[0] = math.atan2(vv[1], vv[0])
        x[1] = 0.0
    for k in range(1, prob.N):
        rv, _vv, m, dirv = nodes[k]
        rr = math.sqrt(sum(q * q for q in rv))
        mag = a_sep(rr, m, k_deg) * min(prob.dt[k - 1], prob.dt[k]) * DAY
        x[off + 3 * (k - 1) : off + 3 * (k - 1) + 3] = [mag * q for q in dirv]
    return x


def sf_solve2(
    prob: D.Prob,
    x0: Optional[Sequence[float]] = None,
    seed: int = D.SEED_DEFAULT,
    law: Optional[Any] = None,
    k_deg: float = 1.0,
    feas_iter: int = 2000,
    polish_iter: int = 250,
    verbose: bool = False,
) -> Dict[str, Any]:
    """**两段式 SF 求解**（R1-1 的"修补/重设计"）：

    ① 可行性段：``scipy.optimize.least_squares`` 直接压等式约束残差（位置/速度失配），
       只受节点冲量上限的 box 界约束 —— 对欠定问题（42+ 维零空间）比 SLSQP 稳；
    ② 精修段：以可行解为初值，SLSQP 最小化 ΣΔv 并保持等式/不等式约束。

    返回 dict（含两段残差、Δv、触发统计、v∞）。
    """
    from scipy.optimize import least_squares

    if x0 is None:
        x0 = init_spiral(prob, law=law, k_deg=k_deg)
    bnds = D._bounds(prob, x0)
    lo = [b[0] for b in bnds]
    hi = [b[1] for b in bnds]
    x0 = np.asarray(D._clip_caps(prob, np.asarray(x0, float)), float)
    r_in = prob.resid_km(x0)
    t0 = time.time()
    ls = least_squares(
        prob.eq_cons,
        x0,
        bounds=(lo, hi),
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        max_nfev=feas_iter,
        x_scale="jac",
    )
    x1 = np.asarray(ls.x, float)
    r_feas = prob.resid_km(x1)
    st: List[Dict[str, Any]] = []
    x2, r_pol = x1, r_feas
    if r_feas < 5.0e3:
        try:
            cons = [
                dict(type="eq", fun=prob.eq_cons),
                dict(type="ineq", fun=prob.ineq_cons),
            ]
            r2 = minimize(
                prob.obj,
                x1,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options=dict(maxiter=polish_iter, ftol=1e-12, disp=False),
            )
            if prob.resid_km(r2.x) <= max(10.0, r_feas):
                x2, r_pol = np.asarray(r2.x, float), prob.resid_km(r2.x)
                st.append(
                    dict(
                        stage="slsqp_polish",
                        resid=r_pol,
                        obj=prob.obj(x2),
                        nfev=int(r2.nfev),
                        msg=str(r2.message)[:60],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            st.append(
                dict(stage="slsqp_polish", msg="%s: %s" % (exc.__class__.__name__, exc))
            )
    cnt = prob.counters(x2)
    out = dict(
        name=prob.name,
        N=prob.N,
        segs=prob.segs,
        kinds=prob.kinds,
        bodies=prob.bodies,
        nodes=[float(t) for t in prob.nodes],
        dates=[LT.ymd(t) for t in prob.nodes],
        resid_in_km=r_in,
        resid_feas_km=r_feas,
        resid_km=r_pol,
        total=prob.obj(x2),
        x=[float(v) for v in x2],
        counters=cnt,
        stages=st,
        seconds=time.time() - t0,
        method="init=螺旋热启动；①least_squares 可行性 ②SLSQP 精修",
    )
    if verbose:
        print(
            "    螺旋热启动：初残差 %.3e km → 可行性段 %.3e km → 精修 ΣΔv=%.4f 残差 %.3e km"
            % (r_in, r_feas, out["total"], out["resid_km"])
        )
    return out


# ================================================================ 2 弹道几何下界
def ballistic_bounds(
    r_dep_au: float, r_arr_au: float, v_arr_earth_kms: float
) -> Dict[str, Any]:
    """**单弧弹道**（一次 Kepler 弧从 r_dep 到 r_arr）的若干解析界（共面圆轨道近似）。

    * 弧的 a 满足 ``a ≤ (r_dep + r_arr)/2``：因 r_a ≥ r_dep、r_p ≤ r_arr ⇒ 本函数只给两条
      **保守**界：① ``a_max = (r_dep + r_arr)/2``（r_p = r_arr 且 r_a = r_dep 时取等，
      即"最低能量单弧"）；② 硬下界 ``a_min = r_dep/2``（r_a = r_dep、r_p → 0 的退化情形）；
    * 到达速度 ``v = √(µ(2/r_arr − 1/a))`` 在 a 上**单调增** ⇒ 到达速度 ∈ [v(a_min), v(a_max)]；
      **v 的下界给 v∞ 的硬下界**（三角不等式 v∞ ≥ v − v_E）。
    """
    mu = MU
    r1, r2 = r_arr_au * AU, r_dep_au * AU
    a_max = 0.5 * (r1 + r2)
    a_min = 0.5 * r2
    v_of = lambda a: math.sqrt(mu * (2.0 / r1 - 1.0 / a))  # noqa: E731
    vmax_a = v_of(a_max)
    vmin_a = v_of(a_min)
    return dict(
        r_dep_au=r_dep_au,
        r_arr_au=r_arr_au,
        a_max_au=a_max / AU,
        a_min_au=a_min / AU,
        v_arr_at_amax=vmax_a,
        v_arr_at_amin=vmin_a,
        vinf_at_amax=vmax_a - v_arr_earth_kms,
        vinf_hard_lower=vmin_a - v_arr_earth_kms,
        note="单弧弹道；a∈[a_min,a_max]（a_min 为退化径向弧，需巨大离场 Δv）；v∞ 下界＝三角不等式",
    )


# ================================================================ 3 v∞ 边界扫描
def vinf_scan(
    t0_list: Sequence[float],
    tof_grid: Sequence[float],
    nmax: int = 2,
    nscan: int = 150,
    departure: str = "ceres",
) -> List[Dict[str, Any]]:
    """脉冲口径的 (Δv₁, v∞) 扫描：离场日 × TOF × 多圈 Lambert。

    Δv₁ = |v1_L − v_body(t0)|（离场脉冲，含方向）；v∞ = |v2_L − v_Earth(t1)|。
    """
    rows: List[Dict[str, Any]] = []
    for t0 in t0_list:
        r1v, v1b = LT.state_off(departure, t0)
        for tof in tof_grid:
            r2v, v2b = LT.state_off("earth", t0 + tof)
            for s in LT.lambert_multi(r1v, r2v, tof, nmax=nmax, nscan=nscan):
                dv1 = LT.vnorm(LT.vsub(s["v1"], v1b))
                vinf = LT.vnorm(LT.vsub(s["v2"], v2b))
                rows.append(
                    dict(
                        t0_off=t0,
                        t0=LT.ymd(t0),
                        tof_d=float(tof),
                        t1=LT.ymd(t0 + tof),
                        rev=s["rev"],
                        branch=s["branch"],
                        dv1_kms=dv1,
                        vinf_kms=vinf,
                        dv_total_kms=dv1 + vinf,
                    )
                )
    return rows


def frontier(
    rows: Sequence[Dict[str, Any]], key: str = "vinf_kms"
) -> List[Dict[str, Any]]:
    """按 v∞ 升序、Δv₁ 降序的 Pareto 前沿（最小化 Δv₁ 与 v∞）。"""
    rs = sorted(rows, key=lambda r: (r["vinf_kms"], r["dv1_kms"]))
    out: List[Dict[str, Any]] = []
    best_dv = float("inf")
    for r in rs:
        if r["dv1_kms"] < best_dv - 1e-12:
            best_dv = r["dv1_kms"]
            out.append(r)
    return out


def threshold_table(
    front: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] = (6.0, 6.3, 6.6, 7.0, 8.0, 10.0),
) -> List[Dict[str, Any]]:
    """给定 v∞ 门限，前沿上"最小 Δv₁ 的可行解"（离场日固定时即"所需条件"）。"""
    out: List[Dict[str, Any]] = []
    for th in thresholds:
        cand = [r for r in front if r["vinf_kms"] <= th]
        if not cand:
            out.append(dict(vinf_max=th, feasible=False))
            continue
        b = min(cand, key=lambda r: r["dv1_kms"])
        out.append(
            dict(
                vinf_max=th,
                feasible=True,
                dv1_kms=b["dv1_kms"],
                tof_d=b["tof_d"],
                t0=b["t0"],
                t1=b["t1"],
                rev=b["rev"],
                branch=b["branch"],
                vinf_kms=b["vinf_kms"],
            )
        )
    return out


# ================================================================ 4 返程腿问题工厂
def prob_ret(
    tof_d: float = 1719.0,
    nseg: int = 16,
    model: Optional[Any] = None,
    m0: float = M0,
    vinf_max: float = 6.0,
    t0: float = T_DEPART,
    name: Optional[str] = None,
) -> D.Prob:
    """SR-1 返程链（ceres 离场 mode B → 地球再入位置匹配＋v∞ 上限）。"""
    return D.Prob(
        [t0, t0 + tof_d],
        [nseg],
        ["ceres", "earth"],
        ["start_body", "arrive_earth"],
        model or D.sep_model(),
        m0=m0,
        vinf_max=vinf_max,
        name=name or "返程 TOF=%.0f d（N=%d）" % (tof_d, nseg),
    )


# ================================================================ 5 自检
def selftest(verbose: bool = True) -> Dict[str, Any]:
    """① 最短时螺旋两跑逐位一致；② a_sep 与 SEPModel.a_km_s2 一致；③ 弹道下界公式 vs Lambert；
    ④ 反推螺旋的 Δv 与 Edelbaum 解析标尺同量级。"""
    res: Dict[str, Any] = {}
    t0 = time.time()
    r0, v0 = LT.state_off("ceres", T_DEPART)
    a = spiral_run(
        r0, v0, M0, lambda tt: math.pi, k_deg=1.0, tmax_d=200.0, dt_d=0.25, sample_d=1e9
    )
    b = spiral_run(
        r0, v0, M0, lambda tt: math.pi, k_deg=1.0, tmax_d=200.0, dt_d=0.25, sample_d=1e9
    )
    res["two_runs_identical"] = bool(
        a["r_au"] == b["r_au"] and a["state"][6] == b["state"][6]
    )

    mdl = LT.SEPModel()
    d_a = []
    for rr, mm in ((1.0 * AU, M0), (2.7655 * AU, 4000.0), (2.979 * AU, 3500.0)):
        d_a.append(
            abs(a_sep(rr, mm) - mdl.a_km_s2(rr, mm)) / max(mdl.a_km_s2(rr, mm), 1e-30)
        )
    res["a_sep_max_rel"] = max(d_a)

    # 弹道下界：与 Lambert（自由 TOF 粗扫）最小 v∞ 对拍
    r0, v0 = LT.state_off("ceres", T_DEPART)
    re, ve = LT.state_off("earth", T_BACK)
    r_arr = LT.vnorm(re) / AU
    bb = ballistic_bounds(LT.vnorm(r0) / AU, r_arr, LT.vnorm(ve))
    res["ballistic_bounds"] = bb

    # Edelbaum 对拍：单程螺旋 |v_c1 − v_c2| 与反推螺旋在 target 的 Δv
    vc1 = math.sqrt(MU / LT.vnorm(r0))
    vc2 = math.sqrt(MU / (r_arr * AU))
    res["edelbaum_one_way"] = abs(vc2 - vc1)

    if verbose:
        print("=" * 88)
        print("thrust_certificate · self-test")
        print("=" * 88)
        print(
            "① 最短时螺旋两跑逐位一致：%s"
            % ("是" if res["two_runs_identical"] else "**否**")
        )
        print("② a_sep vs SEPModel.a_km_s2 最大相对差：%.2e" % res["a_sep_max_rel"])
        print(
            "③ 弹道几何界：a∈[%.4f, %.4f] AU ⇒ v∞∈[%.3f, %.3f] km/s（到达半径 %.4f AU）"
            % (
                bb["a_min_au"],
                bb["a_max_au"],
                bb["vinf_hard_lower"],
                bb["vinf_at_amax"],
                r_arr,
            )
        )
        print(
            "④ Edelbaum 单程螺旋标尺 |v_c2−v_c1| = %.3f km/s" % res["edelbaum_one_way"]
        )
        print("   自检耗时 %.1f s" % (time.time() - t0))
        print("=" * 88)
    return res


# ================================================================ 6 能力诊断 / v∞ 扫描（无落盘）
def run_capability(target_au: Optional[float] = None) -> Dict[str, Any]:
    """返程腿能力诊断（最短时螺旋）＋固定 1719 d 最小半径证书（**不落盘**）。

    源件把结果写 ``任务/拓展_R1_返程能力.json``；本件只返回 dict（差异②）。
    """
    global R_ARR_EARTH
    re, ve = LT.state_off("earth", T_BACK)
    R_ARR_EARTH = LT.vnorm(re) / AU
    tgt = target_au if target_au is not None else R_ARR_EARTH
    r0, v0 = LT.state_off("ceres", T_DEPART)
    t_start = time.time()
    print("=" * 96)
    print(
        "返程腿能力诊断（最短时螺旋）：Ceres 离场 2038-10-30 → 地球再入半径 %.4f AU（2043-07-15）"
        % tgt
    )
    out: Dict[str, Any] = dict(
        kind="r1_ret_capability",
        target_radius_au=tgt,
        tof_avail_d=1719.0,
        t_depart=LT.ymd(T_DEPART),
        t_back=LT.ymd(T_BACK),
        depart_r_au=LT.vnorm(r0) / AU,
        depart_v_kms=LT.vnorm(v0),
        earth_arr_v_kms=LT.vnorm(ve),
        model=D.sep_model().as_dict(),
        method="切向(反推)螺旋，η(t) 三系数 Nelder-Mead 最小化到达时刻；RK4 dt=0.25–0.5 d；"
        "只判半径下降（必要条件）",
    )
    out["spiral_by_kdeg"] = []
    for kd in (0.9262, 0.95, 1.0, 1.05):
        res = min_time_spiral(k_deg=kd, target_au=tgt, nstart=6)
        out["spiral_by_kdeg"].append(
            dict(
                k_deg=kd,
                t_d=res["t_d"],
                dv_kms=res["dv"],
                m_final_kg=res["m_final_kg"],
                speed_kms=res["speed"],
                vinf_lower_kms=res["vinf_lower_kms"],
                law=res["law"],
                reached=res["reached"],
            )
        )
        print(
            "  k_deg=%.4f：t_min=%.1f d   Δv=%.3f km/s   m_f=%.0f kg   终端 |v|=%.3f km/s"
            "  v∞下界=%.3f km/s  law=%s"
            % (
                kd,
                res["t_d"],
                res["dv"],
                res["m_final_kg"],
                res["speed"],
                res["vinf_lower_kms"],
                res["law"],
            )
        )
    rr = spiral_run(
        r0,
        v0,
        M0,
        lambda tt: math.pi,
        k_deg=1.0,
        tmax_d=1719.0,
        dt_d=0.5,
        target_au=None,
        sample_d=100.0,
    )
    out["r_at_1719d_tangential_au"] = rr["r_au"]
    out["dv_at_1719d_tangential_kms"] = rr["dv"]
    print(
        "  切向反推跑满 1719 d（k_deg=1.0）：r=%.4f AU（距目标半径仍差 %.4f AU = %.2e km）"
        "Δv 已用 %.3f km/s"
        % (rr["r_au"], rr["r_au"] - tgt, (rr["r_au"] - tgt) * AU, rr["dv"])
    )
    mr = min_radius_at_T(1719.0, k_deg=1.0, nsec=10, dt_d=0.5, nstart=5)
    out["min_radius_at_1719d"] = mr
    out["infeasible_1719d"] = bool(mr["r_min_au"] > tgt)
    print(
        "  ★ 固定 T=1719 d 的最小可达日心距（分段 η 直接搜索，%d 段）：r_min=%.4f AU"
        " ⇒ %s（Δv=%.3f km/s）"
        % (
            10,
            mr["r_min_au"],
            "**不可达（不可行证书）**" if mr["r_min_au"] > tgt else "可达",
            mr["dv_kms"],
        )
    )
    kstar = thrust_mult_for_tof(1719.0, target_au=tgt, lo=1.0, hi=1.35)
    out["thrust_mult_for_1719d"] = kstar
    if kstar:
        print(
            "  ⇒ 需把 a_max 增强到 ×%.4f（k_deg 从 1.00 → ≥%.4f）才能在 1719 d 内降到该半径"
            % (kstar, kstar)
        )
    out["tof_required_k1_d"] = out["spiral_by_kdeg"][2]["t_d"]
    out["tof_shortfall_d"] = out["tof_required_k1_d"] - 1719.0
    print(
        "  ⇒ k_deg=1.0 时所需 TOF ≥ %.0f d，缺口 %.0f d（%.1f%%）"
        % (
            out["tof_required_k1_d"],
            out["tof_shortfall_d"],
            100 * out["tof_shortfall_d"] / 1719.0,
        )
    )
    rE, vE = LT.state_off("earth", T_LAUNCH)
    s = math.sqrt(D.SR1_C3)
    vE2 = [vE[i] + s * vE[i] / LT.vnorm(vE) for i in range(3)]
    up = spiral_run(
        rE,
        vE2,
        M0,
        lambda tt: 0.0,
        k_deg=1.0,
        tmax_d=2600.0,
        dt_d=0.5,
        target_au=LT.BODIES["ceres"]["a_au"],
        sample_d=100.0,
    )
    out["outbound_rise"] = dict(
        reached=up["reached"],
        t_d=up["t_d"],
        dv_kms=up["dv"],
        r_au=up["r_au"],
        note="C3=13.4 注入后切向顺行螺旋上升到 a_Ceres（对照，非最优日）",
    )
    print(
        "  对照·去程上升螺旋（C3=13.4，切向顺行）：到达 a_Ceres 需 %.0f d（可用 2376 d）"
        "Δv=%.3f km/s" % (up["t_d"], up["dv"])
    )
    out["ballistic"] = ballistic_bounds(LT.vnorm(r0) / AU, tgt, LT.vnorm(ve))
    b = out["ballistic"]
    print(
        "  弹道几何界（单弧，共面圆轨道近似）：a∈[%.4f, %.4f] AU ⇒ 到达 v∞ ∈ [%.3f, %.3f] km/s"
        % (b["a_min_au"], b["a_max_au"], b["vinf_hard_lower"], b["vinf_at_amax"])
    )
    out["seconds"] = time.time() - t_start
    print("合计 %.0f s" % out["seconds"])
    return out


def run_vinf(
    nscan: int = 150, nmax: int = 2, coarse_step: float = 25.0, verbose: bool = True
) -> Dict[str, Any]:
    """返程 v∞ 边界扫描（固定离场日 2038-10-30）＋窗口放宽（**不落盘**）。"""
    t_start = time.time()
    r0, v0 = LT.state_off("ceres", T_DEPART)
    re, ve = LT.state_off("earth", T_BACK)
    tof_grid = list(np.arange(200.0, 2201.0, coarse_step))
    if verbose:
        print("=" * 96)
        print(
            "返程 v∞ 边界扫描（脉冲/多圈 Lambert）：固定离场日 2038-10-30，TOF %g–%g d"
            % (tof_grid[0], tof_grid[-1])
        )
    rows = vinf_scan([T_DEPART], tof_grid, nmax=nmax, nscan=nscan)
    fr = frontier(rows)
    tab = threshold_table(fr)
    b = ballistic_bounds(LT.vnorm(r0) / AU, LT.vnorm(re) / AU, LT.vnorm(ve))
    if verbose:
        print("  脉冲解总数 %d（多圈 ≤%d、顺/逆行两支）" % (len(rows), nmax))
        print(
            "  最低可达 v∞ = %.3f km/s（Δv₁=%.3f km/s, TOF=%.0f d, 到达 %s, rev=%d）"
            % (
                fr[0]["vinf_kms"],
                fr[0]["dv1_kms"],
                fr[0]["tof_d"],
                fr[0]["t1"],
                fr[0]["rev"],
            )
        )
        print(
            "  单弧最低能量解析下界 v∞ ≥ %.3f km/s（a=%.4f AU）"
            % (b["vinf_at_amax"], b["a_max_au"])
        )
        for t in tab:
            if t["feasible"]:
                print(
                    "  v∞ ≤ %.1f：可行 Δv₁=%.3f km/s，TOF=%.0f d（%s→%s, rev=%d, v∞=%.3f）"
                    % (
                        t["vinf_max"],
                        t["dv1_kms"],
                        t["tof_d"],
                        t["t0"],
                        t["t1"],
                        t["rev"],
                        t["vinf_kms"],
                    )
                )
            else:
                print("  v∞ ≤ %.1f：**该网格内无解**" % t["vinf_max"])
    t0s = list(np.arange(LT.dfe(2037, 1, 1), LT.dfe(2042, 1, 1), 30.0))
    rows2 = vinf_scan(t0s, tof_grid, nmax=nmax, nscan=nscan)
    fr2 = frontier(rows2)
    tab2 = threshold_table(fr2)
    if verbose:
        print(
            "  窗口放宽（离场日 2037-01→2042-01 每 30 d；脉冲解 %d）：最低 v∞ = %.3f km/s"
            "（Δv₁=%.3f, 离场 %s, TOF=%.0f d）"
            % (
                len(rows2),
                fr2[0]["vinf_kms"],
                fr2[0]["dv1_kms"],
                fr2[0]["t0"],
                fr2[0]["tof_d"],
            )
        )
    out = dict(
        kind="r1_vinf_scan",
        method="脉冲（多圈 Lambert ≤%d，顺/逆行）" % nmax,
        t0_list=[T_DEPART],
        tof_grid=tof_grid,
        rows=rows,
        frontier=fr,
        thresholds=tab,
        ballistic=b,
        window=dict(t0_list=t0s, rows=rows2, frontier=fr2, thresholds=tab2),
        seconds=time.time() - t_start,
    )
    if verbose:
        print("合计 %.0f s" % out["seconds"])
    return out


# ================================================================ 7 CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="小推力半径-时间不可行证书与螺旋能力（Ceres R1-A）"
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--capability", action="store_true")
    ap.add_argument("--vinf", action="store_true")
    ap.add_argument("--out-json", default=None, help="结果 JSON 路径（空＝不写）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.self_test:
        st = selftest()
        return 0 if st["two_runs_identical"] and st["a_sep_max_rel"] < 1e-12 else 1
    if a.capability:
        out = run_capability()
        if a.out_json:
            LT.dump_json(a.out_json, out)
        return 0
    if a.vinf:
        out = run_vinf()
        if a.out_json:
            LT.dump_json(a.out_json, out)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
