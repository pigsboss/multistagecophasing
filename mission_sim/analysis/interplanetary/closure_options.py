# mission_sim/analysis/interplanetary/closure_options.py
# -*- coding: utf-8 -*-
"""返程闭合方案矩阵四案（Ceres R5 能力沉淀件）。

来源：谷神星探测专题 自由探索 R5（``工具/s4_lt3_r5.py`` v0.1）。四案的算法、常数与报数口径
不变（对拍见 ``tests/test_interplanetary_certificate_closure.py``：与岛内机读结果
``任务/拓展_R5_*.json`` 逐位一致）。

四案
----
* **① 时间延后**（:func:`opt1`）：返程 TOF 延后下"最小半径-时间"闭合扫描
  （切向反推螺旋 t_min ＋ ``thrust_certificate.min_radius_at_T`` 不可行证书）＋ SF 抽验
  （:func:`sf_solve2_local` 两段式）。
* **② 推力/功率提升**（:func:`opt2`）：k_deg∈{…} 对 t_min 的影响、闭合阈值 k* 与梯度线性度。
* **③ 返程地球借力 EFB**（:func:`opt3_efb`）：脉冲（多圈 Lambert）口径的两弧链
  "Ceres 离场 → 地球飞掠（保 |v∞|、转动受 r_p 双曲界约束）→ 地球再入（v∞≤6）"，
  给 v∞ 前后对照与 Δv 账（含飞掠转角可行界 δ_max）。
* **④ 化学离场脉冲/混合**（:func:`opt4_chem`）：Ceres 离场加单个集中脉冲 Δv_c（≤~1 km/s 级）
  后仍切向反推螺旋 ⇒ 对 148 d 缺口与 v∞ 的影响（**救得了/救不了，如实**）。
* **R5-2 v∞ 前沿**（:func:`run_vinf`）：窗口 × 离场日（≤25 d 步长）× 多圈≤3 的
  min v∞／Δv 扫描，给 ≤6.0／6.3 可达性判定。

与源件差异（仅宿主适配；算法、常数、容差与随机种子**未改**）
------------------------------------------------------------
① 依赖内联：岛内 ``s4_lt_common``／``s4_lt_design``／``s4_lt2_r1`` →
   本仓 ``low_thrust``／``sf_design``／``thrust_certificate``。
② 不再向岛内 ``任务/*.json`` 落盘：``opt1``／``sf_checks`` 只返回结果（源件 ``sf_checks``
   默认 ``merge=True`` 会改写既有 ``拓展_R5_时间延后.json``）；需要落盘时由调用方用
   ``low_thrust.dump_json`` 处理。
③ :func:`init_spiral_local`／:func:`sf_solve2_local` **原样搬运**：这是源件为**修正** R1 工具
   ``init_spiral`` 的"单腿节点索引缺陷"而写的本件版（单腿问题按 ``prob.dt`` 正确构造节点时间）；
   本仓两版并存（另见 ``thrust_certificate.init_spiral`` 的 warning）。
④ 未搬运 ``figs``（matplotlib 出图，岛内路径）——列为【待裁定】。

用法::

  python3 -m mission_sim.analysis.interplanetary.closure_options --opt1
  python3 -m mission_sim.analysis.interplanetary.closure_options --opt2
  python3 -m mission_sim.analysis.interplanetary.closure_options --opt3
  python3 -m mission_sim.analysis.interplanetary.closure_options --opt4
  python3 -m mission_sim.analysis.interplanetary.closure_options --vinf
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from mission_sim.analysis.interplanetary import low_thrust as LT  # noqa: E402
from mission_sim.analysis.interplanetary import sf_design as D  # noqa: E402
from mission_sim.analysis.interplanetary import thrust_certificate as TC  # noqa: E402

AU = LT.AU_KM
MU = LT.MU_SUN
DAY = LT.DAY
M0 = D.M0_KG
ISP = D.ISP_S
MU_EARTH = 398600.4418  # km³/s²（飞掠双曲界用）

T_DEPART = LT.dfe(2038, 10, 30)  # SR-1 离场谷神星
T_BACK = LT.dfe(2043, 7, 15)  # SR-1 再入地球
TGT_AU = LT.vnorm(LT.state_off("earth", T_BACK)[0]) / AU  # 地球再入点半径 ≈1.0165 AU
DATE_2043_12_09 = LT.dfe(2043, 12, 9)

K_REF = 1.0  # 标称 k_deg


# ================================================================ ① 时间延后闭合
def tangential_tmin(
    k_deg: float = 1.0,
    impulse_kms: float = 0.0,
    impulse: str = "retro",
    tmax_d: float = 2400.0,
) -> Dict[str, Any]:
    """切向（反推）螺旋到地球再入半径的**最短时** t_min 与 Δv。

    ``impulse_kms>0`` 时在离场瞬间先加一个集中脉冲（``retro``＝沿瞬时轨道面内逆行切向 −t̂；
    ``pro``＝顺行 +t̂；``rad_in``＝朝日径向），随后仍以 η=π 的逆行切向推力螺旋。
    """
    r0, v0 = LT.state_off("ceres", T_DEPART)
    v0n = list(v0)
    if impulse_kms:
        dirv = TC._thrust_dir(
            r0,
            v0,
            (
                math.pi
                if impulse == "retro"
                else (0.0 if impulse == "pro" else -math.pi / 2.0)
            ),
        )
        v0n = [v0[i] + impulse_kms * dirv[i] for i in range(3)]
    res = TC.spiral_run(
        r0,
        v0n,
        M0,
        lambda tt: math.pi,
        k_deg=k_deg,
        tmax_d=tmax_d,
        dt_d=0.5,
        target_au=TGT_AU,
        sample_d=25.0,
    )
    if res["reached"]:
        t_end = T_DEPART + res["t_d"]
        _, vE = LT.state_off("earth", t_end)
        res["vinf_lower_kms"] = res["speed"] - LT.vnorm(vE)
    res["k_deg"] = k_deg
    res["impulse_kms"] = impulse_kms
    res["arrival"] = LT.ymd(T_DEPART + res["t_d"]) if res["reached"] else None
    return res


# ---- 两段式 SF（本件自建；**修正** R1 工具 ``thrust_certificate.init_spiral`` 的单腿索引缺陷）----
def init_spiral_local(
    prob: D.Prob, law: Optional[Any] = None, k_deg: float = 1.0, dt_d: float = 0.5
):
    """螺旋热启动初值（本件版）。

    R1 ``thrust_certificate.init_spiral`` 用 ``[prob.nodes[i] for i in range(prob.N+1)]``
    取节点时间——对**单腿**问题（nodes 仅 2 个）会 IndexError（该函数从未在单腿问题上成功跑过）。
    本件按 ``prob.dt``（每段时长）正确构造 N+1 个节点时间。
    """
    law = law or (lambda tt: math.pi)
    r0, v0 = prob.ic(None, None)
    cum = [0.0]
    for j, n in enumerate(prob.segs):
        d = (prob.nodes[j + 1] - prob.nodes[j]) / n
        for _ in range(n):
            cum.append(cum[-1] + d)
    nodes = TC.spiral_nodes(r0, v0, prob.m0, cum, law, k_deg=k_deg, dt_d=dt_d)
    off = 2 if prob.bstart else 0
    x = np.zeros(prob.nv)
    if prob.bstart:
        vv = LT.vunit(nodes[0][1])
        x[0] = math.atan2(vv[1], vv[0])
        x[1] = 0.0
    for k in range(1, prob.N):
        rv, _vv, m, dirv = nodes[k]
        rr = math.sqrt(sum(q * q for q in rv))
        mag = TC.a_sep(rr, m, k_deg) * min(prob.dt[k - 1], prob.dt[k]) * DAY
        x[off + 3 * (k - 1) : off + 3 * (k - 1) + 3] = [mag * q for q in dirv]
    return x


def sf_solve2_local(
    prob: D.Prob,
    x0: Optional[Sequence[float]] = None,
    k_deg: float = 1.0,
    feas_iter: int = 2000,
    polish_iter: int = 250,
    verbose: bool = False,
) -> Dict[str, Any]:
    """两段式 SF：① least_squares 压等式残差（可行性）；② SLSQP 精修 ΣΔv。"""
    from scipy.optimize import least_squares, minimize as _min

    if x0 is None:
        x0 = init_spiral_local(prob, k_deg=k_deg)
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
    x2, r_pol = x1, r_feas
    if r_feas < 5.0e3:
        try:
            cons = [
                dict(type="eq", fun=prob.eq_cons),
                dict(type="ineq", fun=prob.ineq_cons),
            ]
            r2 = _min(
                prob.obj,
                x1,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options=dict(maxiter=polish_iter, ftol=1e-12, disp=False),
            )
            if prob.resid_km(r2.x) <= max(10.0, r_feas):
                x2, r_pol = np.asarray(r2.x, float), prob.resid_km(r2.x)
        except Exception:  # noqa: BLE001
            pass
    return dict(
        name=prob.name,
        resid_in_km=r_in,
        resid_feas_km=r_feas,
        resid_km=r_pol,
        total=prob.obj(x2),
        x=[float(v) for v in x2],
        counters=prob.counters(x2),
        seconds=time.time() - t0,
        method="本件 init_spiral_local + least_squares + SLSQP（修正 R1 单腿索引缺陷）",
    )


def opt1(
    ts: Sequence[float] = (1719.0, 1800.0, 1867.0, 1950.0, 2000.0, 2100.0),
    sf_tofs: Sequence[float] = (1867.0, 2000.0),
    sf_nseg: int = 16,
    run_sf: bool = True,
) -> Dict[str, Any]:
    """① 时间延后：闭合扫描＋不可行证书＋SF 抽验（**不落盘**）。"""
    t_start = time.time()
    print("=" * 96)
    print("① 时间延后闭合（返程 TOF 延后；切向反推螺旋＋r_min(T) 证书）")
    out: Dict[str, Any] = dict(
        kind="r5_opt1_time_delay",
        target_radius_au=TGT_AU,
        depart=LT.ymd(T_DEPART),
        date_req_2043_12_09=LT.ymd(DATE_2043_12_09),
        tof_avail_d=1719.0,
    )
    sp = tangential_tmin(k_deg=K_REF, tmax_d=2400.0)
    out["tangential_tmin"] = dict(
        reached=sp["reached"],
        t_min_d=sp["t_d"],
        dv_kms=sp["dv"],
        m_final_kg=sp["state"][6],
        speed_kms=sp["speed"],
        vinf_lower_kms=sp.get("vinf_lower_kms"),
        arrival=sp["arrival"],
    )
    print(
        "  切向反推螺旋 t_min=%.1f d（到达 %s）Δv=%.3f km/s ⇒ 需 TOF ≥ %.0f d（原定 1719 d）"
        % (sp["t_d"], sp["arrival"], sp["dv"], sp["t_d"])
    )
    hist = sp["hist"]
    out["r_curve"] = [dict(t_d=h["t_d"], r_au=h["r_au"]) for h in hist]
    out["min_radius_at_T"] = []
    for T in ts:
        mr = TC.min_radius_at_T(T, k_deg=K_REF, nsec=10, dt_d=0.5, nstart=2)
        closed = bool(mr["r_min_au"] <= TGT_AU)
        out["min_radius_at_T"].append(
            dict(
                T_d=T,
                reentry_date=LT.ymd(T_DEPART + T),
                r_min_au=mr["r_min_au"],
                dv_kms=mr["dv_kms"],
                speed_kms=mr["speed_kms"],
                vinf_lower_kms=mr["vinf_lower_kms"],
                closed=closed,
            )
        )
        print(
            "    r_min(TOF=%.0f d = 再入 %s) = %.5f AU（目标 %.5f）⇒ %s"
            % (
                T,
                LT.ymd(T_DEPART + T),
                mr["r_min_au"],
                TGT_AU,
                "闭合" if closed else "**不可达**",
            )
        )
    curve = sorted(out["min_radius_at_T"], key=lambda r: r["T_d"])
    Tstar = None
    for i in range(1, len(curve)):
        if curve[i - 1]["r_min_au"] > TGT_AU >= curve[i]["r_min_au"]:
            a, b = curve[i - 1], curve[i]
            Tstar = a["T_d"] + (b["T_d"] - a["T_d"]) * (a["r_min_au"] - TGT_AU) / (
                a["r_min_au"] - b["r_min_au"]
            )
    out["T_smallest_closed_d"] = Tstar
    out["reentry_date_smallest_closed"] = LT.ymd(T_DEPART + Tstar) if Tstar else None
    if Tstar:
        print(
            "  ★ 半径必要条件闭合阈值 T* ≈ %.0f d ⇒ 再入日 ≥ %s（原定 2043-07-15，延后 %.0f d）"
            % (Tstar, LT.ymd(T_DEPART + Tstar), Tstar - 1719.0)
        )
    out["sf_checks"] = []
    if run_sf:
        for tof in sf_tofs:
            print(
                "  SF 抽验（两段式 sf_solve2）TOF=%.0f d（再入 %s），N=%d …"
                % (tof, LT.ymd(T_DEPART + tof), sf_nseg)
            )
            p = TC.prob_ret(tof_d=tof, nseg=sf_nseg, vinf_max=10.0)
            r = sf_solve2_local(p, feas_iter=1500, polish_iter=200, verbose=True)
            enc = r["counters"]["enc"][-1]
            out["sf_checks"].append(
                dict(
                    tof_d=tof,
                    reentry=LT.ymd(T_DEPART + tof),
                    resid_km=r["resid_km"],
                    resid_feas_km=r["resid_feas_km"],
                    total_kms=r["total"],
                    vinf_kms=enc["vinf_kms"],
                    r_miss_km=enc["r_miss_km"],
                    n_binding=r["counters"]["n_binding"],
                    util=r["counters"]["util"],
                    seconds=r["seconds"],
                )
            )
            print(
                "     ⇒ ΣΔv=%.4f km/s 残差 %.3e km v∞=%.3f km/s 触界 %d 用时 %.0f s"
                % (
                    r["total"],
                    r["resid_km"],
                    enc["vinf_kms"],
                    r["counters"]["n_binding"],
                    r["seconds"],
                )
            )
    out["seconds"] = time.time() - t_start
    print("合计 %.0f s" % out["seconds"])
    return out


def sf_checks(
    sf_tofs: Sequence[float] = (1784.0, 1867.0, 2000.0),
    sf_nseg: int = 16,
    vinf_max: float = 10.0,
) -> List[Dict[str, Any]]:
    """① 的 SF 抽验（两段式 ``sf_solve2_local``）：返程腿 TOF≥T* 的实跑。

    .. note::
       源件 ``merge=True`` 会把结果并入岛内 ``任务/拓展_R5_时间延后.json``；本件**只返回结果**
       （差异②），由调用方决定是否落盘。
    """
    t_start = time.time()
    print("=" * 96)
    print("① SF 抽验（两段式 sf_solve2；返程腿实跑）")
    res: List[Dict[str, Any]] = []
    for tof in sf_tofs:
        print(
            "  SF: TOF=%.0f d（再入 %s），N=%d …"
            % (tof, LT.ymd(T_DEPART + tof), sf_nseg)
        )
        p = TC.prob_ret(tof_d=tof, nseg=sf_nseg, vinf_max=vinf_max)
        r = sf_solve2_local(p, feas_iter=1500, polish_iter=200, verbose=True)
        enc = r["counters"]["enc"][-1]
        res.append(
            dict(
                tof_d=tof,
                reentry=LT.ymd(T_DEPART + tof),
                resid_km=r["resid_km"],
                resid_feas_km=r["resid_feas_km"],
                resid_in_km=r["resid_in_km"],
                total_kms=r["total"],
                vinf_kms=enc["vinf_kms"],
                r_miss_km=enc["r_miss_km"],
                n_binding=r["counters"]["n_binding"],
                util=r["counters"]["util"],
                seconds=r["seconds"],
            )
        )
        print(
            "     ⇒ ΣΔv=%.4f km/s 残差 %.3e km（可行性段 %.3e）v∞=%.3f km/s 触界 %d 用时 %.0f s"
            % (
                r["total"],
                r["resid_km"],
                r["resid_feas_km"],
                enc["vinf_kms"],
                r["counters"]["n_binding"],
                r["seconds"],
            )
        )
    print("合计 %.0f s" % (time.time() - t_start))
    return res


# ================================================================ ② 推力/功率提升
def opt2(
    kdegs: Sequence[float] = (1.00, 1.02, 1.03, 1.035, 1.04, 1.06, 1.08, 1.10),
    tof_avail: float = 1719.0,
) -> Dict[str, Any]:
    """② k_deg 对 t_min 的影响、闭合阈值与梯度线性度。"""
    t_start = time.time()
    print("=" * 96)
    print("② 推力/功率提升（k_deg 对返程 t_min 与闭合阈值）")
    out: Dict[str, Any] = dict(
        kind="r5_opt2_kdeg", tof_avail_d=tof_avail, target_radius_au=TGT_AU
    )
    rows = []
    for kd in kdegs:
        sp = tangential_tmin(k_deg=kd, tmax_d=2400.0)
        rows.append(
            dict(
                k_deg=kd,
                t_min_d=sp["t_d"],
                dv_kms=sp["dv"],
                m_final_kg=sp["state"][6],
                speed_kms=sp["speed"],
                vinf_lower_kms=sp.get("vinf_lower_kms"),
                reached=sp["reached"],
                arrival=LT.ymd(T_DEPART + sp["t_d"]) if sp["reached"] else None,
            )
        )
        print(
            "   k_deg=%.3f：t_min=%.1f d Δv=%.3f km/s 终端|v|=%.3f ⇒ %s（相对可用 %.0f d %+d d）"
            % (
                kd,
                sp["t_d"],
                sp["dv"],
                sp["speed"],
                "闭合" if sp["t_d"] <= tof_avail else "**不足**",
                tof_avail,
                round(sp["t_d"] - tof_avail),
            )
        )
    out["rows"] = rows
    kstar = None
    rs = sorted(rows, key=lambda r: r["k_deg"])
    for i in range(1, len(rs)):
        if rs[i - 1]["t_min_d"] > tof_avail >= rs[i]["t_min_d"]:
            a, b = rs[i - 1], rs[i]
            kstar = a["k_deg"] + (b["k_deg"] - a["k_deg"]) * (
                a["t_min_d"] - tof_avail
            ) / (a["t_min_d"] - b["t_min_d"])
    out["k_star_for_1719d"] = kstar
    if kstar:
        print(
            "  ★ 闭合 1719 d 需 k_deg ≥ %.4f（即 a_max ×%.4f；按两点线性插值）"
            % (kstar, kstar)
        )
    ks = np.array([r["k_deg"] for r in rs])
    ts_ = np.array([r["t_min_d"] for r in rs])
    A = np.vstack([ks, np.ones_like(ks)]).T
    slope, icept = np.linalg.lstsq(A, ts_, rcond=None)[0]
    pred = slope * ks + icept
    rms = float(np.sqrt(np.mean((pred - ts_) ** 2)))
    out["gradient_d_per_kdeg"] = float(slope)
    out["gradient_lstsq_rms_d"] = rms
    out["gradient_r1_ref_d"] = -2630.0
    out["local_gradients"] = [
        dict(
            seg=[rs[i - 1]["k_deg"], rs[i]["k_deg"]],
            dtdk=(rs[i]["t_min_d"] - rs[i - 1]["t_min_d"])
            / (rs[i]["k_deg"] - rs[i - 1]["k_deg"]),
        )
        for i in range(1, len(rs))
    ]
    print(
        "  ∂t_min/∂k_deg（最小二乘）= %.0f d（R1 参考 −2630 d）；拟合 RMS=%.1f d ⇒ %s"
        % (slope, rms, "近线性" if rms < 25 else "**非线性**")
    )
    for lg in out["local_gradients"]:
        print(
            "     局部 [%.2f,%.2f]：∂t_min/∂k_deg = %.0f d" % (*lg["seg"], lg["dtdk"])
        )
    out["seconds"] = time.time() - t_start
    print("合计 %.0f s" % out["seconds"])
    return out


# ================================================================ ③ 返程地球借力（EFB）
def flyby_dvmax(vinf_kms: float, rp_km: float, mu: float = MU_EARTH) -> float:
    """飞掠可实现的 v∞ 矢量最大转角 δ_max = 2·asin(1/e)，e = 1 + r_p·v∞²/µ。"""
    e = 1.0 + rp_km * vinf_kms * vinf_kms / mu
    if e <= 1.0:
        return math.pi
    return 2.0 * math.asin(min(1.0, 1.0 / e))


def _angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = LT.vnorm(a), LT.vnorm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, LT.vdot(a, b) / (na * nb)))
    return math.acos(c)


def opt3_efb(
    t1_grid: Optional[Sequence[float]] = None,
    tof2_grid: Optional[Sequence[float]] = None,
    nmax: int = 3,
    nscan: int = 120,
    rp_alt_km: float = 200.0,
    vinf2_max: float = 6.0,
    match_tol: float = 0.05,
    top_t1: int = 20,
    t0_step: float = 15.0,
) -> Dict[str, Any]:
    """③ 返程地球借力 EFB：脉冲（多圈 Lambert）口径两弧链。

    弧1：Ceres(t0) → 地球(t1)（多圈**弹道**，离场脉冲 Δv₁）；弧2：地球(t1) → 地球(t2)（多圈弹道）。
    飞掠在 t1 处**保 |v∞|**（无推力；转动角须 ≤ δ_max = 2·asin(1/e)，e = 1 + r_p·v∞²/µ_E）。
    产出：① **自由飞掠可达性判定**（弧1 到达 v∞ 分布 vs 弧2 所需 v∞ 分布 ⇒ 匹配证书）；
    ② 动力飞掠（允许 t1 处 Δv，非自由）的 Δv 账。给 v∞ 前后对照（v∞_in 飞掠前 → v∞_2 再入）。
    """
    t_start = time.time()
    print("=" * 96)
    print("③ 返程地球借力（EFB）：Ceres 离场 → 地球飞掠 → 地球再入（多圈弹道）")
    rp_km = 6378.137 + rp_alt_km
    if t1_grid is None:
        t1_grid = np.arange(LT.dfe(2039, 1, 1), LT.dfe(2046, 1, 1), 30.0)
    if tof2_grid is None:
        tof2_grid = np.arange(250.0, 1550.0, 25.0)
    arc2 = []
    for t1 in t1_grid:
        r1, vE1 = LT.state_off("earth", t1)
        for tof2 in tof2_grid:
            t2 = t1 + tof2
            r2, vE2 = LT.state_off("earth", t2)
            for s in LT.lambert_multi(r1, r2, tof2, nmax=nmax, nscan=nscan):
                vinf2 = LT.vnorm(LT.vsub(s["v2"], vE2))
                if vinf2 > vinf2_max:
                    continue
                vinf_out = LT.vsub(s["v1"], vE1)  # 飞掠后 v∞ 矢量（进入弧2）
                arc2.append(
                    dict(
                        t1=float(t1),
                        t2=float(t2),
                        tof2_d=float(tof2),
                        rev=s["rev"],
                        branch=s["branch"],
                        vinf2_kms=vinf2,
                        m_need_kms=LT.vnorm(vinf_out),
                        v1=s["v1"],
                        vinf_out=vinf_out,
                    )
                )
    print(
        "  弧2（地球→地球，v∞≤%.1f）解数：%d（t1 覆盖 %d 档）"
        % (vinf2_max, len(arc2), len(t1_grid))
    )
    mneed_max = max((a["m_need_kms"] for a in arc2), default=0.0)
    out: Dict[str, Any] = dict(
        kind="r5_opt3_efb",
        vinf2_max=vinf2_max,
        nmax=nmax,
        nscan=nscan,
        rp_alt_km=rp_alt_km,
        rp_km=rp_km,
        arc2_count=len(arc2),
        n_t1=len(t1_grid),
        arc2_mneed_max_kms=mneed_max,
        note="弧2＝地球→地球弹道，其到达 v∞≤%.1f 对应**离场 v∞** 最大 %.3f km/s"
        "（＝飞掠前 |v∞_in| 必须等于的值）" % (vinf2_max, mneed_max),
    )
    if not arc2:
        out["status"] = "no_arc2"
        print(
            "  ★ 无 v∞≤%.1f 的地球→地球弹道解 ⇒ EFB 该网格**不可行（不可行证书）**"
            % vinf2_max
        )
        out["seconds"] = time.time() - t_start
        return out
    arc2_by_t1: Dict[float, List[Dict[str, Any]]] = {}
    for a in arc2:
        arc2_by_t1.setdefault(a["t1"], []).append(a)
    cand = sorted(
        (min(v, key=lambda a: a["vinf2_kms"]) for v in arc2_by_t1.values()),
        key=lambda a: a["vinf2_kms"],
    )[:top_t1]
    arc1_by_t1: Dict[float, List[Dict[str, Any]]] = {}
    vE1_by_t1: Dict[float, List[float]] = {}
    vinf_in_min = 1e9
    for a2 in cand:
        t1 = a2["t1"]
        rE, vE1 = LT.state_off("earth", t1)
        vE1_by_t1[t1] = vE1
        lst = []
        for t0 in np.arange(t1 - 1500.0, t1 - 180.0, t0_step):
            tof1 = t1 - t0
            rC, vC = LT.state_off("ceres", t0)
            for s in LT.lambert_multi(rC, rE, tof1, nmax=nmax, nscan=nscan):
                lst.append(
                    dict(
                        t0=float(t0),
                        tof1_d=float(tof1),
                        rev=s["rev"],
                        dv1_kms=LT.vnorm(LT.vsub(s["v1"], vC)),
                        v2=s["v2"],
                    )
                )
                vinf_in_min = min(vinf_in_min, LT.vnorm(LT.vsub(s["v2"], vE1)))
        arc1_by_t1[t1] = lst
    out["arc1_vinf_in_min_kms"] = vinf_in_min
    matches = []
    for a2 in cand:
        t1 = a2["t1"]
        m_need, vinf_out, vE1 = a2["m_need_kms"], a2["vinf_out"], vE1_by_t1[t1]
        dmax = flyby_dvmax(m_need, rp_km)
        for a1 in arc1_by_t1[t1]:
            vinf_in = LT.vsub(a1["v2"], vE1)
            if abs(LT.vnorm(vinf_in) - m_need) > match_tol:
                continue
            turn = _angle_between(vinf_in, vinf_out)
            if turn > dmax:
                continue
            matches.append(
                dict(
                    t0_date=LT.ymd(a1["t0"]),
                    flyby_date=LT.ymd(t1),
                    reentry_date=LT.ymd(a2["t2"]),
                    tof1_d=a1["tof1_d"],
                    tof2_d=a2["tof2_d"],
                    dv1_kms=a1["dv1_kms"],
                    vinf_in_kms=LT.vnorm(vinf_in),
                    vinf2_kms=a2["vinf2_kms"],
                    turn_deg=math.degrees(turn),
                    turn_max_deg=math.degrees(dmax),
                    rev1=a1["rev"],
                    rev2=a2["rev"],
                )
            )
    matches.sort(key=lambda b: b["dv1_kms"])
    out["n_match_free"] = len(matches)
    out["matches_free"] = matches[:10]
    powered = None
    for a2 in cand:
        t1 = a2["t1"]
        for a1 in arc1_by_t1[t1]:
            dsw = LT.vnorm(LT.vsub(a2["v1"], a1["v2"]))
            tot = a1["dv1_kms"] + dsw
            if powered is None or tot < powered["dv_total_kms"]:
                powered = dict(
                    t0_date=LT.ymd(a1["t0"]),
                    flyby_date=LT.ymd(t1),
                    reentry_date=LT.ymd(a2["t2"]),
                    tof1_d=a1["tof1_d"],
                    tof2_d=a2["tof2_d"],
                    dv1_kms=a1["dv1_kms"],
                    dv_switch_kms=dsw,
                    dv_total_kms=tot,
                    vinf_in_kms=LT.vnorm(LT.vsub(a1["v2"], vE1_by_t1[t1])),
                    vinf2_kms=a2["vinf2_kms"],
                    rev1=a1["rev"],
                    rev2=a2["rev"],
                )
    out["powered_best"] = powered
    out["status"] = "free_infeasible" if not matches else "ok"
    print(
        "  弧1 到达 v∞ 最小 = %.3f km/s；弧2 所需（离场）v∞ 最大 = %.3f km/s"
        % (vinf_in_min, mneed_max)
    )
    if not matches:
        print(
            "  ★ **自由飞掠不可行证书**：弧2 到达 v∞≤%.1f 要求其离场 v∞ ≤ %.3f km/s；而弧1"
            "（Ceres→地球）到达 v∞ ≥ %.3f km/s ⇒ 两区间**无交集**（飞掠保 |v∞|，无法把 v∞ 降下来）"
            % (vinf2_max, mneed_max, vinf_in_min)
        )
    else:
        b = matches[0]
        print(
            "  ★ 自由 EFB 最优：离场 %s（Δv₁=%.3f）→ 飞掠 %s（v∞=%.3f）→ 再入 %s（v∞=%.3f）"
            % (
                b["t0_date"],
                b["dv1_kms"],
                b["flyby_date"],
                b["vinf_in_kms"],
                b["reentry_date"],
                b["vinf2_kms"],
            )
        )
    if powered:
        p = powered
        print(
            "    动力飞掠变体（t1 处 Δv，非自由）：最小 ΣΔv=%.3f km/s（Δv₁=%.3f＋Δv_飞掠=%.3f），"
            "离场 %s 飞掠 %s 再入 %s（v∞_2=%.3f）"
            % (
                p["dv_total_kms"],
                p["dv1_kms"],
                p["dv_switch_kms"],
                p["t0_date"],
                p["flyby_date"],
                p["reentry_date"],
                p["vinf2_kms"],
            )
        )
    out["seconds"] = time.time() - t_start
    print("合计 %.0f s" % out["seconds"])
    return out


# ================================================================ ④ 化学离场脉冲
def opt4_chem(
    dvs: Sequence[float] = (0.0, 0.30, 0.37, 0.50, 0.75, 1.00, 1.50, 2.00),
    isp_mono: float = 230.0,
    isp_bipro: float = 320.0,
) -> Dict[str, Any]:
    """④ 化学离场脉冲（混合）：单个集中脉冲 Δv_c 后仍切向反推螺旋。

    评价：对 148 d 缺口与 v∞ 的影响（**救得了/救不了，如实**）；并给脉冲推进剂账
    （单组元 N₂H₄ Isp≈230 s／双组元 Isp≈320 s；【待核】——岛内 S10 记 370 m/s 肼）。
    """
    t_start = time.time()
    print("=" * 96)
    print("④ 化学离场脉冲/混合（Δv_c 集中脉冲 + 切向反推螺旋）")
    out: Dict[str, Any] = dict(
        kind="r5_opt4_chem",
        tof_avail_d=1719.0,
        target_radius_au=TGT_AU,
        gap_to_close_d=148.0,
        isp_mono_s=isp_mono,
        isp_bipro_s=isp_bipro,
    )
    rows = []
    for dvc in dvs:
        sp = tangential_tmin(
            k_deg=K_REF, impulse_kms=dvc, impulse="retro", tmax_d=2400.0
        )
        # v_e = g0·Isp/1000 (km/s)；推进剂以"到达离场脉冲时的整器质量" m0 计
        ve_m = isp_mono * LT.G0 / 1000.0
        ve_b = isp_bipro * LT.G0 / 1000.0
        prop_m = M0 * (1.0 - math.exp(-dvc / ve_m)) if dvc > 0 else 0.0
        prop_b = M0 * (1.0 - math.exp(-dvc / ve_b)) if dvc > 0 else 0.0
        rows.append(
            dict(
                dv_chem_kms=dvc,
                t_min_d=sp["t_d"],
                dv_sep_kms=sp["dv"],
                m_final_kg=sp["state"][6],
                speed_kms=sp["speed"],
                vinf_lower_kms=sp.get("vinf_lower_kms"),
                reached=sp["reached"],
                arrival=LT.ymd(T_DEPART + sp["t_d"]) if sp["reached"] else None,
                prop_mono_kg=prop_m,
                prop_bipro_kg=prop_b,
                closed=bool(sp["reached"] and sp["t_d"] <= 1719.0),
            )
        )
        print(
            "   Δv_c=%.2f km/s：t_min=%.1f d（Δ vs 无脉冲 %+.0f d）Δv_SEP=%.3f km/s 终端|v|=%.3f"
            " 推进剂(肼)=%.0f kg ⇒ %s"
            % (
                dvc,
                sp["t_d"],
                sp["t_d"] - rows[0]["t_min_d"],
                sp["dv"],
                sp["speed"],
                prop_m,
                "闭合 1719 d" if rows[-1]["closed"] else "不足",
            )
        )
    out["rows"] = rows
    if len(rows) >= 3 and rows[1]["dv_chem_kms"] > 0:
        gains = [
            (rows[i - 1]["t_min_d"] - rows[i]["t_min_d"])
            / (rows[i]["dv_chem_kms"] - rows[i - 1]["dv_chem_kms"])
            for i in range(1, len(rows))
        ]
        out["dt_per_dvchem_d_per_kms"] = gains
        print(
            "   时间收益 ∂t_min/∂Δv_c（d per km/s）：%s"
            % " ".join("%.0f" % g for g in gains)
        )
    need = None
    for i in range(1, len(rows)):
        if rows[i - 1]["t_min_d"] > 1719.0 >= rows[i]["t_min_d"]:
            a, b = rows[i - 1], rows[i]
            need = a["dv_chem_kms"] + (b["dv_chem_kms"] - a["dv_chem_kms"]) * (
                a["t_min_d"] - 1719.0
            ) / (a["t_min_d"] - b["t_min_d"])
    out["dv_chem_for_1719d_kms"] = need
    if need:
        ve_m = isp_mono * LT.G0 / 1000.0
        print(
            "   ★ 若单靠化学离场脉冲闭合 1719 d：需 Δv_c ≈ %.3f km/s（肼推进剂 ≈%.0f kg）"
            % (need, M0 * (1 - math.exp(-need / ve_m)))
        )
    out["seconds"] = time.time() - t_start
    print("合计 %.0f s" % out["seconds"])
    return out


# ================================================================ R5-2 v∞ 前沿细化
def run_vinf(
    t0_step: float = 25.0,
    tof_step: float = 25.0,
    nmax: int = 3,
    nscan: int = 150,
    verbose: bool = True,
) -> Dict[str, Any]:
    """R5-2：窗口×离场日（≤25 d 步长）×多圈≤3 的 v∞/Δv 前沿。"""
    t_start = time.time()
    if verbose:
        print("=" * 96)
        print(
            "R5-2 v∞ 前沿细化（窗口 2037-01→2043-01 × 离场日步长 %.0f d × 多圈≤%d）"
            % (t0_step, nmax)
        )
    t0_list = list(np.arange(LT.dfe(2037, 1, 1), LT.dfe(2043, 1, 1) + 1e-9, t0_step))
    tof_grid = list(np.arange(200.0, 2201.0, tof_step))
    rows = TC.vinf_scan(t0_list, tof_grid, nmax=nmax, nscan=nscan)
    if verbose:
        print(
            "  脉冲解总数 %d（离场日 %d 档 × TOF %d 档 × ≤%d 圈 × 双支）"
            % (len(rows), len(t0_list), len(tof_grid), nmax + 1)
        )
    fr = TC.frontier(rows)
    tab = TC.threshold_table(fr, thresholds=(6.0, 6.3, 6.6, 7.0, 8.0, 9.0, 10.0))
    r0, v0 = LT.state_off("ceres", T_DEPART)
    re, ve = LT.state_off("earth", T_BACK)
    b = TC.ballistic_bounds(LT.vnorm(r0) / AU, LT.vnorm(re) / AU, LT.vnorm(ve))
    if verbose:
        print(
            "  最低可达 v∞ = %.3f km/s（Δv₁=%.3f, 离场 %s, TOF=%.0f d, 到达 %s, rev=%d）"
            % (
                fr[0]["vinf_kms"],
                fr[0]["dv1_kms"],
                fr[0]["t0"],
                fr[0]["tof_d"],
                fr[0]["t1"],
                fr[0]["rev"],
            )
        )
        print(
            "  单弧弹道几何下界：v∞ ≥ %.3f（硬）／最低能量单弧 %.3f km/s"
            % (b["vinf_hard_lower"], b["vinf_at_amax"])
        )
        for t in tab:
            if t["feasible"]:
                print(
                    "  v∞ ≤ %.1f：可行 Δv₁=%.3f km/s，离场 %s，TOF=%.0f d，rev=%d，v∞=%.3f"
                    % (
                        t["vinf_max"],
                        t["dv1_kms"],
                        t["t0"],
                        t["tof_d"],
                        t["rev"],
                        t["vinf_kms"],
                    )
                )
            else:
                print("  v∞ ≤ %.1f：**该网格内无解（不可达证书）**" % t["vinf_max"])
    keep = [r for r in rows if r["vinf_kms"] <= 9.0]
    out = dict(
        kind="r5_vinf_frontier",
        method="脉冲（多圈 Lambert ≤%d，顺/逆行）" % nmax,
        t0_step_d=t0_step,
        tof_step_d=tof_step,
        n_t0=len(t0_list),
        n_tof=len(tof_grid),
        n_rows=len(rows),
        n_rows_kept=len(keep),
        t0_range=[LT.ymd(t0_list[0]), LT.ymd(t0_list[-1])],
        tof_range=[tof_grid[0], tof_grid[-1]],
        min_vinf=fr[0],
        frontier=fr[:40],
        thresholds=tab,
        ballistic=b,
        rows_le9=keep,
        seconds=time.time() - t_start,
    )
    if verbose:
        print("合计 %.0f s" % out["seconds"])
    return out


# ================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="返程闭合方案矩阵四案（Ceres R5）")
    ap.add_argument("--opt1", action="store_true", help="时间延后闭合")
    ap.add_argument("--opt2", action="store_true", help="推力/功率提升")
    ap.add_argument("--opt3", action="store_true", help="返程地球借力 EFB")
    ap.add_argument("--opt4", action="store_true", help="化学离场脉冲/混合")
    ap.add_argument("--vinf", action="store_true", help="v∞ 前沿细化")
    ap.add_argument("--sfonly", action="store_true", help="只跑① 的 SF 抽验")
    ap.add_argument("--no-sf", action="store_true", help="--opt1 跳过 SF 抽验")
    ap.add_argument("--out-dir", default=None, help="结果 JSON 目录（空＝不写）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    ran = False

    def _dump(name: str, obj: Any) -> None:
        if a.out_dir:
            LT.dump_json(os.path.join(a.out_dir, name), obj)

    if a.sfonly:
        _dump("拓展_R5_时间延后_sf.json", sf_checks())
        ran = True
    if a.opt1:
        _dump("拓展_R5_时间延后.json", opt1(run_sf=not a.no_sf))
        ran = True
    if a.opt2:
        _dump("拓展_R5_kdeg提升.json", opt2())
        ran = True
    if a.opt3:
        _dump("拓展_R5_地球借力EFB.json", opt3_efb())
        ran = True
    if a.opt4:
        _dump("拓展_R5_化学离场脉冲.json", opt4_chem())
        ran = True
    if a.vinf:
        _dump("拓展_R5_vinf前沿.json", run_vinf())
        ran = True
    if not ran:
        print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
