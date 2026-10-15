# mission_sim/analysis/interplanetary/flyby_mars.py
# -*- coding: utf-8 -*-
"""去程火星借力链（Ceres R6-1／R6-2 能力沉淀件）。

来源：谷神星探测专题 自由探索 R6（``工具/s4_lt4_r6.py`` v0.1）。算法、常数与报数口径不变
（对拍见 ``tests/test_interplanetary_certificate_closure.py``：与岛内机读结果
``任务/拓展_R6_去程火星借力.json``／``任务/拓展_R6_返程2099复核.json`` 逐位一致）。

本件做两件事
------------
* **R6-1 去程火星借力链**（:func:`opt1`／:func:`run_scan`）：
  ① **脉冲（Lambert）口径的"自由飞掠可行性"扫描**（:func:`ga_scan`）——在飞掠窗
     （003 的 2031-08-13 ±60 d）上构造 ``Earth(t0)→Mars(t1)→Ceres(t2)`` 两弧链，逐对计算
     入射/出射 v∞ 与转角，判定"|v∞_in|≈|v∞_out| 且 转角 ≤ δ_max(r_p)"；
  ② **SEP 小推力链 SF 求解**（:func:`chain_sf`，``E→M→C``，飞掠点＝位置匹配节点）：给 ΣΔv、
     飞掠节点冲量、飞掠前后 v∞、转角与 δ_max；**推进 Δv ＝ ΣΔv − 飞掠节点冲量**（飞掠的
     日心速度改向由火星引力"免费"提供，故该节点冲量计入自由飞掠那一笔）。
* **R6-2 返程 2099.4 d／v∞6.00 复核**（:func:`opt2`）：半径-时间证书
  ``r_min(T=2099.4 d)``（复用 ``thrust_certificate.min_radius_at_T``）＋脉冲 v∞ 扫描＋SF 抽验。

火星星历（**本件新增，源岛内无此件**）
--------------------------------------
源岛内 ``s4_lt_common.BODIES`` **不含 Mars**。本件用 **JPL《Approximate Positions of the
Planets》(Standish) Table 1**（1800–2050 表）的线性根数，在天历起点 **2034-01-01** 处取值，
以本包 ``two_body.Ephemeris``（平根数二体传播器）实例化：
    a,e,I,L,ϖ,Ω ＋ 每世纪变率 da,de,dI,dL,dϖ,dΩ
    Ω=ln；ω=ϖ−Ω；M=L−ϖ；n=(dL−dϖ)/36525 deg·d⁻¹
**口径告示【待核】**：该根数为**外部近似表**（非源岛内登记值），且**未含摄动**；Mars 位置误差
按该表固有精度约 **O(10⁻³–10⁻² AU)**，对本件的飞掠几何判读（v∞ 模长、转角）属可接受量级，
但**不得当精密星历引用**。

与源件差异（仅宿主适配；算法、常数、容差与随机种子**未改**）
------------------------------------------------------------
① 依赖内联：岛内 ``s4_lt_common``／``s4_lt_design``／``s4_lt2_r1``／``s4_lt3_r5``／
   ``ceres_transfer`` → 本仓 ``low_thrust``／``sf_design``／``thrust_certificate``／
   ``closure_options``／``two_body``（前者数值口径已在 93b68ea／d56418b 对拍为逐位一致）。
② :func:`register_mars` 只改内存中的 ``low_thrust.BODIES``／``EPH``（源件同）；本件**不落盘**
   任何岛内 ``任务/*.json``——需要落盘时由调用方用 ``low_thrust.dump_json`` 处理。
③ ``--timeout``／``--quick`` 等源件 CLI 旋钮**未搬**；见 :func:`opt1` 的 ``t2_scan``／
   ``nseg_scan`` 形参（与源件同名同默认）。
④ 未搬运 ``figs``（matplotlib 出图，岛内路径）与 ``--chain``（已独立沉淀为
   ``chain_accounting``）——列为【待裁定】。

用法::

  python3 -m mission_sim.analysis.interplanetary.flyby_mars --opt1   # R6-1（GA 扫描＋SEP SF 链）
  python3 -m mission_sim.analysis.interplanetary.flyby_mars --scan   # 到达日扫描（便宜档）
  python3 -m mission_sim.analysis.interplanetary.flyby_mars --opt2   # R6-2 返程 2099.4 d 复核
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

from mission_sim.analysis.interplanetary import closure_options as CO  # noqa: E402
from mission_sim.analysis.interplanetary import low_thrust as LT  # noqa: E402
from mission_sim.analysis.interplanetary import sf_design as D  # noqa: E402
from mission_sim.analysis.interplanetary import thrust_certificate as TC  # noqa: E402

AU = LT.AU_KM
M0 = D.M0_KG
ISP = D.ISP_S
MU_MARS = 42828.0  # km³/s²（火星 GM；JPL DE440 量级）
R_MARS_KM = 3396.2  # 火星平均半径（IAU；用于飞掠高度）

# SR-1 时间线锚（沿 E1／R1／mission.yaml）
T_LAUNCH = LT.dfe(2030, 12, 15)
T_ARRIVE = LT.dfe(2037, 6, 17)
T_DEPART = LT.dfe(2038, 10, 30)  # Ceres 离场（SR-1）
T_BACK = LT.dfe(2043, 7, 15)  # 地球再入（SR-1）
TGT_AU = LT.vnorm(LT.state_off("earth", T_BACK)[0]) / AU  # ≈1.01650 AU

# 003 Figure 10 图内日期链
F003 = dict(
    dep=("2030-12-20", 4663.5),
    fly=("2031-08-13", 4549.4),
    arr=("2037-04-07", 3913.0),
    dep2=("2038-09-04", 2713.0),
    back=("2044-06-03", 2263.5),
    vinf_mars=4.85,
    alt_mars_km=300.0,
    tof_mars=235.7,
    tof_arr=2300.1,
    stay_d=515.0,
    tof_ret=2099.4,
    vinf_earth=6.00,
)
DV_003_ROUNDTRIP = 14.0
DV_NOFLYBY_KMS = 17.075  # E1 去程 SF 收敛解（无借力）
XE_BUDGET_KG = 1320.0
VE = ISP * LT.G0 / 1000.0


# ================================================================ 火星星历（源岛内无此件）
MARS_JPL_T1 = dict(
    a=1.52371034,
    e=0.09339410,
    i=1.84969142,
    L=-4.55343205,
    lp=-23.94362959,
    ln=49.55953891,
    da=0.00001847,
    de=0.00007882,
    di=-0.00813131,
    dL=19140.30268499,
    dlp=0.44441088,
    dln=-0.29257343,
)


def mars_spec(epoch_jd: float) -> Dict[str, Any]:
    """火星平根数（JPL Standish 近似表，在 epoch_jd 处取值）。返回 Ephemeris 规格 dict。"""
    T = (epoch_jd - 2451545.0) / 36525.0
    a = MARS_JPL_T1["a"] + MARS_JPL_T1["da"] * T
    e = MARS_JPL_T1["e"] + MARS_JPL_T1["de"] * T
    I = MARS_JPL_T1["i"] + MARS_JPL_T1["di"] * T
    L = MARS_JPL_T1["L"] + MARS_JPL_T1["dL"] * T
    lp = MARS_JPL_T1["lp"] + MARS_JPL_T1["dlp"] * T
    ln = MARS_JPL_T1["ln"] + MARS_JPL_T1["dln"] * T
    return dict(
        name="mars(JPL近似根数@%s)" % LT.iso_from_jd(epoch_jd),
        a_au=a,
        e=e,
        i_deg=I,
        om_deg=ln % 360.0,
        w_deg=(lp - ln) % 360.0,
        ma_deg=(L - lp) % 360.0,
        n_deg_day=(MARS_JPL_T1["dL"] - MARS_JPL_T1["dlp"]) / 36525.0,
        epoch_jd=epoch_jd,
    )


def register_mars(epoch_ymd: Sequence[int] = (2034, 1, 1)) -> Dict[str, Any]:
    """把火星注册进 ``LT.BODIES``／``LT.EPH``（只在内存中；不改任何文件）。"""
    ep = LT.dfe(*epoch_ymd)
    spec = mars_spec(LT.JD0 + ep)
    LT.BODIES["mars"] = spec
    LT.EPH["mars"] = LT.Ephemeris(spec)
    return spec


# ================================================================ 飞掠几何工具
def dmax_deg(
    vinf_kms: float, alt_km: float, mu: float = MU_MARS, r_body: float = R_MARS_KM
) -> float:
    """自由飞掠可实现的最大 v∞ 转角（度）：δ_max = 2·asin(1/e)，e = 1 + r_p·v∞²/µ。"""
    e = 1.0 + (r_body + alt_km) * vinf_kms * vinf_kms / mu
    return math.degrees(2.0 * math.asin(min(1.0, 1.0 / e)))


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = LT.vnorm(a), LT.vnorm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, LT.vdot(a, b) / (na * nb)))))


def helio_arc_deg(kind: Sequence[str], t0: float, t1: float) -> float:
    """日心角距（度）：出发点/到达点天体位置矢量夹角（用于"日心弧"账）。"""
    ra, _ = LT.state_off(kind[0], t0)
    rb, _ = LT.state_off(kind[1], t1)
    return angle_deg(ra, rb)


# ================================================================ R6-1 ① GA 可行性扫描（脉冲口径）
def ga_scan(
    t1_grid: Optional[Sequence[float]] = None,
    dt0_grid: Optional[Sequence[float]] = None,
    dt2_grid: Optional[Sequence[float]] = None,
    nmax: int = 1,
    nscan: int = 200,
    vinf_lo: float = 4.35,
    vinf_hi: float = 5.35,
    alt_km: float = 300.0,
    mag_tol: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """脉冲（Lambert）口径的"自由火星飞掠"可行性扫描。

    两弧链：弧1 ``Earth(t0=t1−dt0)→Mars(t1)``；弧2 ``Mars(t1)→Ceres(t2=t1+dt2)``。
    自由飞掠条件：**|v∞_in| = |v∞_out|**（模长守恒）且 **转角 ≤ δ_max(r_p, |v∞|)**。
    本函数逐对枚举，给：配对总数、模长匹配对数、转角可行对数、最小转角、rev0 的最小 |v∞_out|。

    .. note::
       需先 :func:`register_mars`（否则 ``state_off("mars", …)`` 会 KeyError）。
    """
    t_start = time.time()
    if t1_grid is None:
        t1_grid = [LT.dfe(2031, 6, 14) + 4.0 * i for i in range(31)]  # 2031-08-13 ±60 d
    if dt0_grid is None:
        dt0_grid = [120.0 + 10.0 * i for i in range(29)]  # 120…400 d
    if dt2_grid is None:
        dt2_grid = [1400.0 + 25.0 * i for i in range(45)]  # 1400…2500 d
    pairs = 0
    lam = 0
    matched = []
    vout_min: Dict[Tuple[int, str], Tuple[float, str, float]] = {}
    for t1 in t1_grid:
        rM, vM = LT.state_off("mars", t1)
        vin = []
        for dt0 in dt0_grid:
            t0 = t1 - dt0
            if t0 < LT.dfe(2029, 1, 1):
                continue
            rE, vE = LT.state_off("earth", t0)
            for s in LT.lambert_multi(rE, rM, dt0, nmax=nmax, nscan=nscan):
                lam += 1
                vi = LT.vsub(s["v2"], vM)
                vin.append(
                    (
                        t0,
                        LT.vnorm(vi),
                        vi,
                        LT.vnorm(LT.vsub(s["v1"], vE)),
                        s["rev"],
                        s["branch"],
                        dt0,
                    )
                )
        vout = []
        for dt2 in dt2_grid:
            t2 = t1 + dt2
            rC, vC = LT.state_off("ceres", t2)
            for s in LT.lambert_multi(rM, rC, dt2, nmax=nmax, nscan=nscan):
                lam += 1
                vo = LT.vsub(s["v1"], vM)
                vout.append(
                    (
                        t2,
                        LT.vnorm(vo),
                        vo,
                        LT.vnorm(LT.vsub(s["v2"], vC)),
                        s["rev"],
                        s["branch"],
                        dt2,
                    )
                )
                k = (s["rev"], s["branch"])
                if k not in vout_min or LT.vnorm(vo) < vout_min[k][0]:
                    vout_min[k] = (LT.vnorm(vo), LT.ymd(t2), dt2)
        for t0, mi, vi, vd, r1, b1, dt0 in vin:
            if not (vinf_lo <= mi <= vinf_hi):
                continue
            for t2, mo, vo, va, r2, b2, dt2 in vout:
                if not (vinf_lo <= mo <= vinf_hi):
                    continue
                pairs += 1
                th = angle_deg(vi, vo)
                vv = 0.5 * (mi + mo)
                dm = dmax_deg(vv, alt_km)
                matched.append(
                    dict(
                        t0=LT.ymd(t0),
                        t1=LT.ymd(t1),
                        t2=LT.ymd(t2),
                        tof1_d=t1 - t0,
                        tof2_d=t2 - t1,
                        vinf_in=mi,
                        vinf_out=mo,
                        turn_deg=th,
                        dmax_deg=dm,
                        feasible=(abs(mi - mo) <= mag_tol and th <= dm),
                        rev1=r1,
                        rev2=r2,
                        branch1=b1,
                        branch2=b2,
                        dv_dep=vd,
                        dv_arr=va,
                        pulse_total=vd + va,
                    )
                )
    matched.sort(
        key=lambda r: (abs(r["vinf_in"] - r["vinf_out"]) > mag_tol, r["turn_deg"])
    )
    feas = [r for r in matched if r["feasible"]]
    out = dict(
        kind="r6_ga_scan",
        nmax=nmax,
        nscan=nscan,
        alt_km=alt_km,
        vinf_window=[vinf_lo, vinf_hi],
        mag_tol=mag_tol,
        t1_range=[LT.ymd(t1_grid[0]), LT.ymd(t1_grid[-1])],
        n_lambert=lam,
        n_magnitude_matched=len(matched),
        n_feasible=len(feas),
        min_turn=(min(matched, key=lambda r: r["turn_deg"]) if matched else None),
        best_feasible=(feas[0] if feas else None),
        vout_min_by_rev={
            ("rev%d" % k[0]) + k[1]: dict(vinf_kms=v[0], date=v[1], tof_d=v[2])
            for k, v in sorted(vout_min.items())
        },
        sample_top=(feas[:10] if feas else matched[:400]),
        seconds=time.time() - t_start,
    )
    if verbose:
        print(
            "  GA 扫描：Lambert %d 次｜模长匹配对 %d｜**自由飞掠可行对 %d**｜用时 %.1f s"
            % (lam, len(matched), len(feas), out["seconds"])
        )
        if matched:
            m = out["min_turn"]
            print(
                "    最小转角对：%s→%s→%s  v∞ %.2f/%.2f  转角 %.1f°（δ_max %.1f°）"
                % (
                    m["t0"],
                    m["t1"],
                    m["t2"],
                    m["vinf_in"],
                    m["vinf_out"],
                    m["turn_deg"],
                    m["dmax_deg"],
                )
            )
        for k, v in sorted(vout_min.items(), key=lambda q: q[1][0])[:6]:
            print(
                "    Mars→Ceres 最低出射 v∞：rev%d%s = %.2f km/s（%s, TOF %.0f d）"
                % (k[0], k[1], v[0], v[1], v[2])
            )
    return out


# ================================================================ R6-1 ② SEP 链 SF
def init_chain(prob: D.Prob, seed: int = D.SEED_DEFAULT):
    """链式问题的初值：切向打靶（``D.init_tangential``）→ 按 ``D._bounds`` 夹紧（防
    least_squares 报 "Initial guess is outside of provided bounds"——
    ``closure_options.sf_solve2_local`` 对 bstart 问题不做 α/β 夹紧）。"""
    x0, _ = D.init_tangential(prob, seed=seed)
    bb = D._bounds(prob, x0)
    return np.clip(np.asarray(x0, float), [b[0] for b in bb], [b[1] for b in bb])


def chain_sf(
    t0: float,
    t1: float,
    t2: float,
    n1: int = 16,
    n2: int = 16,
    c3: float = D.SR1_C3,
    feas_iter: int = 1200,
    polish_iter: int = 250,
    label: str = "",
) -> Dict[str, Any]:
    """SEP 小推力链 ``Earth(depart_earth) → Mars(flyby) → Ceres(rendezvous)``。

    返回 ΣΔv、飞掠节点冲量（＝自由飞掠那一笔）、飞掠前/后 v∞ 矢量与转角、δ_max、推进 Δv。
    """
    p = D.Prob(
        [t0, t1, t2],
        [n1, n2],
        ["earth", "mars", "ceres"],
        ["depart_earth", "flyby", "rendezvous"],
        D.sep_model(),
        m0=M0,
        c3=c3,
        name="E→M→C %s/%s/%s" % (LT.ymd(t0), LT.ymd(t1), LT.ymd(t2)),
    )
    t_start = time.time()
    x0 = init_chain(p)
    r = CO.sf_solve2_local(p, x0=x0, feas_iter=feas_iter, polish_iter=polish_iter)
    tr = p.propagate(r["x"])
    k = n1
    _, vb = p.body_state[1]
    vpost = tr["vnode"][k]
    dvn = tr["dv"][k - 1]
    vpre = [vpost[i] - dvn[i] for i in range(3)]
    vin = LT.vsub(vpre, vb)
    vout = LT.vsub(vpost, vb)
    imp = LT.vnorm(dvn)
    vi_n, vo_n = LT.vnorm(vin), LT.vnorm(vout)
    turn = angle_deg(vin, vout)
    dm300 = dmax_deg(vi_n, 300.0)
    dm500 = dmax_deg(vi_n, 500.0)
    legs = [float(l["dv"]) for l in r["counters"]["legs"]]
    out = dict(
        label=label,
        t0=LT.ymd(t0),
        t1=LT.ymd(t1),
        t2=LT.ymd(t2),
        t0_off=t0,
        t1_off=t1,
        t2_off=t2,
        tof1_d=t1 - t0,
        tof2_d=t2 - t1,
        tof_tot_d=t2 - t0,
        nseg=[n1, n2],
        nv=p.nv,
        dv_total_kms=r["total"],
        dv_propulsive_kms=r["total"] - imp,
        flyby_impulse_kms=imp,
        legs_dv=legs,
        resid_km=r["resid_km"],
        resid_feas_km=r["resid_feas_km"],
        flyby=dict(
            vinf_in_kms=vi_n,
            vinf_out_kms=vo_n,
            turn_deg=turn,
            dmax_deg_alt300=dm300,
            dmax_deg_alt500=dm500,
            magnitude_ok=bool(abs(vi_n - vo_n) <= 0.5),
            turn_ok=bool(turn <= dm300),
            free_flyby_ok=bool(abs(vi_n - vo_n) <= 0.5 and turn <= dm300),
        ),
        r_miss_flyby_km=LT.vnorm(LT.vsub(tr["rnode"][k], vb)),
        seconds=time.time() - t_start,
    )
    return out


def direct_sf(
    t0: float,
    t1: float,
    nseg: int = 16,
    c3: float = D.SR1_C3,
    maxiter: int = 80,
    multistart: int = 2,
) -> Dict[str, Any]:
    """无借力直迁对照（``D.sf_solve``，E1 口径）。"""
    p = D.prob_sr1_out(nseg=nseg, t0=t0, t1=t1)
    r = D.sf_solve(
        p, maxiter=maxiter, multistart=multistart, seed=D.SEED_DEFAULT, verbose=False
    )
    return dict(
        t0=LT.ymd(t0),
        t1=LT.ymd(t1),
        tof_d=t1 - t0,
        nseg=nseg,
        dv_total_kms=r["total"],
        resid_km=r["resid_km"],
        n_binding=r["counters"]["n_binding"],
        util=r["counters"]["util"],
        seconds=r["seconds"],
    )


# ================================================================ R6-1 主流程
# 选定链路（由 --scan 的到达日扫描 + 003 图内日期链挑出）
FLYBY_CHAIN_RUNS = [
    dict(
        label="003图内日期链（dep 2030-12-20 / fly 2031-08-13 / arr 2037-04-07）",
        t0=(2030, 12, 20),
        t1=(2031, 8, 13),
        t2=(2037, 4, 7),
        n1=16,
        n2=16,
    ),
    dict(
        label="候选A·到达日 2037-03-21（到达日扫描 2036–2038 带内最优）",
        t0=(2030, 12, 16),
        t1=(2031, 8, 13),
        t2=(2037, 3, 21),
        n1=16,
        n2=16,
    ),
    dict(
        label="候选B·到达日 2037-01-20（扫描次优）",
        t0=(2030, 12, 16),
        t1=(2031, 8, 13),
        t2=(2037, 1, 20),
        n1=16,
        n2=16,
    ),
]
# 「同日期直迁」对照的日期（与候选A 同链同总 TOF，用于隔离'借力'这一步的净效应）
DIRECT_SAME_DATES = ((2030, 12, 16), (2037, 3, 21))


def opt1(
    verbose: bool = True, t2_scan: bool = True, nseg_scan: int = 8
) -> Dict[str, Any]:
    """R6-1：GA 可行性扫描 + SEP 链 SF + 到达日扫描 + 无借力对照（**不落盘**）。"""
    t_start = time.time()
    register_mars()
    print("=" * 96)
    print("R6-1 去程火星借力链（Mars 星历＝JPL 近似根数@2034-01-01【待核】）")
    out: Dict[str, Any] = dict(
        kind="r6_opt1_mars_flyby",
        mars_spec=LT.BODIES["mars"],
        f003=F003,
        dv_no_flyby_ref_kms=DV_NOFLYBY_KMS,
        no_flyby_ref_note="E1 轮去程 SF 收敛解（N=16，残差 0.59 km）",
    )
    print("\n① 脉冲口径自由飞掠可行性扫描（两弧链，多圈 ≤1 prograde）")
    out["ga_scan"] = ga_scan()
    print("   （宽窗对照：|v∞| ∈ [0.5, 8.0]、±1.0）")
    out["ga_scan_wide"] = ga_scan(vinf_lo=0.5, vinf_hi=8.0, mag_tol=1.0, verbose=False)
    gw = out["ga_scan_wide"]
    print(
        "    宽窗：匹配对 %d｜可行对 %d｜最小转角 %.1f°（δ_max %.1f°）"
        % (
            gw["n_magnitude_matched"],
            gw["n_feasible"],
            gw["min_turn"]["turn_deg"],
            gw["min_turn"]["dmax_deg"],
        )
    )
    print("\n② 无借力「直迁」对照（E1 口径 SF）")
    try:
        ref = direct_sf(
            LT.dfe(2030, 12, 15),
            LT.dfe(2037, 6, 17),
            nseg=16,
            maxiter=300,
            multistart=3,
        )
        out["direct_sr1_ref"] = ref
        print(
            "    SR-1 直迁 2030-12-15→2037-06-17（%.0f d, maxiter=300）：ΣΔv=%.3f 残差 %.2e km"
            "（E1 参考 17.075，残差 0.59 km）"
            % (ref["tof_d"], ref["dv_total_kms"], ref["resid_km"])
        )
    except Exception as exc:  # noqa: BLE001
        out["direct_sr1_ref"] = dict(error="%s" % exc)
    try:
        d = direct_sf(
            LT.dfe(*DIRECT_SAME_DATES[0]), LT.dfe(*DIRECT_SAME_DATES[1]), nseg=16
        )
        out["direct_same_dates"] = d
        print(
            "    同日期直迁 %s→%s（%.0f d）：ΣΔv=%.3f 残差 %.2e km"
            % (d["t0"], d["t1"], d["tof_d"], d["dv_total_kms"], d["resid_km"])
        )
    except Exception as exc:  # noqa: BLE001
        out["direct_same_dates"] = dict(error="%s" % exc)
    print("\n③ SEP 链 SF（E→M→C，飞掠＝位置匹配节点）")
    out["chains"] = []
    for spec in FLYBY_CHAIN_RUNS:
        print("   ── %s" % spec["label"])
        c = chain_sf(
            LT.dfe(*spec["t0"]),
            LT.dfe(*spec["t1"]),
            LT.dfe(*spec["t2"]),
            n1=spec["n1"],
            n2=spec["n2"],
            label=spec["label"],
        )
        out["chains"].append(c)
        print(
            "      ΣΔv=%.3f（推进 %.3f＋飞掠 %.3f）残差 %.2e km｜飞掠 v∞ %.2f→%.2f 转角 %.1f°"
            "（δ_max %.1f°，%s）"
            % (
                c["dv_total_kms"],
                c["dv_propulsive_kms"],
                c["flyby_impulse_kms"],
                c["resid_km"],
                c["flyby"]["vinf_in_kms"],
                c["flyby"]["vinf_out_kms"],
                c["flyby"]["turn_deg"],
                c["flyby"]["dmax_deg_alt300"],
                "自由飞掠可行" if c["flyby"]["free_flyby_ok"] else "**飞掠几何不可行**",
            )
        )
    if t2_scan:
        print("\n④ 到达日扫描（飞掠 2031-08-13、离地 2030-12-16 固定；N=8+8 便宜档）")
        rows = []
        for i in range(16):
            t2 = LT.dfe(2035, 4, 1) + 60.0 * i
            c = chain_sf(
                LT.dfe(2030, 12, 16),
                LT.dfe(2031, 8, 13),
                t2,
                n1=nseg_scan,
                n2=nseg_scan,
                feas_iter=800,
                polish_iter=150,
                label="arr %s" % LT.ymd(t2),
            )
            rows.append(c)
            print(
                "      arr %s TOF2=%4.0f：ΣΔv=%.3f 推进=%.3f 残差 %.1e｜v∞ %.2f→%.2f 转角 %.1f°"
                % (
                    c["t2"],
                    c["tof2_d"],
                    c["dv_total_kms"],
                    c["dv_propulsive_kms"],
                    c["resid_km"],
                    c["flyby"]["vinf_in_kms"],
                    c["flyby"]["vinf_out_kms"],
                    c["flyby"]["turn_deg"],
                )
            )
        out["arrival_scan"] = rows
    out["seconds"] = time.time() - t_start
    print("\nR6-1 合计 %.0f s" % out["seconds"])
    return out


def run_scan(nseg: int = 8, verbose: bool = True) -> Dict[str, Any]:
    """只跑到达日扫描（供挑链路；便宜档）。"""
    t_start = time.time()
    register_mars()
    rows = []
    for i in range(16):
        t2 = LT.dfe(2035, 4, 1) + 60.0 * i
        c = chain_sf(
            LT.dfe(2030, 12, 16),
            LT.dfe(2031, 8, 13),
            t2,
            n1=nseg,
            n2=nseg,
            feas_iter=800,
            polish_iter=150,
            label="arr %s" % LT.ymd(t2),
        )
        rows.append(c)
        print(
            "arr %s TOF2=%4.0f ΣΔv=%.3f 推进=%.3f 残差 %.1e v∞ %.2f→%.2f 转角 %.1f/%s"
            % (
                c["t2"],
                c["tof2_d"],
                c["dv_total_kms"],
                c["dv_propulsive_kms"],
                c["resid_km"],
                c["flyby"]["vinf_in_kms"],
                c["flyby"]["vinf_out_kms"],
                c["flyby"]["turn_deg"],
                ("%.1f" % c["flyby"]["dmax_deg_alt300"]),
            ),
            flush=True,
        )
    out = dict(
        kind="r6_arrival_scan", nseg=nseg, rows=rows, seconds=time.time() - t_start
    )
    print("合计 %.0f s" % out["seconds"])
    return out


def f003_subset() -> Dict[str, Any]:
    """003 图内日期链的标量摘要（供 :mod:`chain_accounting` 与报告引用）。"""
    return dict(
        get_dep=F003["dep"][0],
        fly=F003["fly"][0],
        arr=F003["arr"][0],
        dep2=F003["dep2"][0],
        back=F003["back"][0],
        vinf_mars=F003["vinf_mars"],
        alt_mars_km=F003["alt_mars_km"],
        tof_ret=F003["tof_ret"],
        vinf_earth=F003["vinf_earth"],
    )


# ================================================================ R6-2 返程复核
def opt2(
    tof: float = 2099.4,
    tofs_sf: Sequence[float] = (2099.4,),
    nsec: int = 10,
    nseg: int = 16,
    tdep: float = T_DEPART,
    scan_tof: Sequence[float] = (1800.0, 2200.0, 25.0),
    t0_alt: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """R6-2：返程 `TOF=2099.4 d、v∞=6.00` 复核（半径证书＋脉冲 v∞＋SF 抽验）。

    ``t0_alt``＝另一起算的离场日（默认同时算 003 图内离场日 2038-09-04），用于"δ日期"对账。
    """
    t_start = time.time()
    register_mars()
    print("=" * 96)
    print(
        "R6-2 返程复核：TOF=%.1f d（出发 %s）、003 图内要求 v∞=6.00 km/s"
        % (tof, LT.ymd(tdep))
    )
    out: Dict[str, Any] = dict(
        kind="r6_opt2_return_2099",
        tof_d=tof,
        depart=LT.ymd(tdep),
        reentry=LT.ymd(tdep + tof),
        target_radius_au=TGT_AU,
        f003=f003_subset(),
        dv_003_roundtrip=DV_003_ROUNDTRIP,
    )
    print("\n① 半径-时间证书（固定 T 内可达到的最小日心距）")
    mr = TC.min_radius_at_T(tof, k_deg=1.0, nsec=nsec, dt_d=0.5, nstart=2)
    closed = bool(mr["r_min_au"] <= TGT_AU)
    out["certificate"] = dict(
        T_d=tof,
        reentry=LT.ymd(tdep + tof),
        r_min_au=mr["r_min_au"],
        margin_au=TGT_AU - mr["r_min_au"],
        target_au=TGT_AU,
        closed=closed,
        dv_kms=mr["dv_kms"],
        speed_kms=mr["speed_kms"],
        vinf_lower_kms=mr["vinf_lower_kms"],
        law=mr["law"],
        nsec=nsec,
        note="r_min 是「最小半径解」（会把半径压过头），非 Δv 最优",
    )
    print(
        "    r_min(T=%.1f d) = %.5f AU（目标 %.5f，余量 %+.5f AU）⇒ %s；该点 Δv=%.3f km/s"
        % (
            tof,
            mr["r_min_au"],
            TGT_AU,
            TGT_AU - mr["r_min_au"],
            "**闭合**" if closed else "**不可达**",
            mr["dv_kms"],
        )
    )
    print("\n② 脉冲 v∞ 扫描（多圈 Lambert ≤3；固定离场日）")
    t0_list = [tdep] + ([t0_alt] if t0_alt else [])
    tof_grid = list(np.arange(scan_tof[0], scan_tof[1] + 1e-9, scan_tof[2]))
    if tof not in tof_grid:
        tof_grid.append(float(tof))
        tof_grid.sort()
    rows = TC.vinf_scan(t0_list, tof_grid, nmax=3, nscan=150)
    near = [r for r in rows if abs(r["tof_d"] - tof) < 1.0]
    fr = TC.frontier(rows)
    out["vinf_scan"] = dict(
        n_rows=len(rows),
        tof_grid=[tof_grid[0], tof_grid[-1]],
        n_tof=len(tof_grid),
        min_vinf_overall=fr[0] if fr else None,
        thresholds=TC.threshold_table(fr, thresholds=(6.0, 6.3, 6.6, 7.0)),
        near_2099=dict(
            n=len(near),
            min_vinf=(min(near, key=lambda r: r["vinf_kms"]) if near else None),
        ),
        departures=[LT.ymd(t) for t in t0_list],
    )
    if near:
        b = min(near, key=lambda r: r["vinf_kms"])
        print(
            "    固定离场日 %s 上 TOF=%.1f d：脉冲解 %d 个，最低 v∞=%.3f km/s（Δv₁=%.3f, rev=%d）"
            % (LT.ymd(tdep), tof, len(near), b["vinf_kms"], b["dv1_kms"], b["rev"])
        )
        print(
            "      ⇒ v∞ ≤ 6.00 %s" % ("可达" if b["vinf_kms"] <= 6.0 else "**不可达**")
        )
    for t in out["vinf_scan"]["thresholds"]:
        print(
            "    全网格 v∞ ≤ %.1f：%s"
            % (
                t["vinf_max"],
                (
                    (
                        "可行 Δv₁=%.3f（%s, TOF %.0f d）"
                        % (t["dv1_kms"], t["t0"], t["tof_d"])
                    )
                    if t["feasible"]
                    else "**无解（不可达证书）**"
                ),
            )
        )
    print("\n③ SF 抽验（返程腿，两段式；N=%d）" % nseg)
    out["sf_checks"] = []
    for tf in tofs_sf:
        p = TC.prob_ret(tof_d=tf, nseg=nseg, vinf_max=10.0, t0=tdep)
        r = CO.sf_solve2_local(p, feas_iter=1500, polish_iter=200)
        enc = r["counters"]["enc"][-1]
        rec = dict(
            tof_d=tf,
            reentry=LT.ymd(tdep + tf),
            resid_km=r["resid_km"],
            resid_feas_km=r["resid_feas_km"],
            dv_total_kms=r["total"],
            vinf_kms=enc["vinf_kms"],
            r_miss_km=enc["r_miss_km"],
            n_binding=r["counters"]["n_binding"],
            util=r["counters"]["util"],
            seconds=r["seconds"],
            vinf_within_6=bool(enc["vinf_kms"] <= 6.0),
        )
        out["sf_checks"].append(rec)
        print(
            "    TOF=%.1f d（再入 %s）：ΣΔv=%.3f km/s，残差 %.2e km，v∞=%.3f km/s"
            "（≤6.0？%s），触界 %d，用时 %.0f s"
            % (
                tf,
                rec["reentry"],
                rec["dv_total_kms"],
                rec["resid_km"],
                rec["vinf_kms"],
                "是" if rec["vinf_within_6"] else "否",
                rec["n_binding"],
                rec["seconds"],
            )
        )
    out["text_vs_figure"] = dict(
        figure_tof_d=2099.4,
        figure_years=2099.4 / 365.25,
        figure_date="2044-06-03",
        text_tof="4.7 yr",
        text_years=4.7,
        text_tof_d=4.7 * 365.25,
        gap_d=2099.4 - 4.7 * 365.25,
        sentence="图版 TOF 2099.4 d（＝5.75 yr）与正文「4.7 yr to return」（＝1717 d）相差 +382 d；"
        "两者不自洽【待核】，但**方向一致**：均远超 SR-1 的 1719 d，"
        "且岛内的半径必要条件闭合阈值 T*≈1768–1784 d 更靠近正文口径的下沿。",
    )
    out["seconds"] = time.time() - t_start
    print("\nR6-2 合计 %.0f s" % out["seconds"])
    return out


# ================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="去程火星借力链（Ceres R6-1／R6-2）")
    ap.add_argument("--opt1", action="store_true", help="R6-1 去程火星借力链")
    ap.add_argument("--scan", action="store_true", help="R6-1 到达日扫描（便宜档）")
    ap.add_argument("--opt2", action="store_true", help="R6-2 返程 2099.4 d 复核")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--nseg", type=int, default=16)
    ap.add_argument("--out-dir", default=None, help="结果 JSON 目录（空＝不写）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    ran = False

    def _dump(name: str, obj: Any) -> None:
        if a.out_dir:
            LT.dump_json(os.path.join(a.out_dir, name), obj)

    if a.scan:
        _dump("拓展_R6_到达日扫描.json", run_scan(nseg=8))
        ran = True
    if a.opt1:
        _dump("拓展_R6_去程火星借力.json", opt1(t2_scan=not a.no_scan))
        ran = True
    if a.opt2:
        _dump(
            "拓展_R6_返程2099复核.json",
            opt2(tof=2099.4, nseg=a.nseg, t0_alt=LT.dfe(2038, 9, 4)),
        )
        ran = True
    if not ran:
        print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
