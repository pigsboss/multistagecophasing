# mission_sim/analysis/interplanetary/low_thrust.py
# -*- coding: utf-8 -*-
"""低推力（SEP）动力学原语：通用变量 Kepler / 多圈 Lambert / 功率-推力模型 / 质量链。

来源：谷神星探测专题 E1（``工具/s4_lt_common.py``）。本模块只搬运其**模型内核**，算法与
口径不变（对拍见 ``tests/test_interplanetary_low_thrust.py``；与岛内机读结果逐位一致）。

口径（与源件一致，不自造新模型）
--------------------------------
* 星历【近似】：二体平根数传播，复用本包 ``two_body.Ephemeris``（与源件
  ``ceres_transfer.Ephemeris`` 逐位一致，见 landing 证据对拍）；不含摄动 / N 体 / 相对论。
  谷神星与候选目标根数取 ``target_scan.TARGETS``（JPL SBDB full-prec 快照，
  epoch = JD 2461200.5）；巡访池修正口径可行件见 ``POOL_TARGETS``。
* 段间传播【算】：通用变量法 Kepler（Stumpff C/S + 阻尼 Newton），标量内环。
* 转移腿【算】：日心二体 Lambert（通用变量、**含多圈**）；无借力 / 无 DSM。
* 推力模型：``P(r) = P0·k_deg / r_AU²``、``T = 2ηP/v_e``、``a = T/m``，质量随 |Δv|
  累积按齐奥尔科夫斯基下降；仅作量级判断，绝对值受 η【假设】约束（见下）。

推力模型参数（沿源件与 ``S4_能源口径统一说明.md`` §3.1）
------------------------------------------------------
* ``P0 = 27 500 W``（515 Table 3-6 BOL @1 AU）；``Isp = 4 000 s``（NEXT 实值）；
  ``η = 0.70``【假设】；``k_deg = 1.00``（标称；真值【待核】）。**无新增重依赖**
  （仅 numpy/scipy 已声明）。

单位：长度 km、时间 s（天数以 ``_d`` 结尾显式区分）、质量 kg；角度对外为 deg，内部 rad。

用法::

  python3 -m mission_sim.analysis.interplanetary.low_thrust --self-test
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mission_sim.analysis.interplanetary import target_scan as ts
from mission_sim.analysis.interplanetary.two_body import (
    AU_KM,
    EARTH_J2000,
    JD_J2000,
    MU_SUN,
    SEC_PER_DAY,
    Ephemeris,
    iso_from_jd,
    jd_from_iso,
    jd_from_ymd,
    lambert as lambert_robust,
)

Vec3 = List[float]

DAY = SEC_PER_DAY
G0 = 9.80665  # m/s^2（标准重力加速度）
JD0 = JD_J2000  # 偏移零点（与源件 ceres_transfer.JD_J2000 一致）

KEPLER_ITERS = 40
KEPLER_TOL = 1e-13
CS_SMALL = 1e-8  # Stumpff 级数切换阈值（源件低推力变体）
CS_HYPER_MAX = 350.0  # 双曲支 s 上限（超此值判溢出 → 返回入参）
LAMBERT_MULTI_TOL = 1e-13
LAMBERT_MULTI_ITMAX = 140


# ============================ 巡访池候选（修正口径） ============================
# 【文据】JPL SBDB full-prec 直连 2026-10-15，epoch = JD 2461200.5（与具名表同历元）。
# 选用依据：专题岛 ``学习/2026-10-11_学习包_r3/30_成文/31_代理精度校核.md`` §五
# （矢量修正代理·相位自由）—— (487) Venetia / (348) May / (579) Sidonia 为该 48 目标
# 抽样的"真解（飞越）"最省者，即"巡访池内可行件"。
POOL_TARGETS: Dict[str, Dict[str, Any]] = {
    "venetia487": {
        "name": "巡访池 (487) Venetia",
        "a_au": 2.670731541585185,
        "e": 0.08659362236484976,
        "i_deg": 10.2535301615586,
        "om_deg": 114.7511402084405,
        "w_deg": 282.0665763195366,
        "ma_deg": 54.8335794091689,
        "n_deg_day": 0.2258180852556619,
        "epoch_jd": 2461200.5,
    },
    "may348": {
        "name": "巡访池 (348) May",
        "a_au": 2.969281545758611,
        "e": 0.07280948191249735,
        "i_deg": 9.758044911624239,
        "om_deg": 89.89538366417813,
        "w_deg": 13.30257242122793,
        "ma_deg": 355.5109531087361,
        "n_deg_day": 0.1926313663806428,
        "epoch_jd": 2461200.5,
    },
    "sidonia579": {
        "name": "巡访池 (579) Sidonia",
        "a_au": 3.012663149322648,
        "e": 0.07597408543388411,
        "i_deg": 11.01635840044058,
        "om_deg": 82.62229813280224,
        "w_deg": 229.2640157186637,
        "ma_deg": 119.0833662743837,
        "n_deg_day": 0.1884856150391535,
        "epoch_jd": 2461200.5,
    },
}


# ============================ 儒略日 ↔ 离 J2000 偏日数 ============================
def dfe(y: int, m: int, d: float) -> float:
    """公历日期 → 相对 J2000 的偏日数 [d]。【算】"""
    return jd_from_ymd(y, m, d) - JD0


def ymd(off: float) -> str:
    """偏日数 → ``'YYYY-MM-DD'``。"""
    return iso_from_jd(JD0 + off)


# ============================ 天体登记表 ============================
def _register_bodies() -> Dict[str, Dict[str, Any]]:
    """天体表：地球（J2000 平均根数）＋具名目标（快照）＋巡访池可行件。"""
    bodies: Dict[str, Dict[str, Any]] = {"earth": dict(EARTH_J2000)}
    for key in ts.DEFAULT_ORDER:
        bodies[key] = ts.spec_of(key)
    for key, spec in POOL_TARGETS.items():
        bodies[key] = dict(spec)
    return bodies


BODIES: Dict[str, Dict[str, Any]] = _register_bodies()
EPH: Dict[str, Ephemeris] = {k: Ephemeris(v) for k, v in BODIES.items()}


def is_body(key: str) -> bool:
    """键是否已在登记表中。"""
    return key in BODIES


def register_body(key: str, spec: Mapping[str, Any]) -> None:
    """登记一个额外天体（供上游扩展候选池；不覆盖已有键的星历复用）。"""
    BODIES[key] = dict(spec)
    EPH[key] = Ephemeris(spec)


def state(key: Any, jd: float) -> Tuple[Vec3, Vec3]:
    """日心 J2000 黄道状态 (r[km], v[km/s])。``key`` 可为键名或根数规格。"""
    eph = EPH[key] if isinstance(key, str) else Ephemeris(key)
    p, v = eph.state(jd)
    return [p[0], p[1], p[2]], [v[0], v[1], v[2]]


def state_off(key: Any, off: float) -> Tuple[Vec3, Vec3]:
    """偏日数（天）口径的 :func:`state`。"""
    return state(key, JD0 + off)


# ============================ 向量小工具（list[3]） ============================
def vsub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vadd(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vmul(a: Sequence[float], s: float) -> Vec3:
    return [a[0] * s, a[1] * s, a[2] * s]


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vnorm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vunit(a: Sequence[float]) -> Vec3:
    n = vnorm(a)
    return [a[0] / n, a[1] / n, a[2] / n] if n > 0 else [0.0, 0.0, 1.0]


def vcross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


# ============================ 通用变量 Kepler（标量） ============================
def stumpff_cs(z: float) -> Optional[Tuple[float, float]]:
    """Stumpff C(z), S(z)（源件低推力变体：级数阈值 ±1e-8；双曲支溢出返回 None）。

    注：与本包 ``two_body.stumpff_cs`` 是**两套等价表述**（阈值 ±1e-6、无 None 分支）；
    低推力线沿源件变体以求逐位可复算。
    """
    if z > CS_SMALL:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z, (s - math.sin(s)) / (s * s * s)
    if z < -CS_SMALL:
        s = math.sqrt(-z)
        if s > CS_HYPER_MAX:
            return None
        return (math.cosh(s) - 1.0) / (-z), (math.sinh(s) - s) / (s * s * s)
    return 0.5 - z / 24.0 + z * z / 720.0, 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def kep(
    r0: Sequence[float],
    v0: Sequence[float],
    dt_days: float,
    mu: float = MU_SUN,
    iters: int = KEPLER_ITERS,
    tol: float = KEPLER_TOL,
) -> Tuple[Vec3, Vec3]:
    """通用变量 Kepler 传播（标量，math 内环）：返回 (r[km], v[km/s])。【算】"""
    x0, y0, z0 = r0
    u0, u1, u2 = v0
    r0n = math.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
    v0n2 = u0 * u0 + u1 * u1 + u2 * u2
    vr0 = (x0 * u0 + y0 * u1 + z0 * u2) / r0n
    dt = dt_days * DAY
    alpha = 2.0 / r0n - v0n2 / mu
    sq = math.sqrt(mu)
    x = sq * dt / r0n if abs(alpha) < 1e-14 else sq * abs(alpha) * dt
    for _ in range(iters):
        cs = stumpff_cs(alpha * x * x)
        if cs is None:
            return list(r0), list(v0)
        c_, s_ = cs
        f = (
            r0n * vr0 / sq * x * x * c_
            + (1.0 - alpha * r0n) * x**3 * s_
            + r0n * x
            - sq * dt
        )
        df = (
            r0n * vr0 / sq * x * (1.0 - alpha * x * x * s_)
            + (1.0 - alpha * r0n) * x * x * c_
            + r0n
        )
        dx = f / df
        lim = 0.5 * max(1.0, abs(x))
        if dx > lim:
            dx = lim
        elif dx < -lim:
            dx = -lim
        x -= dx
        if abs(dx) < tol * max(1.0, abs(x)):
            break
    cs = stumpff_cs(alpha * x * x)
    if cs is None:
        return list(r0), list(v0)
    c_, s_ = cs
    f = 1.0 - x * x / r0n * c_
    g = dt - x**3 / sq * s_
    rx = f * x0 + g * u0
    ry = f * y0 + g * u1
    rz = f * z0 + g * u2
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    fd = sq / (rn * r0n) * (alpha * x**3 * s_ - x)
    gd = 1.0 - x * x / rn * c_
    return [rx, ry, rz], [fd * x0 + gd * u0, fd * y0 + gd * u1, fd * z0 + gd * u2]


def kep_series(
    r0: Sequence[float],
    v0: Sequence[float],
    dt_days_list: Sequence[float],
    dv_nodes: Optional[Sequence[Optional[Sequence[float]]]] = None,
) -> Tuple[List[Vec3], List[Vec3]]:
    """给定 (r0, v0) 与一串段长（天）顺序传播，返回逐节点 (rs, vs)（含起点）。

    ``dv_nodes``：长度 = ``len(dt_days_list) - 1`` 的节点冲量列表（在第 i+1 段**之前**
    施加于节点 i+1），或 None（纯弹道）。
    """
    rs: List[Vec3] = [list(r0)]
    vs: List[Vec3] = [list(v0)]
    r, v = list(r0), list(v0)
    for i, dt in enumerate(dt_days_list):
        if dv_nodes is not None and i > 0 and dv_nodes[i - 1] is not None:
            d = dv_nodes[i - 1]
            v = [v[0] + d[0], v[1] + d[1], v[2] + d[2]]
        r, v = kep(r, v, dt)
        rs.append(r)
        vs.append(v)
    return rs, vs


# ============================ Lambert（自实现 · 高容差 · 含多圈） ============================
def lambert(
    r1v: Sequence[float],
    r2v: Sequence[float],
    tof_days: float,
    prograde: bool = True,
    tol: float = LAMBERT_MULTI_TOL,
    itmax: int = LAMBERT_MULTI_ITMAX,
) -> Optional[Tuple[Vec3, Vec3]]:
    """单圈日心二体 Lambert（通用变量法、z 二分）。返回 (v1, v2) 或 None。【算】

    与 ``two_body.lambert``（tol 1e-8）是**双路独立实现**，互为对拍。
    |sinΔν| ≤ 1e-6 的 180° 退化情形转 ``two_body.lambert`` 的**拱点稳健分支**。
    """
    r1 = vnorm(r1v)
    r2 = vnorm(r2v)
    cr = vcross(r1v, r2v)
    cosd = max(-1.0, min(1.0, vdot(r1v, r2v) / (r1 * r2)))
    dnu = math.acos(cosd)
    if (prograde and cr[2] < 0.0) or ((not prograde) and cr[2] > 0.0):
        dnu = 2.0 * math.pi - dnu
    sd = math.sin(dnu)
    if abs(sd) <= 1e-6:  # 180° 退化 → 用 two_body 的稳健分支
        try:
            v1, v2 = lambert_robust(
                tuple(r1v), tuple(r2v), float(tof_days) * DAY, prograde=prograde
            )
            return [v1[0], v1[1], v1[2]], [v2[0], v2[1], v2[2]]
        except Exception:  # noqa: BLE001
            return None
    a_const = math.sqrt(2.0 * r1 * r2) * math.cos(dnu / 2.0)
    tof_s = float(tof_days) * DAY
    sq = math.sqrt(MU_SUN)

    def t_of(z: float) -> Optional[float]:
        cs = stumpff_cs(z)
        if cs is None:
            return None
        c_, s_ = cs
        if c_ <= 0.0:
            return None
        y = r1 + r2 + a_const * (z * s_ - 1.0) / math.sqrt(c_)
        if y <= 0.0:
            return None
        xx = math.sqrt(y / c_)
        return (xx * xx * xx * s_ + a_const * math.sqrt(y)) / sq

    t0_ = t_of(0.0)
    if t0_ is None:
        return None
    zmax = 4.0 * math.pi * math.pi * (1.0 - 1e-7)
    if tof_s >= t0_:
        lo, hi = 0.0, zmax
        th = t_of(hi)
        if th is not None and th < tof_s:
            return None
    else:
        lo_in, z, step = None, 0.0, 1.0
        while z > -1.0e7:
            zn = z - step
            tn = t_of(zn)
            if tn is None:
                step *= 0.5
                if step < 1e-14:
                    return None
                continue
            if tn <= tof_s:
                lo_in = zn
                break
            z, step = zn, max(step * 1.5, 1.0)
        if lo_in is None:
            return None
        lo, hi = lo_in, z
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        tm = t_of(mid)
        if tm is None:
            lo = mid
            continue
        if tm < tof_s:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) <= tol * max(1.0, abs(mid)):
            break
    z = 0.5 * (lo + hi)
    cs = stumpff_cs(z)
    if cs is None:
        return None
    c_, s_ = cs
    y = r1 + r2 + a_const * (z * s_ - 1.0) / math.sqrt(c_)
    if y <= 0.0:
        return None
    f = 1.0 - y / r1
    g = a_const * math.sqrt(y / MU_SUN)
    gdot = 1.0 - y / r2
    v1 = vmul(vsub(r2v, vmul(r1v, f)), 1.0 / g)
    v2 = vmul(vsub(vmul(r2v, gdot), r1v), 1.0 / g)
    return v1, v2


def lambert2(
    r1v: Sequence[float], r2v: Sequence[float], tof_days: float
) -> List[Dict[str, Any]]:
    """单圈 Lambert 两族（顺/逆行）解列表。"""
    out: List[Dict[str, Any]] = []
    for pro in (True, False):
        s = lambert(r1v, r2v, tof_days, prograde=pro)
        if s is not None:
            out.append(
                {
                    "branch": "pro" if pro else "ret",
                    "rev": 0,
                    "v1": s[0],
                    "v2": s[1],
                    "r1": vnorm(r1v),
                    "r2": vnorm(r2v),
                }
            )
    return out


def lambert_multi(
    r1v: Sequence[float],
    r2v: Sequence[float],
    tof_days: float,
    nmax: int = 3,
    nscan: int = 600,
) -> List[Dict[str, Any]]:
    """**多圈** Lambert：枚举圈数 N=0..nmax × 顺/逆行两支，返回全部满足 TOF 的解。【算】

    做法：在 z 区间 (4π²N², 4π²(N+1)²) 上密集采样 t(z)，找变号区间后二分（与
    :func:`lambert` 同 t(z) 定义）。低推力口径下"固定日期 2 冲量参照"须允许多圈才有意义。
    """
    out: List[Dict[str, Any]] = []
    r1, r2 = vnorm(r1v), vnorm(r2v)
    tof_s = float(tof_days) * DAY
    sq = math.sqrt(MU_SUN)
    for pro in (True, False):
        cr = vcross(r1v, r2v)
        cosd = max(-1.0, min(1.0, vdot(r1v, r2v) / (r1 * r2)))
        dnu = math.acos(cosd)
        if (pro and cr[2] < 0.0) or ((not pro) and cr[2] > 0.0):
            dnu = 2.0 * math.pi - dnu
        if abs(math.sin(dnu)) <= 1e-6:
            continue
        a_const = math.sqrt(2.0 * r1 * r2) * math.cos(dnu / 2.0)

        def t_of(z: float) -> Optional[float]:
            cs = stumpff_cs(z)
            if cs is None:
                return None
            cc, s_ = cs
            if cc <= 0.0:
                return None
            y = r1 + r2 + a_const * (z * s_ - 1.0) / math.sqrt(cc)
            if y <= 0.0:
                return None
            xx = math.sqrt(y / cc)
            return (xx * xx * xx * s_ + a_const * math.sqrt(y)) / sq

        for n_rev in range(0, nmax + 1):
            lo = 4.0 * math.pi * math.pi * n_rev * n_rev + 1e-9
            hi = 4.0 * math.pi * math.pi * (n_rev + 1) ** 2 - 1e-9
            if hi <= lo:
                continue
            zs = np.linspace(lo, hi, nscan)
            ts_arr = np.array(
                [t_of(float(z)) if t_of(float(z)) is not None else np.nan for z in zs]
            )
            for i in range(nscan - 1):
                a_, b_ = ts_arr[i], ts_arr[i + 1]
                if not (np.isfinite(a_) and np.isfinite(b_)):
                    continue
                if (a_ - tof_s) * (b_ - tof_s) > 0:
                    continue
                zlo, zhi = zs[i], zs[i + 1]
                for _ in range(90):
                    mid = 0.5 * (zlo + zhi)
                    tm = t_of(float(mid))
                    if tm is None:
                        zlo = mid
                        continue
                    if tm < tof_s:
                        zlo = mid
                    else:
                        zhi = mid
                z = 0.5 * (zlo + zhi)
                cs = stumpff_cs(z)
                if cs is None:
                    continue
                cc, s_ = cs
                y = r1 + r2 + a_const * (z * s_ - 1.0) / math.sqrt(cc)
                if y <= 0.0:
                    continue
                f = 1.0 - y / r1
                g = a_const * math.sqrt(y / MU_SUN)
                gd = 1.0 - y / r2
                v1 = vmul(vsub(r2v, vmul(r1v, f)), 1.0 / g)
                v2 = vmul(vsub(vmul(r2v, gd), r1v), 1.0 / g)
                out.append(
                    {
                        "branch": "pro" if pro else "ret",
                        "rev": n_rev,
                        "z": float(z),
                        "v1": v1,
                        "v2": v2,
                        "r1": r1,
                        "r2": r2,
                    }
                )
    return out


# ============================ SEP 功率-推力模型 ============================
class SEPModel:
    """SEP 功率-推力模型：``P(r)=P0·k_deg/r_AU²``；``T=2ηP/v_e``；``a=T/m``。

    参数默认：P0＝27 500 W（515 Table 3-6 BOL @1 AU）；Isp＝4 000 s（NEXT 实值）；
    η＝0.70【假设】；k_deg＝1.00（标称；真值【待核】）。
    """

    def __init__(
        self,
        P0_W: float = 27500.0,
        isp_s: float = 4000.0,
        eta: float = 0.70,
        k_deg: float = 1.00,
    ) -> None:
        self.P0 = P0_W
        self.isp = isp_s
        self.eta = eta
        self.k_deg = k_deg
        self.ve = isp_s * G0 / 1000.0  # km/s

    def P_W(self, r_km: float) -> float:
        return self.P0 * self.k_deg / (r_km / AU_KM) ** 2

    def T_N(self, r_km: float) -> float:
        return 2.0 * self.eta * self.P_W(r_km) / (self.ve * 1000.0)

    def a_km_s2(self, r_km: float, m_kg: float) -> float:
        return self.T_N(r_km) / (
            max(m_kg, 1.0) * 1000.0
        )  # m/s² → km/s²（质量下限守卫）

    def mass(self, m0: float, dv_km_s: float) -> float:
        return m0 * math.exp(-dv_km_s / self.ve)

    def as_dict(self) -> Dict[str, float]:
        return {
            "P0_W": self.P0,
            "isp_s": self.isp,
            "eta": self.eta,
            "k_deg": self.k_deg,
            "ve_km_s": self.ve,
        }


class ConstAModel:
    """定加速度模型（与专题岛 C1/E6 LT 工作同口径：a＝常数 1e-7 km/s²）。"""

    def __init__(self, a_km_s2: float = 1e-7) -> None:
        self.a = a_km_s2
        self.ve = 4000.0 * G0 / 1000.0

    def a_km_s2(self, r_km: float, m_kg: float) -> float:
        return self.a

    def P_W(self, r_km: float) -> float:
        return float("nan")

    def T_N(self, r_km: float) -> float:
        return float("nan")

    def mass(self, m0: float, dv_km_s: float) -> float:
        return m0 * math.exp(-dv_km_s / self.ve)

    def as_dict(self) -> Dict[str, Any]:
        return {"a_km_s2": self.a, "kind": "const"}


# ============================ 质量链 ============================
def rocket_prop(
    m0_kg: float, dv_km_s: float, isp_s: float = 4000.0
) -> Tuple[float, float]:
    """齐奥尔科夫斯基：返回 (m_final, 推进剂)。【算】"""
    ve = isp_s * G0 / 1000.0
    mf = m0_kg * math.exp(-dv_km_s / ve)
    return mf, m0_kg - mf


def dv_from_prop(m0_kg: float, prop_kg: float, isp_s: float = 4000.0) -> float:
    """由推进剂质量反解 Δv（km/s）。【算】"""
    ve = isp_s * G0 / 1000.0
    return ve * math.log(m0_kg / (m0_kg - prop_kg))


# ============================ RK4 参考解（独立于通用变量法） ============================
def rk4(
    r0: Sequence[float],
    v0: Sequence[float],
    dt_days: float,
    n: int = 20000,
    mu: float = MU_SUN,
) -> Tuple[Vec3, Vec3]:
    """定步长 RK4 二体递推（参考解，用于传播器校验）。返回 (r, v)。【算】"""
    r = list(r0)
    v = list(v0)
    dt = float(dt_days) * DAY / n

    def acc(rr: Sequence[float]) -> Vec3:
        d3 = (rr[0] ** 2 + rr[1] ** 2 + rr[2] ** 2) ** 1.5
        k = -mu / d3
        return [k * rr[0], k * rr[1], k * rr[2]]

    for _ in range(n):
        k1v = acc(r)
        k1r = v
        k2r = vadd(v, vmul(k1v, 0.5 * dt))
        k2v = acc(vadd(r, vmul(k1r, 0.5 * dt)))
        k3r = vadd(v, vmul(k2v, 0.5 * dt))
        k3v = acc(vadd(r, vmul(k2r, 0.5 * dt)))
        k4r = vadd(v, vmul(k3v, dt))
        k4v = acc(vadd(r, vmul(k3r, dt)))
        r = vadd(
            r,
            vmul(vadd(vadd(k1r, vmul(k2r, 2.0)), vadd(vmul(k3r, 2.0), k4r)), dt / 6.0),
        )
        v = vadd(
            v,
            vmul(vadd(vadd(k1v, vmul(k2v, 2.0)), vadd(vmul(k3v, 2.0), k4v)), dt / 6.0),
        )
    return r, v


def specific_energy(
    r: Sequence[float], v: Sequence[float], mu: float = MU_SUN
) -> float:
    """二体比能 E = v²/2 − μ/r。"""
    return 0.5 * vdot(v, v) - mu / vnorm(r)


# ============================ 落盘 ============================
def dump_json(path: str, obj: Any, indent: int = 1) -> str:
    """把结果写为 JSON（UTF-8；``path`` 原样使用，自动建父目录）。返回绝对路径。"""
    import json
    import os

    p = os.path.abspath(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=float)
    return p


# ============================ 常量对拍 ============================
# 【文据】源件登记值：AU/μ☉ 取专题岛 任务/常量.yaml；地球取 J2000 平均根数；
# 谷神星/候选目标取 JPL SBDB full-prec 快照（epoch JD 2461200.5）。
_REF = {
    "AU_KM": 149597870.7,
    "MU_SUN": 1.32712440018e11,
    "JD_J2000": 2451545.0,
    "SEC_PER_DAY": 86400.0,
    "earth.a_au": 1.00000011,
    "earth.e": 0.01671022,
    "earth.i_deg": 0.00005,
    "earth.om_deg": -11.26064,
    "earth.w_deg": 114.20783,
    "earth.L_deg": 100.46435,
    "earth.n_deg_day": 360.0 / 365.256,
    "ceres.a_au": 2.765552595034094,
    "ceres.e": 0.07969229514816586,
    "ceres.i_deg": 10.58802780183462,
    "ceres.om_deg": 80.24862682043221,
    "ceres.w_deg": 73.29421453021587,
    "ceres.ma_deg": 274.4193463761342,
    "ceres.n_deg_day": 0.21430445064843,
    "ceres.epoch_jd": 2461200.5,
}


def constants_audit() -> List[Dict[str, Any]]:
    """内联常量 vs 源件登记值（``任务/常量.yaml`` + SBDB 快照）的逐项对拍。【算】

    作用：把"落位件与源件同口径"写成可复算断言（源件侧由 landing 对拍另行逐位核）。
    """
    mine = {
        "AU_KM": AU_KM,
        "MU_SUN": MU_SUN,
        "JD_J2000": JD_J2000,
        "SEC_PER_DAY": SEC_PER_DAY,
        "earth.a_au": EARTH_J2000["a_au"],
        "earth.e": EARTH_J2000["e"],
        "earth.i_deg": EARTH_J2000["i_deg"],
        "earth.om_deg": EARTH_J2000["om_deg"],
        "earth.w_deg": EARTH_J2000["w_deg"],
        "earth.L_deg": EARTH_J2000["L_deg"],
        "earth.n_deg_day": EARTH_J2000["n_deg_day"],
        "ceres.a_au": ts.TARGETS["ceres"]["a_au"],
        "ceres.e": ts.TARGETS["ceres"]["e"],
        "ceres.i_deg": ts.TARGETS["ceres"]["i_deg"],
        "ceres.om_deg": ts.TARGETS["ceres"]["om_deg"],
        "ceres.w_deg": ts.TARGETS["ceres"]["w_deg"],
        "ceres.ma_deg": ts.TARGETS["ceres"]["ma_deg"],
        "ceres.n_deg_day": ts.TARGETS["ceres"]["n_deg_day"],
        "ceres.epoch_jd": ts.SBDB_SNAPSHOT_EPOCH,
    }
    return [
        dict(
            item=k,
            mine=mine[k],
            ref=ref,
            rel=(mine[k] - ref) / max(abs(ref), 1e-30),
        )
        for k, ref in _REF.items()
    ]


# ============================ self-test / CLI ============================
def run_self_test() -> int:
    """公共件自检：① 常量对拍；② 传播器 vs RK4＋守恒量；③ Lambert 双路对拍；
    ④ 星历内部一致性（kep 用 μ☉ vs 平根数自身的 n）。【算】

    判据：① 最大相对差 < 1e-6；② |Δr|(vs RK4) < 1e-3 km；③ 端点在 300–2376 d 全闭合。
    """
    print("=" * 88)
    print("low_thrust · self-test（常量 / 传播器 / Lambert 双路 / 星历一致性）")
    print("=" * 88)
    rows = constants_audit()
    rel_max = max(abs(r["rel"]) for r in rows)
    print(
        f"① 内联常量 vs two_body 口径对拍（{len(rows)} 项，最大相对差 {rel_max:.2e}）"
    )

    prop_rows = []
    for key in ("ceres", "earth"):
        for dt in (100.0, 400.0, 1200.0):
            t = dfe(2037, 6, 17)
            r0, v0 = state_off(key, t)
            r1, v1 = kep(r0, v0, dt)
            rk, vk = rk4(r0, v0, dt, n=max(4000, int(dt * 20)))
            prop_rows.append(
                dict(
                    body=key,
                    dt=dt,
                    dr_km=vnorm(vsub(r1, rk)),
                    dv_kms=vnorm(vsub(v1, vk)),
                    energy_rel=(specific_energy(r1, v1) - specific_energy(r0, v0))
                    / abs(specific_energy(r0, v0)),
                    h_rel=(vnorm(vcross(r1, v1)) - vnorm(vcross(r0, v0)))
                    / vnorm(vcross(r0, v0)),
                )
            )
    dr_max = max(p["dr_km"] for p in prop_rows)
    e_max = max(abs(p["energy_rel"]) for p in prop_rows)
    h_max = max(abs(p["h_rel"]) for p in prop_rows)
    print(
        f"② 传播器 vs RK4（独立数值积分）：最大 |Δr| = {dr_max:.3e} km｜"
        f"能量相对漂移 ≤ {e_max:.1e}｜角动量相对漂移 ≤ {h_max:.1e}"
    )

    lam_ok = True
    print("③ Lambert：自实现 vs two_body.lambert（双路）＋端点闭合")
    for tof in (300.0, 800.0, 1500.0, 2376.0):
        r1, _ = state_off("earth", dfe(2030, 12, 15))
        r2, _ = state_off("ceres", dfe(2030, 12, 15) + tof)
        s = lambert(r1, r2, tof)
        if s is None:
            lam_ok = False
            print(f"   TOF={tof:7.0f} d : 无解")
            continue
        try:
            v1c, _ = lambert_robust(tuple(r1), tuple(r2), tof * DAY)
            dv_ref: Optional[float] = vnorm(vsub(s[0], [v1c[0], v1c[1], v1c[2]]))
        except Exception:  # noqa: BLE001
            dv_ref = None
        rr, _ = kep(r1, s[0], tof)
        dr = vnorm(vsub(rr, r2))
        ok = dr < 1.0 and (dv_ref is None or dv_ref < 1e-6)
        lam_ok = lam_ok and ok
        print(
            f"   TOF={tof:7.0f} d : |Δr|={dr:.3e} km｜双路 |Δv1|="
            f"{'n/a' if dv_ref is None else f'{dv_ref:.1e} km/s'}"
        )

    print("④ 星历内部一致性（kep 用 μ☉ vs 平根数自身 n；口径限见源件）")
    for key in ("ceres", "earth"):
        for dt in (400.0, 1200.0):
            t = dfe(2037, 6, 17)
            r0, v0 = state_off(key, t)
            r1, _ = kep(r0, v0, dt)
            r2, _ = state_off(key, t + dt)
            print(f"   {key:<6} dt={dt:6.0f} d : 漂移 {vnorm(vsub(r1, r2)):.3e} km")

    ok_all = rel_max < 1e-6 and dr_max < 1e-3 and lam_ok
    print("-" * 88)
    print(
        "判据：常量逐位一致 & |Δr|(RK4)<1e-3 km & Lambert 端点闭合 ⇒",
        "通过" if ok_all else "**未通过**",
    )
    return 0 if ok_all else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="低推力（SEP）动力学原语 · 自检")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    print(__doc__)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
