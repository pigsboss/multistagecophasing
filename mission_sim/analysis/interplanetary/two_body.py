# mission_sim/analysis/interplanetary/two_body.py
# -*- coding: utf-8 -*-
"""日心二体星历与 Lambert 转移内核（Ceres S4 能力沉淀件）。

来源：谷神星探测专题 S4/CML5 仿真线的 ``工具/ceres_transfer.py``（二体平根数星历 +
日心二体 Lambert 通用变量法）与 ``工具/s4_target_scan.py``（多目标窗口扫描）。本模块
只搬运两者共用的**模型内核**，算法与判据不变（对拍见 tests/test_interplanetary_window_scan.py）。

口径（与源件一致，不自造新模型）
--------------------------------
* 星历【近似】：二体平根数传播，不含摄动 / N 体 / 相对论。
* 转移腿【算】：日心二体 Lambert（通用变量、单圈、脉冲）；无借力 / 无 DSM / 无有限推力。
* 该口径用于"档次与窗口"的量级判断，**不用于制导**，与含借力的文献口径不可直接对标。

单位：长度 km、时间 s（天数显式以 ``_d`` 结尾）、角度对外为 deg、内部为 rad。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# ============================ 参数集中区（便于改） ============================
AU_KM = 149597870.7  # km（IAU 2012 定义）【文据】
MU_SUN = 1.32712440018e11  # km^3/s^2（日心引力常数，DE440 量级）【文据】
JD_J2000 = 2451545.0  # 2000-01-01 12:00 TT
SEC_PER_DAY = 86400.0
DEG = math.pi / 180.0

LAMBERT_TOL = 1e-8  # Lambert 相对收敛容差
SIN_DEGEN = 1e-6  # |sinΔν| ≤ 此值 → 走 Δν≈180° 拱点稳健分支
Z_MAX_FRAC = 1e-8  # 椭圆分支 z 上限 = 4π²·(1−此值)
KEPLER_TOL = 1e-13

# 地球 J2000 平均根数（与源件 EARTH 同值）
EARTH_J2000: Dict[str, Any] = {
    "name": "地球(J2000 平均根数)",
    "a_au": 1.00000011,
    "e": 0.01671022,
    "i_deg": 0.00005,
    "om_deg": -11.26064,
    "w_deg": 114.20783,
    "L_deg": 100.46435,
    "n_deg_day": 360.0 / 365.256,
    "epoch_jd": JD_J2000,
}


# ============================ 向量小工具 ============================
def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


# ============================ 儒略日 ↔ 日期 ============================
def jd_from_ymd(y: int, m: int, d: float) -> float:
    """公历日期 → 儒略日（00:00，格里高利历）。【算】"""
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return float(
        math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5
    )


def jd_from_iso(s: str) -> float:
    """``'YYYY-MM-DD'`` → 儒略日。【算】"""
    y, m, d = (int(p) for p in s.split("-"))
    return jd_from_ymd(y, m, d)


def iso_from_jd(jd: float) -> str:
    """儒略日 → ``'YYYY-MM-DD'``。"""
    z = jd + 0.5
    zi = int(math.floor(z))
    f = z - zi
    if zi < 2299161:
        a = zi
    else:
        alpha = int((zi - 1867216.25) / 36524.25)
        a = zi + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    dd = int(365.25 * c)
    e = int((b - dd) / 30.6001)
    day = b - dd - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return f"{year:04d}-{month:02d}-{int(math.floor(day)):02d}"


# ============================ 二体星历（平根数） ============================
def solve_kepler(mean_anomaly: float, e: float) -> float:
    """开普勒方程 M = E − e·sinE（牛顿法）。【算】"""
    M = math.fmod(mean_anomaly, 2.0 * math.pi)
    E = M + e * math.sin(M) if e < 0.85 else math.pi
    for _ in range(80):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < KEPLER_TOL:
            break
    return E


class Ephemeris:
    """平根数二体星历：给定儒略日 → 日心 J2000 黄道位置/速度（km, km/s）。【近似】

    Args:
        spec: 根数字典，需含 ``name``、``a_au``、``e``、``i_deg``、``om_deg``、``w_deg``、
            ``n_deg_day``、``epoch_jd``，以及平近点角 ``ma_deg``；若给的是平黄经 ``L_deg``
            （地球口径），则按 M = L − ω − Ω 换算。
    """

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.name: str = spec["name"]
        self.a: float = spec["a_au"] * AU_KM
        self.e: float = spec["e"]
        self.i: float = spec["i_deg"] * DEG
        self.om: float = spec["om_deg"] * DEG
        self.w: float = spec["w_deg"] * DEG
        self.n: float = spec["n_deg_day"] * DEG
        self.epoch: float = spec["epoch_jd"]
        if "ma_deg" in spec:
            self.M0: float = spec["ma_deg"] * DEG
        else:
            # M = L − ω − Ω（与源件口径一致）
            self.M0 = (spec["L_deg"] - spec["w_deg"] - spec["om_deg"]) * DEG

    def state(self, jd: float) -> Tuple[Vec3, Vec3]:
        """给定儒略日返回 (位置 km, 速度 km/s)。"""
        M = self.M0 + self.n * (jd - self.epoch)
        E = solve_kepler(M, self.e)
        cE, sE = math.cos(E), math.sin(E)
        r = self.a * (1.0 - self.e * cE)
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + self.e) * math.sin(E / 2.0),
            math.sqrt(1.0 - self.e) * math.cos(E / 2.0),
        )
        p = self.a * (1.0 - self.e * self.e)
        cnu, snu = math.cos(nu), math.sin(nu)
        xp, yp = r * cnu, r * snu
        vxp = -math.sqrt(MU_SUN / p) * snu
        vyp = math.sqrt(MU_SUN / p) * (self.e + cnu)
        co, so = math.cos(self.om), math.sin(self.om)
        cw, sw = math.cos(self.w), math.sin(self.w)
        ci, si = math.cos(self.i), math.sin(self.i)
        R11 = co * cw - so * sw * ci
        R12 = -co * sw - so * cw * ci
        R21 = so * cw + co * sw * ci
        R22 = -so * sw + co * cw * ci
        R31 = sw * si
        R32 = cw * si
        pos = (R11 * xp + R12 * yp, R21 * xp + R22 * yp, R31 * xp + R32 * yp)
        vel = (R11 * vxp + R12 * vyp, R21 * vxp + R22 * vyp, R31 * vxp + R32 * vyp)
        return pos, vel


# ============================ Lambert（通用变量法，单圈） ============================
def stumpff_cs(z: float) -> Tuple[float, float]:
    """Stumpff 函数 C(z), S(z)（级数展开避开 z≈0 的数值病态）。"""
    if z > 1e-6:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z, (sz - math.sin(sz)) / (z * sz)
    if z < -1e-6:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z), (math.sinh(sz) - sz) / ((-z) * sz)
    return (
        0.5 - z / 24.0 + z * z / 720.0,
        1.0 / 6.0 - z / 120.0 + z * z / 5040.0,
    )


def lambert(
    r1_vec: Sequence[float],
    r2_vec: Sequence[float],
    tof_s: float,
    mu: float = MU_SUN,
    prograde: bool = True,
    tol: float = LAMBERT_TOL,
) -> Tuple[Vec3, Vec3]:
    """日心二体 Lambert（单圈）：给 r1, r2 与飞行时间，返回出发/抵达速度矢量（km/s）。【算】

    通用变量法：A = √(2·r1·r2)·cos(Δν/2)；y(z) = r1 + r2 + A(zS−1)/√C；
    解 √μΔt = x³S + A√y。|sinΔν| ≤ SIN_DEGEN 时走拱点稳健分支。

    Raises:
        ValueError: 椭圆分支 TOF 过大、双曲分支无解或夹逼失败。
    """
    r1 = _norm(r1_vec)
    r2 = _norm(r2_vec)
    cr = _cross(r1_vec, r2_vec)
    cosd = max(-1.0, min(1.0, _dot(r1_vec, r2_vec) / (r1 * r2)))
    dnu = math.acos(cosd)
    if prograde:
        if cr[2] < 0.0:
            dnu = 2.0 * math.pi - dnu
    else:
        if cr[2] > 0.0:
            dnu = 2.0 * math.pi - dnu

    sin_dnu = math.sin(dnu)
    one_minus_cos = 2.0 * math.sin(dnu / 2.0) ** 2  # = 1 − cosΔν，数值稳定
    A = math.sqrt(2.0 * r1 * r2) * math.cos(dnu / 2.0)

    def t_of(z: float) -> float | None:
        C, S = stumpff_cs(z)
        if C <= 0.0:
            return None
        y = r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)
        if y <= 0.0:
            return None
        x = math.sqrt(y / C)
        return (x * x * x * S + A * math.sqrt(y)) / math.sqrt(mu)

    # ---- 夹逼 + 二分求 z ----
    t_zero = t_of(0.0)
    if t_zero is None:
        raise ValueError("Lambert: t(z=0) 不可求")
    z_max = 4.0 * math.pi * math.pi * (1.0 - Z_MAX_FRAC)
    if tof_s >= t_zero:
        lo, hi = 0.0, z_max
        if (t_of(hi) or -1.0) < tof_s:
            raise ValueError("Lambert: 椭圆分支无解（TOF 过大）")
    else:
        lo = None
        z, step = 0.0, 1.0
        while z > -1.0e7:
            zn = z - step
            tn = t_of(zn)
            if tn is None:
                step *= 0.5
                if step < 1e-13:
                    raise ValueError("Lambert: 双曲分支夹逼失败")
                continue
            if tn <= tof_s:
                lo = zn
                break
            z, step = zn, max(step * 1.5, 1.0)
        if lo is None:
            raise ValueError("Lambert: 双曲分支无解")
        hi = z

    for _ in range(300):
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

    C, S = stumpff_cs(z)
    y = r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)
    p = r1 * r2 * one_minus_cos / y  # 半通径（对 180° 亦稳健）

    if abs(sin_dnu) > SIN_DEGEN:
        f = 1.0 - y / r1
        g = A * math.sqrt(y / mu)
        gdot = 1.0 - y / r2
        v1 = _scale(_sub(r2_vec, _scale(r1_vec, f)), 1.0 / g)
        v2 = _scale(_sub(_scale(r2_vec, gdot), r1_vec), 1.0 / g)
    else:
        # Δν≈180° 稳健分支：两端按拱点（径向速度=0），纯横向速度
        r1h = _scale(r1_vec, 1.0 / r1)
        kh = (0.0, 0.0, 1.0)
        n_hat = _sub(
            kh, _scale(r1h, _dot(kh, r1h))
        )  # 拱点线正交平面内最接近黄道极的方向
        nn = _norm(n_hat)
        if nn < 1e-12:
            n_hat = (0.0, 1.0, 0.0)
        else:
            n_hat = _scale(n_hat, 1.0 / nn)
        r2h = _scale(r2_vec, 1.0 / r2)
        v1 = _scale(_cross(n_hat, r1h), math.sqrt(mu * p) / r1)
        v2 = _scale(_cross(n_hat, r2h), math.sqrt(mu * p) / r2)
    return v1, v2


# ============================ 转移腿 ============================
def direct_leg(
    launch_jd: float, tof_d: float, earth: Ephemeris, target: Ephemeris
) -> Dict[str, float]:
    """地球→目标直飞单腿：返回 C3 = |v∞_dep|²（km²/s²）等。【算】"""
    r1_vec, v1_vec = earth.state(launch_jd)
    r2_vec, v2_vec = target.state(launch_jd + tof_d)
    v1_sol, v2_sol = lambert(r1_vec, r2_vec, tof_d * SEC_PER_DAY)
    dv1 = _sub(v1_sol, v1_vec)
    dv2 = _sub(v2_sol, v2_vec)
    c3 = _dot(dv1, dv1)
    return {
        "launch_jd": launch_jd,
        "arrival_jd": launch_jd + tof_d,
        "tof_d": tof_d,
        "c3": c3,
        "vinf_dep": math.sqrt(max(c3, 0.0)),
        "vinf_arr": _norm(dv2),
    }


def return_leg(
    depart_jd: float, tof_d: float, earth: Ephemeris, target: Ephemeris
) -> Dict[str, float]:
    """目标→地球返回单腿。【算】"""
    r1_vec, v1_vec = target.state(depart_jd)
    r2_vec, v2_vec = earth.state(depart_jd + tof_d)
    v1_sol, v2_sol = lambert(r1_vec, r2_vec, tof_d * SEC_PER_DAY)
    dv1 = _sub(v1_sol, v1_vec)
    dv2 = _sub(v2_sol, v2_vec)
    return {
        "depart_jd": depart_jd,
        "arrival_jd": depart_jd + tof_d,
        "tof_d": tof_d,
        "vinf_dep": _norm(dv1),
        "vinf_arr": _norm(dv2),
    }


def hohmann_transfer(r1_au: float, r2_au: float) -> Dict[str, float]:
    """圆轨道共面 Hohmann 解析解（地球↔目标；作下界参照）。【算】"""
    r1, r2 = r1_au * AU_KM, r2_au * AU_KM
    v_c1, v_c2 = math.sqrt(MU_SUN / r1), math.sqrt(MU_SUN / r2)
    at = 0.5 * (r1 + r2)
    vt1 = math.sqrt(MU_SUN * (2.0 / r1 - 1.0 / at))
    vt2 = math.sqrt(MU_SUN * (2.0 / r2 - 1.0 / at))
    return {
        "v_c1": v_c1,
        "v_c2": v_c2,
        "vt1": vt1,
        "vt2": vt2,
        "vinf_dep": vt1 - v_c1,
        "c3": (vt1 - v_c1) ** 2,
        "vinf_arr": v_c2 - vt2,
        "tof_d": math.pi * math.sqrt(at**3 / MU_SUN) / SEC_PER_DAY,
    }


def lambert_on_180(
    r1_au: float, r2_au: float, tof_s: float, delta_deg: float
) -> Tuple[float, float, float]:
    """在夹角 (180°−δ) 的共面转移上跑 Lambert，返回 (v∞dep, C3, v∞arr)。【算】

    用于以 Lambert 复现 Hohmann 解析下界（Δν→180° 拱点分支的判据锚）。
    """
    r1, r2 = r1_au * AU_KM, r2_au * AU_KM
    d = delta_deg * DEG
    r1_vec = (r1, 0.0, 0.0)
    r2_vec = (-r2 * math.cos(d), r2 * math.sin(d), 0.0)  # 与 +x 夹角 180°−δ
    v1_sol, v2_sol = lambert(r1_vec, r2_vec, tof_s)
    v_earth = (0.0, math.sqrt(MU_SUN / r1), 0.0)
    v_target = (0.0, -math.sqrt(MU_SUN / r2), 0.0)
    dv1 = _sub(v1_sol, v_earth)
    c3 = _dot(dv1, dv1)
    return math.sqrt(max(c3, 0.0)), c3, _norm(_sub(v2_sol, v_target))


# ============================ 网格小工具 ============================
def tof_grid(t_min_d: float, t_max_d: float, t_step_d: float) -> List[float]:
    """闭区间 [t_min, t_max] 上步长 t_step 的 TOF 列表（含两端）。"""
    n = int(round((t_max_d - t_min_d) / t_step_d))
    return [t_min_d + k * t_step_d for k in range(n + 1)]


def iter_julian_days(j0: float, j1: float, step_d: float) -> Iterator[float]:
    """闭区间 [j0, j1] 上步长 step_d 的儒略日迭代器（含两端）。"""
    n = int(math.floor((j1 - j0) / step_d + 1e-9))
    for k in range(n + 1):
        yield j0 + k * step_d
