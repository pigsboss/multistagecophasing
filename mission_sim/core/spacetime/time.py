"""
高精度时间系统转换模块（更随和版）

轻松搞定 UTC 字符串 (ISO 8601) 与常用时间系统间的双向转换：
  - TAI (International Atomic Time)
  - TT  (Terrestrial Time)
  - TDB (Barycentric Dynamical Time)
  - Julian Date (UTC)
  - Unix timestamp
  - 平滑 UTC 秒 (不含闰秒，自 J2000.0 UTC 起算)

内置最新的闰秒表 (截至 2025 年 4 月)，并支持动态更新。
所有时间系统的内部表示均为 **自 J2000.0 历元起算的连续秒数**，
遵循 MCPC 统一时间轴约定。

依赖：仅使用 Python 标准库 (math, datetime)。
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Union

# ---------------------------------------------------------------------------
# 历元常量
# ---------------------------------------------------------------------------
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # J2000.0 UTC
J2000_JD = 2451545.0  # Julian Date of J2000.0 (TT 历元，但常用作参考)
TAI_OFFSET_AT_J2000 = 32.0  # 在 J2000.0 UTC 时，TAI - UTC = 32 s

# ---------------------------------------------------------------------------
# 闰秒表 (UTC datetime 对象列表，按时间排序)
# 每次新增闰秒时在此处追加即可。
# source: IERS Bulletin C
# ---------------------------------------------------------------------------
_LEAP_SECONDS_DATES = [
    datetime(1972, 1, 1, tzinfo=timezone.utc),
    datetime(1972, 7, 1, tzinfo=timezone.utc),
    datetime(1973, 1, 1, tzinfo=timezone.utc),
    datetime(1974, 1, 1, tzinfo=timezone.utc),
    datetime(1975, 1, 1, tzinfo=timezone.utc),
    datetime(1976, 1, 1, tzinfo=timezone.utc),
    datetime(1977, 1, 1, tzinfo=timezone.utc),
    datetime(1978, 1, 1, tzinfo=timezone.utc),
    datetime(1979, 1, 1, tzinfo=timezone.utc),
    datetime(1980, 1, 1, tzinfo=timezone.utc),
    datetime(1981, 7, 1, tzinfo=timezone.utc),
    datetime(1982, 7, 1, tzinfo=timezone.utc),
    datetime(1983, 7, 1, tzinfo=timezone.utc),
    datetime(1985, 7, 1, tzinfo=timezone.utc),
    datetime(1988, 1, 1, tzinfo=timezone.utc),
    datetime(1990, 1, 1, tzinfo=timezone.utc),
    datetime(1991, 1, 1, tzinfo=timezone.utc),
    datetime(1992, 7, 1, tzinfo=timezone.utc),
    datetime(1993, 7, 1, tzinfo=timezone.utc),
    datetime(1994, 7, 1, tzinfo=timezone.utc),
    datetime(1996, 1, 1, tzinfo=timezone.utc),
    datetime(1997, 7, 1, tzinfo=timezone.utc),
    datetime(1999, 1, 1, tzinfo=timezone.utc),
    datetime(2006, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 1, 1, tzinfo=timezone.utc),
    datetime(2012, 7, 1, tzinfo=timezone.utc),
    datetime(2015, 7, 1, tzinfo=timezone.utc),
    datetime(2017, 1, 1, tzinfo=timezone.utc),
]


# ---------------------------------------------------------------------------
# 闰秒管理
# ---------------------------------------------------------------------------
def leap_seconds(utc_time: datetime) -> int:
    """
    返回给定 UTC datetime 之前的累计闰秒数 (TAI - UTC)。
    若时间早于 1972-01-01，则返回 0 (闰秒制度引入前)。
    """
    cnt = 0
    for d in _LEAP_SECONDS_DATES:
        if d <= utc_time:
            cnt += 1
    return cnt


def add_leap_second(date_str: str):
    """
    添加一个未来的闰秒日期到内部表中。

    Parameters
    ----------
    date_str : str
        ISO 格式日期, 如 "2026-01-01"
    """
    new_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    _LEAP_SECONDS_DATES.append(new_date)
    _LEAP_SECONDS_DATES.sort()


# ---------------------------------------------------------------------------
# 核心转换函数
# ---------------------------------------------------------------------------
def utc_string_to_utc_smooth(utc_iso: str) -> float:
    """
    将 UTC 字符串转换为平滑 UTC 秒数 (不含闰秒，自 J2000.0 UTC)。

    Parameters
    ----------
    utc_iso : str
        ISO 8601 格式日期时间，如 "2023-06-15T08:30:00"

    Returns
    -------
    float : 秒数 (s)
    """
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return (dt - J2000_UTC).total_seconds()


def utc_smooth_to_utc_string(utc_smooth_sec: float) -> str:
    """反向：将平滑 UTC 秒数转换为 UTC ISO 字符串"""
    dt = J2000_UTC + timedelta(seconds=utc_smooth_sec)
    return dt.isoformat()


def utc_string_to_tai(utc_iso: str) -> float:
    """
    UTC -> TAI 秒 (自 J2000.0 TAI 历元)

    TAI 参考历元为 2000-01-01 12:00:00 TAI，此刻对应 UTC 平滑秒 + 闰秒 = 0。
    因此计算：TAI_sec = utc_smooth_sec + leap - TAI_OFFSET_AT_J2000
    """
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    utc_smooth = utc_string_to_utc_smooth(utc_iso)  # 复用避免重复
    leap = leap_seconds(dt)
    return utc_smooth + leap - TAI_OFFSET_AT_J2000


def tai_to_utc_string(tai_sec: float) -> str:
    """
    TAI 秒 -> UTC ISO 字符串
    逆过程：解出 UTC 平滑秒，然后转换为 datetime。
    注意闰秒边界处理：在闰秒时刻两边可能有多解性，这里采用常见约定：
    直接由 TAI 秒反算 UTC 平滑秒，并取最近的闰秒计数。
    """
    # 回溯法：逐步寻找合适的 leap
    # 简单方式：从预估开始
    guess_utc_smooth = tai_sec + TAI_OFFSET_AT_J2000 - 37  # 尝试
    # 转换为 datetime 并验证 leap
    # 由于 leap 随日期单调，可以二分查找
    # 但闰秒数量少，直接遍历
    for leap in range(0, 60):  # 最多 60 个闰秒
        utc_smooth = tai_sec - leap + TAI_OFFSET_AT_J2000
        # 转换为 datetime
        dt = J2000_UTC + timedelta(seconds=utc_smooth)
        # 检查此 dt 对应的 leap 是否等于假设的 leap
        if leap_seconds(dt) == leap:
            return dt.isoformat()
    raise ValueError(f"无法找到合法的 UTC 时间对应 TAI={tai_sec}")


def utc_string_to_tt(utc_iso: str) -> float:
    """UTC -> TT 秒 (自 J2000.0 TT 历元)"""
    tai = utc_string_to_tai(utc_iso)
    return tai + 32.184  # TAI 与 TT 的固定偏移


def tt_to_utc_string(tt_sec: float) -> str:
    """TT 秒 -> UTC 字符串"""
    tai = tt_sec - 32.184
    return tai_to_utc_string(tai)


def utc_string_to_tdb(utc_iso: str) -> float:
    """
    UTC -> TDB 秒 (自 J2000.0 TDB 历元)

    使用简化解析近似 (Fairhead & Bretagnon, 1990)，精度 < 1 μs。
    """
    # 先获取 TT 秒数 (J2000 TT)
    tt = utc_string_to_tt(utc_iso)
    # 计算从 J2000 TT 起算的儒略世纪数
    t_tt = tt / 86400.0 / 36525.0  # 自 J2000.0 TT 的儒略世纪
    # 经典公式中的 g 基于 UTC 或 TT，这里采用 TT (影响可忽略)
    g = (357.528 + 35999.05 * t_tt) * math.radians(1)  # 转换为弧度
    tdb_offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))  # 秒
    return tt + tdb_offset


def tdb_to_utc_string(tdb_sec: float) -> str:
    """TDB 秒 -> UTC 字符串，迭代求解 TT 逆函数"""
    # 使用简单迭代：从 TT = TDB 出发，逐步逼近
    tt = tdb_sec  # 初始猜测
    for _ in range(5):
        t_tt = tt / 86400.0 / 36525.0
        g = (357.528 + 35999.05 * t_tt) * math.radians(1)
        offset = 0.001658 * math.sin(g + 0.0167 * math.sin(g))
        tt = tdb_sec - offset
    return tt_to_utc_string(tt)


def utc_string_to_jd(utc_iso: str) -> float:
    """UTC -> Julian Date (UTC 尺度的连续 JD，不考虑闰秒跳变)"""
    utc_smooth = utc_string_to_utc_smooth(utc_iso)
    return J2000_JD + utc_smooth / 86400.0


def jd_to_utc_string(jd: float) -> str:
    """Julian Date (UTC) -> UTC 字符串"""
    utc_smooth = (jd - J2000_JD) * 86400.0
    return utc_smooth_to_utc_string(utc_smooth)


def utc_string_to_unix(utc_iso: str) -> float:
    """UTC -> Unix timestamp (POSIX 秒)"""
    dt = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def unix_to_utc_string(unixtime: float) -> str:
    """Unix timestamp -> UTC 字符串"""
    dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# 兼容旧式函数 (如 astro.py 中的名称)
# ---------------------------------------------------------------------------
def utc2tai(utc_jd: float) -> float:
    """
    旧接口：UTC Julian date -> TAI Julian date
    (假设输入为连续的 UTC JD)
    """
    # 计算隐含的 datetime
    utc_smooth = (utc_jd - J2000_JD) * 86400.0
    dt = J2000_UTC + timedelta(seconds=utc_smooth)
    leap = leap_seconds(dt)
    # TAI JD = UTC JD + leap/86400
    return utc_jd + leap / 86400.0


def utc2tdt(utc_jd: float) -> float:
    """UTC JD -> TDT (TT) JD"""
    return utc2tai(utc_jd) + 32.184 / 86400.0


def utc2tdb(utc_jd: float) -> float:
    """UTC JD -> TDB JD (简化)"""
    tdt = utc2tdt(utc_jd)
    jc = (utc_jd - J2000_JD) / 36525.0  # 近似使用 UTC
    g = 2.0 * math.pi * (357.528 + 35999.05 * jc) / 360.0
    return tdt + 0.001658 * math.sin(g + 0.0167 * math.sin(g)) / 86400.0


def unix2utc(t: float) -> float:
    """Unix timestamp -> UTC Julian day"""
    return t / 86400.0 + 2440587.5


def utc2unix(utc_jd: float) -> float:
    """UTC Julian day -> Unix timestamp"""
    return (utc_jd - 2440587.5) * 86400.0
