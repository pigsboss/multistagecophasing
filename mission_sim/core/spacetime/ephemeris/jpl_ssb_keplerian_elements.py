# mission_sim/core/spacetime/ephemeris/jpl_ssb_keplerian_elements.py
"""
长时段太阳系天体轨道根数 (Table 2a) 与外行星平近点角修正 (Table 2b)

数据来源：
  JPL Solar System Dynamics Group
  Orbital Ephemerides of the Sun, Moon, and Planets (Standish & Williams, 1992)
  (相对于 J2000.0 平均黄道和春分点，有效范围 3000 BC – 3000 AD)

本模块提供：
- 由儒略日计算自 J2000.0 起算的儒略世纪数
- 获取任意时刻指定天体的 6 个密切开普勒根数 (半长轴已转为米，角度为弧度)
- 对外行星自动加入 Table 2b 的长期/周期摄动项
- 直接计算天体在 J2000 黄道坐标系中的笛卡尔状态 (m, m/s)
- 批量获取所有行星的笛卡尔状态 (可用于 N 体传播器初始化)
"""

import numpy as np
from mission_sim.utils.solvers.keplerian import kepler_elements_to_cartesian_batch

# ---------------------------------------------------------------------------
# 天文学常数
# ---------------------------------------------------------------------------
_AU = 149597870700.0          # 天文单位 (m)
_MU_SUN = 1.32712440018e20    # 太阳引力常数 (m³/s²)
_DEG2RAD = np.pi / 180.0
_CENTURY_DAYS = 36525.0        # 儒略世纪 (日)

# ---------------------------------------------------------------------------
# Table 2a – 原始平均根数 (J2000.0 历元, 角度及变率单位: 度)
# ---------------------------------------------------------------------------
_TABLE2A = {
    "Mercury": {
        "a0": 0.38709843,  "a_dot": 0.00000000,
        "e0": 0.20563661,  "e_dot": 0.00002123,
        "I0":  7.00559432, "I_dot": -0.00590158,
        "L0": 252.25166724, "L_dot": 149472.67486623,
        "lp0": 77.45771895, "lp_dot": 0.15940013,   # 近日点经度
        "ln0": 48.33961819, "ln_dot": -0.12214182,  # 升交点经度
    },
    "Venus": {
        "a0": 0.72332102,  "a_dot": -0.00000026,
        "e0": 0.00676399,  "e_dot": -0.00005107,
        "I0": 3.39777545,  "I_dot": 0.00043494,
        "L0": 181.97970850, "L_dot": 58517.81560260,
        "lp0": 131.76755713, "lp_dot": 0.05679648,
        "ln0": 76.67261496, "ln_dot": -0.27274174,
    },
    "EM Bary": {
        "a0": 1.00000018,  "a_dot": -0.00000003,
        "e0": 0.01673163,  "e_dot": -0.00003661,
        "I0": -0.00054346, "I_dot": -0.01337178,
        "L0": 100.46691572, "L_dot": 35999.37306329,
        "lp0": 102.93005885, "lp_dot": 0.31795260,
        "ln0": -5.11260389, "ln_dot": -0.24123856,
    },
    "Mars": {
        "a0": 1.52371243,  "a_dot": 0.00000097,
        "e0": 0.09336511,  "e_dot": 0.00009149,
        "I0": 1.85181869,  "I_dot": -0.00724757,
        "L0": -4.56813164, "L_dot": 19140.29934243,
        "lp0": -23.91744784, "lp_dot": 0.45223625,
        "ln0": 49.71320984, "ln_dot": -0.26852431,
    },
    "Jupiter": {
        "a0": 5.20248019,  "a_dot": -0.00002864,
        "e0": 0.04853590,  "e_dot": 0.00018026,
        "I0": 1.29861416,  "I_dot": -0.00322699,
        "L0": 34.33479152, "L_dot": 3034.90371757,
        "lp0": 14.27495244, "lp_dot": 0.18199196,
        "ln0": 100.29282654, "ln_dot": 0.13024619,
    },
    "Saturn": {
        "a0": 9.54149883,  "a_dot": -0.00003065,
        "e0": 0.05550825,  "e_dot": -0.00032044,
        "I0": 2.49424102,  "I_dot": 0.00451969,
        "L0": 50.07571329, "L_dot": 1222.11494724,
        "lp0": 92.86136063, "lp_dot": 0.54179478,
        "ln0": 113.63998702, "ln_dot": -0.25015002,
    },
    "Uranus": {
        "a0": 19.18797948, "a_dot": -0.00020455,
        "e0": 0.04685740,  "e_dot": -0.00001550,
        "I0": 0.77298127,  "I_dot": -0.00180155,
        "L0": 314.20276625, "L_dot": 428.49512595,
        "lp0": 172.43404441, "lp_dot": 0.09266985,
        "ln0": 73.96250215, "ln_dot": 0.05739699,
    },
    "Neptune": {
        "a0": 30.06952752, "a_dot": 0.00006447,
        "e0": 0.00895439,  "e_dot": 0.00000818,
        "I0": 1.77005520,  "I_dot": 0.00022400,
        "L0": 304.22289287, "L_dot": 218.46515314,
        "lp0": 46.68158724, "lp_dot": 0.01009938,
        "ln0": 131.78635853, "ln_dot": -0.00606302,
    },
}

# ---------------------------------------------------------------------------
# Table 2b – 外行星平近点角附加摄动项
#   b: 二次项系数 [度/世纪²]
#   c: 余弦振幅 [度]
#   s: 正弦振幅 [度]
#   f: 频率 [度/世纪]
# 注意：函数内部会将 f 转换为弧度/世纪，用于三角函数计算
# ---------------------------------------------------------------------------
_TABLE2B = {
    "Jupiter": {"b": -0.00012452, "c":  0.06064060, "s": -0.35635438, "f": 38.35125000},
    "Saturn":  {"b":  0.00025899, "c": -0.13434469, "s":  0.87320147, "f": 38.35125000},
    "Uranus":  {"b":  0.00058331, "c": -0.97731848, "s":  0.17689245, "f":  7.67025000},
    "Neptune": {"b": -0.00041348, "c":  0.68346318, "s": -0.10162547, "f":  7.67025000},
}

# ---------------------------------------------------------------------------
# 时间转换
# ---------------------------------------------------------------------------
def jd_to_t(jd: float) -> float:
    """
    将儒略日转换为自 J2000.0 起算的儒略世纪数。

    Parameters
    ----------
    jd : float
        儒略日 (任意尺度)。

    Returns
    -------
    t_cy : float
        儒略世纪数。
    """
    return (jd - 2451545.0) / _CENTURY_DAYS

# ---------------------------------------------------------------------------
# 轨道根数计算
# ---------------------------------------------------------------------------
def get_elements(body: str, t_cy: float) -> dict:
    """
    返回指定天体在儒略世纪 t_cy 时的密切开普勒根数 (Table 2a + 2b)。

    Parameters
    ----------
    body : str
        天体名称 ("Mercury", "Venus", "EM Bary", "Mars", "Jupiter",
        "Saturn", "Uranus", "Neptune")
    t_cy : float
        自 J2000.0 起算的儒略世纪数。

    Returns
    -------
    dict
        包含键:
        - 'a'   : 半长轴 (m)
        - 'e'   : 偏心率
        - 'i'   : 倾角 (rad)
        - 'Omega' : 升交点经度 (rad)
        - 'omega' : 近日点幅角 (rad)
        - 'M'     : 平近点角 (rad), 已加修正
    """
    d = _TABLE2A[body]

    # 1. 线性部分 (角度为度)
    a_au   = d["a0"]   + d["a_dot"]   * t_cy
    e      = d["e0"]   + d["e_dot"]   * t_cy
    I_deg  = d["I0"]   + d["I_dot"]   * t_cy
    L_deg  = d["L0"]   + d["L_dot"]   * t_cy
    lp_deg = d["lp0"]  + d["lp_dot"]  * t_cy
    ln_deg = d["ln0"]  + d["ln_dot"]  * t_cy

    # 2. 外行星附加摄动项 (Table 2b)
    if body in _TABLE2B:
        b2, c2, s2, f2 = (_TABLE2B[body]["b"],
                          _TABLE2B[body]["c"],
                          _TABLE2B[body]["s"],
                          _TABLE2B[body]["f"])
        # 将频率 f 转换为弧度/世纪，用于三角函数
        f2_rad = f2 * _DEG2RAD
        delta_M_deg = (b2 * t_cy * t_cy
                       + c2 * np.cos(f2_rad * t_cy)
                       + s2 * np.sin(f2_rad * t_cy))
    else:
        delta_M_deg = 0.0

    M_deg = L_deg - lp_deg + delta_M_deg

    # 3. 角度归化并计算 ω
    # 将 M 限制在 (-180°, +180°] 区间，与参考文献一致
    M_deg = np.fmod(M_deg + 180.0, 360.0) - 180.0

    Omega_deg = np.fmod(ln_deg, 360.0)
    omega_deg = np.fmod(lp_deg - ln_deg, 360.0)

    # 4. 转换为 SI 及弧度
    a_m    = a_au * _AU
    i_rad  = I_deg   * _DEG2RAD
    Omega  = Omega_deg * _DEG2RAD
    omega  = omega_deg * _DEG2RAD
    M      = M_deg   * _DEG2RAD

    return {
        "a": a_m, "e": e, "i": i_rad,
        "Omega": Omega, "omega": omega, "M": M
    }


def get_elements_cartesian(body: str, t_cy: float, mu: float = _MU_SUN) -> np.ndarray:
    """
    直接返回天体在 J2000 黄道坐标系中的笛卡尔状态 (m, m/s)。

    Parameters
    ----------
    body : str
        天体名称。
    t_cy : float
        自 J2000.0 起算的儒略世纪数。
    mu : float
        中心天体引力常数 (默认为太阳)。

    Returns
    -------
    state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz]  (黄道坐标系)
    """
    el = get_elements(body, t_cy)
    result = kepler_elements_to_cartesian_batch(
        np.array([el["a"]]),
        np.array([el["e"]]),
        np.array([el["i"]]),
        np.array([el["Omega"]]),
        np.array([el["omega"]]),
        np.array([el["M"]]),
        mu,
    )
    return result[0]


def get_all_planet_states(t_cy: float, mu: float = _MU_SUN) -> dict:
    """
    返回所有行星 (含 EM Bary) 在时刻 t_cy 的黄道笛卡尔状态。

    Parameters
    ----------
    t_cy : float
        儒略世纪数。
    mu : float
        引力常数。

    Returns
    -------
    dict
        键为天体名称，值为 6 元素状态数组。
    """
    bodies = list(_TABLE2A.keys())
    states = {}
    for body in bodies:
        states[body] = get_elements_cartesian(body, t_cy, mu)
    return states
