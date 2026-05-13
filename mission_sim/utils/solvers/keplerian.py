"""
Kepler 方程求解器及轨道根数 → 笛卡尔坐标批量转换
利用 Numba 进行实时编译与向量化并行加速，适用于大批量轨道计算。
支持椭圆 (e < 1)、双曲 (e > 1) 和近抛物 (e ≈ 1) 轨道。

架构：
- 三个专用批量函数：_cartesian_elliptic_batch, _cartesian_hyperbolic_batch, _cartesian_parabolic_batch
  每个函数内部无分支，可独立测试和重用。
- 公共旋转辅助函数 _rotate_orbital_to_inertial，被上述函数复用。
- 统一入口 kepler_elements_to_cartesian_batch 根据偏心率分组，分别调用专用函数后合并结果。
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# 公共辅助函数
# ---------------------------------------------------------------------------

@njit
def _rotate_orbital_to_inertial(x_orb, y_orb, vx_orb, vy_orb,
                                cos_i, sin_i, cos_Omega, sin_Omega,
                                cos_omega, sin_omega):
    """
    将轨道面内的位置和速度旋转到惯性坐标系。
    复用三类轨道的共同计算，避免重复代码。
    """
    # 位置旋转
    x1 = x_orb * cos_omega - y_orb * sin_omega
    y1 = x_orb * sin_omega + y_orb * cos_omega
    y2 = y1 * cos_i
    z1 = y1 * sin_i
    x = x1 * cos_Omega - y2 * sin_Omega
    y = x1 * sin_Omega + y2 * cos_Omega
    z = z1

    # 速度旋转
    vx1 = vx_orb * cos_omega - vy_orb * sin_omega
    vy1 = vx_orb * sin_omega + vy_orb * cos_omega
    vy2 = vy1 * cos_i
    vz1 = vy1 * sin_i
    vx = vx1 * cos_Omega - vy2 * sin_Omega
    vy = vx1 * sin_Omega + vy2 * cos_Omega
    vz = vz1

    return x, y, z, vx, vy, vz


# ---------------------------------------------------------------------------
# 椭圆批量： e < 1
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _cartesian_elliptic_batch(a_arr, e_arr, i_arr, Omega_arr, omega_arr, M_arr, mu):
    """
    批量转换椭圆轨道到笛卡尔状态。
    所有输入数组长度相同，仅包含椭圆轨道 (e < 1)。
    """
    n = a_arr.shape[0]
    states = np.empty((n, 6))

    for idx in prange(n):
        a = a_arr[idx]
        e = e_arr[idx]
        M = M_arr[idx]

        # 牛顿法求解偏近点角 E
        E = M
        for _ in range(50):
            sin_E = np.sin(E)
            cos_E = np.cos(E)
            dE = (E - e * sin_E - M) / (1.0 - e * cos_E)
            E -= dE
            if abs(dE) < 1e-12:
                break

        cos_E = np.cos(E)
        sin_E = np.sin(E)

        # 真近点角 nu
        sqrt_1pe = np.sqrt(1.0 + e)
        sqrt_1me = np.sqrt(1.0 - e)
        nu = 2.0 * np.arctan2(sqrt_1pe * np.sin(0.5 * E),
                              sqrt_1me * np.cos(0.5 * E))

        # 向径
        r = a * (1.0 - e * cos_E)

        # 轨道面内速度
        sqrt_1me2 = np.sqrt(1.0 - e * e)
        n_motion = np.sqrt(mu * a) / r
        vx_orb = -n_motion * sin_E
        vy_orb = n_motion * sqrt_1me2 * cos_E

        # 旋转到惯性系
        x, y, z, vx, vy, vz = _rotate_orbital_to_inertial(
            r * np.cos(nu), r * np.sin(nu),
            vx_orb, vy_orb,
            np.cos(i_arr[idx]), np.sin(i_arr[idx]),
            np.cos(Omega_arr[idx]), np.sin(Omega_arr[idx]),
            np.cos(omega_arr[idx]), np.sin(omega_arr[idx])
        )

        states[idx, 0] = x
        states[idx, 1] = y
        states[idx, 2] = z
        states[idx, 3] = vx
        states[idx, 4] = vy
        states[idx, 5] = vz

    return states


# ---------------------------------------------------------------------------
# 双曲批量： e > 1
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _cartesian_hyperbolic_batch(a_arr, e_arr, i_arr, Omega_arr, omega_arr, M_arr, mu):
    """
    批量转换双曲轨道到笛卡尔状态。
    注意：对于双曲轨道，半长轴 a 为负值，但我们仍直接传递并用于公式。
    """
    n = a_arr.shape[0]
    states = np.empty((n, 6))

    for idx in prange(n):
        a = a_arr[idx]
        e = e_arr[idx]
        M = M_arr[idx]

        # 牛顿法求解双曲偏近点角 H
        # 初始值
        H = np.log(2.0 * abs(M) / e + 1.8)
        for _ in range(50):
            sinh_H = np.sinh(H)
            cosh_H = np.cosh(H)
            dH = (e * sinh_H - H - M) / (e * cosh_H - 1.0)
            H -= dH
            if abs(dH) < 1e-12:
                break

        sinh_H = np.sinh(H)
        cosh_H = np.cosh(H)

        # 真近点角 nu
        sqrt_e_plus_1 = np.sqrt(e + 1.0)
        sqrt_e_minus_1 = np.sqrt(e - 1.0)
        nu = 2.0 * np.arctan2(sqrt_e_plus_1 * np.sinh(0.5 * H),
                              sqrt_e_minus_1 * np.cosh(0.5 * H))

        # 向径
        r = a * (1.0 - e * cosh_H)  # 注意 a 为负，所以 r 正

        # 轨道面内速度（双曲超速）
        sqrt_e2_minus1 = np.sqrt(e * e - 1.0)
        n_motion = np.sqrt(-mu * a) / r  # mu * a 为负时取平方根需处理，实际上用 -a
        vx_orb = -n_motion * sinh_H
        vy_orb = n_motion * sqrt_e2_minus1 * cosh_H

        # 旋转到惯性系
        x, y, z, vx, vy, vz = _rotate_orbital_to_inertial(
            r * np.cos(nu), r * np.sin(nu),
            vx_orb, vy_orb,
            np.cos(i_arr[idx]), np.sin(i_arr[idx]),
            np.cos(Omega_arr[idx]), np.sin(Omega_arr[idx]),
            np.cos(omega_arr[idx]), np.sin(omega_arr[idx])
        )

        states[idx, 0] = x
        states[idx, 1] = y
        states[idx, 2] = z
        states[idx, 3] = vx
        states[idx, 4] = vy
        states[idx, 5] = vz

    return states


# ---------------------------------------------------------------------------
# 抛物批量： e ≈ 1
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _cartesian_parabolic_batch(p_arr, i_arr, Omega_arr, omega_arr, M_arr, mu):
    """
    批量转换抛物轨道到笛卡尔状态。
    输入 p_arr 为半通径 (semi-latus rectum)，即近拱点距离的2倍。
    M = 平近点角（对于抛物轨道常有定义）
    使用 Barker 方程。
    """
    n = p_arr.shape[0]
    states = np.empty((n, 6))

    for idx in prange(n):
        p = p_arr[idx]
        M = M_arr[idx]

        # Barker 方程： M = q + q^3/3
        # 直接代数解
        # 先计算判别式
        D = M * 0.5
        # 解三次方程 q^3 + 3q - 3M = 0
        # 使用标准解析法
        # 代入 q = s - 1/s, 其中 s = (3M + sqrt(9M^2+4)/2)^(1/3) ？
        # 为了精确，使用稳定公式
        # 设 A = (3M + sqrt(9M^2+4))/2
        # 则 q = A^(1/3) - A^(-1/3)
        # 更稳： q = 2 / ( ( sqrt(9M^2+4) + 3M )^(1/3) - ... ) 不好。
        # 使用通用公式（来自 Vallado）：
        sqrt_term = np.sqrt(M * M + 1.0)
        B = M * 0.5
        # 采用双曲三角表示？不用。直接计算。
        # 利用 tan(nu/2) = q,  nu = 2*atan(q)
        # 下面使用文献中的解析解：
        S = (M + np.sqrt(M * M + 1.0)) ** (1.0 / 3.0)
        q = S - 1.0 / S

        # 真近点角
        nu = 2.0 * np.arctan(q)

        # 向径
        r = p / (1.0 + np.cos(nu))

        # 速度大小 v = sqrt(2*mu/r)
        v = np.sqrt(2.0 * mu / r)
        # 径向速度 vr = sqrt(mu/p) * sin(nu)
        # 横向速度 vt = sqrt(mu/p) * (1 + cos(nu))
        sqrt_mu_over_p = np.sqrt(mu / p)
        vr = sqrt_mu_over_p * np.sin(nu)
        vt = sqrt_mu_over_p * (1.0 + np.cos(nu))

        # 轨道面内速度分量
        vx_orb = vr * np.cos(nu) - vt * np.sin(nu)
        vy_orb = vr * np.sin(nu) + vt * np.cos(nu)

        # 旋转到惯性系
        x, y, z, vx, vy, vz = _rotate_orbital_to_inertial(
            r * np.cos(nu), r * np.sin(nu),
            vx_orb, vy_orb,
            np.cos(i_arr[idx]), np.sin(i_arr[idx]),
            np.cos(Omega_arr[idx]), np.sin(Omega_arr[idx]),
            np.cos(omega_arr[idx]), np.sin(omega_arr[idx])
        )

        states[idx, 0] = x
        states[idx, 1] = y
        states[idx, 2] = z
        states[idx, 3] = vx
        states[idx, 4] = vy
        states[idx, 5] = vz

    return states


# ---------------------------------------------------------------------------
# 统一入口：按偏心率分组，调用专用函数
# ---------------------------------------------------------------------------

@njit(parallel=True)
def kepler_elements_to_cartesian_batch(a_arr, e_arr, i_arr, Omega_arr, omega_arr, M_arr, mu):
    """
    统一批量转换函数：根据偏心率自动选择椭圆、双曲或抛物算法。
    输入数组长度相同。为抛物轨道利用半通径 p 替代 a，但本函数中抛物线仍传入
    实参 a_arr 表示半通径（因为 a 对抛物无定义），调用者需调整输入。
    """
    n = a_arr.shape[0]
    states = np.empty((n, 6))

    # 分组
    idx_ell = np.where(e_arr < 0.9999)[0]
    idx_hyp = np.where(e_arr > 1.0001)[0]
    mask_par = (e_arr >= 0.9999) & (e_arr <= 1.0001)
    idx_par = np.where(mask_par)[0]

    # 分别调用专用批量函数，并填充结果
    if len(idx_ell) > 0:
        states_ell = _cartesian_elliptic_batch(
            a_arr[idx_ell], e_arr[idx_ell], i_arr[idx_ell],
            Omega_arr[idx_ell], omega_arr[idx_ell], M_arr[idx_ell], mu
        )
        for i, orig in enumerate(idx_ell):
            states[orig] = states_ell[i]

    if len(idx_hyp) > 0:
        states_hyp = _cartesian_hyperbolic_batch(
            a_arr[idx_hyp], e_arr[idx_hyp], i_arr[idx_hyp],
            Omega_arr[idx_hyp], omega_arr[idx_hyp], M_arr[idx_hyp], mu
        )
        for i, orig in enumerate(idx_hyp):
            states[orig] = states_hyp[i]

    if len(idx_par) > 0:
        # 对于抛物线，需要半通径 p，这里假设传入的 a_arr 部分已经是 p
        states_par = _cartesian_parabolic_batch(
            a_arr[idx_par], i_arr[idx_par], Omega_arr[idx_par],
            omega_arr[idx_par], M_arr[idx_par], mu
        )
        for i, orig in enumerate(idx_par):
            states[orig] = states_par[i]

    return states
