import numpy as np
from scipy.linalg import solve_continuous_are

def get_lqr_gain(A, B, Q, R):
    """求解连续时间 LQR 增益矩阵 K"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def get_lvlh_dcm(pos_chief, vel_chief, omega_frame=1.991e-7):
    """
    计算从绝对旋转系到 LVLH 坐标系的方向余弦矩阵 (DCM)
    
    参数:
        pos_chief: 主星在旋转系下的位置 [x, y, z]
        vel_chief: 主星在旋转系下的速度 [vx, vy, vz]
        omega_frame: 旋转系的基础角速度 (日地系统为 1.991e-7 rad/s)
    返回:
        3x3 的 DCM 矩阵
    """
    # 1. 旋转系的角速度矢量 (绕 Z 轴旋转)
    omega_vec = np.array([0.0, 0.0, omega_frame])

    # 2. 计算 Chief 在惯性系下的真实速度 (运动学基本定理)
    # v_inertial = v_rot + omega x r
    vel_inertial = vel_chief + np.cross(omega_vec, pos_chief)

    # 3. 构造 LVLH 三轴单位矢量
    # X轴 (Radial): 沿位置矢量方向
    r_norm = np.linalg.norm(pos_chief)
    x_lvlh = pos_chief / r_norm

    # Z轴 (Normal): 沿惯性轨道角动量方向 h = r x v
    h_inertial = np.cross(pos_chief, vel_inertial)
    z_lvlh = h_inertial / np.linalg.norm(h_inertial)

    # Y轴 (Along-track): 构成右手系 Y = Z x X
    y_lvlh = np.cross(z_lvlh, x_lvlh)

    # 4. 组装 DCM 矩阵 (从 旋转系 转换到 LVLH 系)
    # 因为 x, y, z 已经是行向量，直接 vstack 即可构成旋转矩阵
    dcm = np.vstack((x_lvlh, y_lvlh, z_lvlh))
    return dcm

def absolute_to_lvlh(state_chief, state_deputy, omega_frame=1.991e-7):
    """
    将 Deputy 的绝对状态转换为相对于 Chief 的 LVLH 状态。
    
    参数:
        state_chief: 主星的 6D 绝对状态 (旋转系)
        state_deputy: 从星的 6D 绝对状态 (旋转系)
    返回:
        relative_state_lvlh: 6D 相对状态 [x, y, z, vx, vy, vz] (LVLH系)
    """
    r_c, v_c = state_chief[0:3], state_chief[3:6]
    r_d, v_d = state_deputy[0:3], state_deputy[3:6]

    # 1. 计算在旋转系下的相对状态
    delta_r_rot = r_d - r_c
    delta_v_rot = v_d - v_c

    # 2. 获取当前时刻的转换矩阵 DCM
    dcm = get_lvlh_dcm(r_c, v_c, omega_frame)

    # 3. 将相对位置投影到 LVLH 坐标轴上
    delta_r_lvlh = dcm @ delta_r_rot

    # 4. 将相对速度投影到 LVLH 坐标轴上
    # (注：这里做了一个针对 L2 编队的极简运动学近似。
    #  严格来说需减去 LVLH 系自身的旋转牵连速度，但在 L2 编队中 DCM 变化极慢，此近似精度足够)
    delta_v_lvlh = dcm @ delta_v_rot

    return np.concatenate((delta_r_lvlh, delta_v_lvlh))

def get_hcw_matrices(n: float, mass: float):
    """
    生成经典 C-W (Clohessy-Wiltshire) 相对运动方程的状态空间 A 和 B 矩阵。
    适用于近圆轨道或近似常数角速度的 LVLH 系相对编队控制。
    
    参数:
        n: 主星轨道的平均角速度 (rad/s)。
           (注: 若在日地 L2 点，n 近似为日地系统旋转角速度 omega)
        mass: 从星的质量 (kg)，用于计算推力加速度映射
        
    返回:
        A (6x6): 相对运动状态转移矩阵
        B (6x3): 控制输入映射矩阵
    """
    n2 = n ** 2
    
    # A 矩阵: 相对状态演化 X_dot = A * X
    # 状态向量 X = [x, y, z, vx, vy, vz]^T in LVLH
    A = np.zeros((6, 6))
    
    # 运动学关系: 速度是位置的导数
    A[0:3, 3:6] = np.eye(3)
    
    # 动力学关系: 加速度项 (引力梯度与离心力)
    A[3, 0] = 3 * n2   # X 轴 (径向) 引力梯度
    A[4, 0] = 0.0      # Y 轴 (切向)
    A[5, 2] = -n2      # Z 轴 (法向) 回复力
    
    # 科氏力耦合项
    A[3, 4] = 2 * n
    A[4, 3] = -2 * n
    
    # B 矩阵: 控制输入映射 X_dot = A * X + B * U
    # 控制输入 U = [Fx, Fy, Fz]^T 
    B = np.zeros((6, 3))
    B[3:6, 0:3] = np.eye(3) / mass
    
    return A, B
