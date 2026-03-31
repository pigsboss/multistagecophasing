# mission_sim/core/physics/spacecraft.py
import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame

class SpacecraftPointMass:
    """
    航天器质点模型 (Level 1)
    职责：维护航天器动力学状态，处理外力输入，提供状态导数，累加 ΔV，并强制执行坐标系一致性校验。
    """
    def __init__(self, sc_id: str, initial_state: list | np.ndarray, frame: CoordinateFrame, initial_mass: float = 1000.0):
        """
        :param sc_id: 航天器唯一标识符
        :param initial_state: 初始状态 [x, y, z, vx, vy, vz] (SI单位: m, m/s)
        :param frame: 该初始状态所在的坐标系 (强契约绑定)
        :param initial_mass: 初始干重+推进剂总质量 (kg)
        """
        self.id = sc_id
        self.state = np.array(initial_state, dtype=np.float64)
        self.frame = frame  # 显式声明航天器本体当前所在的坐标系
        self.mass = float(initial_mass)
        
        self.external_accel = np.zeros(3, dtype=np.float64)
        self.consumed_fuel = 0.0
        
        # [L1核心增量]：连续积分的 ΔV 统计器，用于输出轨道维持(Station Keeping)基准账单
        self.accumulated_dv = 0.0 

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def apply_thrust(self, force_vector: np.ndarray, force_frame: CoordinateFrame):
        """
        施加推力并转化为本体加速度。
        [信息域 -> 物理域] 接口，强制校验坐标系契约。
        
        :param force_vector: 推力向量 [Fx, Fy, Fz] (Newton)
        :param force_frame: 该推力向量所在的坐标系
        """
        if force_frame != self.frame:
            raise ValueError(
                f"[{self.id} 物理域拒收] 推力坐标系不匹配！航天器处于 {self.frame.name}，"
                f"但 GNC 传递的推力基于 {force_frame.name}。"
            )
            
        accel = np.array(force_vector, dtype=np.float64) / self.mass
        self.external_accel += accel

    def get_derivative(self, gravity_accel: np.ndarray, gravity_frame: CoordinateFrame) -> np.ndarray:
        """
        获取状态导数，供外部的数值积分器 (如 RK4, Euler) 使用。
        [环境引擎 -> 航天器本体] 接口，强制校验引力场坐标系。
        
        :param gravity_accel: 环境引擎提供的总扰动加速度 [ax, ay, az]
        :param gravity_frame: 该加速度所在的坐标系
        :return: 状态导数 [vx, vy, vz, ax_total, ay_total, az_total]
        """
        if gravity_frame != self.frame:
            raise ValueError(
                f"[{self.id} 物理域崩溃] 动力学基准冲突！航天器处于 {self.frame.name}，"
                f"但环境引力场注入的加速度基于 {gravity_frame.name}。"
            )
            
        v = self.velocity
        a_total = np.array(gravity_accel, dtype=np.float64) + self.external_accel
        
        return np.concatenate([v, a_total])

    def integrate_dv(self, dt: float):
        """
        精确核算 ΔV 消耗。
        必须在当前时间步的推力施加完毕 (apply_thrust) 且完成数值积分后，
        调用 clear_thrust 之前执行此方法。
        """
        # ΔV = |a_thrust| * dt
        self.accumulated_dv += np.linalg.norm(self.external_accel) * dt

    def clear_thrust(self):
        """清空本时间步的推力加速度累加器"""
        self.external_accel = np.zeros(3, dtype=np.float64)

    def consume_mass(self, m_dot: float, dt: float):
        """更新质量模型 (推进剂消耗)"""
        dm = m_dot * dt
        self.mass -= dm
        self.consumed_fuel += dm

    def __repr__(self):
        return (f"Spacecraft[{self.id}] | Frame: {self.frame.name} | "
                f"Mass: {self.mass:.1f}kg | ΔV: {self.accumulated_dv:.4f} m/s")
