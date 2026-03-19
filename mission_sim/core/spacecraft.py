import numpy as np
from mission_sim.core.types import CoordinateFrame

class SpacecraftPointMass:
    """
    航天器质点模型 (Level 1)
    职责：维护航天器动力学状态，处理外力输入，提供状态导数，并强制执行坐标系一致性校验。
    """
    def __init__(self, sc_id: str, initial_state: list, frame: CoordinateFrame, initial_mass: float = 1000.0):
        """
        :param sc_id: 航天器唯一标识符
        :param initial_state: 初始状态 [x, y, z, vx, vy, vz] (SI单位: m, m/s)
        :param frame: 该初始状态所在的坐标系
        :param initial_mass: 初始干重+燃料总质量 (kg)
        """
        self.id = sc_id
        self.state = np.array(initial_state, dtype=np.float64)
        
        # 显式声明航天器当前所在的坐标系
        self.frame = frame
        
        self.mass = float(initial_mass)
        self.external_accel = np.zeros(3, dtype=np.float64)
        self.consumed_fuel = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def apply_thrust(self, force_vector: np.ndarray, force_frame: CoordinateFrame):
        """
        施加推力 (通常由 GNC 模块调用)
        :param force_vector: 推力向量 [Fx, Fy, Fz] (Newton)
        :param force_frame: 该推力向量所在的坐标系
        """
        # 【防呆校验】确保推力的坐标系与航天器本体所在坐标系一致
        if force_frame != self.frame:
            raise ValueError(
                f"[{self.id}] 推力坐标系不匹配！航天器处于 {self.frame.name}，"
                f"但接收到的推力基于 {force_frame.name}。"
            )
            
        accel = np.array(force_vector, dtype=np.float64) / self.mass
        self.external_accel += accel

    def clear_thrust(self):
        """
        清空推力加速度累加器。在每个控制/积分周期结束时调用。
        """
        self.external_accel = np.zeros(3, dtype=np.float64)

    def consume_mass(self, m_dot: float, dt: float):
        """
        更新质量 (燃料消耗)
        """
        dm = m_dot * dt
        self.mass -= dm
        self.consumed_fuel += dm

    def get_derivative(self, gravity_accel: np.ndarray, gravity_frame: CoordinateFrame) -> np.ndarray:
        """
        获取状态导数，供数值积分器使用。
        :param gravity_accel: 环境引力场提供的加速度 [ax, ay, az]
        :param gravity_frame: 该加速度所在的坐标系
        :return: 状态导数 [vx, vy, vz, ax_total, ay_total, az_total]
        """
        # 【防呆校验】确保引力场加速度的坐标系与航天器一致
        if gravity_frame != self.frame:
            raise ValueError(
                f"[{self.id}] 动力学坐标系不匹配！航天器处于 {self.frame.name}，"
                f"但环境引力场基于 {gravity_frame.name}。"
            )
            
        v = self.velocity
        a_total = np.array(gravity_accel, dtype=np.float64) + self.external_accel
        
        return np.concatenate([v, a_total])

    def __repr__(self):
        return (f"Spacecraft[{self.id}] | Frame: {self.frame.name} | "
                f"Mass: {self.mass:.1f}kg | Pos: {self.position} m")
