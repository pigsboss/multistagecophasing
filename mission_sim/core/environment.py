import numpy as np
from mission_sim.core.types import CoordinateFrame

class CelestialEnvironment:
    """
    天体力学环境类 (Level 1)
    职责：提供特定空间区域的引力及惯性力加速度，强制校验计算基准坐标系。
    """
    
    # --- 物理常数 (SI 单位制：米, 秒, 千克) ---
    GM_SUN = 1.32712440018e20
    GM_EARTH = 3.986004418e14
    AU = 1.495978707e11
    OMEGA = 1.990986e-7  # 地球绕日公转平均角速度 (rad/s)

    def __init__(self, region: str = "SUN_EARTH_L2", initial_epoch: float = 0.0):
        """
        :param region: 仿真的空间区域标识 (默认日地 L2)
        :param initial_epoch: 初始历元时间 (秒)
        """
        self.region = region
        self.epoch = float(initial_epoch)
        
        # 【新增】显式绑定该物理区域的计算坐标系
        if region == "SUN_EARTH_L2":
            self.computation_frame = CoordinateFrame.SUN_EARTH_ROTATING
        else:
            raise NotImplementedError(f"区域 {region} 尚未配置默认坐标系！")
        
        # --- 预计算 CRTBP 框架参数 ---
        self.mu = self.GM_EARTH / (self.GM_SUN + self.GM_EARTH)
        self.pos_sun = np.array([-self.mu * self.AU, 0.0, 0.0], dtype=np.float64)
        self.pos_earth = np.array([(1.0 - self.mu) * self.AU, 0.0, 0.0], dtype=np.float64)

    def step_time(self, dt: float):
        """推进环境历元"""
        self.epoch += dt

    def get_gravity_acceleration(self, sc_state: np.ndarray, sc_frame: CoordinateFrame) -> tuple[np.ndarray, CoordinateFrame]:
        """
        获取环境加速度。
        【防呆校验】拒绝为坐标系不匹配的状态提供物理计算！
        
        :param sc_state: 航天器状态 [x, y, z, vx, vy, vz]
        :param sc_frame: 该状态所在的坐标系标签
        :return: (加速度向量 [ax, ay, az], 加速度所在的坐标系标签)
        """
        if sc_frame != self.computation_frame:
            raise ValueError(
                f"[环境类报错] 坐标系冲突！动力学计算需要 {self.computation_frame.name}，"
                f"但传入的航天器状态基于 {sc_frame.name}。"
            )
            
        if self.region == "SUN_EARTH_L2":
            accel = self._compute_crtbp_acceleration(sc_state)
            # 返回时“盖上印章”，确保航天器也能反向校验
            return accel, self.computation_frame
        else:
            return np.zeros(3, dtype=np.float64), self.computation_frame

    def _compute_crtbp_acceleration(self, state: np.ndarray) -> np.ndarray:
        """
        内部计算：日地受限三体动力学 (CRTBP) 加速度。
        包含：万有引力 + 旋转系惯性力 (离心力、科氏力)。
        """
        pos = state[0:3]
        vel = state[3:6]
        
        r_sun_vec = pos - self.pos_sun
        r_earth_vec = pos - self.pos_earth
        
        r_sun_mag3 = np.linalg.norm(r_sun_vec)**3
        r_earth_mag3 = np.linalg.norm(r_earth_vec)**3
        
        # 1. 太阳与地球的保守引力
        accel_grav_sun = -self.GM_SUN * r_sun_vec / r_sun_mag3
        accel_grav_earth = -self.GM_EARTH * r_earth_vec / r_earth_mag3
        
        # 2. 旋转系引入的表观惯性力
        accel_centrifugal = np.array([
            self.OMEGA**2 * pos[0],
            self.OMEGA**2 * pos[1],
            0.0
        ], dtype=np.float64)
        
        accel_coriolis = np.array([
            2.0 * self.OMEGA * vel[1],
            -2.0 * self.OMEGA * vel[0],
            0.0
        ], dtype=np.float64)
        
        return accel_grav_sun + accel_grav_earth + accel_centrifugal + accel_coriolis

    def __repr__(self):
        return f"Environment[{self.region} | Frame={self.computation_frame.name} | Epoch={self.epoch:.1f}s]"
