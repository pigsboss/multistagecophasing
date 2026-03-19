import numpy as np
from mission_sim.core.types import CoordinateFrame, Telecommand

class GroundStation:
    """
    地面测控站类 (Level 1)
    职责：模拟地面站定轨系统，向航天器发送带有坐标系标签的带噪遥测数据和遥控指令。
    """
    def __init__(self, 
                 name: str, 
                 operating_frame: CoordinateFrame, 
                 pos_noise_std: float = 5.0, 
                 vel_noise_std: float = 0.005):
        """
        :param name: 测控站名称 (如 "Beijing_DCC")
        :param operating_frame: 地面站解算星历所在的工作坐标系
        :param pos_noise_std: 定轨位置误差标准差 (1 sigma, m)
        :param vel_noise_std: 定轨速度误差标准差 (1 sigma, m/s)
        """
        self.name = name
        self.operating_frame = operating_frame
        self.pos_noise_std = float(pos_noise_std)
        self.vel_noise_std = float(vel_noise_std)

    def track_spacecraft(self, true_state: np.ndarray, sc_frame: CoordinateFrame) -> tuple[np.ndarray, CoordinateFrame]:
        """
        模拟跟踪测量：将航天器的真实物理状态转化为地面定轨解算结果。
        
        :param true_state: 航天器绝对真实状态 [x, y, z, vx, vy, vz]
        :param sc_frame: 该真实状态所在的坐标系
        :return: (带噪声的观测状态估计值, 地面站的工作坐标系)
        """
        # 【防呆校验】确保能够“看到”航天器 (L1 阶段不包含坐标转换模块，强制要求同系)
        if sc_frame != self.operating_frame:
            raise ValueError(
                f"[测控站报错] 坐标系不匹配！地面站 {self.name} 在 {self.operating_frame.name} 下工作，"
                f"无法直接处理基于 {sc_frame.name} 的航天器状态。需引入坐标转换。"
            )

        # 模拟定轨系统的解算误差 (高斯白噪声)
        pos_noise = np.random.normal(0.0, self.pos_noise_std, 3)
        vel_noise = np.random.normal(0.0, self.vel_noise_std, 3)
        noise_vector = np.concatenate([pos_noise, vel_noise])
        
        # 观测状态 = 真实状态 + 误差
        observed_state = true_state + noise_vector
        
        # 返回数据并盖上地面站的坐标系印章
        return observed_state, self.operating_frame

    def generate_telecommand(self, cmd_type, target_state, target_frame):
        return Telecommand(
            cmd_type=cmd_type, 
            target_state=target_state, 
            frame=target_frame
        )    

    def __repr__(self):
        return (f"GroundStation[{self.name}] | Frame: {self.operating_frame.name} | "
                f"Noise(P/V): {self.pos_noise_std}m, {self.vel_noise_std}m/s")
