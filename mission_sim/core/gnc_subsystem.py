import numpy as np
from mission_sim.core.types import CoordinateFrame, Telecommand

class GNC_Subsystem:
    """
    航天器 GNC 子系统 (Level 1)
    职责：接收地面指令与遥测，执行严格的坐标系校验，计算控制偏差，并输出带坐标系标签的推力指令。
    """
    def __init__(self, sc_id: str, operating_frame: CoordinateFrame):
        """
        :param sc_id: 航天器标识符
        :param operating_frame: GNC 算法内部运算所依赖的基准坐标系
        """
        self.sc_id = sc_id
        self.operating_frame = operating_frame
        
        # 内部状态寄存器
        self.estimated_state: np.ndarray = None
        self.target_state: np.ndarray = None
        
        # 控制状态机
        self.is_active = True
        self.control_mode = "STANDBY"

    def process_telecommand(self, packet: Telecommand):
        """处理地面上行的遥控指令，此时 packet 已经是 Telecommand 对象"""
        
        # 1. 强制契约校验：如果指令坐标系与 GNC 运行坐标系不符，直接拒收！
        if packet.frame != self.operating_frame:
            raise ValueError(
                f"[GNC Error] Frame Mismatch! "
                f"GNC is in {self.operating_frame.name}, "
                f"but Command is in {packet.frame.name}."
            )
            
        # 2. 解析对象属性 (不再使用字典的 .get())
        if packet.cmd_type == "ORBIT_MAINTENANCE":
            self.target_state = np.copy(packet.target_state)
            self.control_mode = packet.cmd_type
            self.is_active = True
            print(f"[{self.sc_id} GNC] Accepted command: {packet.cmd_type}")

    def update_navigation(self, obs_state: np.ndarray, obs_frame: CoordinateFrame):
        """
        更新导航滤波器状态 (L1 阶段为直接透传)
        :param obs_state: 带有噪声的观测状态 [x, y, z, vx, vy, vz]
        :param obs_frame: 该观测状态所在的坐标系
        """
        # 【防呆校验 2】拦截坐标系错误的测量数据
        if obs_frame != self.operating_frame:
            raise ValueError(
                f"[GNC 报错] 测量坐标系拒收！无法将基于 {obs_frame.name} 的观测值"
                f"直接融合进 {self.operating_frame.name} 的导航滤波器中。"
            )
            
        self.estimated_state = np.array(obs_state, dtype=np.float64)

    def compute_control_force(self, k_matrix: np.ndarray) -> tuple[np.ndarray, CoordinateFrame]:
        """
        计算所需的推力控制量，并附带 GNC 的坐标系印章
        :param k_matrix: 外部传入的控制增益矩阵 (例如 LQR 增益)
        :return: (推力向量 [Fx, Fy, Fz] (N), 推力向量所在的坐标系)
        """
        # 安全检查：未激活或数据不全时，输出零推力
        if not self.is_active or self.estimated_state is None or self.target_state is None:
            return np.zeros(3, dtype=np.float64), self.operating_frame

        # 1. 计算状态偏差 (此时由于前面的严格校验，保证了两者一定在同一坐标系下)
        error = self.estimated_state - self.target_state
        
        # 2. 计算推力 (比例反馈控制: u = -K * x)
        force_cmd = -k_matrix @ error
        
        # 3. 返回推力和坐标系印章 (供 SpacecraftPointMass.apply_thrust 进行最后一次校验)
        return force_cmd, self.operating_frame

    def __repr__(self):
        mode_str = self.control_mode if self.is_active else "DISABLED"
        return f"GNC[{self.sc_id}] | Frame: {self.operating_frame.name} | Mode: {mode_str}"
