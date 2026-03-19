# mission_sim/core/gnc_subsystem.py
import numpy as np
from mission_sim.core.types import CoordinateFrame, Telecommand

class GNC_Subsystem:
    """
    制导、导航与控制 (GNC) 子系统 (Information Domain)
    负责在特定参考系下计算控制律，实现绝对轨道维持或相对编队保持。
    """
    def __init__(self, sc_id: str, operating_frame: CoordinateFrame):
        self.sc_id = sc_id
        self.operating_frame = operating_frame
        
        # 逻辑自洽：初始化为全零向量 (6x1)，确保计算安全性
        self.current_nav_state = np.zeros(6)
        self.target_state = np.zeros(6)
        
        # 遥测记录
        self.last_control_force = np.zeros(3)

    def process_telecommand(self, packet: Telecommand):
        """
        处理来自地面站或编队管理器的指令。
        逻辑自洽：指令的参考系必须与 GNC 的运行参考系一致。
        """
        if packet.frame != self.operating_frame:
            print(f"⚠️  [{self.sc_id} GNC] Frame Mismatch! CMD: {packet.frame.name} | GNC: {self.operating_frame.name}")
            return False

        # 支持 L1(绝对) 和 L2(相对) 指令集
        valid_cmds = ["ORBIT_MAINTENANCE", "FORMATION_KEEPING", "ARRAY_RECONFIGURATION"]
        
        if packet.cmd_type in valid_cmds:
            self.target_state = np.copy(packet.target_state)
            print(f"✅ [{self.sc_id} GNC] Target Locked: {self.target_state[0:3]}m in {self.operating_frame.name}")
            return True
        else:
            print(f"❌ [{self.sc_id} GNC] Unknown Command Type: {packet.cmd_type}")
            return False

    def update_navigation(self, obs_state: np.ndarray, frame: CoordinateFrame):
        """更新导航滤波器输出的状态"""
        if frame != self.operating_frame:
            raise ValueError(f"Navigation frame {frame} does not match GNC operating frame.")
        self.current_nav_state = np.copy(obs_state)

    def compute_control_force(self, K_matrix: np.ndarray):
        """
        基于 LQR 增益矩阵计算控制力。
        公式：u = -K * (x - x_target)
        """
        # 误差计算：这是消除 550m 静差的数学核心
        # 如果是从星，这里的 current_nav_state 是相对位移，target_state 是目标相对位移
        error = self.current_nav_state - self.target_state
        
        # 计算控制量 (3x1 推力矢量)
        u = -K_matrix @ error
        
        self.last_control_force = np.copy(u)
        
        # 返回推力及其所在的参考系标签
        return u, self.operating_frame

    def get_status(self):
        """返回 GNC 当前健康状态"""
        return {
            "sc_id": self.sc_id,
            "frame": self.operating_frame.name,
            "error_magnitude": np.linalg.norm(self.current_nav_state[0:3] - self.target_state[0:3])
        }

    def __repr__(self):
        """返回开发者友好的对象描述"""
        return (f"GNC_Subsystem(id='{self.sc_id}', "
                f"frame={self.operating_frame.name}, "
                f"target={self.target_state[0:3]})")

    def __str__(self):
        """返回用户友好的状态摘要"""
        pos_err = np.linalg.norm(self.current_nav_state[0:3] - self.target_state[0:3])
        return (f"[{self.sc_id} GNC] Frame: {self.operating_frame.name} | "
                f"Target: {self.target_state[0:3]} | "
                f"PosError: {pos_err:.2f}m")