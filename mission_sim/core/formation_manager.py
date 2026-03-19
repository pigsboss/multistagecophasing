# mission_sim/core/formation_manager.py
import numpy as np
from mission_sim.core.spacecraft import SpacecraftPointMass
from mission_sim.utils.math_tools import absolute_to_lvlh

class FormationManager:
    """
    L2级 编队协同导航中枢 (Information Domain)
    逻辑自洽核心：必须基于主星(Chief)的实时物理状态计算相对导航。
    """
    def __init__(self, chief: SpacecraftPointMass):
        # 引用主星实体，确保后续读取的是 .state 的实时更新值
        self.chief = chief
        self.deputies = {}

    def add_deputy(self, sc_id: str, spacecraft: SpacecraftPointMass):
        """将从星注册到星间链路网络"""
        if sc_id in self.deputies:
            print(f"[Formation Warning] Deputy '{sc_id}' already exists.")
        self.deputies[sc_id] = spacecraft
        # 修正：统一使用 .id 属性
        print(f"📡 [ISL Network] Deputy '{sc_id}' linked to Chief '{self.chief.id}'.")

    def get_lvlh_relative_state(self, deputy_id: str) -> np.ndarray:
        """
        核心导航解算：
        实现从物理域(Rotating Frame)到信息域(LVLH Frame)的降维投影。
        """
        if deputy_id not in self.deputies:
            raise ValueError(f"Deputy '{deputy_id}' not found in network.")

        deputy = self.deputies[deputy_id]

        relative_state_lvlh = absolute_to_lvlh(
            state_chief=self.chief.state, 
            state_deputy=deputy.state
        )

        return relative_state_lvlh

    def broadcast_all_rel_states(self) -> dict:
        """广播全阵列相对于主星的 LVLH 导航报文"""
        return {d_id: self.get_lvlh_relative_state(d_id) for d_id in self.deputies}