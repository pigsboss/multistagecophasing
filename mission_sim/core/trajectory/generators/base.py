# mission_sim/core/trajectory/generators/base.py
"""标称轨道生成器抽象基类"""

from abc import ABC, abstractmethod
from mission_sim.core.trajectory.ephemeris import Ephemeris


class BaseTrajectoryGenerator(ABC):
    """
    标称轨道生成器抽象基类。
    所有具体生成器必须实现 generate 方法，返回 Ephemeris 对象。
    """

    @abstractmethod
    def generate(self, config: dict) -> Ephemeris:
        """
        生成标称轨道星历。

        Args:
            config: 生成器特定配置字典

        Returns:
            Ephemeris: 标称轨道星历表
        """
        pass