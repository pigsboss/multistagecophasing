# mission_sim/core/gnc/propagator.py
"""
盲区外推器模块 (Propagator)
在测控盲区时，利用简化的动力学模型外推导航状态，避免直接沿用旧值。

设计原则:
    1. 外推器仅依赖当前状态和简单的动力学模型，不引入复杂环境（如摄动）。
    2. 外推器应与 GNC 工作在同一坐标系，坐标系校验由调用方负责。
    3. 提供抽象基类，便于后续扩展更精确的外推模型（如 CRTBP、J2 等）。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Propagator(ABC):
    """
    盲区外推器抽象基类。
    定义外推器的统一接口，所有具体外推器必须实现 propagate 方法。
    """

    @abstractmethod
    def propagate(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        根据当前状态和时间步长外推状态。

        Args:
            state: 当前状态向量 [x, y, z, vx, vy, vz] (长度6)
            dt: 外推时间步长 (s)

        Returns:
            np.ndarray: 外推后的状态向量 [x, y, z, vx, vy, vz] (长度6)
        """
        pass


class SimplePropagator(Propagator):
    """
    简单线性外推器。
    假设无外力作用，速度恒定，位置线性变化。

    适用于极短盲区或对精度要求不高的场景。
    """

    def propagate(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        线性外推: 新位置 = 当前位置 + 当前速度 * dt
                  速度保持不变

        Args:
            state: 当前状态 [x, y, z, vx, vy, vz]
            dt: 时间步长 (s)

        Returns:
            np.ndarray: 外推后的状态
        """
        pos = state[:3]
        vel = state[3:6]

        new_pos = pos + vel * dt
        new_state = np.concatenate([new_pos, vel])

        return new_state.astype(np.float64)


class KeplerPropagator(Propagator):
    """
    二体开普勒外推器。
    假设只受中心天体引力，基于当前状态计算下一时刻的状态。
    适用于地心惯性系 (J2000_ECI) 或日心惯性系等中心力场占主导的场景。

    使用四阶 Runge-Kutta 积分器进行外推，确保一定精度。
    """

    def __init__(self, mu: float):
        """
        初始化二体外推器。

        Args:
            mu: 中心天体引力常数 (m³/s²)
        """
        self.mu = mu

    def propagate(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        使用 RK4 积分二体运动方程，外推状态。

        Args:
            state: 当前状态 [x, y, z, vx, vy, vz]
            dt: 时间步长 (s)

        Returns:
            np.ndarray: 外推后的状态
        """
        # 定义二体运动方程导数
        def derivatives(s: np.ndarray) -> np.ndarray:
            pos = s[:3]
            vel = s[3:6]
            r = np.linalg.norm(pos)
            # 防止除零
            if r < 1e-10:
                r = 1e-10
            acc = -self.mu * pos / (r ** 3)
            return np.concatenate([vel, acc])

        # RK4 积分
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * dt * k1)
        k3 = derivatives(state + 0.5 * dt * k2)
        k4 = derivatives(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_state.astype(np.float64)


# 可选：CRTBP 外推器（日地旋转系），后续 L2 可能需要，这里暂不实现
# class CRTBPPropagator(Propagator):
#     pass