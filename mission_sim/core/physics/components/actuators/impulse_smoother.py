# mission_sim/core/physics/components/actuators/impulse_smoother.py
"""
MCPC L2: 冲量平滑器 (Impulse Smoother)
--------------------------------------
铁律 3（L2 架构文档）：物理全局积分步长与推力器最小冲量比特（MIB）存在跨尺度冲突；
严禁为凑 MIB 缩短全局步长。本组件悬挂于物理执行器入口，把高频/低幅指令等效为
低频平均推力（MIB 整数倍脉冲），保证冲量守恒：
    任意时间段内输出冲量 = 指令冲量 − 残留（|残留| < 1 MIB）。
"""

import numpy as np


class ImpulseSmoother:
    """将连续力指令量化为 MIB 整数倍脉冲（以等效平均力输出）。"""

    def __init__(self, mib_ns: float = 0.01, max_thrust_n: float = 1.0):
        """
        Args:
            mib_ns: 最小冲量比特 MIB (N·s)
            max_thrust_n: 单步最大输出力 (N)（与推力器饱和一致）
        """
        self.mib_ns = float(mib_ns)
        self.max_thrust_n = float(max_thrust_n)
        self._pending = np.zeros(3, dtype=np.float64)

    def smooth(self, commanded_force: np.ndarray, dt: float) -> np.ndarray:
        """累积期望冲量并量化输出；返回本步的等效平均力 (N)。"""
        desired = np.asarray(commanded_force, dtype=np.float64) * float(dt)
        self._pending = self._pending + desired

        pending_mag = float(np.linalg.norm(self._pending))
        if pending_mag < self.mib_ns:
            return np.zeros(3, dtype=np.float64)

        n_pulses = int(pending_mag // self.mib_ns)
        max_impulse = self.max_thrust_n * float(dt)
        emitted = n_pulses * self.mib_ns
        if emitted > max_impulse:
            emitted = (max_impulse // self.mib_ns) * self.mib_ns
        if emitted <= 0.0:
            return np.zeros(3, dtype=np.float64)

        out_impulse = self._pending * (emitted / pending_mag)
        self._pending = self._pending - out_impulse
        return out_impulse / float(dt)

    def reset(self) -> None:
        """清空待发冲量（如安全模式/复位）。"""
        self._pending[:] = 0.0

    def __repr__(self) -> str:
        return f"ImpulseSmoother(MIB={self.mib_ns} N·s, MaxF={self.max_thrust_n} N)"
