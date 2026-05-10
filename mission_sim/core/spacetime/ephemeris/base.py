# mission_sim/core/trajectory/ephemeris.py
import numpy as np
from mission_sim.core.spacetime.ids import CoordinateFrame

class Ephemeris:
    """
    全域离散星历表契约类 (Level 1)
    职责：作为预处理阶段与 GNC 阶段之间的数据契约，封装随时间变化的标称状态，
          并提供状态查询接口。
    子类可重载 get_state 以实现不同的获取逻辑（如解析 Kepler 计算、SPICE 直接查询等）。
    """
    def __init__(self, times: list | np.ndarray, states: list | np.ndarray, frame: CoordinateFrame):
        """
        :param times: 离散时间序列 1D array (秒)
        :param states: 对应的状态序列 2D array (N, 6) [x, y, z, vx, vy, vz]
        :param frame: 该星历表所在的目标坐标系 (强契约约束)
        """
        self.times = np.array(times, dtype=np.float64) if times is not None else None
        self.states = np.array(states, dtype=np.float64) if states is not None else None
        
        if not isinstance(frame, CoordinateFrame):
            raise TypeError(f"[Ephemeris] 坐标系字段必须是 CoordinateFrame 枚举，当前为: {type(frame)}")
            
        if self.states is not None:
            if self.states.ndim != 2 or self.states.shape[1] != 6:
                raise ValueError(f"[Ephemeris] states 必须是形状为 (N, 6) 的数组，当前形状: {self.states.shape}")
                
            if len(self.times) != len(self.states):
                raise ValueError(f"[Ephemeris] times 和 states 的长度不匹配: {len(self.times)} != {len(self.states)}")
                
            # 确保时间序列是单调递增的，否则插值器会崩溃
            if len(self.times) > 1 and not np.all(np.diff(self.times) > 0):
                raise ValueError("[Ephemeris] 输入的时间序列必须是严格单调递增的。")

        self.frame = frame

    def get_state(self, t: float) -> np.ndarray:
        """
        获取指定时刻的轨道状态 (供 GNC 模块高频调用)。
        默认实现为线性插值（外推允许）。
        
        :param t: 目标时刻 (s)
        :return: 6x1 的状态向量 [x, y, z, vx, vy, vz]
        """
        if self.times is None or self.states is None:
            raise ValueError("[Ephemeris] 没有可用的时间/状态数据进行插值，子类需重写 get_state。")
        
        n = len(self.times)
        if n == 0:
            raise ValueError("[Ephemeris] 时间序列为空。")
        if n == 1:
            return self.states[0].copy()
        
        # 线性插值（含外推）
        idx = np.searchsorted(self.times, t, side='right') - 1
        idx = np.clip(idx, 0, n - 2)
        t0, t1 = self.times[idx], self.times[idx + 1]
        s0, s1 = self.states[idx], self.states[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        result = s0 + alpha * (s1 - s0)
        
        # 边界告警
        if t < self.times[0] or t > self.times[-1]:
            print(f"⚠️ [Ephemeris Warning] 请求时间 {t:.1f}s 超出星历覆盖范围 "
                  f"[{self.times[0]:.1f}s, {self.times[-1]:.1f}s]。正在进行高风险外推！")
        
        return result

    def get_interpolated_state(self, t: float) -> np.ndarray:
        """
        已弃用：请使用 get_state。
        """
        import warnings
        warnings.warn(
            "get_interpolated_state is deprecated, use get_state instead",
            DeprecationWarning, stacklevel=2
        )
        return self.get_state(t)

    def __repr__(self):
        if self.times is None or len(self.times) == 0:
            return f"Ephemeris(Frame={self.frame.name} | No data)"
        duration_hours = (self.times[-1] - self.times[0]) / 3600.0 if len(self.times) > 1 else 0.0
        return (f"Ephemeris(Frame={self.frame.name} | "
                f"Points={len(self.times)} | "
                f"Duration={duration_hours:.2f}h)")
