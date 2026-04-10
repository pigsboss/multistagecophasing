# mission_sim/core/trajectory/ephemeris.py
import numpy as np
from scipy.interpolate import CubicSpline
from mission_sim.core.spacetime.ids import CoordinateFrame

class Ephemeris:
    """
    全域离散星历表契约类 (Level 1)
    职责：作为预处理阶段与 GNC 阶段之间的数据契约，封装随时间变化的标称状态，并提供高精度的高阶插值。
    """
    def __init__(self, times: list | np.ndarray, states: list | np.ndarray, frame: CoordinateFrame):
        """
        :param times: 离散时间序列 1D array (秒)
        :param states: 对应的状态序列 2D array (N, 6) [x, y, z, vx, vy, vz]
        :param frame: 该星历表所在的目标坐标系 (强契约约束)
        """
        self.times = np.array(times, dtype=np.float64)
        self.states = np.array(states, dtype=np.float64)
        
        # --- 数据格式防呆校验 ---
        if not isinstance(frame, CoordinateFrame):
            raise TypeError(f"[Ephemeris] 坐标系字段必须是 CoordinateFrame 枚举，当前为: {type(frame)}")
            
        if self.states.ndim != 2 or self.states.shape[1] != 6:
            raise ValueError(f"[Ephemeris] states 必须是形状为 (N, 6) 的数组，当前形状: {self.states.shape}")
            
        if len(self.times) != len(self.states):
            raise ValueError(f"[Ephemeris] times 和 states 的长度不匹配: {len(self.times)} != {len(self.states)}")
            
        # 确保时间序列是单调递增的，否则插值器会崩溃
        if len(self.times) > 1 and not np.all(np.diff(self.times) > 0):
            raise ValueError("[Ephemeris] 输入的时间序列必须是严格单调递增的。")

        self.frame = frame
        
        # --- 构建高精度三次样条插值器 ---
        # 仅当数据点≥2时创建插值器（SPICE模式可能只有1个dummy点）
        if len(self.times) >= 2:
            # bc_type='natural' 意味着边界的二阶导数为 0 (自然边界条件)
            # CubicSpline 会自动沿着 axis=0 对 6 个维度分别生成插值函数
            self._interpolator = CubicSpline(self.times, self.states, bc_type='natural')
        else:
            # 对于单点（SPICE模式占位），禁用插值器
            self._interpolator = None

    def get_interpolated_state(self, t: float) -> np.ndarray:
        """
        获取指定时刻的标称参考状态 (由 GNC 模块高频调用)。
        
        :param t: 目标时刻 (s)
        :return: 6x1 的状态向量 [x, y, z, vx, vy, vz]
        """
        if self._interpolator is None:
            # 如果只有单点数据（如SPICE模式占位），返回该点状态
            if len(self.times) == 1:
                return self.states[0].copy()
            raise ValueError("[Ephemeris] 插值器未初始化且没有可用的状态数据")
            
        # 边界告警防护：真实工程中，星历外推(Extrapolation)是高风险操作
        if t < self.times[0] or t > self.times[-1]:
            # 这里允许执行外推运算，但抛出控制台警告，提醒总体规划人员星历长度不足
            print(f"⚠️ [Ephemeris Warning] 请求时间 {t:.1f}s 超出星历覆盖范围 "
                  f"[{self.times[0]:.1f}s, {self.times[-1]:.1f}s]。正在进行高风险外推！")
            
        # 调用底层插值器，返回形状为 (6,) 的 numpy array
        return self._interpolator(t)

    def __repr__(self):
        duration_hours = (self.times[-1] - self.times[0]) / 3600.0 if len(self.times) > 1 else 0.0
        return (f"Ephemeris(Frame={self.frame.name} | "
                f"Points={len(self.times)} | "
                f"Duration={duration_hours:.2f}h)")
