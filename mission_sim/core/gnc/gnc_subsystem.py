# mission_sim/core/gnc/gnc_subsystem.py
"""
制导、导航与控制 (GNC) 子系统 - 增强版（集成外推器）
"""

import numpy as np
from typing import Tuple, Optional

from mission_sim.core.types import CoordinateFrame, Telecommand
from mission_sim.core.trajectory.ephemeris import Ephemeris
from mission_sim.core.gnc.propagator import Propagator


class GNC_Subsystem:
    """
    制导、导航与控制 (GNC) 子系统 (Level 1) - 增强版
    职责：接收带有坐标系契约的导航状态，读取动态星历标称轨迹，
          计算追踪误差，并基于控制律输出补偿推力。
          支持在测控盲区时使用外推器更新导航状态。
    """

    def __init__(self, sc_id: str, operating_frame: CoordinateFrame, verbose: bool = True):
        """
        初始化 GNC 子系统

        Args:
            sc_id: 航天器标识符
            operating_frame: GNC 算法的工作基准坐标系 (强契约)
            verbose: 是否输出详细信息
        """
        self.sc_id = sc_id
        self.operating_frame = operating_frame
        self.verbose = verbose

        # 导航滤波器输出状态 (初始为空)
        self.current_nav_state = np.zeros(6, dtype=np.float64)

        # 标称轨道星历表
        self.ref_ephemeris: Optional[Ephemeris] = None

        # 遥测记录暂存区
        self.last_control_force = np.zeros(3, dtype=np.float64)
        self.last_tracking_error = np.zeros(6, dtype=np.float64)
        self.last_target_state = np.zeros(6, dtype=np.float64)

        # 盲区外推器（初始为空）
        self._propagator: Optional[Propagator] = None

        # 调试和统计信息
        self.total_control_calls = 0
        self.force_shape_warnings = 0

        if self.verbose:
            print(f"✅ [{self.sc_id} GNC] 初始化完成，工作坐标系: {self.operating_frame.name}")

    def set_propagator(self, propagator: Propagator):
        """
        注入盲区外推器。

        Args:
            propagator: 外推器实例（继承自 Propagator）
        """
        self._propagator = propagator
        if self.verbose:
            print(f"✅ [{self.sc_id} GNC] 已注入外推器: {propagator.__class__.__name__}")

    def load_reference_trajectory(self, eph: Ephemeris) -> None:
        """
        加载由预处理阶段生成的标称星历

        Args:
            eph: 标称轨道星历对象

        Raises:
            ValueError: 如果星历坐标系与GNC工作坐标系不匹配
        """
        if eph.frame != self.operating_frame:
            raise ValueError(
                f"[{self.sc_id} GNC 崩溃] 标称星历坐标系不匹配！\n"
                f"  GNC 运行在: {self.operating_frame.name}\n"
                f"  星历基准为: {eph.frame.name}\n"
                f"  请检查轨道生成器的输出配置。"
            )

        self.ref_ephemeris = eph
        if self.verbose:
            duration_hours = (eph.times[-1] - eph.times[0]) / 3600.0
            print(f"✅ [{self.sc_id} GNC] 成功锁定动态基准星历")
            print(f"   星历时长: {duration_hours:.1f} 小时")
            print(f"   星历点数: {len(eph.times)}")
            print(f"   时间范围: {eph.times[0]:.1f} 到 {eph.times[-1]:.1f} 秒")

    def update_navigation(self, obs_state: Optional[np.ndarray], frame: CoordinateFrame, dt: float = 0.0) -> None:
        """
        更新来自 GroundStation 的导航状态。
        如果 obs_state 为 None（盲区）且存在外推器，则使用外推器更新状态。

        Args:
            obs_state: 观测状态向量 [x, y, z, vx, vy, vz] 或 None
            frame: 观测状态所在的坐标系
            dt: 时间步长（秒），仅当盲区外推时使用

        Raises:
            ValueError: 如果观测状态坐标系与GNC工作坐标系不匹配
            TypeError: 如果观测状态不是numpy数组且不为 None
            ValueError: 如果观测状态形状不正确
        """
        # 盲区处理：无观测且存在外推器
        if obs_state is None and self._propagator is not None:
            # 外推导航状态
            if dt > 0:
                self.current_nav_state = self._propagator.propagate(self.current_nav_state, dt)
            return

        # 有观测时正常更新
        if obs_state is None:
            # 既无观测也无外推器，保持原状态（警告仅一次）
            if not hasattr(self, "_warned_no_propagator"):
                self._warned_no_propagator = True
                if self.verbose:
                    print(f"⚠️ [{self.sc_id} GNC] 盲区且无外推器，导航状态冻结")
            return

        # 类型校验
        if not isinstance(obs_state, np.ndarray):
            raise TypeError(f"[{self.sc_id} GNC] 观测状态必须是 numpy 数组，当前类型: {type(obs_state)}")

        # 形状校验
        if obs_state.shape != (6,):
            raise ValueError(f"[{self.sc_id} GNC] 观测状态必须是形状为 (6,) 的向量，当前形状: {obs_state.shape}")

        # 坐标系校验
        if frame != self.operating_frame:
            raise ValueError(
                f"[{self.sc_id} GNC 拒收] 导航状态坐标系不匹配！\n"
                f"  期望坐标系: {self.operating_frame.name}\n"
                f"  实际坐标系: {frame.name}\n"
                f"  请检查地面站配置或坐标系转换逻辑。"
            )

        # 更新导航状态
        self.current_nav_state = np.copy(obs_state.astype(np.float64))

        # 调试信息（仅当 verbose 开启）
        if self.verbose and self.total_control_calls % 1000 == 0:
            pos_norm = np.linalg.norm(obs_state[0:3])
            vel_norm = np.linalg.norm(obs_state[3:6])
            print(f"  [{self.sc_id} GNC] 导航更新: 位置={pos_norm:.1f}m, 速度={vel_norm:.4f}m/s")

    def compute_control_force(self, epoch: float, K_matrix: np.ndarray) -> Tuple[np.ndarray, CoordinateFrame]:
        """
        计算站位维持的补偿推力。

        核心逻辑: e(t) = X_nav(t) - X_nominal(t)
                  u(t) = -K * e(t)

        Args:
            epoch: 当前仿真历元时间 (s)
            K_matrix: 最优控制反馈增益矩阵，期望形状 (3, 6)

        Returns:
            Tuple[np.ndarray, CoordinateFrame]:
                - 推力向量 [Fx, Fy, Fz] (形状始终为 (3,))
                - 推力所在的坐标系标签
        """
        self.total_control_calls += 1

        if self.ref_ephemeris is None:
            raise RuntimeError(f"[{self.sc_id} GNC] 未加载标称星历，无法计算动态追踪偏差！")

        # 动态获取当前时刻的绝对标称目标状态
        try:
            target_state = self.ref_ephemeris.get_interpolated_state(epoch)
            self.last_target_state = np.copy(target_state)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [{self.sc_id} GNC] 星历插值失败: {e}")
            target_state = self.last_target_state

        # 计算追踪误差
        error = self.current_nav_state - target_state
        self.last_tracking_error = np.copy(error)

        # 验证K矩阵形状
        K_matrix = self._validate_and_fix_K_matrix(K_matrix)

        # 线性反馈控制律
        try:
            raw_force = -K_matrix @ error
            control_force = self._standardize_control_force(raw_force)
            self.last_control_force = np.copy(control_force)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [{self.sc_id} GNC] 控制力计算失败: {e}")
            control_force = np.zeros(3, dtype=np.float64)
            self.last_control_force = control_force

        # 调试输出（仅当 verbose 开启）
        if self.verbose and self.total_control_calls % 1000 == 0:
            err_pos = np.linalg.norm(error[0:3])
            err_vel = np.linalg.norm(error[3:6]) * 1000
            force_norm = np.linalg.norm(control_force)
            print(f"  [{self.sc_id} GNC] 控制计算: 位置误差={err_pos:.2f}m, "
                  f"速度误差={err_vel:.2f}mm/s, 控制力={force_norm:.4f}N")

        return control_force, self.operating_frame

    def _validate_and_fix_K_matrix(self, K_matrix: np.ndarray) -> np.ndarray:
        """
        验证并修复K矩阵形状，确保为 (3,6)。
        【L1 稳健性升级】：移除静默妥协，遇到无法安全修复的维度错误时立即抛出异常 (Fail-Fast)。
        """
        if not isinstance(K_matrix, np.ndarray):
            K_matrix = np.array(K_matrix, dtype=np.float64)

        expected_shape = (3, 6)
        if K_matrix.shape == expected_shape:
            return K_matrix

        if self.verbose:
            print(f"⚠️ [{self.sc_id} GNC] K矩阵形状不匹配: 当前 {K_matrix.shape}，期望 {expected_shape}。尝试进行安全重塑...")

        # 尝试安全的形状修正逻辑 (仅处理明确可推导的降维/展平情况)
        try:
            if K_matrix.shape == (1, 6) or K_matrix.shape == (6,):
                # 如果是 1D 数组，假定 3 个轴使用相同的反馈增益
                if K_matrix.shape == (6,):
                    K_matrix = K_matrix.reshape(1, 6)
                K_matrix = np.tile(K_matrix, (3, 1))
                if self.verbose:
                    print(f"  [{self.sc_id} GNC] 已将 1D 增益广播为 3D: {K_matrix.shape}")
                return K_matrix

            # 如果是纯粹的展平数组且元素数量匹配，则重塑
            if K_matrix.size == 18:
                K_matrix = K_matrix.reshape(expected_shape)
                if self.verbose:
                    print(f"  [{self.sc_id} GNC] 已重塑为标准维度: {K_matrix.shape}")
                return K_matrix

        except Exception as e:
            # 捕获任何在重塑过程中发生的意外错误
            raise ValueError(f"[{self.sc_id} GNC 致命错误] K 矩阵重塑失败: {e}")

        # 如果走到这里，说明矩阵维度完全不可控，必须 Fail-Fast
        raise ValueError(
            f"[{self.sc_id} GNC 致命错误] 拒绝执行控制指令！\n"
            f"  传入的 K 矩阵形状 {K_matrix.shape} 无法安全转换为所需的 {expected_shape}。\n"
            f"  请检查 _design_control_law 中的 LQR/反馈逻辑设计。"
        )

    def _standardize_control_force(self, raw_force) -> np.ndarray:
        """标准化控制力输入，确保输出为 (3,) 数组"""
        if isinstance(raw_force, (int, float, np.number)):
            return np.array([float(raw_force), 0.0, 0.0], dtype=np.float64)
        elif isinstance(raw_force, np.ndarray):
            if raw_force.shape == ():
                return np.array([float(raw_force), 0.0, 0.0], dtype=np.float64)
            elif raw_force.shape == (1,):
                return np.array([float(raw_force[0]), 0.0, 0.0], dtype=np.float64)
            elif raw_force.shape == (3,):
                return raw_force.astype(np.float64)
            elif len(raw_force) >= 3:
                return raw_force[:3].astype(np.float64)
            else:
                try:
                    if raw_force.size == 3:
                        return raw_force.reshape(3).astype(np.float64)
                except:
                    pass
        if self.verbose and self.force_shape_warnings < 5:
            print(f"⚠️ [{self.sc_id} GNC] 控制力格式异常: {type(raw_force)}")
            self.force_shape_warnings += 1
        return np.zeros(3, dtype=np.float64)

    def get_tracking_error(self, epoch: float) -> np.ndarray:
        """获取当前时刻的追踪误差（不计算控制力）"""
        if self.ref_ephemeris is None:
            return np.zeros(6, dtype=np.float64)
        try:
            target_state = self.ref_ephemeris.get_interpolated_state(epoch)
            return self.current_nav_state - target_state
        except:
            return self.last_tracking_error

    def get_performance_metrics(self) -> dict:
        """获取GNC性能指标"""
        return {
            "position_error_m": float(np.linalg.norm(self.last_tracking_error[0:3])),
            "velocity_error_mps": float(np.linalg.norm(self.last_tracking_error[3:6])),
            "control_force_norm": float(np.linalg.norm(self.last_control_force)),
            "total_control_calls": self.total_control_calls,
            "force_shape_warnings": self.force_shape_warnings
        }

    def reset(self) -> None:
        """重置GNC状态"""
        self.current_nav_state = np.zeros(6, dtype=np.float64)
        self.last_control_force = np.zeros(3, dtype=np.float64)
        self.last_tracking_error = np.zeros(6, dtype=np.float64)
        self.total_control_calls = 0
        self.force_shape_warnings = 0
        if self.verbose:
            print(f"[{self.sc_id} GNC] 已重置")

    def __repr__(self) -> str:
        metrics = self.get_performance_metrics()
        return (f"GNC[{self.sc_id}] | Frame: {self.operating_frame.name} | "
                f"PosErr: {metrics['position_error_m']:.2f}m | "
                f"VelErr: {metrics['velocity_error_mps']*1000:.2f}mm/s | "
                f"Calls: {metrics['total_control_calls']}")


# 兼容性别名
GNCSubsystem = GNC_Subsystem