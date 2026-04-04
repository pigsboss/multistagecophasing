"""三体场景仿真基类"""
import numpy as np
from mission_sim.simulation.base import BaseSimulation
from mission_sim.core.spacetime.ids import CoordinateFrame
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.cyber.platform_gnc.ground_station import GroundStation
from mission_sim.core.cyber.platform_gnc.gnc_subsystem import GNCSubsystem
from mission_sim.core.cyber.models.threebody.base import CRTBP
from mission_sim.core.physics.spacecraft import SpacecraftPointMass


class ThreeBodyBaseSimulation(BaseSimulation):
    def __init__(self, config):
        super().__init__(config)
        # 从配置读取 μ, L, ω
        mu = config.get("mu", 3.00348e-6)
        L = config.get("L", 1.495978707e11)
        omega = config.get("omega", 1.990986e-7)
        self.crtbp = CRTBP(mu, L, omega)

    def _initialize_physical_domain(self):
        # 创建环境，注册 CRTBP 力模型
        from mission_sim.core.physics.models.gravity import GravityCRTBP
        self.environment = CelestialEnvironment(
            computation_frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_epoch=0.0,
            verbose=self.verbose
        )
        self.environment.register_force(GravityCRTBP())

        # 获取配置中的注入误差
        injection = self.config.get("injection_error")
        # 如果是字符串（来自命令行），尝试解析
        if isinstance(injection, str):
            try:
                import json
                # 将 "[0,0,0,0,0,0]" 转换为 list 再转为 array
                injection = np.array(json.loads(injection))
            except:
                print("⚠️ 注入误差解析失败，回退到默认值")
                injection = np.array([2000, 2000, -1000, 0.01, -0.01, 0.005])
        
        # 如果配置里没有（或为 None），使用默认值
        if injection is None:
            injection = np.array([2000, 2000, -1000, 0.01, -0.01, 0.005])
        nom0_phys = self.ephemeris.get_interpolated_state(0.0)
        true0_phys = nom0_phys + injection
        # ... 后续初始化 ...
        
        self.spacecraft = SpacecraftPointMass(
            sc_id="SC",
            initial_state=true0_phys,
            frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_mass=self.config.get("spacecraft_mass", 6200.0)
        )

    def _initialize_information_domain(self):
        frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.ground_station = GroundStation(
            name="DSN",
            operating_frame=frame,
            pos_noise_std=self.config.get("pos_noise_std", 5.0),
            vel_noise_std=self.config.get("vel_noise_std", 0.005),
            sampling_rate_hz=self.config.get("sampling_rate_hz", 0.1),
            visibility_windows=self.config.get("visibility_windows")
        )
        self.gnc_system = GNCSubsystem("SC", operating_frame=frame, verbose=self.verbose)
        self.gnc_system.load_reference_trajectory(self.ephemeris)

        # 用航天器的初始状态初始化导航状态（避免初始巨大误差）
        self.gnc_system.current_nav_state = self.spacecraft.state.copy()

        # 外推器可选
        prop_type = self.config.get("propagator_type")
        if prop_type == "simple":
            from mission_sim.core.cyber.platform_gnc.propagator import SimplePropagator
            self.gnc_system.set_propagator(SimplePropagator())
        elif prop_type == "kepler":
            from mission_sim.core.cyber.platform_gnc.propagator import KeplerPropagator
            self.gnc_system.set_propagator(
                KeplerPropagator(self.config.get("propagator_mu", 1.32712440018e20))
            )
