"""三体场景仿真基类"""
from mission_sim.simulation.base import BaseSimulation
from mission_sim.core.physics.environment import CelestialEnvironment
from mission_sim.core.gnc.ground_station import GroundStation
from mission_sim.core.gnc.gnc_subsystem import GNC_Subsystem
from mission_sim.models.threebody.base import CRTBP

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
        from mission_sim.core.physics.models.gravity_crtbp import Gravity_CRTBP
        self.environment = CelestialEnvironment(
            computation_frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_epoch=0.0
        )
        self.environment.register_force(Gravity_CRTBP())
        # 初始化航天器（带偏差）
        nom0_phys = self.ephemeris.get_interpolated_state(0.0)
        injection = self.config.get("injection_error", np.array([2000,2000,-1000,0.01,-0.01,0.005]))
        true0_phys = nom0_phys + injection
        from mission_sim.core.physics.spacecraft import SpacecraftPointMass
        self.spacecraft = SpacecraftPointMass(
            sc_id="SC", initial_state=true0_phys,
            frame=CoordinateFrame.SUN_EARTH_ROTATING,
            initial_mass=self.config.get("spacecraft_mass", 6200.0)
        )

    def _initialize_information_domain(self):
        frame = CoordinateFrame.SUN_EARTH_ROTATING
        self.ground_station = GroundStation(
            name="DSN", operating_frame=frame,
            pos_noise_std=self.config.get("pos_noise_std", 5.0),
            vel_noise_std=self.config.get("vel_noise_std", 0.005),
            sampling_rate_hz=self.config.get("sampling_rate_hz", 0.1),
            visibility_windows=self.config.get("visibility_windows")
        )
        self.gnc_system = GNC_Subsystem("SC", operating_frame=frame)
        self.gnc_system.load_reference_trajectory(self.ephemeris)
        # 外推器可选
        prop_type = self.config.get("propagator_type")
        if prop_type == "simple":
            from mission_sim.core.gnc.propagator import SimplePropagator
            self.gnc_system.set_propagator(SimplePropagator())
        elif prop_type == "kepler":
            from mission_sim.core.gnc.propagator import KeplerPropagator
            self.gnc_system.set_propagator(KeplerPropagator(self.config.get("propagator_mu", 1.32712440018e20)))