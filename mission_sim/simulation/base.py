"""顶层仿真基类"""
from abc import ABC, abstractmethod
import os
import numpy as np
from mission_sim.utils.logger import HDF5Logger
from mission_sim.utils.visualizer_L1 import L1Visualizer

class BaseSimulation(ABC):
    def __init__(self, config):
        self.config = config
        self.ephemeris = None
        self.environment = None
        self.spacecraft = None
        self.ground_station = None
        self.gnc_system = None
        self.logger = None
        self.k_matrix = None
        # 输出目录等
        ...

    @abstractmethod
    def _generate_nominal_orbit(self): pass

    @abstractmethod
    def _initialize_physical_domain(self): pass

    @abstractmethod
    def _initialize_information_domain(self): pass

    @abstractmethod
    def _design_control_law(self): pass

    def run(self):
        self._generate_nominal_orbit()
        self._initialize_physical_domain()
        self._initialize_information_domain()
        self._design_control_law()
        self._initialize_data_logging()
        self._execute_simulation_loop()
        return self._finalize_simulation()