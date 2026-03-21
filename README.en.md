# MCPC Framework

**Multi-stage Co-Phasing Control** – An industrial‑grade spacecraft dynamics and control simulation framework for distributed spaceborne arrays.  
MCPC supports nested control from kilometer‑level orbits to nanometer‑level wavefront phasing, serving as the digital backbone for space distributed synthetic aperture interferometry missions (e.g., the *Miyin* mission).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

<div align="center">
  <p>English | <a href="./README.md">中文</a></p>
</div>

---

## 🎯 Vision & Multi‑level Control Objectives

MCPC adopts a progressive model‑fidelity strategy that gradually approaches real‑world complexity:

| Level  | Focus                     | Engineering Goal                                             | Status                              |
| ------ | ------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| **L1** | Baseline calibration      | Absolute orbit maintenance, dominant perturbations, ideal thrust, ground‑based tracking | ✅ **Completed** (Sun‑Earth L2 Halo) |
| **L2** | Cooperative efficiency    | Relative motion, formation reconfiguration, inter‑satellite links | 🔄 In progress                       |
| **L3** | Principle validation      | Platform‑payload multi‑body coupling, error budget allocation | 📋 Planned                           |
| **L4** | Engineering qualification | Full 6‑DOF, hardware nonlinearities, hardware‑in‑the‑loop    | 📋 Planned                           |
| **L5** | Digital twin              | Flexible multi‑body, extreme environments, in‑orbit identification | 📋 Planned                           |

Each level is strictly defined with its own dynamics, measurement, actuation and control assumptions, enabling you to select the right fidelity for your phase of development.

---

## ✨ Key Architectural Features

- **Orthogonal decomposition**  
  Missions are classified by their dynamical nature (**two‑body** vs. **three‑body**) and fidelity level (**L1**–**L5**). This ensures maximum code reuse while maintaining precise model granularity.

- **Strong coordinate frame contract**  
  Every piece of data exchanged across modules carries a `CoordinateFrame` tag (e.g., `J2000_ECI`, `SUN_EARTH_ROTATING`, `LVLH`). Inconsistent frames raise immediate exceptions, eliminating reference‑frame bugs at the source.

- **Physical / Information domain decoupling**  
  - **Physical domain** (`core/physics`) simulates the objective universe: spacecraft only integrate forces, never read control outputs directly.  
  - **Information domain** (`core/gnc`) mimics onboard computers: navigation filtering and control laws run independently, affecting the physical domain through contract‑enforced commands.

- **Real‑world units in optimal control**  
  LQR controllers are designed using true SI units (e.g., Sun‑Earth angular velocity ~2×10⁻⁷ rad/s), avoiding numerical issues from non‑dimensionalization.

- **High‑performance data pipeline**  
  `HDF5Logger` uses memory buffering and compression to prevent I/O bottlenecks and memory overflow, even for month‑long simulations with high sampling rates.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/mcpc.git
cd mcpc
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a sample simulation (Sun‑Earth L2 Halo orbit, L1 level)

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 1 --time_step 60
```

Or use a YAML configuration file (e.g., `config/halo_example.yaml`):

```yaml
mission_name: "Halo L1 Test"
simulation_days: 1
time_step: 60.0
Az: 0.05
```

```bash
python run.py --scene sun_earth_l2 --level 1 --config config/halo_example.yaml
```

### 4. View results

Output is written to the `data/` directory:

- **`*.h5`** – HDF5 file with epochs, nominal/true/nav states, tracking errors, control forces, and accumulated ΔV.
- **`fuel_bill_*.csv`** – Station‑keeping fuel summary.
- **`*_trajectory.png`**, **`*_errors.png`**, **`*_control.png`** – Plots (if `enable_visualization` is `True`).

You can also generate a complete HTML report:

```bash
python visualize.py data/simulation.h5 --report
```

---

## 📂 Project Structure

```
mission_sim/
├── core/                     # Core domain models
│   ├── dynamics/             # Equations of motion (two‑body / three‑body)
│   ├── physics/              # Physical domain (environment, spacecraft, force models)
│   ├── gnc/                  # Information domain (GNC, ground station, propagators)
│   ├── trajectory/           # Ephemeris & orbit generators
│   └── types.py              # Global types (CoordinateFrame, Telecommand)
├── simulation/               # Simulation controllers (by scenario)
│   ├── base.py               # Abstract base class (template method)
│   ├── threebody/            # Three‑body scenarios (Sun‑Earth L2, etc.)
│   └── twobody/              # Two‑body scenarios (LEO, GEO, etc.)
├── utils/                    # Infrastructure
│   ├── logger.py             # HDF5Logger (buffered, compressed)
│   ├── math_tools.py         # LQR, LVLH, orbital element conversions
│   ├── differential_correction.py
│   └── visualizer_*.py
├── tests/                    # Unit and integration tests
├── analysis/                 # Post‑processing scripts (Monte Carlo, fuel analysis)
├── config/                   # Example YAML configurations
├── run.py                    # Unified simulation entry point
├── visualize.py              # Data visualization tool
└── requirements.txt
```

---

## 🛠️ Extending the Framework

### Adding a new scenario (e.g., LEO)

1. Create a new file `simulation/twobody/leo.py`.
2. Inherit from `BaseSimulation` (or `TwoBodyBaseSimulation` when available).
3. Implement the abstract methods:  
   - `_generate_nominal_orbit()` – use `KeplerianGenerator` or `J2KeplerianGenerator`.  
   - `_initialize_physical_domain()` – register appropriate force models (`J2Gravity`, `AtmosphericDrag`).  
   - `_initialize_information_domain()` – create `GroundStation` and `GNC_Subsystem`.  
   - `_design_control_law()` – compute feedback gain matrix.
4. Register the scenario in `run.py` under `SCENE_MODULE_MAP`.

### Adding a new force model

1. Create a class in `core/physics/models/` that implements `IForceModel`.
2. Implement `compute_accel(state, epoch)`.
3. Register it with `CelestialEnvironment` in your simulation's `_initialize_physical_domain()`.

### Adding a new fidelity level

Within an existing scenario folder, create a new simulation class (e.g., `SunEarthL2L2Simulation`) that inherits from the base scenario class and overrides the relevant methods (e.g., to add relative dynamics for L2).

---

## 📊 Analysis Tools

The `analysis/` folder contains scripts for deeper investigation:

- **Control robustness Monte Carlo** – `control_robustness_analysis.py`  
  Vary initial errors, measurement noise, control gains, and collect statistics on final error, ΔV consumption, convergence time, etc.

- **Fuel consumption analysis** – `fuel_analysis.py`  
  Scan over orbit types, control gains, and blind intervals to evaluate fuel budgets.

Example usage:

```bash
cd analysis
python control_robustness_analysis.py
```

---

## 🤝 Contributing

We welcome contributions! Please follow the [PEP 8](https://peps.python.org/pep-0008/) style guide and add unit tests for new features.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

---

**MCPC – From orbit to wavelength, progressively closer to reality.**