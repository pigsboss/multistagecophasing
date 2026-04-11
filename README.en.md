# 🌌 MCPC: Multi-stage Co-Phasing Control Simulation Framework

MCPC is a digital twin and high-fidelity simulation framework engineered for space observatories and large-scale distributed spacecraft swarms. It is designed to evaluate Navigation, Guidance, and Control (GNC) performance in multi-body cooperative missions across deep space and near-Earth orbits.

Adhering strictly to aerospace Systems Engineering paradigms, MCPC utilizes a decoupled architecture of **"3 Core Domains (Spacetime, Physics, Cyber) + 2 Support Domains"**, ensuring a physical and informational separation between the objective laws of the physical world and the subjective intelligence of the cybernetic space.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

<div align="center">
  <p>English | <a href="./README.md">中文</a></p>
</div>

---

## 🎯 Mission Statement

Delivering full-system simulations and digital twins from Level 1 to Level 5:

  * **[L1] Foundation**: Constructing high-fidelity single-spacecraft dynamics (SRP, high-degree gravity, multi-body perturbations) and achieving component-level closed-loop control.
  * **[L2] Formation**: Simulating multi-satellite relative navigation and precise configuration maintenance.
  * **[L3] Payload**: Synergizing platform large-angle maneuvers with the payload's micro/nano-meter displacement control.
  * **[L4] Flexibility**: Introducing low-frequency vibration coupling models for large solar arrays and antennas.
  * **[L5] Evaluation**: Generating system-level performance reports based on Monte Carlo shooting and end-of-life analysis.

-----

## 🏛️ Architecture & Directory Map

To ensure rigorous code structures, MCPC divides core logic into independent domain models. All cross-domain interactions MUST pass through data packets defined in the `ids.py` (Interface Definition Specification):

```text
mcpc/
├── run.py                          # [Entry] Main simulation runner
├── visualize.py                    # [Entry] Data visualization and reporting
│
├── mission_sim/                    # ================= Core Package =================
│   ├── config/                     # YAML configuration files
│   │
│   ├── core/                       # 🌟 [Foreground]: System Core Logic
│   │   ├── spacetime/              # 🌌 [Spacetime] Absolute base and reference frames
│   │   │   ├── ids.py              # 📜 Global Contracts: CoordinateFrame, Telecommand, FormationState
│   │   │   ├── ephemeris/          # Ephemeris Engine: Absolute celestial truths
│   │   │   └── generators/         # Reference Generators: Halo, Keplerian, etc.
│   │   │
│   │   ├── physics/                # 🪐 [Physics] Spacecraft forces and objective laws
│   │   │   ├── ids.py              # 📜 Physics Code: Constants, Units, Health Status
│   │   │   ├── environment.py      # Environment Factory: Gravity and SRP integration
│   │   │   ├── spacecraft.py       # System Entity: Mass and force integration interface
│   │   │   └── components/         # Component Models (Deadbands, Friction, Noise, NO algorithms)
│   │   │       ├── actuators/      # Actuator units (Thrusters, Reaction Wheels)
│   │   │       ├── sensors/        # Sensor units (Star Trackers, ISL hardware)
│   │   │       └── mechanisms/     # Mechanism units (Fast Steering Mirrors, Delay Lines)
│   │   │
│   │   └── cyber/                  # 🧠 [Cyber] Subjective Intelligence (Sensing, Comm, Compute)
│   │       ├── ids.py              # 📜 Cyber Code: State machines, network protocols
│   │       ├── models/             # Cognitive Models: Predictive math (CW, STM)
│   │       ├── networks/           # Network Protocols: ISL data flow and routing delays
│   │       └── platform_gnc/       # GNC Brain: Filters, Control Laws, Mode Logic
│   │
│   ├── simulation/                 # 🎬 [Background]: Orchestration & Clock
│   │   ├── base.py                 # Main loop and event-driven engine
│   │   ├── threebody/              # Deep space scenario assembly (L2 Halo, etc.)
│   │   └── twobody/                # Near-Earth scenario assembly (LEO/GEO, etc.)
│   │
│   ├── analysis/                   # ⚖️ [Background]: Engineering Referee
│   │   └── fuel_analysis.py        # Offline performance and Delta-V billing
│   │
│   └── utils/                      # 🛠️ [Infrastructure Layer]
│       ├── math_tools.py           # Core Math: Rotations, LQR solvers
│       └── logger.py               # HDF5 high-frequency data bus
│
└── tests/                          # 🛡️ Contract-driven unit test suite
```

**Architecture Principles:**
- **Spacetime Domain**: Provides the absolute spacetime foundation, including celestial ephemeris, coordinate frames, and reference orbit generation. Interactions between natural celestial bodies (e.g., Earth orbiting the Sun) are implicitly provided through ephemeris.
- **Physics Domain**: Responsible for computing all forces acting on the spacecraft (gravitational, solar radiation pressure, atmospheric drag, etc.), using celestial positions provided by the Spacetime domain as input. Does not compute interactions between natural celestial bodies.
- **Cyber Domain**: Implements spacecraft perception, communication, and control intelligence.

-----

## ⚙️ Quick Start

### 1\. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/multistagecophasing.git
cd multistagecophasing

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r mission_sim/requirements.txt
```

### 2\. Run First Simulation

For example, to run an L1 Sun-Earth L2 Halo station-keeping simulation based on the CRTBP model:

```bash
python run.py --scene sun_earth_l2 --level 1 --simulation_days 30
```

### 3\. Visualization

Upon completion, full-lifecycle states are recorded into HDF5 files. Generate charts using:

```bash
python visualize.py --input data/logs/simulation_xxx.h5
```

-----

## 🛡️ Test-Driven Engineering

MCPC adheres to strict verification standards. Ensure all workflow tests pass before committing:

```bash
pytest tests/
```

-----

**MCPC – From orbit to wavelength, progressively closer to reality.**
