# Mixed-Variable SPSA Drone Simulator

This project implements a 2D drone simulator with inertia and trajectory optimization using the Mixed-Variable Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm as described in "Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications".

## Overview

The simulator demonstrates how SPSA can be used to optimize drone trajectories in real-time, taking into account:
- Physical constraints (inertia, momentum)
- Obstacle avoidance
- Target following
- Smooth movement patterns

## Key Features

### 1. Mixed-Variable SPSA Optimizer
- **Dual optimization**: Separate optimization for speed and direction parameters
- **Stochastic gradient estimation**: Uses simultaneous perturbation for gradient approximation
- **Configurable parameters**: Step sizes, perturbation scales, convergence rates
- **Target following**: Dynamic loss function that adapts to current position and obstacles

### 2. Physics-Based Drone Model
- **Inertia simulation**: Realistic response to control inputs
- **2D dynamics**: Position, velocity, and acceleration tracking
- **Constraints**: Maximum speed, physical limits
- **Trajectory history**: Complete path recording for analysis

### 3. Real-Time Visualization
- Live trajectory plotting
- Speed and direction monitoring
- Loss function convergence tracking
- Multiple drones in same environment

## Project Structure

```
.
├── src/
│   └── drone_simulator/
│       ├── __init__.py              # Package exports
│       ├── core/
│       │   ├── __init__.py          # Core exports
│       │   ├── drone.py             # Drone physics and state
│       │   └── simulation.py        # Simulation engine and visualization
│       ├── optimizers/
│       │   ├── __init__.py          # Optimizer exports
│       │   ├── spsa.py              # SPSA algorithm implementation
│       │   └── gradient.py          # Gradient Descent implementation
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── simulation_viz.py    # Matplotlib visualization
│       └── utils/
│           ├── __init__.py
│           └── config_loader.py     # JSON configuration loader
├── tests/
│   ├── test_drone_simulator.py      # Unit tests for simulator
│   ├── test_spsa_optimizer.py       # Unit tests for SPSA
│   └── test_gradient_optimizer.py   # Unit tests for GD
├── examples/
│   └── comparison.py                # SPSA vs GD live simulation
├── configs/
│   ├── simulation/
│   │   └── default.json             # Simulation, physics, obstacles
│   ├── spsa/
│   │   └── default.json             # SPSA hyper-parameters
│   └── gd/
│       └── default.json             # GD hyper-parameters
├── scripts/
│   └── setup_venv.sh                # Environment setup script
├── results/                         # Output directory (created at runtime)
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package configuration
├── README.md                        # This file
└── ARCHITECTURE.md                  # Architecture overview
```

## Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib
```

## Quick Start

Run the live simulation with SPSA vs GD comparison:
```bash
python examples/comparison.py
```

## Running Tests

```bash
python -m pytest tests/
```

Or run individual test files:
```bash
python -m pytest tests/test_spsa_optimizer.py
python -m pytest tests/test_gradient_optimizer.py
python -m pytest tests/test_drone_simulator.py
```

## Configuration

The simulator supports extensive configuration:

### SPSA Parameters
- `a`: Step size amplitude
- `c`: Perturbation amplitude
- `alpha`: Step size decay exponent
- `gamma`: Perturbation decay exponent

### Gradient Descent Parameters
- `lr`: Learning rate (step size)
- `epsilon`: Finite difference step for numerical gradient

### Drone Physics
- `inertia_coefficient`: Higher values = more momentum (0.0-1.0)
- `max_speed`: Maximum velocity limit
- `response_time`: Control system time constant

### Loss Function Weights
- `obstacle_weight`: Penalty for collisions
- `speed_smooth_weight`: Encourages smooth acceleration
- `dir_smooth_weight`: Reduces sharp turns
- `energy_weight`: Penalizes high speeds

## Usage in Your Code

```python
from src.drone_simulator import Drone, DroneConfig, DroneSimulator, SimulationConfig

# Create a drone with SPSA (stochastic, 2 loss evals/step)
drone_spsa = Drone([0.0, 0.0], DroneConfig(optimizer_type="spsa"))
drone_spsa.set_target([20.0, 15.0])
drone_spsa.set_obstacles([[5.0, 5.0, 2.0], [10.0, 8.0, 1.5]])

# Or with Gradient Descent (deterministic, 4 loss evals/step for d=2)
drone_gd = Drone([0.0, 0.0], DroneConfig(optimizer_type="gd"))
drone_gd.set_target([20.0, 15.0])
drone_gd.set_obstacles([[5.0, 5.0, 2.0], [10.0, 8.0, 1.5]])

# Run simulation
sim = DroneSimulator([drone_spsa], SimulationConfig(duration=30.0))
sim.run(visualize=True)
```

## Mathematical Foundation

Based on SPSA optimization theory, the algorithm uses simultaneous perturbation for gradient estimation:

```
g(θ) ≈ [L(θ + cΔ) - L(θ - cΔ)] / (2cΔ)
```

Where Δ are random perturbations and c controls perturbation size.

The mixed-variable extension allows different update mechanisms for speed and direction parameters while maintaining convergence guarantees.

## References

Salishev, S. (2024). Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications. IEEE Access.
