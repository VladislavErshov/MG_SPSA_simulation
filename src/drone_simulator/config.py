"""
Unified configuration dictionary for the drone simulator.
All parameters are centralized here for easy modification.

Environment: Python 3.14+
Author: Senior Python Developer & Control Theory Researcher
"""

from typing import Dict, List, Tuple

# Unified configuration dictionary
CONFIG = {
    # Physics parameters (from technical spec line 10)
    "physics": {
        "inertia_alpha": 0.5,  # α ∈ (0, 1) - inertia coefficient, larger = smoother response
        "max_speed": 10.0,  # m/s - maximum drone speed
        "max_acceleration": 5.0,  # m/s² - maximum acceleration (optional realism)
        "dt": 0.05,  # s - simulation time step
        "collision_detection": True,  # Enable obstacle collision detection
    },

    # Simulation parameters
    "simulation": {
        "duration": 30.0,  # s - max simulation time
        "update_interval": 1,  # steps - position update frequency
        "plot_interval": 5,  # steps - visualization update frequency
    },

    # Initial conditions
    "initial_position": [0.0, 0.0],  # [x, y] in meters
    "target_position": [20.0, 15.0],  # [x, y] in meters

    # Obstacles (circular, format: [x, y, radius])
    "obstacles": [
        [5.0, 5.0, 2.0],
        [10.0, 8.0, 1.5],
        [15.0, 12.0, 2.5],
        [8.0, 15.0, 2.0],
        [18.0, 10.0, 1.8],
    ],

    # SPSA optimizer parameters (aligned with article theory)
    "spsa_optimizer": {
        # Practical Spall formula: alpha_n = a / (A + n)^alpha
        "a": 2.0,  # Step size amplitude
        "alpha": 0.602,  # Decay exponent (standard SPSA value)
        "A": 50.0,  # Stability constant (prevents freeze in online control)

        # Corollary 4.2 balanced gamma for q=1 effective defect: gamma = 1/4
        "c": 0.5,  # Perturbation amplitude (increased for better obstacle sensing)
        "gamma": 0.25,  # Perturbation decay exponent

        # From Lemma 6.1 / Theorem 3.1
        "num_perturbations": 8,  # N perturbations per iteration (more averaging)
        "decorrelation_exponent": 1.0,  # Rho for variance decay
        "gradient_momentum": 0.25,  # EMA momentum for g_phi smoothing

        # Exact gradient channel (w block)
        "epsilon_w": 0.01,  # Finite-difference step for exact speed gradient

        # Parameter constraints
        "speed_min": 0.0,  # Minimum speed (m/s)
        "speed_max": 10.0,  # Maximum speed (m/s)
        "direction_min": -3.141592653589793,  # Minimum direction (rad)
        "direction_max": 3.141592653589793,  # Maximum direction (rad)
    },

    # Gradient Descent optimizer parameters
    "gd_optimizer": {
        "lr": 0.1,  # Learning rate
        "epsilon": 0.01,  # Finite difference step size
        "max_grad_norm": 5.0,  # Gradient clipping for barrier function

        # Parameter constraints
        "speed_min": 0.0,
        "speed_max": 10.0,
        "direction_min": -3.141592653589793,
        "direction_max": 3.141592653589793,
    },

    # Performance metrics thresholds
    "metrics": {
        "target_tolerance": 1.0,  # m - distance to target considered "reached"
        "collision_penalty": 100.0,  # Loss function weight for collisions
        "avoidance_strength": 2.0,  # Soft obstacle avoidance strength
        "speed_smooth_weight": 0.2,  # Speed smoothing penalty weight
        "dir_smooth_weight": 0.1,  # Direction smoothing penalty weight
        "energy_weight": 0.05,  # Energy consumption penalty weight
        "look_ahead_time": 0.2,  # s - prediction horizon for obstacle avoidance
        "safety_margin": 0.3,  # m - extra radius added to obstacles in loss
    },
}

# Optimizer mapping
OPTIMIZER_TYPES = {
    "spsa": "TargetFollowingSPSA",
    "gd": "TargetFollowingGD",
}

# Type hints for clarity
PhysicsConfig = Dict[str, float]
SimulationConfig = Dict[str, float]
OptimizerConfig = Dict[str, float]


def get_config(section: str) -> Dict:
    """Get configuration section by name"""
    return CONFIG.get(section, {})
