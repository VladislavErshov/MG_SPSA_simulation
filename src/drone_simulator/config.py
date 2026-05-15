"""
Unified configuration loader for the drone simulator.
All parameters are loaded from JSON files in the configs/ directory.

Environment: Python 3.14+
Author: Senior Python Developer & Control Theory Researcher
"""

import json
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_json(rel_path: str) -> dict:
    """Load a JSON config relative to the project root."""
    with open(_PROJECT_ROOT / rel_path, 'r') as f:
        return json.load(f)


def load_configs(simulation_json: str = "configs/simulation/default.json") -> Dict:
    """Load full configuration from simulation JSON + linked spsa/gd JSONs.

    Returns a dict with the same structure as the legacy CONFIG:
        physics, simulation, initial_position, target_position,
        obstacles, wind, mixed_optimizer, gd_optimizer, metrics
    """
    sim = _load_json(simulation_json)

    spsa = _load_json(sim["spsa_config"])
    gd = _load_json(sim["gd_config"])

    return {
        "physics": sim["physics"],
        "simulation": {
            "duration": sim["duration"],
            "dt": sim["dt"],
            "update_interval": sim["update_interval"],
            "plot_interval": sim["plot_interval"],
        },
        "initial_position": sim["initial_position"],
        "target_position": sim["target_position"],
        "obstacles": sim["obstacles"],
        "wind": sim["wind"],
        "mixed_optimizer": spsa,
        "gd_optimizer": gd,
        "metrics": sim["metrics"],
    }


# Default global config loaded from JSON
CONFIG = load_configs()

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
