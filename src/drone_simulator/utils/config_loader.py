"""Load simulation and optimizer configurations from JSON files or unified config."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..core import DroneConfig, SimulationConfig
from ..optimizers import (
    GradientDescentConfig,
    SPSAConfig,
    TargetFollowingGD,
    TargetFollowingSPSA,
)

# Import unified configuration
from ..config import CONFIG, PhysicsConfig, SimulationConfig as SimDict



def load_simulation_config(config_dir: Path = Path("configs")) -> Dict[str, Any]:
    """Load full simulation configuration from the default JSON files.

    Expects the following layout under *config_dir*::

        simulation/default.json   – simulation, physics and obstacle parameters
        spsa/default.json         – SPSA hyper-parameters
        gd/default.json           – Gradient Descent hyper-parameters

    Returns a dict with ready-to-use objects:
        - 'simulation': SimulationConfig
        - 'initial_position': np.ndarray
        - 'target_position': np.ndarray
        - 'obstacles': list[list[float]]
        - 'physics': dict
        - 'spsa_config': SPSAConfig
        - 'gd_config': GradientDescentConfig
        - 'spsa_optimizer': TargetFollowingSPSA instance
        - 'gd_optimizer': TargetFollowingGD instance
    """
    return load_simulation_config_unified()


def load_simulation_config_unified() -> Dict[str, Any]:
    """
    Load configuration from unified CONFIG dictionary.
    This is the new preferred method as per technical specification.
    """
    return {
        "simulation": SimulationConfig(
            duration=CONFIG["simulation"]["duration"],
            dt=CONFIG["physics"]["dt"],  # dt is in physics config
            update_interval=CONFIG["simulation"]["update_interval"],
            plot_interval=CONFIG["simulation"]["plot_interval"],
        ),
        "initial_position": np.array(CONFIG["initial_position"]),
        "target_position": np.array(CONFIG["target_position"]),
        "obstacles": CONFIG["obstacles"],
        "physics": CONFIG["physics"],
        "spsa_config": SPSAConfig(**CONFIG["spsa_optimizer"]),
        "gd_config": GradientDescentConfig(**CONFIG["gd_optimizer"]),
        "spsa_optimizer": TargetFollowingSPSA(SPSAConfig(**CONFIG["spsa_optimizer"])),
        "gd_optimizer": TargetFollowingGD(GradientDescentConfig(**CONFIG["gd_optimizer"])),
        "metrics": CONFIG["metrics"],
    }

