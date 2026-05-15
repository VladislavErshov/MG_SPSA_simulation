"""Load simulation and optimizer configurations from JSON files or unified config."""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..core import DroneConfig, SimulationConfig
from ..optimizers import (
    MixedOptimizerConfig,
    TargetFollowingSPSA,
)

# Import unified configuration
from ..config import CONFIG, PhysicsConfig, SimulationConfig as SimDict


def load_simulation_config(config_dir: Path = Path("configs")) -> Dict[str, Any]:
    """Load full simulation configuration from the default JSON files.

    Expects the following layout under *config_dir*::

        simulation/default.json   – simulation, physics and obstacle parameters
        mixed/default.json        – Mixed optimizer hyper-parameters

    Returns a dict with ready-to-use objects:
        - 'simulation': SimulationConfig
        - 'initial_position': np.ndarray
        - 'target_position': np.ndarray
        - 'obstacles': list[list[float]]
        - 'physics': dict
        - 'mixed_config': MixedOptimizerConfig
        - 'mixed_optimizer': TargetFollowingSPSA instance
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
        "mixed_config": MixedOptimizerConfig(**CONFIG["mixed_optimizer"]),
        "mixed_optimizer": TargetFollowingSPSA(MixedOptimizerConfig(**CONFIG["mixed_optimizer"])),
        "metrics": CONFIG["metrics"],
    }
