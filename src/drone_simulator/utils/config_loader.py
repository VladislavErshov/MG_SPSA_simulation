"""Load simulation and optimizer configurations from JSON files."""

import json
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
    sim_path = config_dir / "simulation" / "default.json"
    with open(sim_path, "r", encoding="utf-8") as f:
        sim_cfg: Dict[str, Any] = json.load(f)

    # Load optimizer configs referenced inside the simulation config
    spsa_path = config_dir / "spsa" / "default.json"
    gd_path = config_dir / "gd" / "default.json"

    with open(spsa_path, "r", encoding="utf-8") as f:
        spsa_json = json.load(f)
    with open(gd_path, "r", encoding="utf-8") as f:
        gd_json = json.load(f)

    spsa_cfg = SPSAConfig(**spsa_json)
    gd_cfg = GradientDescentConfig(**gd_json)

    return {
        "simulation": SimulationConfig(
            duration=sim_cfg["duration"],
            dt=sim_cfg["dt"],
            update_interval=sim_cfg["update_interval"],
            plot_interval=sim_cfg["plot_interval"],
        ),
        "initial_position": np.array(sim_cfg["initial_position"]),
        "target_position": np.array(sim_cfg["target_position"]),
        "obstacles": sim_cfg["obstacles"],
        "physics": sim_cfg["physics"],
        "spsa_config": spsa_cfg,
        "gd_config": gd_cfg,
        "spsa_optimizer": TargetFollowingSPSA(spsa_cfg),
        "gd_optimizer": TargetFollowingGD(gd_cfg),
    }
