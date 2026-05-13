"""
Mixed-Variable SPSA Drone Simulator

A 2D drone simulator with inertia and trajectory optimization using
the Mixed-Gradient SPSA framework from the article.
"""

from .core import (
    Drone,
    DroneConfig,
    DroneSimulator,
    SimulationConfig,
)
from .optimizers import (
    MixedOptimizer,
    MixedOptimizerConfig,
    BlockConfig,
    TargetFollowingSPSA,
    create_test_scenario,
    GradientDescent,
    GradientDescentConfig,
    TargetFollowingGD,
)

__all__ = [
    "Drone",
    "DroneConfig",
    "DroneSimulator",
    "SimulationConfig",
    "MixedOptimizer",
    "MixedOptimizerConfig",
    "BlockConfig",
    "TargetFollowingSPSA",
    "create_test_scenario",
    "GradientDescent",
    "GradientDescentConfig",
    "TargetFollowingGD",
]
