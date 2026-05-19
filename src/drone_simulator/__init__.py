"""
Mixed-Gradient SPSA Drone Simulator

A 2D drone simulator with inertia and trajectory optimization using
the Mixed-Gradient SPSA framework from the article.
"""

from .core import (
    Drone,
    DroneConfig,
)
from .optimizers import (
    MixedOptimizer,
    MixedOptimizerConfig,
    BlockConfig,
    TargetFollowingSPSA,
)

__all__ = [
    "Drone",
    "DroneConfig",
    "MixedOptimizer",
    "MixedOptimizerConfig",
    "BlockConfig",
    "TargetFollowingSPSA",
]
