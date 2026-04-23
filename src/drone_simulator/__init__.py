"""
Mixed-Variable SPSA Drone Simulator

A 2D drone simulator with inertia and trajectory optimization using
the Mixed-Variable Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm.
"""

from .core import (
    Drone,
    DroneConfig,
    DroneSimulator,
    SimulationConfig,
)
from .optimizers import (
    MixedVariableSPSA,
    SPSAConfig,
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
    "MixedVariableSPSA",
    "SPSAConfig",
    "TargetFollowingSPSA",
    "create_test_scenario",
    "GradientDescent",
    "GradientDescentConfig",
    "TargetFollowingGD",
]
