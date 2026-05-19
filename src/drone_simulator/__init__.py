"""
Maneuver-learning drone simulator.

A 2D drone simulator with constant speed, blind obstacles,
and learnable maneuver parameters (d_back, omega_turn, alpha_evade).
"""

from .core import Drone
from .optimizers import ManeuverOptimizer, ManeuverOptimizerConfig

__all__ = [
    "Drone",
    "ManeuverOptimizer",
    "ManeuverOptimizerConfig",
]
