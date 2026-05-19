"""
Abstract base class for drone trajectory optimizers.
Provides unified interface for Gradient Descent, SPSA, and future MPC implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, List

import numpy as np


@dataclass
class BlockConfig:
    """Configuration for a single parameter block within a mixed optimizer.

    Attributes:
        param_slice: slice object selecting this block's indices from theta.
        method: gradient estimation method — 'exact', 'spsa_off_center', or 'spsa_centered'.
        q: defect order for SPSA blocks (1 for off-center, 2+ for centered stencil).
             Ignored for 'exact' method.
    """
    param_slice: slice
    method: str  # 'exact', 'spsa_off_center', 'spsa_centered'
    q: int = 1


@dataclass
class OptimizerConfig:
    """Base configuration for all optimizers"""
    speed_min: float = 0.0
    speed_max: float = 10.0
    direction_min: float = -np.pi
    direction_max: float = np.pi
    wind_estimate_min: float = -5.0
    wind_estimate_max: float = 5.0
    max_direction_delta: float = 0.1  # Maximum direction change per step (rad)


class BaseOptimizer(ABC):
    """
    Abstract base class for drone trajectory optimizers.

    All optimizers must implement:
    - step(): Perform one optimization step
    - get_speed(): Get current speed command
    - get_direction(): Get current direction command
    - update_state(): Update drone state for loss calculation
    """

    def __init__(self, config: OptimizerConfig, loss_function: Callable):
        self.config = config
        self.loss_function = loss_function
        self.iteration = 0
        self.param_dim = 2  # [speed, direction] by default

        # History tracking
        self.history = {
            'parameters': [],  # [speed, direction] pairs
            'loss': [],
            'gradients': []
        }

        # Initialize parameters to reasonable defaults
        self.theta = np.array([
            (config.speed_max + config.speed_min) / 2,
            0.0  # Straight ahead
        ])

    @abstractmethod
    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute one optimization step.

        Returns:
            - Updated parameters [speed, direction]
            - Current loss value
            - Gradient estimate (if available)
        """
        pass

    def get_speed(self) -> float:
        """Get current speed command"""
        return self.theta[0]

    def get_direction(self) -> float:
        """Get current direction command in radians"""
        return self.theta[1] if len(self.theta) > 1 else 0.0

    def set_parameters(self, speed: Optional[float] = None, direction: Optional[float] = None):
        """Manually set parameters"""
        if speed is not None:
            self.theta[0] = np.clip(speed, self.config.speed_min, self.config.speed_max)
        if direction is not None and len(self.theta) > 1:
            self.theta[1] = direction

    def get_velocity_command(self) -> np.ndarray:
        """Get current velocity command as [vx, vy] vector"""
        speed = self.get_speed()
        direction = self.get_direction()
        return np.array([
            speed * np.cos(direction),
            speed * np.sin(direction)
        ])

    def _rate_limit_direction(self, theta: np.ndarray) -> np.ndarray:
        """Limit direction change per step to avoid sharp turns"""
        if len(self.history['parameters']) == 0 or len(theta) <= 1:
            return theta
        max_delta = getattr(self.config, 'max_direction_delta', None)
        if max_delta is None:
            return theta
        theta_limited = theta.copy()
        prev_dir = self.history['parameters'][-1][1]
        delta = np.arctan2(
            np.sin(theta_limited[1] - prev_dir),
            np.cos(theta_limited[1] - prev_dir)
        )
        delta = np.clip(delta, -max_delta, max_delta)
        theta_limited[1] = prev_dir + delta
        return theta_limited

    def _clip_parameters(self, theta: np.ndarray) -> np.ndarray:
        """Clip parameters to valid ranges"""
        theta_clipped = theta.copy()

        # Clip speed
        theta_clipped[0] = np.clip(
            theta_clipped[0],
            self.config.speed_min,
            self.config.speed_max
        )

        # Wrap direction angle
        if len(theta_clipped) > 1:
            theta_clipped[1] = np.arctan2(
                np.sin(theta_clipped[1]),
                np.cos(theta_clipped[1])
            )

        # Clip wind estimate
        if len(theta_clipped) > 2:
            theta_clipped[2] = np.clip(
                theta_clipped[2],
                self.config.wind_estimate_min,
                self.config.wind_estimate_max
            )

        return theta_clipped

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: List):
        """Update drone state for loss calculation. Override in subclasses if needed."""
        pass

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                       obstacles: List, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute step with updated state"""
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)
