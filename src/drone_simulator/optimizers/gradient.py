"""
Ordinary Gradient Descent Optimizer for Drone Trajectory Optimization
Used for comparison with SPSA.
"""

from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import numpy as np


@dataclass
class GradientDescentConfig:
    """Configuration for Gradient Descent optimizer"""
    # Learning rate
    lr: float = 0.1  # Step size

    # Finite difference epsilon
    epsilon: float = 0.01  # For numerical gradient

    # Parameter constraints
    speed_min: float = 0.0
    speed_max: float = 10.0
    direction_min: float = -np.pi
    direction_max: float = np.pi


class GradientDescent:
    """
    Ordinary Gradient Descent Optimizer using central finite differences.
    Requires 2*d loss evaluations per step (deterministic).
    """

    def __init__(self, config: GradientDescentConfig, loss_function: Callable):
        self.config = config
        self.loss_function = loss_function
        self.iteration = 0
        self.param_dim = 2  # [speed, direction]

        # Initialize parameters
        self.theta = np.array([
            (config.speed_max + config.speed_min) / 2,
            0.0
        ])

        self.history = {
            'parameters': [],
            'loss': [],
            'gradients': []
        }

    def _compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute gradient via central finite differences"""
        eps = self.config.epsilon
        grad = np.zeros(self.param_dim)

        for i in range(self.param_dim):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps

            # Clip to valid ranges
            theta_plus = self._clip_parameters(theta_plus)
            theta_minus = self._clip_parameters(theta_minus)

            loss_plus = self.loss_function(theta_plus)
            loss_minus = self.loss_function(theta_minus)

            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    def _clip_parameters(self, theta: np.ndarray) -> np.ndarray:
        """Clip parameters to valid ranges"""
        theta_clipped = theta.copy()
        theta_clipped[0] = np.clip(
            theta_clipped[0],
            self.config.speed_min,
            self.config.speed_max
        )
        if len(theta_clipped) > 1:
            theta_clipped[1] = np.arctan2(
                np.sin(theta_clipped[1]),
                np.cos(theta_clipped[1])
            )
        return theta_clipped

    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute one gradient descent step"""
        self.iteration += 1

        # Compute gradient (deterministic)
        gradient = self._compute_gradient(self.theta)

        # Update parameters
        theta_new = self.theta - self.config.lr * gradient
        theta_new = self._clip_parameters(theta_new)

        # Compute loss
        loss = self.loss_function(theta_new, **loss_kwargs)

        self.theta = theta_new
        self.history['parameters'].append(self.theta.copy())
        self.history['loss'].append(loss)
        self.history['gradients'].append(gradient.copy())

        return self.theta.copy(), loss, gradient.copy()

    def get_speed(self) -> float:
        return self.theta[0]

    def get_direction(self) -> float:
        return self.theta[1]

    def set_parameters(self, speed: Optional[float] = None, direction: Optional[float] = None):
        if speed is not None:
            self.theta[0] = np.clip(speed, self.config.speed_min, self.config.speed_max)
        if direction is not None:
            self.theta[1] = direction


class TargetFollowingGD(GradientDescent):
    """
    Specialized Gradient Descent for target following with dynamic loss function.
    Mirror of TargetFollowingSPSA for fair comparison.
    """

    def __init__(self, config: GradientDescentConfig):
        super().__init__(config, self._dynamic_loss)
        self.current_position = np.array([0.0, 0.0])
        self.target_position = np.array([10.0, 10.0])
        self.obstacles = []

    def _dynamic_loss(self, params: np.ndarray, **kwargs) -> float:
        """Dynamic loss function (identical to TargetFollowingSPSA)"""
        speed = params[0]
        direction = params[1]

        # Predict next position
        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        next_position = self.current_position + np.array([dx, dy])

        loss = 0.0

        # 1. Distance to target
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        loss += dist_to_target

        # 2. Obstacle avoidance
        obstacle_weight = kwargs.get('obstacle_weight', 100.0)
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0
            dist_to_obstacle = np.linalg.norm(next_position - obs_pos)
            if dist_to_obstacle < obs_radius:
                loss += obstacle_weight * (obs_radius - dist_to_obstacle) ** 2
            else:
                avoidance_strength = kwargs.get('avoidance_strength', 1.0)
                loss += avoidance_strength / (dist_to_obstacle - obs_radius + 1e-6)

        # 3. Speed smoothness
        if len(self.history['parameters']) > 0:
            prev_speed = self.history['parameters'][-1][0]
            speed_smooth_weight = kwargs.get('speed_smooth_weight', 0.1)
            loss += speed_smooth_weight * (speed - prev_speed) ** 2

        # 4. Direction smoothness
        if len(self.history['parameters']) > 0:
            prev_direction = self.history['parameters'][-1][1]
            direction_diff = np.arctan2(
                np.sin(direction - prev_direction),
                np.cos(direction - prev_direction)
            )
            dir_smooth_weight = kwargs.get('dir_smooth_weight', 0.1)
            loss += dir_smooth_weight * direction_diff ** 2

        # 5. Energy efficiency
        energy_weight = kwargs.get('energy_weight', 0.01)
        loss += energy_weight * speed ** 2

        return loss

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: list):
        self.current_position = position.copy()
        self.target_position = target.copy()
        self.obstacles = obstacles.copy()

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                        obstacles: list, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)
