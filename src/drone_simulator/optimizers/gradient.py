"""
Ordinary Gradient Descent Optimizer for Drone Trajectory Optimization
Used for comparison with SPSA.
"""

from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List

import numpy as np

from .base import BaseOptimizer, OptimizerConfig


@dataclass
class GradientDescentConfig(OptimizerConfig):
    """Configuration for Gradient Descent optimizer"""
    # Learning rate
    lr: float = 0.1  # Step size

    # Finite difference epsilon
    epsilon: float = 0.01  # For numerical gradient

    # Gradient clipping to handle barrier function explosions
    max_grad_norm: float = 10.0


class GradientDescent(BaseOptimizer):
    """
    Ordinary Gradient Descent Optimizer using central finite differences.
    Requires 2*d loss evaluations per step (deterministic).
    """

    def __init__(self, config: GradientDescentConfig, loss_function: Callable):
        super().__init__(config, loss_function)
        self.config = config
        self.param_dim = 2  # [speed, direction]

    def _compute_gradient(self, theta: np.ndarray, **loss_kwargs) -> np.ndarray:
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

            loss_plus = self.loss_function(theta_plus, **loss_kwargs)
            loss_minus = self.loss_function(theta_minus, **loss_kwargs)

            grad[i] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute one gradient descent step"""
        self.iteration += 1

        # Compute gradient (deterministic)
        gradient = self._compute_gradient(self.theta, **loss_kwargs)

        # Clip gradient norm to avoid explosion from barrier function
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.config.max_grad_norm:
            gradient = gradient * (self.config.max_grad_norm / grad_norm)

        # Update parameters
        theta_new = self.theta - self.config.lr * gradient
        theta_new = self._rate_limit_direction(theta_new)
        theta_new = self._clip_parameters(theta_new)

        # Compute loss
        loss = self.loss_function(theta_new, **loss_kwargs)

        self.theta = theta_new
        self.history['parameters'].append(self.theta.copy())
        self.history['loss'].append(loss)
        self.history['gradients'].append(gradient.copy())

        return self.theta.copy(), loss, gradient.copy()


class TargetFollowingGD(GradientDescent):
    """
    Specialized Gradient Descent for target following
    Mirrors TargetFollowingSPSA for fair comparison
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

        # Predict position with look-ahead horizon
        look_ahead_time = kwargs.get('look_ahead_time', 0.5)
        dx = speed * np.cos(direction) * look_ahead_time
        dy = speed * np.sin(direction) * look_ahead_time
        next_position = self.current_position + np.array([dx, dy])

        loss = 0.0

        # 1. Distance to target (quadratic for stronger attraction)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        loss += dist_to_target ** 2

        # 2. Obstacle avoidance (barrier function with safety margin)
        obstacle_weight = kwargs.get('obstacle_weight', 100.0)
        avoidance_strength = kwargs.get('avoidance_strength', 10.0)
        safety_margin = kwargs.get('safety_margin', 1.0)

        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0
            effective_radius = obs_radius + safety_margin

            dist_to_obstacle = np.linalg.norm(next_position - obs_pos)
            if dist_to_obstacle < effective_radius:
                loss += obstacle_weight * (effective_radius - dist_to_obstacle) ** 2
            else:
                loss += avoidance_strength / (dist_to_obstacle - effective_radius + 1e-6)

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

        # 6. Near-target braking (prevent overshoot)
        dist_to_target_current = np.linalg.norm(self.current_position - self.target_position)
        if dist_to_target_current < 5.0:
            braking_weight = (5.0 - dist_to_target_current) * 0.5
            loss += braking_weight * speed

        return loss

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: List):
        self.current_position = position.copy()
        self.target_position = target.copy()
        self.obstacles = obstacles.copy()

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                        obstacles: List, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)
