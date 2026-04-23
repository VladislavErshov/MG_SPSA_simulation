"""
Core drone physics and state.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

from ..optimizers.spsa import SPSAConfig, TargetFollowingSPSA
from ..optimizers.gradient import GradientDescentConfig, TargetFollowingGD


@dataclass
class DroneConfig:
    """Configuration for drone physics"""
    mass: float = 1.0
    max_thrust: float = 20.0
    max_speed: float = 10.0
    inertia_coefficient: float = 0.9
    response_time: float = 0.1
    optimizer_type: str = "spsa"


class Drone:
    """2D Drone with inertia and optimizer-based control"""

    def __init__(self, position: np.ndarray, config: Optional[DroneConfig] = None,
                 optimizer_config=None):
        self.config = config or DroneConfig()

        self.position = np.array(position).astype(float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])

        self.target_speed = 0.0
        self.target_direction = 0.0

        self.trajectory = [self.position.copy()]
        self.speed_history = [0.0]
        self.direction_history = [0.0]
        self.time_history = [0.0]

        self.time = 0.0
        self.dt = 0.05

        if optimizer_config is not None:
            self.optimizer = optimizer_config
        elif self.config.optimizer_type.lower() == "gd":
            gd_config = GradientDescentConfig(
                lr=0.5, epsilon=0.05,
                speed_min=0.0, speed_max=self.config.max_speed,
            )
            self.optimizer = TargetFollowingGD(gd_config)
        else:
            spsa_config = SPSAConfig(
                a=2.0, c=0.2,
                speed_min=0.0, speed_max=self.config.max_speed,
            )
            self.optimizer = TargetFollowingSPSA(spsa_config)

    def set_target(self, target_position: np.ndarray):
        self.target_position = np.array(target_position).astype(float)

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        self.obstacles = [list(obs) for obs in obstacles] if obstacles else []

    def _compute_desired_control(self) -> Tuple[float, float]:
        self.optimizer.update_state(
            self.position, self.target_position, self.obstacles
        )
        params, loss, gradient = self.optimizer.step_with_state(
            self.position, self.target_position, self.obstacles,
            obstacle_weight=50.0, avoidance_strength=1.0,
            speed_smooth_weight=0.2, dir_smooth_weight=0.1,
            energy_weight=0.05
        )
        return self.optimizer.get_speed(), self.optimizer.get_direction()

    def _apply_control_with_inertia(self, desired_speed: float, desired_direction: float):
        desired_vx = desired_speed * np.cos(desired_direction)
        desired_vy = desired_speed * np.sin(desired_direction)
        desired_velocity = np.array([desired_vx, desired_vy])

        inertia = self.config.inertia_coefficient
        self.velocity = inertia * self.velocity + (1 - inertia) * desired_velocity

        speed = np.linalg.norm(self.velocity)
        if speed > self.config.max_speed:
            self.velocity = self.velocity / speed * self.config.max_speed

        self.acceleration = (self.velocity - np.array([
            self.speed_history[-1] * np.cos(self.direction_history[-1]) if self.speed_history else 0,
            self.speed_history[-1] * np.sin(self.direction_history[-1]) if self.speed_history else 0,
        ])) / self.dt

        self.target_speed = desired_speed
        self.target_direction = desired_direction

    def step(self):
        desired_speed, desired_direction = self._compute_desired_control()
        self._apply_control_with_inertia(desired_speed, desired_direction)
        self.position += self.velocity * self.dt
        self.time += self.dt

        self.trajectory.append(self.position.copy())
        self.speed_history.append(np.linalg.norm(self.velocity))
        self.direction_history.append(np.arctan2(self.velocity[1], self.velocity[0]))
        self.time_history.append(self.time)

        current_params = np.array([
            np.linalg.norm(self.velocity),
            np.arctan2(self.velocity[1], self.velocity[0])
        ])
        self.optimizer.history['parameters'].append(current_params)

    def get_state(self) -> Dict[str, any]:
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'speed': np.linalg.norm(self.velocity),
            'direction': np.arctan2(self.velocity[1], self.velocity[0]),
            'target_speed': self.target_speed,
            'target_direction': self.target_direction,
            'time': self.time,
        }

    def get_trajectory(self) -> np.ndarray:
        return np.array(self.trajectory)

    def reached_target(self, tolerance: float = 1.0) -> bool:
        return np.linalg.norm(self.position - self.target_position) < tolerance
