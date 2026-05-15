"""
Core drone physics and state with inertia and collision handling.
Implements the updated dynamics V_{t+1} = V_t + alpha * (V_cmd - V_t).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

from ..optimizers import BaseOptimizer, MixedOptimizerConfig, TargetFollowingSPSA


@dataclass
class DroneConfig:
    """Configuration for drone physics"""
    # Inertia coefficient for velocity smoothing - equation at line 10 of technical spec
    inertia_alpha: float = 0.5  # alpha ∈ (0, 1), larger = more inertia

    # Maximum speed limit
    max_speed: float = 10.0

    # Maximum acceleration limit for realistic dynamics
    max_acceleration: float = 5.0

    # Time step
    dt: float = 0.05

    # Obstacle collision detection
    collision_detection: bool = True

    # Optimizer type
    optimizer_type: str = "spsa"


class Drone:
    """
    2D Drone with linear inertia smoothing and collision detection.

    Physics implementation:
    - Velocity update: V_{t+1} = V_t + alpha * (V_cmd - V_t)
    - Position update: X_{t+1} = X_t + (V_{t+1} + wind) * dt
    - Collision: If next position is inside obstacle, position is not updated;
      instead velocity is set to point away from obstacle.
    """

    def __init__(self, position: np.ndarray, config: Optional[DroneConfig] = None,
                 optimizer_config=None):
        self.config = config or DroneConfig()

        # State variables
        self.position = np.array(position).astype(float)
        self.velocity = np.array([0.0, 0.0])  # V_t - current velocity
        self.command_velocity = np.array([0.0, 0.0])  # V_cmd - desired velocity from optimizer
        self.wind = np.array([0.0, 0.0])  # Constant wind vector [vx, vy]

        # History for analysis
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.command_velocity_history = [self.command_velocity.copy()]
        self.in_collision_history = [False]
        self.time_history = [0.0]

        # Maintain backward compatibility for speed/direction history
        self.speed_history = [0.0]
        self.direction_history = [0.0]

        self.time = 0.0
        self.consecutive_collisions = 0
        self.stuck_steps = 0

        # Initialize optimizer
        if optimizer_config is not None:
            self.optimizer = optimizer_config
        else:
            spsa_config = MixedOptimizerConfig(
                a=20.0, c=0.2, burn_in=50,
                speed_min=0.0, speed_max=self.config.max_speed,
            )
            self.optimizer = TargetFollowingSPSA(spsa_config)

    def set_target(self, target_position: np.ndarray):
        """Set target position"""
        self.target_position = np.array(target_position).astype(float)

    def set_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Set circular obstacles (x, y, radius)"""
        self.obstacles = [list(obs) for obs in obstacles] if obstacles else []

    def set_wind(self, wind_vector: np.ndarray):
        """Set constant wind vector [vx, vy] in m/s"""
        self.wind = np.array(wind_vector).astype(float)
        if hasattr(self.optimizer, 'set_wind'):
            self.optimizer.set_wind(self.wind)

    def _compute_desired_control(self) -> Tuple[float, float]:
        """
        Compute desired speed and direction from optimizer.
        This is V_cmd converted to speed/direction representation.
        """
        self.optimizer.update_state(
            self.position, self.target_position, self.obstacles
        )
        params, loss, gradient = self.optimizer.step_with_state(
            self.position, self.target_position, self.obstacles,
            obstacle_weight=50.0, avoidance_strength=2.0,
            speed_smooth_weight=0.2, dir_smooth_weight=0.1,
            energy_weight=0.1, look_ahead_time=0.5, safety_margin=0.3
        )
        return self.optimizer.get_speed(), self.optimizer.get_direction()

    def _check_collision(self, position: np.ndarray) -> bool:
        """
        Check if position is inside any obstacle.
        Returns True if collision detected.
        """
        if not self.config.collision_detection:
            return False

        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0

            dist_to_obstacle = np.linalg.norm(position - obs_pos)
            if dist_to_obstacle < obs_radius:
                return True

        return False

    def _bounce_from_obstacle(self, position: np.ndarray, bounce_speed: float) -> np.ndarray:
        """Return velocity pointing away from nearest obstacle."""
        min_dist = float('inf')
        away_vec = np.array([1.0, 0.0])
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            diff = position - obs_pos
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                if dist > 1e-8:
                    away_vec = diff / dist
                else:
                    away_vec = np.array([1.0, 0.0])
        return away_vec * max(bounce_speed, 0.2)

    def _resolve_collision(self, position: np.ndarray, margin: float = 0.05) -> np.ndarray:
        """Push position to outside of all obstacles with small margin."""
        resolved = position.copy()
        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0
            diff = resolved - obs_pos
            dist = np.linalg.norm(diff)
            if dist < obs_radius + margin:
                if dist > 1e-8:
                    resolved = obs_pos + diff / dist * (obs_radius + margin)
                else:
                    resolved = obs_pos + np.array([obs_radius + margin, 0.0])
        return resolved

    def _apply_physics_step(self):
        """
        Apply physics step with inertia and collision handling.

        Equation implementation:
        1. Update velocity: V_{t+1} = V_t + alpha * (V_cmd - V_t)
        2. Check collision at new position: X_{t+1} = X_t + V_{t+1} * dt
        3. If collision: V_{t+1} = 0 (instant stop penalty)
        4. Update position even during collision (drone attempts to escape)
        """
        # Step 1: Compute command velocity from optimizer
        speed_cmd, direction_cmd = self._compute_desired_control()
        vx_cmd = speed_cmd * np.cos(direction_cmd)
        vy_cmd = speed_cmd * np.sin(direction_cmd)
        self.command_velocity = np.array([vx_cmd, vy_cmd])

        # Step 2: Apply inertia smoothing
        # V_{t+1} = V_t + alpha * (V_cmd - V_t)
        alpha = self.config.inertia_alpha
        self.velocity = self.velocity + alpha * (self.command_velocity - self.velocity)

        # Step 3: Limit maximum speed and acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.config.max_speed:
            self.velocity = self.velocity / speed * self.config.max_speed

        # Acceleration limiting (optional, for realism)
        if len(self.velocity_history) > 0:
            prev_velocity = self.velocity_history[-1]
            acceleration = (self.velocity - prev_velocity) / self.config.dt
            accel_mag = np.linalg.norm(acceleration)
            if accel_mag > self.config.max_acceleration:
                acceleration = acceleration / accel_mag * self.config.max_acceleration
                self.velocity = prev_velocity + acceleration * self.config.dt

        # Step 4: Predict next position (ground velocity = air velocity + wind)
        ground_velocity = self.velocity + self.wind
        next_position = self.position + ground_velocity * self.config.dt

        # Step 5: Advance position
        self.position += ground_velocity * self.config.dt

        # Step 6: If inside obstacle, push to boundary and bounce
        in_collision = self._check_collision(self.position)
        if in_collision:
            self.consecutive_collisions += 1
            # Push to outside of obstacle
            self.position = self._resolve_collision(self.position)
            # Strong bounce: must overcome wind to escape
            bounce_speed = max(
                np.linalg.norm(self.velocity) * 0.5,
                np.linalg.norm(self.wind) * 2.0,
                2.0,
            )
            self.velocity = self._bounce_from_obstacle(self.position, bounce_speed)
            bounce_dir = np.arctan2(self.velocity[1], self.velocity[0])
            self.optimizer.theta[1] = bounce_dir
            if self.consecutive_collisions > 3:
                self.optimizer.iteration = max(self.optimizer.iteration - 30, 0)
        else:
            self.consecutive_collisions = 0

        # Detect stuck state: low physical speed for many steps
        if np.linalg.norm(self.velocity) < 0.5:
            self.stuck_steps += 1
            if self.stuck_steps > 15:
                # Random perturbation to escape local minimum
                self.optimizer.theta[0] = 1.5
                self.optimizer.theta[1] += float(np.random.choice([-1, 1])) * 0.5
                if len(self.optimizer.theta) > 2:
                    self.optimizer.theta[2] += float(np.random.choice([-1, 1])) * 0.3
                self.stuck_steps = 0
        else:
            self.stuck_steps = 0

        self.time += self.config.dt

        # Store history
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.command_velocity_history.append(self.command_velocity.copy())
        self.in_collision_history.append(in_collision)
        self.time_history.append(self.time)

        # Update backward compatibility fields
        current_speed = np.linalg.norm(self.velocity)
        current_direction = np.arctan2(self.velocity[1], self.velocity[0])
        self.speed_history.append(current_speed)
        self.direction_history.append(current_direction)

    def step(self):
        """Execute one simulation step"""
        return self._apply_physics_step()

    def get_state(self) -> Dict[str, any]:
        """Get current drone state"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'command_velocity': self.command_velocity.copy(),
            'speed': np.linalg.norm(self.velocity),
            'direction': np.arctan2(self.velocity[1], self.velocity[0]),
            'command_speed': np.linalg.norm(self.command_velocity),
            'command_direction': np.arctan2(self.command_velocity[1], self.command_velocity[0]),
            'time': self.time,
            'in_collision': self.in_collision_history[-1] if self.in_collision_history else False,
        }

    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory as Nx2 array"""
        return np.array(self.trajectory)

    def get_collision_count(self) -> int:
        """Get total number of collision events"""
        return sum(self.in_collision_history)

    def get_min_obstacle_distance(self) -> float:
        """Get minimum distance to any obstacle over entire trajectory"""
        min_dist = float('inf')
        trajectory = self.get_trajectory()

        for pos in trajectory:
            for obs in self.obstacles:
                obs_pos = np.array(obs[:2])
                obs_radius = obs[2] if len(obs) > 2 else 1.0
                dist = np.linalg.norm(pos - obs_pos) - obs_radius
                min_dist = min(min_dist, dist)

        return min_dist

    def reached_target(self, tolerance: float = 1.0) -> bool:
        """Check if drone reached target"""
        return np.linalg.norm(self.position - self.target_position) < tolerance

    def get_trajectory_length(self) -> float:
        """Calculate total trajectory length in meters"""
        trajectory = self.get_trajectory()
        if len(trajectory) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))

    def get_time_to_target(self, tolerance: float = 1.0) -> Optional[float]:
        """
        Get time to reach target if reached, None otherwise
        """
        trajectory = self.get_trajectory()
        for i, pos in enumerate(trajectory):
            if np.linalg.norm(pos - self.target_position) < tolerance:
                return self.time_history[i]
        return None
