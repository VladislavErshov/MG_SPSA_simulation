"""
Mixed-Variable SPSA Optimizer for Drone Trajectory Optimization
Based on "Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications"
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SPSAConfig:
    """Configuration for Mixed-Variable SPSA optimizer"""
    # Step size parameters
    a: float = 1.0  # Step size amplitude
    alpha: float = 0.602  # Step size exponent (standard SPSA value)

    # Perturbation size parameters
    c: float = 0.1  # Perturbation amplitude
    gamma: float = 0.101  # Perturbation exponent (standard SPSA value)

    # Mixed-variable specific parameters
    num_perturbations: int = 2  # Number of perturbations per iteration (N)
    decorrelation_exponent: float = 1.0  # Rho value for variance decay

    # Parameter block sizes
    speed_params: int = 1  # Number of speed control parameters
    direction_params: int = 1  # Number of direction control parameters

    # Constraints
    speed_min: float = 0.1
    speed_max: float = 10.0
    direction_min: float = -np.pi
    direction_max: float = np.pi


class MixedVariableSPSA:
    """
    Mixed-Variable SPSA Optimizer for Drone Control

    Optimizes mixed parameters with potentially different update mechanisms:
    - Speed parameters: Direct gradient estimation
    - Direction parameters: SPSA-based perturbation
    """

    def __init__(self, config: SPSAConfig, loss_function: Callable):
        self.config = config
        self.loss_function = loss_function
        self.iteration = 0

        # Total parameter dimension
        self.param_dim = config.speed_params + config.direction_params

        # Initialize parameter vector [speed, direction]
        self.theta = np.array([
            (config.speed_max + config.speed_min) / 2,  # Initial speed
            0.0  # Initial direction (straight ahead)
        ])

        # History for tracking
        self.history = {
            'parameters': [],
            'loss': [],
            'gradients': []
        }

    def _compute_step_size(self) -> float:
        """Compute step size a_n = a / n^alpha"""
        n = self.iteration + 1
        return self.config.a / (n ** self.config.alpha)

    def _compute_perturbation_size(self) -> float:
        """Compute perturbation size c_n = c / n^gamma"""
        n = self.iteration + 1
        return self.config.c / (n ** self.config.gamma)

    def _generate_perturbation_vector(self) -> np.ndarray:
        """
        Generate random perturbation vector for SPSA
        Uses Bernoulli ±1 distribution as recommended in the paper
        """
        return np.random.choice([-1, 1], size=self.param_dim)

    def _compute_gradient_estimate(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient estimate using simultaneous perturbation
        Implements the mixed SPSA operator from Lemma 6.1
        """
        n_perturb = self.config.num_perturbations
        beta_n = self._compute_perturbation_size()

        # Initialize gradient estimate
        gradient = np.zeros(self.param_dim)

        # For each perturbation (j = 1 to N)
        perturbation_sum = np.zeros(self.param_dim)

        for j in range(n_perturb):
            # Generate random perturbation direction
            delta = self._generate_perturbation_vector()

            # Perturbed parameters: theta + beta_n * delta
            theta_plus = theta + beta_n * delta
            theta_minus = theta - beta_n * delta

            # Clip parameters to valid ranges
            theta_plus = self._clip_parameters(theta_plus)
            theta_minus = self._clip_parameters(theta_minus)

            # Evaluate loss function at perturbed points
            loss_plus = self.loss_function(theta_plus)
            loss_minus = self.loss_function(theta_minus)

            # SPSA gradient estimate for this perturbation
            # g = (loss_plus - loss_minus) / (2 * beta_n * delta)
            # Using simultaneous perturbation formula
            perturbation_sum += delta * (loss_plus - loss_minus)

        # Average over all perturbations
        gradient = perturbation_sum / (2 * beta_n * n_perturb)

        return gradient

    def _clip_parameters(self, theta: np.ndarray) -> np.ndarray:
        """Clip parameters to valid ranges"""
        theta_clipped = theta.copy()

        # Clip speed
        theta_clipped[0] = np.clip(
            theta_clipped[0],
            self.config.speed_min,
            self.config.speed_max
        )

        # Clip direction (handle angle wrapping)
        if self.config.direction_params > 0:
            theta_clipped[1] = np.arctan2(
                np.sin(theta_clipped[1]),
                np.cos(theta_clipped[1])
            )

        return theta_clipped

    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute one optimization step

        Returns:
            - Updated parameters
            - Current loss
            - Gradient estimate
        """
        self.iteration += 1

        # Compute gradient estimate
        gradient = self._compute_gradient_estimate(self.theta)

        # Compute step size
        alpha_n = self._compute_step_size()

        # Update parameters: theta = theta - alpha_n * gradient
        theta_new = self.theta - alpha_n * gradient

        # Clip to valid range
        theta_new = self._clip_parameters(theta_new)

        # Compute loss at new parameters
        loss = self.loss_function(theta_new, **loss_kwargs)

        # Update stored parameters
        self.theta = theta_new

        # Store history
        self.history['parameters'].append(self.theta.copy())
        self.history['loss'].append(loss)
        self.history['gradients'].append(gradient.copy())

        return self.theta.copy(), loss, gradient.copy()

    def get_speed(self) -> float:
        """Get current speed parameter"""
        return self.theta[0]

    def get_direction(self) -> float:
        """Get current direction parameter (radians)"""
        return self.theta[1] if self.config.direction_params > 0 else 0.0

    def set_parameters(self, speed: Optional[float] = None, direction: Optional[float] = None):
        """Manually set parameters"""
        if speed is not None:
            self.theta[0] = np.clip(speed, self.config.speed_min, self.config.speed_max)
        if direction is not None and self.config.direction_params > 0:
            self.theta[1] = direction


class TargetFollowingSPSA(MixedVariableSPSA):
    """
    Specialized SPSA for target following with dynamic loss function
    """

    def __init__(self, config: SPSAConfig):
        # Initialize with a placeholder loss function
        super().__init__(config, self._dynamic_loss)

        # Target following specific attributes
        self.current_position = np.array([0.0, 0.0])
        self.target_position = np.array([10.0, 10.0])
        self.obstacles = []
        self.waypoints = []

    def _dynamic_loss(self, params: np.ndarray, **kwargs) -> float:
        """
        Dynamic loss function for target following
        Combines distance to target, obstacle avoidance, and smoothness
        """
        speed = params[0]
        direction = params[1]

        # Predict next position
        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        next_position = self.current_position + np.array([dx, dy])

        # Loss components
        loss = 0.0

        # 1. Distance to target (primary objective)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        loss += dist_to_target

        # 2. Obstacle avoidance (barrier function)
        obstacle_weight = kwargs.get('obstacle_weight', 100.0)
        min_obstacle_dist = float('inf')

        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0

            dist_to_obstacle = np.linalg.norm(next_position - obs_pos)
            if dist_to_obstacle < obs_radius:
                # Collision penalty
                loss += obstacle_weight * (obs_radius - dist_to_obstacle) ** 2
            else:
                # Soft barrier
                avoidance_strength = kwargs.get('avoidance_strength', 1.0)
                loss += avoidance_strength / (dist_to_obstacle - obs_radius + 1e-6)

        # 3. Speed smoothness (penalize rapid speed changes)
        if len(self.history['parameters']) > 0:
            prev_speed = self.history['parameters'][-1][0]
            speed_smooth_weight = kwargs.get('speed_smooth_weight', 0.1)
            loss += speed_smooth_weight * (speed - prev_speed) ** 2

        # 4. Direction smoothness (penalize sharp turns)
        if len(self.history['parameters']) > 0:
            prev_direction = self.history['parameters'][-1][1]
            # Handle angle wrapping
            direction_diff = np.arctan2(
                np.sin(direction - prev_direction),
                np.cos(direction - prev_direction)
            )
            dir_smooth_weight = kwargs.get('dir_smooth_weight', 0.1)
            loss += dir_smooth_weight * direction_diff ** 2

        # 5. Energy efficiency (penalize high speeds)
        energy_weight = kwargs.get('energy_weight', 0.01)
        loss += energy_weight * speed ** 2

        return loss

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: list):
        """Update the state for loss calculation"""
        self.current_position = position.copy()
        self.target_position = target.copy()
        self.obstacles = obstacles.copy()

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                       obstacles: list, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute step with updated state"""
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)


def create_test_scenario() -> tuple:
    """Create a test scenario with obstacles"""
    # Start position
    start_pos = np.array([0.0, 0.0])

    # Target position
    target_pos = np.array([20.0, 15.0])

    # Obstacles (x, y, radius)
    obstacles = [
        [5.0, 5.0, 2.0],
        [10.0, 8.0, 1.5],
        [15.0, 12.0, 2.5],
        [8.0, 15.0, 2.0],
        [18.0, 10.0, 1.8]
    ]

    return start_pos, target_pos, obstacles


if __name__ == "__main__":
    # Test the optimizer
    config = SPSAConfig()
    scenario = create_test_scenario()

    # Create optimizer
    optimizer = TargetFollowingSPSA(config)

    # Set initial state
    optimizer.update_state(scenario[0], scenario[1], scenario[2])

    # Run optimization for a few steps
    print("Testing SPSA Optimizer:")
    print(f"Initial parameters: speed={optimizer.get_speed():.2f}, direction={optimizer.get_direction():.2f}")

    for i in range(10):
        params, loss, gradient = optimizer.step_with_state(
            scenario[0], scenario[1], scenario[2]
        )
        print(f"Step {i+1}: params={params}, loss={loss:.4f}, grad_norm={np.linalg.norm(gradient):.4f}")
