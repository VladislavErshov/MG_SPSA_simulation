"""
Mixed-Variable SPSA Optimizer for Drone Trajectory Optimization
Based on "Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications"

References to Overleaf project for theoretical foundations.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List

import numpy as np

from .base import BaseOptimizer, OptimizerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SPSAConfig(OptimizerConfig):
    """Configuration for Mixed-Variable SPSA optimizer"""
    # Step size parameters - practical Spall formula: a / (A + n)^alpha
    a: float = 2.0  # Step size amplitude
    alpha: float = 0.602  # Decay exponent (standard SPSA value)
    A: float = 50.0  # Stability constant (prevents freeze in online control)

    # Perturbation size parameters - from Corollary 4.2 (q=1 balanced gamma=1/4)
    c: float = 0.5  # Perturbation amplitude (increased for better obstacle sensing)
    gamma: float = 0.25  # Perturbation exponent for q=1 effective defect

    # Mixed-variable specific parameters - from Lemma 6.1 / Theorem 3.1
    num_perturbations: int = 4  # Number of perturbations per iteration (N)
    decorrelation_exponent: float = 1.0  # Rho value for variance decay

    # Exact gradient channel parameters (w block)
    epsilon_w: float = 0.01  # Finite-difference step for exact gradient on speed

    # Parameter block sizes - from section 3.1 in mixed_variable_spsa.tex
    speed_params: int = 1  # Number of speed control parameters (w block)
    direction_params: int = 1  # Number of direction control parameters (phi block)


class MixedVariableSPSA(BaseOptimizer):
    """
    Mixed-Variable SPSA Optimizer for Drone Control

    Implements the mixed-parameter update from the article:
    - w channel (speed): exact gradient via finite differences
    - phi channel (direction): one-measurement SPSA probing

    Update rule (theta = [w, phi]):
        g_n = [g_n^{(w)}, g_n^{(phi)}]
        theta_n = theta_{n-1} - alpha_n * g_n

    where:
        g_n^{(w)}  = exact gradient of L w.r.t. w  (central FD)
        g_n^{(phi)} = (1/(N*beta_n)) * sum_j Delta_j * (L(w, phi+beta_n*Delta_j) - L(w, phi))
    """

    def __init__(self, config: SPSAConfig, loss_function: Callable):
        super().__init__(config, loss_function)
        self.config = config
        self.w_dim = config.speed_params
        self.phi_dim = config.direction_params
        self.param_dim = self.w_dim + self.phi_dim

    def _compute_step_size(self) -> float:
        """Step size alpha_n = a / (A + n)^alpha (Spall's practical formula)"""
        n = self.iteration + 1
        return self.config.a / ((self.config.A + n) ** self.config.alpha)

    def _compute_perturbation_size(self) -> float:
        """Perturbation size beta_n = c / n^gamma"""
        n = self.iteration + 1
        return self.config.c / (n ** self.config.gamma)

    def _generate_perturbation(self) -> float:
        """Generate scalar Rademacher perturbation for phi channel"""
        return float(np.random.choice([-1, 1]))

    def _compute_exact_gradient_w(self, theta: np.ndarray, **loss_kwargs) -> float:
        """
        Exact gradient for w (speed) via central finite differences.
        This is the 'exact gradient channel' g_n^{(w)} from the article.
        """
        eps = self.config.epsilon_w
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[0] += eps
        theta_minus[0] -= eps
        theta_plus = self._clip_parameters(theta_plus)
        theta_minus = self._clip_parameters(theta_minus)
        return (self.loss_function(theta_plus, **loss_kwargs) -
                self.loss_function(theta_minus, **loss_kwargs)) / (2.0 * eps)

    def _compute_spsa_gradient_phi(self, theta: np.ndarray, beta_n: float, **loss_kwargs) -> float:
        """
        One-measurement SPSA gradient for phi (direction).
        Implements g_n^{(phi)} = (1/(N*beta_n)) * sum_j Delta_j * Y_j
        where Y_j = L(w, phi + beta_n*Delta_j) - L(w, phi) is the probing increment.

        Using the increment (instead of raw loss) removes the O(beta_n^{-1})
        variance blow-up, matching the article's structure where Y_j is an
        observation increment (e.g. lambda*beta*<Delta, Psi> in linear curiosity).
        """
        n_perturb = self.config.num_perturbations
        phi = theta[1]
        loss_baseline = self.loss_function(theta, **loss_kwargs)
        s = 0.0
        for _ in range(n_perturb):
            delta = self._generate_perturbation()
            theta_pert = theta.copy()
            theta_pert[1] = phi + beta_n * delta
            theta_pert = self._clip_parameters(theta_pert)
            y = self.loss_function(theta_pert, **loss_kwargs) - loss_baseline
            s += delta * y
        return s / (n_perturb * beta_n)

    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Execute one mixed-variable SPSA step.

        Returns:
            - Updated parameters [speed, direction]
            - Loss at new parameters
            - Full gradient estimate [g_w, g_phi]
        """
        self.iteration += 1
        alpha_n = self._compute_step_size()
        beta_n = self._compute_perturbation_size()

        g_w = self._compute_exact_gradient_w(self.theta, **loss_kwargs)
        g_phi = self._compute_spsa_gradient_phi(self.theta, beta_n, **loss_kwargs)
        gradient = np.array([g_w, g_phi])

        theta_new = self.theta - alpha_n * gradient
        theta_new = self._rate_limit_direction(theta_new)
        theta_new = self._clip_parameters(theta_new)
        loss = self.loss_function(theta_new, **loss_kwargs)

        self.theta = theta_new
        self.history['parameters'].append(self.theta.copy())
        self.history['loss'].append(loss)
        self.history['gradients'].append(gradient.copy())

        return self.theta.copy(), loss, gradient.copy()


class TargetFollowingSPSA(MixedVariableSPSA):
    """
    Specialized SPSA for target following with dynamic loss function

    This implements the application-specific loss function described in:
    - section 5.2 (Drone Control Application) in applications.tex
    - implements L(theta) = L_target + L_obstacle + L_smooth + L_energy from the paper
    """

    def __init__(self, config: SPSAConfig):
        # Initialize with placeholder loss function
        super().__init__(config, self._dynamic_loss)

        # Target following state (updated each step)
        self.current_position = np.array([0.0, 0.0])
        self.target_position = np.array([0.0, 0.0])
        self.obstacles = []

    def _dynamic_loss(self, params: np.ndarray, **kwargs) -> float:
        """
        Dynamic loss function for target following
        Implements the cost function from section 5.2 of applications.tex:

        L(theta) = w1 * L_target + w2 * L_obstacle + w3 * L_smooth + w4 * L_energy

        where:
        - L_target = distance to target (equation 5.3)
        - L_obstacle = barrier function for obstacle avoidance (equation 5.4)
        - L_smooth = penalty for control input changes (equation 5.5)
        - L_energy = penalty for high speeds (equation 5.6)
        """
        speed = params[0]
        direction = params[1]

        # Predict position with look-ahead horizon
        look_ahead_time = kwargs.get('look_ahead_time', 0.5)
        dx = speed * np.cos(direction) * look_ahead_time
        dy = speed * np.sin(direction) * look_ahead_time
        next_position = self.current_position + np.array([dx, dy])

        loss = 0.0

        # 1. Distance to target (primary objective) - quadratic for stronger attraction
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        loss += dist_to_target ** 2

        # 2. Obstacle avoidance (barrier function with safety margin)
        obstacle_weight = kwargs.get('obstacle_weight', 100.0)
        avoidance_strength = kwargs.get('avoidance_strength', 10.0)
        safety_margin = kwargs.get('safety_margin', 1.0)
        min_obstacle_dist = float('inf')

        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0
            effective_radius = obs_radius + safety_margin

            dist_to_obstacle = np.linalg.norm(next_position - obs_pos)
            if dist_to_obstacle < effective_radius:
                # Collision penalty - quadratic barrier term
                loss += obstacle_weight * (effective_radius - dist_to_obstacle) ** 2
            else:
                # Soft barrier - inverse distance term
                loss += avoidance_strength / (dist_to_obstacle - effective_radius + 1e-6)

            min_obstacle_dist = min(min_obstacle_dist, dist_to_obstacle - obs_radius)

        # 3. Speed smoothness (penalize rapid speed changes)
        # From equation (5.5a) in applications.tex
        if len(self.history['parameters']) > 0:
            prev_speed = self.history['parameters'][-1][0]
            speed_smooth_weight = kwargs.get('speed_smooth_weight', 0.1)
            loss += speed_smooth_weight * (speed - prev_speed) ** 2

        # 4. Direction smoothness (penalize sharp turns)
        # From equation (5.5b) in applications.tex
        if len(self.history['parameters']) > 0:
            prev_direction = self.history['parameters'][-1][1]
            # Handle angle wrapping using arctan2
            direction_diff = np.arctan2(
                np.sin(direction - prev_direction),
                np.cos(direction - prev_direction)
            )
            dir_smooth_weight = kwargs.get('dir_smooth_weight', 0.1)
            loss += dir_smooth_weight * direction_diff ** 2

        # 5. Energy efficiency (penalize high speeds)
        # From equation (5.6) in applications.tex
        energy_weight = kwargs.get('energy_weight', 0.01)
        loss += energy_weight * speed ** 2

        return loss

    def _compute_exact_gradient_w(self, theta: np.ndarray, **loss_kwargs) -> float:
        """
        Analytical exact gradient for w (speed).
        Replaces finite-difference with closed-form derivative of _dynamic_loss.
        """
        speed = theta[0]
        direction = theta[1]
        look_ahead_time = loss_kwargs.get('look_ahead_time', 0.2)

        dir_vec = np.array([np.cos(direction), np.sin(direction)])
        next_position = self.current_position + speed * dir_vec * look_ahead_time

        grad = 0.0

        # 1. Distance to target (quadratic: dist^2)
        diff = next_position - self.target_position
        grad += 2.0 * look_ahead_time * np.dot(diff, dir_vec)

        # 2. Obstacle avoidance
        obstacle_weight = loss_kwargs.get('obstacle_weight', 100.0)
        avoidance_strength = loss_kwargs.get('avoidance_strength', 2.0)
        safety_margin = loss_kwargs.get('safety_margin', 0.3)

        for obs in self.obstacles:
            obs_pos = np.array(obs[:2])
            obs_radius = obs[2] if len(obs) > 2 else 1.0
            effective_radius = obs_radius + safety_margin

            diff_obs = next_position - obs_pos
            dist_obs = np.linalg.norm(diff_obs)
            if dist_obs < 1e-8:
                continue

            d_dist_dspeed = look_ahead_time * np.dot(diff_obs, dir_vec) / dist_obs
            if dist_obs < effective_radius:
                grad += -2.0 * obstacle_weight * (effective_radius - dist_obs) * d_dist_dspeed
            else:
                grad += -avoidance_strength / ((dist_obs - effective_radius + 1e-6) ** 2) * d_dist_dspeed

        # 3. Speed smoothness
        if len(self.history['parameters']) > 0:
            prev_speed = self.history['parameters'][-1][0]
            speed_smooth_weight = loss_kwargs.get('speed_smooth_weight', 0.2)
            grad += 2.0 * speed_smooth_weight * (speed - prev_speed)

        # 4. Energy efficiency
        energy_weight = loss_kwargs.get('energy_weight', 0.05)
        grad += 2.0 * energy_weight * speed

        return grad

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: List):
        """Update state for loss calculation"""
        self.current_position = position.copy()
        self.target_position = target.copy()
        self.obstacles = obstacles.copy()

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                       obstacles: List, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute step with updated state"""
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)


def create_test_scenario() -> tuple:
    """Create a test scenario with obstacles for standalone testing"""
    start_pos = np.array([0.0, 0.0])
    target_pos = np.array([20.0, 15.0])

    obstacles = [
        [5.0, 5.0, 2.0],
        [10.0, 8.0, 1.5],
        [15.0, 12.0, 2.5],
        [8.0, 15.0, 2.0],
        [18.0, 10.0, 1.8]
    ]

    return start_pos, target_pos, obstacles


if __name__ == "__main__":
    # Standalone test
    config = SPSAConfig()
    scenario = create_test_scenario()

    optimizer = TargetFollowingSPSA(config)
    optimizer.update_state(scenario[0], scenario[1], scenario[2])

    print("Testing SPSA Optimizer:")
    print(f"Initial parameters: speed={optimizer.get_speed():.2f}, direction={optimizer.get_direction():.2f}")

    for i in range(10):
        params, loss, gradient = optimizer.step_with_state(
            scenario[0], scenario[1], scenario[2]
        )
        print(f"Step {i+1}: params={params}, loss={loss:.4f}, grad_norm={np.linalg.norm(gradient):.4f}")
