"""
Mixed-Variable Optimizer aligned with article theory.

Implements the modular mixed-gradient framework from
"Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications".

Each parameter block can use its own gradient estimator:
- exact      : conditionally unbiased gradient (e.g. analytical or central FD)
- spsa_off_center : one-measurement SPSA, effective defect order q=1
- spsa_centered   : centered stencil SPSA, defect order q>=2

Gain sequences follow the article exactly:
    alpha_n = a / n                         (step size)
    beta_n  = c / n^{gamma}                 (perturbation size)
where gamma is chosen from the defect order q:
    off-center (q=1) : gamma = 1/4
    centered (q>=2)  : gamma = 1 / (2*(q+1))
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, List

import numpy as np

from .base import BaseOptimizer, OptimizerConfig, BlockConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MixedOptimizerConfig(OptimizerConfig):
    """Configuration for the modular Mixed Optimizer.

    Step size: alpha_n = a / n  (article theory, alpha=1.0)
    Perturbation: beta_n = c / n^{gamma}  (gamma auto-derived from block q)
    """
    a: float = 1.0               # Step-size amplitude
    c: float = 0.2               # Perturbation amplitude
    burn_in: int = 0             # Offset n by burn_in: alpha_n = a/(n+burn_in)
                                 # burn_in=0 gives pure a/n (article theory);
                                 # burn_in>0 stabilises early online-control steps.
    num_perturbations: int = 8   # N probes per SPSA block (Lemma 6.1)
    decorrelation_exponent: float = 1.0  # Rho in N^{-rho} variance decay

    # Exact gradient channel (central FD fallback when no analytic grad is supplied)
    epsilon_exact: float = 0.01

    # Optional EMA smoothing and clipping for SPSA blocks
    gradient_momentum: float = 0.0   # 0 = no EMA (article does not require it)
    max_grad_norm: float = None      # Per-block gradient clipping


class MixedOptimizer(BaseOptimizer):
    """Modular mixed optimizer: each block chooses its own gradient estimator."""

    def __init__(self, config: MixedOptimizerConfig, loss_function: Callable,
                 blocks: Optional[List[BlockConfig]] = None):
        super().__init__(config, loss_function)
        self.config = config
        self.blocks = blocks or []
        self.param_dim = sum(
            (sl.stop or self.param_dim) - (sl.start or 0)
            for sl, _, _ in self._iter_blocks()
        )
        # Re-initialize theta if param_dim changed from BaseOptimizer default
        if self.theta.shape[0] != self.param_dim:
            new_theta = np.zeros(self.param_dim)
            min_dim = min(self.theta.shape[0], self.param_dim)
            new_theta[:min_dim] = self.theta[:min_dim]
            self.theta = new_theta
        # EMA state per SPSA block
        self._ema = {}

    def _iter_blocks(self):
        """Yield (slice, method, q) for every configured block."""
        for b in self.blocks:
            yield b.param_slice, b.method, b.q

    # ------------------------------------------------------------------
    # Gain sequences — exactly as in the article
    # ------------------------------------------------------------------
    def _compute_step_size(self) -> float:
        """alpha_n = a / (n + burn_in).

        burn_in=0 recovers the article's pure a/n (Theorem 3.1 / Corollary 4.2).
        burn_in>0 is a practical stabiliser for online control; it does not
        change the asymptotic rate because n+burn_in ~ n as n -> infinity.
        """
        n = max(self.iteration, 1) + self.config.burn_in
        return self.config.a / n

    @staticmethod
    def _gamma_from_q(method: str, q: int) -> float:
        """Balanced gamma derived from defect order q.

        Off-center (one-measurement) probing has effective q=1 regardless of
        stencil, because the probe-center bias induces a first-order defect.
        Centered stencils eliminate that bias, so the defect order equals the
        stencil order q.

        Balancing condition: 1 - 2*gamma = 2*q*gamma  =>  gamma = 1/(2*(q+1))
        For q=1 off-center: gamma = 1/4.
        """
        if method == "spsa_off_center":
            return 0.25
        if method == "spsa_centered":
            return 1.0 / (2.0 * (q + 1))
        return 0.0

    def _compute_perturbation_size(self, gamma: float) -> float:
        """beta_n = c / (n + burn_in)^{gamma}"""
        n = max(self.iteration, 1) + self.config.burn_in
        return self.config.c / (n ** gamma)

    # ------------------------------------------------------------------
    # Gradient estimators per block
    # ------------------------------------------------------------------
    def _generate_rademacher(self, size: int = 1) -> np.ndarray:
        return np.random.choice([-1.0, 1.0], size=size)

    def _compute_exact_gradient(self, theta: np.ndarray, block_slice: slice,
                                analytic_fn: Optional[Callable] = None,
                                **loss_kwargs) -> np.ndarray:
        """Exact gradient for a block.

        If ``analytic_fn`` is supplied it is called directly;
        otherwise a central finite-difference approximation is used.
        """
        if analytic_fn is not None:
            return analytic_fn(theta, **loss_kwargs)

        eps = self.config.epsilon_exact
        dim = theta[block_slice].shape[0]
        grad = np.zeros(dim)
        idx_start = block_slice.start or 0
        for i in range(dim):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[idx_start + i] += eps
            theta_minus[idx_start + i] -= eps
            theta_plus = self._clip_parameters(theta_plus)
            theta_minus = self._clip_parameters(theta_minus)
            grad[i] = (self.loss_function(theta_plus, **loss_kwargs) -
                       self.loss_function(theta_minus, **loss_kwargs)) / (2.0 * eps)
        return grad

    def _compute_spsa_gradient(self, theta: np.ndarray, block_slice: slice,
                               beta_n: float, q: int, **loss_kwargs) -> np.ndarray:
        """One-measurement SPSA gradient for a block.

        Implements g_n^{(phi)} = (1/(N*beta_n)) * sum_j Delta_j * Y_j
        with the increment Y_j = L(theta_pert) - L(theta) to avoid the
        O(beta_n^{-1}) variance blow-up (article Section 5.2, linear curiosity).
        """
        n_perturb = self.config.num_perturbations
        dim = theta[block_slice].shape[0]
        loss_baseline = self.loss_function(theta, **loss_kwargs)
        grad = np.zeros(dim)

        for _ in range(n_perturb):
            delta = self._generate_rademacher(size=dim)
            theta_pert = theta.copy()
            theta_pert[block_slice] = theta[block_slice] + beta_n * delta
            theta_pert = self._clip_parameters(theta_pert)
            y = self.loss_function(theta_pert, **loss_kwargs) - loss_baseline
            grad += delta * y

        return grad / (n_perturb * beta_n)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Execute one mixed optimization step.

        Returns:
            theta, loss, gradient (full vector)
        """
        self.iteration += 1
        alpha_n = self._compute_step_size()

        gradient = np.zeros_like(self.theta)

        for block_slice, method, q in self._iter_blocks():
            if method == "exact":
                grad = self._compute_exact_gradient(
                    self.theta, block_slice, **loss_kwargs)
            elif method in ("spsa_off_center", "spsa_centered"):
                gamma = self._gamma_from_q(method, q)
                beta_n = self._compute_perturbation_size(gamma)
                grad = self._compute_spsa_gradient(
                    self.theta, block_slice, beta_n, q, **loss_kwargs)

                # Optional EMA smoothing (not required by theory, but
                # useful for high-variance online control)
                if self.config.gradient_momentum > 0.0:
                    key = (block_slice.start, block_slice.stop)
                    prev = self._ema.get(key, 0.0)
                    m = self.config.gradient_momentum
                    grad = m * prev + (1.0 - m) * grad
                    self._ema[key] = grad.copy()

                # Optional clipping
                if self.config.max_grad_norm is not None:
                    norm = np.linalg.norm(grad)
                    if norm > self.config.max_grad_norm:
                        grad = grad * (self.config.max_grad_norm / norm)
            else:
                raise ValueError(f"Unknown block method: {method}")

            gradient[block_slice] = grad

        theta_new = self.theta - alpha_n * gradient
        theta_new = self._rate_limit_direction(theta_new)
        theta_new = self._clip_parameters(theta_new)
        loss = self.loss_function(theta_new, **loss_kwargs)

        self.theta = theta_new
        self.history['parameters'].append(self.theta.copy())
        self.history['loss'].append(loss)
        self.history['gradients'].append(gradient.copy())

        return self.theta.copy(), loss, gradient.copy()


# ===================================================================
# Drone-specific specialization
# ===================================================================

class TargetFollowingSPSA(MixedOptimizer):
    """Drone optimizer: speed + wind by exact gradient, direction by one-measurement SPSA."""

    def __init__(self, config: MixedOptimizerConfig):
        blocks = [
            BlockConfig(slice(0, 1), method="exact", q=0),
            BlockConfig(slice(1, 2), method="spsa_off_center", q=1),
            BlockConfig(slice(2, 3), method="exact", q=0),
        ]
        super().__init__(config, self._dynamic_loss, blocks=blocks)

        # Re-initialize theta for 3 parameters: [speed, direction, wind_estimate]
        self.theta = np.array([
            (config.speed_max + config.speed_min) / 2,
            0.0,
            0.0,
        ])

        self.current_position = np.array([0.0, 0.0])
        self.target_position = np.array([0.0, 0.0])
        self.obstacles = []
        self.wind_vector = np.array([0.0, 0.0])

    def _dynamic_loss(self, params: np.ndarray, **kwargs) -> float:
        """Dynamic loss from article Section 5.2 (Drone Control)."""
        speed = params[0]
        direction = params[1]
        wind_estimate = params[2] if len(params) > 2 else 0.0

        look_ahead_time = kwargs.get('look_ahead_time', 0.5)

        # Wind estimate: assume same direction as true wind, optimize magnitude
        if np.linalg.norm(self.wind_vector) > 1e-8:
            wind_dir = self.wind_vector / np.linalg.norm(self.wind_vector)
        else:
            wind_dir = np.array([0.0, 0.0])
        predicted_wind = wind_estimate * wind_dir

        dx = (speed * np.cos(direction) + predicted_wind[0]) * look_ahead_time
        dy = (speed * np.sin(direction) + predicted_wind[1]) * look_ahead_time
        next_position = self.current_position + np.array([dx, dy])

        loss = 0.0

        # 1. Distance to target (quadratic)
        dist_to_target = np.linalg.norm(next_position - self.target_position)
        loss += dist_to_target ** 2

        # 2. Obstacle avoidance (barrier)
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

        # 6. Near-target braking (prevents overshoot)
        dist_to_target_current = np.linalg.norm(self.current_position - self.target_position)
        if dist_to_target_current < 5.0:
            braking_weight = (5.0 - dist_to_target_current) * 0.5
            loss += braking_weight * speed

        return loss

    def _compute_exact_gradient(self, theta: np.ndarray, block_slice: slice,
                                analytic_fn: Optional[Callable] = None,
                                **loss_kwargs) -> np.ndarray:
        """Override exact gradient to use analytical form for speed and wind blocks."""
        if block_slice == slice(0, 1):
            return np.array([self._analytical_gradient_speed(theta, **loss_kwargs)])
        if block_slice == slice(2, 3):
            return np.array([self._analytical_gradient_wind(theta, **loss_kwargs)])
        return super()._compute_exact_gradient(theta, block_slice, analytic_fn, **loss_kwargs)

    def _analytical_gradient_speed(self, theta: np.ndarray, **loss_kwargs) -> float:
        """Analytical partial derivative dL/d(speed)."""
        speed = theta[0]
        direction = theta[1]
        wind_estimate = theta[2] if len(theta) > 2 else 0.0
        look_ahead_time = loss_kwargs.get('look_ahead_time', 0.5)

        dir_vec = np.array([np.cos(direction), np.sin(direction)])

        if np.linalg.norm(self.wind_vector) > 1e-8:
            wind_dir = self.wind_vector / np.linalg.norm(self.wind_vector)
        else:
            wind_dir = np.array([0.0, 0.0])
        predicted_wind = wind_estimate * wind_dir

        next_position = self.current_position + (speed * dir_vec + predicted_wind) * look_ahead_time

        grad = 0.0

        # 1. Distance to target
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

        # 5. Near-target braking
        dist_to_target_current = np.linalg.norm(self.current_position - self.target_position)
        if dist_to_target_current < 5.0:
            grad += (5.0 - dist_to_target_current) * 0.5

        return grad

    def _analytical_gradient_wind(self, theta: np.ndarray, **loss_kwargs) -> float:
        """Analytical partial derivative dL/d(wind_estimate)."""
        speed = theta[0]
        direction = theta[1]
        wind_estimate = theta[2] if len(theta) > 2 else 0.0
        look_ahead_time = loss_kwargs.get('look_ahead_time', 0.5)

        dir_vec = np.array([np.cos(direction), np.sin(direction)])

        if np.linalg.norm(self.wind_vector) > 1e-8:
            wind_dir = self.wind_vector / np.linalg.norm(self.wind_vector)
        else:
            wind_dir = np.array([0.0, 0.0])
        predicted_wind = wind_estimate * wind_dir

        next_position = self.current_position + (speed * dir_vec + predicted_wind) * look_ahead_time

        grad = 0.0

        # 1. Distance to target
        diff = next_position - self.target_position
        grad += 2.0 * look_ahead_time * np.dot(diff, wind_dir)

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

            d_dist_dwind = look_ahead_time * np.dot(diff_obs, wind_dir) / dist_obs
            if dist_obs < effective_radius:
                grad += -2.0 * obstacle_weight * (effective_radius - dist_obs) * d_dist_dwind
            else:
                grad += -avoidance_strength / ((dist_obs - effective_radius + 1e-6) ** 2) * d_dist_dwind

        return grad

    def set_wind(self, wind_vector: np.ndarray):
        self.wind_vector = np.array(wind_vector).astype(float)

    def update_state(self, position: np.ndarray, target: np.ndarray, obstacles: List):
        self.current_position = position.copy()
        self.target_position = target.copy()
        self.obstacles = obstacles.copy()

    def step_with_state(self, position: np.ndarray, target: np.ndarray,
                        obstacles: List, **loss_kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        self.update_state(position, target, obstacles)
        return self.step(**loss_kwargs)


def create_test_scenario() -> tuple:
    start_pos = np.array([0.0, 0.0])
    target_pos = np.array([20.0, 15.0])
    obstacles = [
        [5.0, 5.0, 2.0],
        [10.0, 8.0, 1.5],
        [15.0, 12.0, 2.5],
        [8.0, 15.0, 2.0],
        [18.0, 10.0, 1.8],
    ]
    return start_pos, target_pos, obstacles


if __name__ == "__main__":
    config = MixedOptimizerConfig()
    scenario = create_test_scenario()

    optimizer = TargetFollowingSPSA(config)
    optimizer.update_state(scenario[0], scenario[1], scenario[2])

    print("Testing Mixed Optimizer:")
    print(f"Initial: speed={optimizer.get_speed():.2f}, dir={optimizer.get_direction():.2f}")

    for i in range(10):
        params, loss, gradient = optimizer.step_with_state(
            scenario[0], scenario[1], scenario[2]
        )
        print(f"Step {i+1}: params={params}, loss={loss:.4f}, grad_norm={np.linalg.norm(gradient):.4f}")
