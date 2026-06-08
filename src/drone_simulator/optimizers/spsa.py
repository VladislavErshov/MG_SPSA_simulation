"""
Maneuver-parameter optimizer.

theta = [d_back, omega_turn, alpha_evade]
- d_back      : exact gradient via central finite difference
- omega_turn  : exact gradient via central finite difference
- alpha_evade : SPSA (one-measurement in spsa1, centered in spsa2)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ManeuverOptimizerConfig:
    """Configuration for maneuver-parameter optimizer."""

    a: float = 5.0  # step-size amplitude
    c: float = 1.0  # perturbation amplitude
    A: float = 5.0  # stability constant for step-size decay
    burn_in: int = 0
    epsilon_exact: float = 0.25  # FD step for exact blocks

    step_size_exponent: float = 1.0  # α_n ∝ n^{-p}  (paper: p=1)
    perturbation_exponent_spsa1: float = 0.25  # β_n ∝ n^{-γ}, q=1 → γ=1/4
    perturbation_exponent_spsa2: float = 1.0 / 6.0  # centered q=2 → γ=1/6

    # Parameter bounds
    d_back_min: float = 0.5
    d_back_max: float = 20.0
    omega_turn_min: float = 0.05
    omega_turn_max: float = 10.0
    alpha_evade_min: float = -2 * np.pi
    alpha_evade_max: float = 2 * np.pi

    # SPSA smoothing
    n_spsa_samples: int = 3

    # Initial values
    d_back_init: float = 2.0
    omega_turn_init: float = 1.0
    alpha_evade_init: float = 1.0


class ManeuverOptimizer:
    """Optimizer for [d_back, omega_turn, alpha_evade]."""

    def __init__(self, config: ManeuverOptimizerConfig):
        self.config = config
        self.theta = np.array(
            [
                config.d_back_init,
                config.omega_turn_init,
                config.alpha_evade_init,
            ]
        )
        self.iteration = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, mode: str, run_fn: Callable[[Dict], float]) -> np.ndarray:
        """
        Compute gradient and update theta.

        Parameters
        ----------
        mode : 'spsa1' | 'spsa2'
        run_fn : callable(theta_dict) -> loss_float
            Must accept a dict with keys 'd_back', 'omega_turn', 'alpha_evade'.

        Returns
        -------
        gradient : np.ndarray of shape (3,)
        """
        self.iteration += 1
        alpha_n = self._step_size()
        beta_n = self._perturbation_size(mode)
        eps = self.config.epsilon_exact

        # --- exact blocks (central finite difference) ---------------
        # d_back  (index 0)
        grad_0 = self._central_fd(run_fn, eps, coord=0)

        # omega_turn  (index 1)
        grad_1 = self._central_fd(run_fn, eps, coord=1)

        # --- SPSA block (alpha_evade, index 2) ----------------------
        grad_2 = 0.0
        for _ in range(self.config.n_spsa_samples):
            delta = float(np.random.choice([-1.0, 1.0]))

            if mode == "spsa1":
                loss_pert = run_fn(
                    self._to_dict(self._perturb_theta(2, beta_n * delta))
                )
                loss_base = run_fn(self._to_dict(self.theta))
                grad_2 += delta * (loss_pert - loss_base) / beta_n
            elif mode == "spsa2":
                loss_plus = run_fn(
                    self._to_dict(self._perturb_theta(2, beta_n * delta))
                )
                loss_minus = run_fn(
                    self._to_dict(self._perturb_theta(2, -beta_n * delta))
                )
                grad_2 += delta * (loss_plus - loss_minus) / (2.0 * beta_n)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        grad_2 /= self.config.n_spsa_samples

        grad = np.array([grad_0, grad_1, grad_2])

        # Normalize by component-wise max to keep relative magnitudes but cap scale
        max_abs = np.max(np.abs(grad))
        if max_abs > 0:
            grad = grad / max_abs

        # --- update -------------------------------------------------
        self.theta = self.theta - alpha_n * grad
        self._clip()

        # --- history ------------------------------------------------
        self.history.append(
            {
                "theta": self.theta.copy(),
                "grad": grad.copy(),
                "loss": run_fn(self._to_dict(self.theta)),
            }
        )

        return grad

    def get_params(self, use_best: bool = False) -> Dict:
        if not use_best or not self.history:
            return self._to_dict(self.theta)
        best = min(self.history, key=lambda h: h["loss"])
        return self._to_dict(best["theta"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _step_size(self) -> float:
        n = max(self.iteration, 1) + self.config.burn_in
        return self.config.a / (n + self.config.A) ** self.config.step_size_exponent

    def _perturbation_size(self, mode: str) -> float:
        n = max(self.iteration, 1) + self.config.burn_in
        if mode == "spsa2":
            gamma = self.config.perturbation_exponent_spsa2
        else:
            gamma = self.config.perturbation_exponent_spsa1
        return self.config.c / (n ** gamma)

    def _perturb_theta(self, coord: int, delta: float) -> np.ndarray:
        t = self.theta.copy()
        t[coord] += delta
        return t

    def _central_fd(
        self, run_fn: Callable[[Dict], float], eps: float, coord: int
    ) -> float:
        loss_plus = run_fn(self._to_dict(self._perturb_theta(coord, eps)))
        loss_minus = run_fn(self._to_dict(self._perturb_theta(coord, -eps)))
        return (loss_plus - loss_minus) / (2.0 * eps)

    def _to_dict(self, theta: np.ndarray) -> Dict:
        return {
            "d_back": float(theta[0]),
            "omega_turn": float(theta[1]),
            "alpha_evade": float(theta[2]),
        }

    def _clip(self):
        cfg = self.config
        self.theta[0] = np.clip(self.theta[0], cfg.d_back_min, cfg.d_back_max)
        self.theta[1] = np.clip(
            self.theta[1], cfg.omega_turn_min, cfg.omega_turn_max
        )
        self.theta[2] = np.clip(
            self.theta[2], cfg.alpha_evade_min, cfg.alpha_evade_max
        )
