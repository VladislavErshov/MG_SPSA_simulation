"""Tests for maneuver optimizer module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.drone_simulator.optimizers.spsa import (
    ManeuverOptimizer,
    ManeuverOptimizerConfig,
)


def test_config_defaults():
    cfg = ManeuverOptimizerConfig()
    assert cfg.a == 1.0
    assert cfg.c == 1.0
    assert cfg.burn_in == 0
    assert cfg.epsilon_exact == 0.25
    assert cfg.d_back_min == 1.0
    assert cfg.n_spsa_samples == 3


def test_optimizer_initialization():
    cfg = ManeuverOptimizerConfig()
    opt = ManeuverOptimizer(cfg)
    params = opt.get_params()
    assert params["d_back"] == pytest.approx(cfg.d_back_init)
    assert params["omega_turn"] == pytest.approx(cfg.omega_turn_init)
    assert params["alpha_evade"] == pytest.approx(cfg.alpha_evade_init)
    assert opt.iteration == 0


def test_parameter_clipping():
    cfg = ManeuverOptimizerConfig(
        d_back_min=0.5,
        d_back_max=5.0,
        omega_turn_min=0.2,
        omega_turn_max=2.0,
        alpha_evade_min=-1.0,
        alpha_evade_max=1.0,
    )
    opt = ManeuverOptimizer(cfg)
    opt.theta = np.array([10.0, 0.05, 2.0])
    opt._clip()
    assert opt.theta[0] == pytest.approx(5.0)
    assert opt.theta[1] == pytest.approx(0.2)
    assert opt.theta[2] == pytest.approx(1.0)


def test_evaluate_spsa1():
    cfg = ManeuverOptimizerConfig(a=1.0, c=0.1, burn_in=0)
    opt = ManeuverOptimizer(cfg)

    def dummy_loss(params):
        d = params["d_back"]
        w = params["omega_turn"]
        a = params["alpha_evade"]
        return d ** 2 + w ** 2 + a ** 2

    np.random.seed(0)
    grad = opt.evaluate("spsa1", dummy_loss)
    assert opt.iteration == 1
    assert grad.shape == (3,)
    assert len(opt.history) == 1


def test_evaluate_spsa2():
    cfg = ManeuverOptimizerConfig(a=1.0, c=0.1, burn_in=0)
    opt = ManeuverOptimizer(cfg)

    def dummy_loss(params):
        d = params["d_back"]
        w = params["omega_turn"]
        a = params["alpha_evade"]
        return d ** 2 + w ** 2 + a ** 2

    np.random.seed(0)
    grad = opt.evaluate("spsa2", dummy_loss)
    assert opt.iteration == 1
    assert grad.shape == (3,)
    assert len(opt.history) == 1


def test_step_size_theory():
    cfg = ManeuverOptimizerConfig(a=2.0)
    opt = ManeuverOptimizer(cfg)

    opt.iteration = 0
    assert opt._step_size() == pytest.approx(2.0 / 1)

    opt.iteration = 10
    assert opt._step_size() == pytest.approx(2.0 / 10)


def test_perturbation_size_theory():
    cfg = ManeuverOptimizerConfig(c=0.5)
    opt = ManeuverOptimizer(cfg)

    opt.iteration = 0
    beta = opt._perturbation_size()
    assert beta == pytest.approx(0.5 / (1 ** 0.25))

    opt.iteration = 16
    beta = opt._perturbation_size()
    assert beta == pytest.approx(0.5 / (16 ** 0.25))


def test_to_dict_and_perturb():
    cfg = ManeuverOptimizerConfig()
    opt = ManeuverOptimizer(cfg)
    t = opt._perturb_theta(0, 0.5)
    d = opt._to_dict(t)
    assert d["d_back"] == pytest.approx(cfg.d_back_init + 0.5)
    assert d["omega_turn"] == pytest.approx(cfg.omega_turn_init)
    assert d["alpha_evade"] == pytest.approx(cfg.alpha_evade_init)
