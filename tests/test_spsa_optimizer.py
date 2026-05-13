"""Tests for the mixed optimizer module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.drone_simulator.optimizers import (
    MixedOptimizer,
    MixedOptimizerConfig,
    BlockConfig,
    TargetFollowingSPSA,
    create_test_scenario,
)


def test_mixed_optimizer_config_defaults():
    config = MixedOptimizerConfig()
    assert config.a == 1.0
    assert config.c == 0.2
    assert config.num_perturbations == 8
    assert config.epsilon_exact == 0.01
    assert config.speed_min == 0.0
    assert config.speed_max == 10.0


def test_mixed_optimizer_initialization():
    config = MixedOptimizerConfig()

    def dummy_loss(params):
        return np.sum(params ** 2)

    blocks = [
        BlockConfig(slice(0, 1), method="exact", q=0),
        BlockConfig(slice(1, 2), method="spsa_off_center", q=1),
    ]
    optimizer = MixedOptimizer(config, dummy_loss, blocks=blocks)
    assert optimizer.iteration == 0
    assert optimizer.get_speed() == pytest.approx((config.speed_max + config.speed_min) / 2)


def test_mixed_optimizer_step():
    config = MixedOptimizerConfig(a=1.0, c=0.1)

    def dummy_loss(params):
        return np.sum(params ** 2)

    blocks = [
        BlockConfig(slice(0, 1), method="exact", q=0),
        BlockConfig(slice(1, 2), method="spsa_off_center", q=1),
    ]
    optimizer = MixedOptimizer(config, dummy_loss, blocks=blocks)
    params, loss, gradient = optimizer.step()

    assert optimizer.iteration == 1
    assert len(optimizer.history['parameters']) == 1
    assert len(optimizer.history['loss']) == 1
    assert len(optimizer.history['gradients']) == 1
    assert gradient.shape == (2,)


def test_parameter_clipping():
    config = MixedOptimizerConfig(speed_min=0.1, speed_max=5.0)

    def dummy_loss(params):
        return np.sum(params ** 2)

    blocks = [BlockConfig(slice(0, 2), method="exact", q=0)]
    optimizer = MixedOptimizer(config, dummy_loss, blocks=blocks)
    optimizer.set_parameters(speed=10.0)
    assert optimizer.get_speed() == pytest.approx(5.0)

    optimizer.set_parameters(speed=0.0)
    assert optimizer.get_speed() == pytest.approx(0.1)


def test_target_following_spsa():
    config = MixedOptimizerConfig()
    optimizer = TargetFollowingSPSA(config)
    assert optimizer.current_position[0] == pytest.approx(0.0)
    assert optimizer.target_position[1] == pytest.approx(0.0)


def test_target_following_init_with_target():
    config = MixedOptimizerConfig()
    optimizer = TargetFollowingSPSA(config)
    position = np.array([0.0, 0.0])
    target = np.array([10.0, 10.0])
    optimizer.update_state(position, target, [])
    assert optimizer.target_position[1] == pytest.approx(10.0)


def test_target_following_step_with_state():
    config = MixedOptimizerConfig()
    optimizer = TargetFollowingSPSA(config)
    position = np.array([0.0, 0.0])
    target = np.array([10.0, 10.0])
    obstacles = []

    params, loss, gradient = optimizer.step_with_state(position, target, obstacles)
    assert loss >= 0.0
    assert params.shape == (2,)


def test_create_test_scenario():
    start, target, obstacles = create_test_scenario()
    assert start.shape == (2,)
    assert target.shape == (2,)
    assert len(obstacles) > 0
    assert len(obstacles[0]) == 3


def test_gamma_from_q():
    """Balanced gamma must match article formulas."""
    assert MixedOptimizer._gamma_from_q("spsa_off_center", 1) == pytest.approx(0.25)
    assert MixedOptimizer._gamma_from_q("spsa_centered", 2) == pytest.approx(1.0 / 6.0)
    assert MixedOptimizer._gamma_from_q("spsa_centered", 4) == pytest.approx(1.0 / 10.0)


def test_step_size_theory():
    """Step size must follow alpha_n = a / n exactly."""
    config = MixedOptimizerConfig(a=2.0)

    def dummy_loss(params):
        return np.sum(params ** 2)

    blocks = [BlockConfig(slice(0, 2), method="exact", q=0)]
    optimizer = MixedOptimizer(config, dummy_loss, blocks=blocks)

    optimizer.iteration = 0
    assert optimizer._compute_step_size() == pytest.approx(2.0 / 1)

    optimizer.iteration = 10
    assert optimizer._compute_step_size() == pytest.approx(2.0 / 10)


def test_perturbation_size_theory():
    """Perturbation size must follow beta_n = c / n^{gamma}."""
    config = MixedOptimizerConfig(c=0.5)

    def dummy_loss(params):
        return np.sum(params ** 2)

    blocks = [BlockConfig(slice(0, 2), method="spsa_off_center", q=1)]
    optimizer = MixedOptimizer(config, dummy_loss, blocks=blocks)

    optimizer.iteration = 0
    beta = optimizer._compute_perturbation_size(0.25)
    assert beta == pytest.approx(0.5 / (1 ** 0.25))

    optimizer.iteration = 16
    beta = optimizer._compute_perturbation_size(0.25)
    assert beta == pytest.approx(0.5 / (16 ** 0.25))
