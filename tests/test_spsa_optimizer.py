"""Tests for the SPSA optimizer module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.drone_simulator.optimizers import (
    MixedVariableSPSA,
    SPSAConfig,
    TargetFollowingSPSA,
    create_test_scenario,
)


def test_spsa_config_defaults():
    config = SPSAConfig()
    assert config.a == 1.0
    assert config.alpha == 0.602
    assert config.c == 0.1
    assert config.gamma == 0.101
    assert config.speed_min == 0.1
    assert config.speed_max == 10.0


def test_spsa_config_custom():
    config = SPSAConfig(a=2.0, c=0.5, speed_max=5.0)
    assert config.a == 2.0
    assert config.c == 0.5
    assert config.speed_max == 5.0


def test_mixed_variable_spsa_initialization():
    config = SPSAConfig()

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = MixedVariableSPSA(config, dummy_loss)
    assert optimizer.iteration == 0
    assert optimizer.param_dim == 2
    assert optimizer.get_speed() == pytest.approx((config.speed_max + config.speed_min) / 2)


def test_mixed_variable_spsa_step():
    config = SPSAConfig(a=1.0, c=0.1)

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = MixedVariableSPSA(config, dummy_loss)
    params, loss, gradient = optimizer.step()

    assert optimizer.iteration == 1
    assert len(optimizer.history['parameters']) == 1
    assert len(optimizer.history['loss']) == 1
    assert len(optimizer.history['gradients']) == 1
    assert gradient.shape == (2,)


def test_parameter_clipping():
    config = SPSAConfig(speed_min=0.1, speed_max=5.0)

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = MixedVariableSPSA(config, dummy_loss)
    optimizer.set_parameters(speed=10.0)
    assert optimizer.get_speed() == pytest.approx(5.0)

    optimizer.set_parameters(speed=0.0)
    assert optimizer.get_speed() == pytest.approx(0.1)


def test_target_following_spsa():
    config = SPSAConfig()
    optimizer = TargetFollowingSPSA(config)
    assert optimizer.current_position[0] == pytest.approx(0.0)
    assert optimizer.target_position[1] == pytest.approx(10.0)


def test_target_following_step_with_state():
    config = SPSAConfig()
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
