"""Tests for the gradient descent optimizer module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.drone_simulator.optimizers import (
    GradientDescent,
    GradientDescentConfig,
    TargetFollowingGD,
)


def test_gd_config_defaults():
    config = GradientDescentConfig()
    assert config.lr == 0.1
    assert config.epsilon == 0.01
    assert config.speed_min == 0.0
    assert config.speed_max == 10.0


def test_gd_initialization():
    config = GradientDescentConfig()

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = GradientDescent(config, dummy_loss)
    assert optimizer.iteration == 0
    assert optimizer.param_dim == 2
    assert optimizer.get_speed() == pytest.approx((config.speed_max + config.speed_min) / 2)


def test_gd_step():
    config = GradientDescentConfig(lr=0.1, epsilon=0.01)

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = GradientDescent(config, dummy_loss)
    params, loss, gradient = optimizer.step()

    assert optimizer.iteration == 1
    assert len(optimizer.history['parameters']) == 1
    assert len(optimizer.history['loss']) == 1
    assert gradient.shape == (2,)


def test_gd_parameter_clipping():
    config = GradientDescentConfig(speed_min=0.1, speed_max=5.0)

    def dummy_loss(params):
        return np.sum(params ** 2)

    optimizer = GradientDescent(config, dummy_loss)
    optimizer.set_parameters(speed=10.0)
    assert optimizer.get_speed() == pytest.approx(5.0)

    optimizer.set_parameters(speed=0.0)
    assert optimizer.get_speed() == pytest.approx(0.1)  # clipped to speed_min


def test_target_following_gd():
    config = GradientDescentConfig()
    optimizer = TargetFollowingGD(config)
    assert optimizer.current_position[0] == pytest.approx(0.0)
    assert optimizer.target_position[1] == pytest.approx(10.0)


def test_target_following_gd_step_with_state():
    config = GradientDescentConfig()
    optimizer = TargetFollowingGD(config)
    position = np.array([0.0, 0.0])
    target = np.array([10.0, 10.0])
    obstacles = []

    params, loss, gradient = optimizer.step_with_state(position, target, obstacles)
    assert loss >= 0.0
    assert params.shape == (2,)


def test_gd_deterministic_gradient():
    """GD should produce the same gradient for the same state"""
    config = GradientDescentConfig(lr=0.1, epsilon=0.01)

    def dummy_loss(params):
        return params[0] ** 2 + 2 * params[1] ** 2

    optimizer = GradientDescent(config, dummy_loss)
    _, _, grad1 = optimizer.step()

    optimizer2 = GradientDescent(config, dummy_loss)
    _, _, grad2 = optimizer2.step()

    np.testing.assert_allclose(grad1, grad2, rtol=1e-5)
