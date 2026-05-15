"""Tests for the drone simulator module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.drone_simulator.core import (
    Drone,
    DroneConfig,
    DroneSimulator,
    SimulationConfig,
)


def test_drone_config_defaults():
    config = DroneConfig()
    assert config.inertia_alpha == 0.5  # New inertia coefficient
    assert config.max_speed == 10.0
    assert config.max_acceleration == 5.0
    assert config.dt == 0.05
    assert config.collision_detection == True
    assert config.optimizer_type == "spsa"


def test_drone_initialization():
    drone = Drone(np.array([0.0, 0.0]))
    assert np.allclose(drone.position, [0.0, 0.0])
    assert np.allclose(drone.velocity, [0.0, 0.0])
    assert drone.time == 0.0
    assert len(drone.trajectory) == 1


def test_drone_set_target():
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target(np.array([10.0, 10.0]))
    assert np.allclose(drone.target_position, [10.0, 10.0])


def test_drone_set_obstacles():
    drone = Drone(np.array([0.0, 0.0]))
    obstacles = [[5.0, 5.0, 2.0], [10.0, 10.0, 1.5]]
    drone.set_obstacles(obstacles)
    assert len(drone.obstacles) == 2


def test_drone_step():
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target(np.array([10.0, 10.0]))
    drone.set_obstacles([])

    initial_pos = drone.position.copy()
    drone.step()

    assert len(drone.trajectory) == 2
    assert drone.time > 0
    assert len(drone.velocity_history) == 2
    assert len(drone.command_velocity_history) == 2
    assert len(drone.in_collision_history) == 2


def test_drone_reached_target():
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target(np.array([10.0, 10.0]))
    assert not drone.reached_target()

    drone.position = np.array([10.0, 10.0])
    assert drone.reached_target()


def test_drone_get_state():
    drone = Drone(np.array([1.0, 2.0]))
    state = drone.get_state()
    assert 'position' in state
    assert 'velocity' in state
    assert 'command_velocity' in state
    assert 'speed' in state
    assert 'direction' in state
    assert 'command_speed' in state
    assert 'command_direction' in state
    assert 'in_collision' in state
    assert np.allclose(state['position'], [1.0, 2.0])


def test_drone_get_trajectory():
    drone = Drone(np.array([0.0, 0.0]))
    traj = drone.get_trajectory()
    assert traj.shape == (1, 2)


def test_simulation_config_defaults():
    config = SimulationConfig()
    assert config.duration == 30.0
    assert config.dt == 0.05
    assert config.update_interval == 1


def test_simulator_initialization():
    drone = Drone(np.array([0.0, 0.0]))
    simulator = DroneSimulator([drone])
    assert len(simulator.drones) == 1
    assert simulator.current_time == 0.0


def test_simulator_step():
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target(np.array([10.0, 10.0]))
    drone.set_obstacles([])
    simulator = DroneSimulator([drone], SimulationConfig(duration=1.0, dt=0.05))

    result = simulator.step()
    assert result is True
    assert simulator.current_time == pytest.approx(0.05)


def test_simulator_step_time_limit():
    drone = Drone(np.array([0.0, 0.0]))
    simulator = DroneSimulator([drone], SimulationConfig(duration=0.0, dt=0.05))
    result = simulator.step()
    assert result is False


def test_simulator_run_no_visualization():
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target(np.array([5.0, 5.0]))
    drone.set_obstacles([])
    simulator = DroneSimulator(
        [drone],
        SimulationConfig(duration=0.2, dt=0.05, update_interval=1)
    )
    result = simulator.run(visualize=False, save_animation=False)
    assert result is True
    assert simulator.current_time >= 0.2


def test_drone_collision_detection():
    """Test collision detection functionality"""
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_obstacles([[2.0, 0.0, 1.0], [5.0, 5.0, 1.5]])

    # Check collision at obstacle center
    assert drone._check_collision(np.array([2.0, 0.0])) == True

    # Check no collision far away
    assert drone._check_collision(np.array([0.0, 0.0])) == False

    # Check collision at edge
    assert drone._check_collision(np.array([2.5, 0.0])) == True


def test_drone_metrics():
    """Test metric calculation functions"""
    drone = Drone(np.array([0.0, 0.0]))
    drone.set_target([10.0, 10.0])
    drone.set_obstacles([[5.0, 5.0, 2.0]])

    # Test initial metrics
    assert drone.get_trajectory_length() == 0.0
    assert drone.get_collision_count() == 0

    # Manual trajectory for testing points: (0,0) -> (1,0) -> (2,1)
    # Distance: 1.0 + sqrt(2) ≈ 1.0 + 1.414 = 2.414
    drone.trajectory = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 1.0])]
    assert drone.get_trajectory_length() == pytest.approx(1.0 + np.sqrt(2), rel=0.01)


def test_drone_with_mixed_optimizer():
    from src.drone_simulator.optimizers import TargetFollowingSPSA
    config = DroneConfig(optimizer_type="spsa")
    drone = Drone(np.array([0.0, 0.0]), config)
    assert drone.config.optimizer_type == "spsa"
    assert isinstance(drone.optimizer, TargetFollowingSPSA)


def test_drone_mixed_step():
    config = DroneConfig(optimizer_type="spsa")
    drone = Drone(np.array([0.0, 0.0]), config)
    drone.set_target(np.array([10.0, 10.0]))
    drone.set_obstacles([])

    drone.step()
    assert len(drone.trajectory) == 2
    assert drone.time > 0


def test_simulator_run_with_mixed():
    config = DroneConfig(optimizer_type="spsa")
    drone = Drone(np.array([0.0, 0.0]), config)
    drone.set_target(np.array([5.0, 5.0]))
    drone.set_obstacles([])
    simulator = DroneSimulator(
        [drone],
        SimulationConfig(duration=0.2, dt=0.05, update_interval=1)
    )
    result = simulator.run(visualize=False, save_animation=False)
    assert result is True
    assert simulator.current_time >= 0.2
