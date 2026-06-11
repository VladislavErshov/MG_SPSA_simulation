"""Tests for Drone physics and collision detection."""

import numpy as np
import pytest

from drone_simulator.core.drone import Drone, ManeuverParams
from drone_simulator.core.obstacles import Circle, Rect, Diamond, Star, Cross


class TestDroneFlight:
    """Tests for basic flight behavior."""

    def test_straight_flight_no_obstacles(self):
        """Given no obstacles, when flying, then drone reaches target quickly."""
        drone = Drone(
            start_pos=[0.0, 0.0],
            target_pos=[10.0, 0.0],
            obstacles=[],
            speed=5.0,
            dt=0.1,
            max_duration=10.0,
        )
        result = drone.fly_episode(ManeuverParams(d_back=2.0, omega_turn=1.0, alpha_evade=1.0))
        assert result["reached"] is True
        assert result["n_collisions"] == 0
        assert result["time"] < 5.0

    def test_flight_with_obstacle_collision(self):
        """Given obstacle on straight path, when flying, then collision occurs."""
        drone = Drone(
            start_pos=[0.0, 0.0],
            target_pos=[20.0, 0.0],
            obstacles=[Circle(5.0, 0.0, 2.0)],
            speed=5.0,
            dt=0.05,
            max_duration=20.0,
        )
        result = drone.fly_episode(ManeuverParams(d_back=2.0, omega_turn=1.0, alpha_evade=1.0))
        assert result["n_collisions"] >= 1

    def test_timeout_when_unreachable(self):
        """Given target inside obstacle, when flying, then drone times out."""
        drone = Drone(
            start_pos=[0.0, 0.0],
            target_pos=[5.0, 0.0],
            obstacles=[Circle(5.0, 0.0, 3.0)],
            speed=5.0,
            dt=0.1,
            max_duration=2.0,
        )
        result = drone.fly_episode(ManeuverParams(d_back=2.0, omega_turn=1.0, alpha_evade=1.0))
        assert result["reached"] is False
        assert result["time"] <= 2.1

    def test_trajectory_starts_at_start(self):
        """Given a drone, when episode starts, then first trajectory point equals start."""
        drone = Drone(
            start_pos=[1.0, 2.0],
            target_pos=[10.0, 10.0],
            obstacles=[],
            speed=5.0,
            dt=0.1,
        )
        result = drone.fly_episode(ManeuverParams(d_back=2.0, omega_turn=1.0, alpha_evade=1.0))
        assert np.allclose(result["trajectory"][0], [1.0, 2.0])


class TestCollisionDetection:
    """Tests for point and segment collision against different shapes."""

    def test_circle_point_collision(self):
        drone = Drone([0, 0], [10, 0], [Circle(5, 0, 2)])
        assert drone.check_collision(np.array([5.0, 0.0])) is True
        assert drone.check_collision(np.array([5.0, 2.5])) is False

    def test_rect_point_collision(self):
        drone = Drone([0, 0], [10, 0], [Rect(5, 0, 4, 2)])
        assert drone.check_collision(np.array([5.0, 0.0])) is True
        assert drone.check_collision(np.array([5.0, 2.0])) is False

    def test_diamond_point_collision(self):
        drone = Drone([0, 0], [10, 0], [Diamond(5, 0, 4, 4)])
        assert drone.check_collision(np.array([5.0, 0.0])) is True
        assert drone.check_collision(np.array([5.0, 3.0])) is False

    def test_star_point_collision(self):
        drone = Drone([0, 0], [10, 0], [Star(5, 0, 2)])
        assert drone.check_collision(np.array([5.0, 0.0])) is True

    def test_cross_point_collision(self):
        drone = Drone([0, 0], [10, 0], [Cross(5, 0, 3, 1)])
        assert drone.check_collision(np.array([5.0, 0.0])) is True
        assert drone.check_collision(np.array([5.0, 3.5])) is False

    def test_circle_segment_collision(self):
        drone = Drone([0, 0], [10, 0], [Circle(5, 0, 2)])
        assert drone.check_segment_collision(np.array([0, 0]), np.array([10, 0])) is True
        assert drone.check_segment_collision(np.array([0, 5]), np.array([10, 5])) is False

    def test_rect_segment_collision(self):
        drone = Drone([0, 0], [10, 0], [Rect(5, 0, 4, 4)])
        assert drone.check_segment_collision(np.array([0, 0]), np.array([10, 0])) is True
        assert drone.check_segment_collision(np.array([0, 5]), np.array([10, 5])) is False

    def test_diamond_segment_collision(self):
        drone = Drone([0, 0], [10, 0], [Diamond(5, 0, 4, 4)])
        assert drone.check_segment_collision(np.array([0, 0]), np.array([10, 0])) is True
        assert drone.check_segment_collision(np.array([0, 5]), np.array([10, 5])) is False

    def test_star_segment_collision(self):
        drone = Drone([0, 0], [10, 0], [Star(5, 0, 2)])
        assert drone.check_segment_collision(np.array([0, 0]), np.array([10, 0])) is True

    def test_cross_segment_collision(self):
        drone = Drone([0, 0], [10, 0], [Cross(5, 0, 3, 1)])
        assert drone.check_segment_collision(np.array([0, 0]), np.array([10, 0])) is True
        assert drone.check_segment_collision(np.array([0, 5]), np.array([10, 5])) is False
