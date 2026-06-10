"""Tests for position randomization and training pipeline."""

import numpy as np
import pytest

from drone_simulator.core.drone import Drone
from drone_simulator.core.obstacles import Circle
from drone_simulator.scenario import create_arena, randomize_positions


def _default_scenario() -> dict:
    return {
        "start": [0.0, 0.0],
        "target": [20.0, 15.0],
        "obstacles": [
            [5.0, 5.0, 2.0],
            [10.0, 8.0, 1.5],
            [15.0, 12.0, 2.5],
        ],
        "speed": 5.0,
        "dt": 0.05,
        "max_duration": 100.0,
    }


# -- Given / When / Then tests --


class TestRandomizePositions:
    """Tests for randomize_positions function."""

    def test_preserves_distance_between_start_and_target(self):
        """Given original start/target, when randomizing, then distance is preserved."""
        scenario = _default_scenario()
        arena = create_arena(scenario)
        orig_dist = np.linalg.norm(
            np.array(scenario["target"]) - np.array(scenario["start"])
        )
        rng = np.random.default_rng(42)

        for _ in range(20):
            new_start, new_target = randomize_positions(arena, scenario, rng)
            new_dist = np.linalg.norm(new_target - new_start)
            assert new_dist == pytest.approx(orig_dist, rel=1e-10)

    def test_neither_point_inside_obstacles(self):
        """Given obstacles, when randomizing, then neither start nor target is inside any obstacle."""
        scenario = _default_scenario()
        arena = create_arena(scenario)
        rng = np.random.default_rng(123)

        for _ in range(30):
            new_start, new_target = randomize_positions(arena, scenario, rng)
            assert not arena._check_collision(new_start)
            assert not arena._check_collision(new_target)

    def test_arena_positions_updated(self):
        """Given an arena, when randomizing, then arena.start_pos and arena.target_pos are updated."""
        scenario = _default_scenario()
        arena = create_arena(scenario)
        rng = np.random.default_rng(7)

        new_start, new_target = randomize_positions(arena, scenario, rng)
        assert np.array_equal(arena.start_pos, new_start)
        assert np.array_equal(arena.target_pos, new_target)

    def test_positions_change_across_calls(self):
        """Given multiple calls, when randomizing with different seeds, then positions differ."""
        scenario = _default_scenario()

        positions = []
        for seed in range(10):
            arena = create_arena(scenario)
            rng = np.random.default_rng(seed)
            start, _ = randomize_positions(arena, scenario, rng)
            positions.append(start.copy())

        unique = len({tuple(p.round(4)) for p in positions})
        assert unique > 1, "Positions should differ across calls with different seeds"

    def test_fallback_when_all_attempts_fail(self):
        """Given scenario where all positions are inside obstacles, when randomizing, then fallback to original."""
        scenario = {
            "start": [0.0, 0.0],
            "target": [2.0, 0.0],
            "obstacles": [
                [-10.0, -10.0, 50.0],  # giant obstacle covering everything
            ],
            "speed": 5.0,
            "dt": 0.05,
            "max_duration": 100.0,
        }
        arena = create_arena(scenario)
        rng = np.random.default_rng(0)

        new_start, new_target = randomize_positions(arena, scenario, rng, max_attempts=5)
        assert np.array_equal(new_start, np.array(scenario["start"]))
        assert np.array_equal(new_target, np.array(scenario["target"]))


class TestTrainReturnsTrajMeta:
    """Tests that train() returns trajectory metadata with start/target positions."""

    def test_train_returns_five_elements(self):
        """Given training, when complete, then returns (optimizer, losses, times, trajectories, traj_meta)."""
        from examples.run_and_plot import train

        scenario = _default_scenario()
        result = train("spsa2", scenario, n_iterations=2, seed=0)
        assert len(result) == 5

    def test_traj_meta_matches_trajectories_count(self):
        """Given training, when complete, then traj_meta length equals trajectories length."""
        from examples.run_and_plot import train

        scenario = _default_scenario()
        _, _, _, trajectories, traj_meta = train("spsa2", scenario, n_iterations=3, seed=1)
        assert len(traj_meta) == len(trajectories) == 3

    def test_traj_meta_contains_start_target_arrays(self):
        """Given training, when complete, then each traj_meta entry is (start, target) arrays."""
        from examples.run_and_plot import train

        scenario = _default_scenario()
        _, _, _, _, traj_meta = train("spsa2", scenario, n_iterations=2, seed=2)
        for start, target in traj_meta:
            assert isinstance(start, np.ndarray)
            assert isinstance(target, np.ndarray)
            assert start.shape == (2,)
            assert target.shape == (2,)
