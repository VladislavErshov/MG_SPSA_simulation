"""Tests for position randomization and training pipeline."""

import numpy as np
import pytest

from drone_simulator.core.drone import Drone


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


def _create_arena(scenario: dict) -> Drone:
    return Drone(
        start_pos=scenario["start"],
        target_pos=scenario["target"],
        obstacles=scenario["obstacles"],
        speed=scenario.get("speed", 5.0),
        dt=scenario.get("dt", 0.05),
        max_duration=scenario.get("max_duration", 100.0),
    )


def randomize_positions(
    arena: Drone,
    scenario: dict,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of run_and_plot.randomize_positions for testing."""
    start_orig = np.array(scenario["start"], dtype=float)
    target_orig = np.array(scenario["target"], dtype=float)
    distance = float(np.linalg.norm(target_orig - start_orig))
    midpoint = (start_orig + target_orig) / 2

    all_x = [obs[0] for obs in scenario["obstacles"]] + [start_orig[0], target_orig[0]]
    all_y = [obs[1] for obs in scenario["obstacles"]] + [start_orig[1], target_orig[1]]
    span_x = max(all_x) - min(all_x)
    span_y = max(all_y) - min(all_y)

    for _ in range(max_attempts):
        angle = rng.uniform(0, 2 * np.pi)
        dx = rng.uniform(-span_x * 0.25, span_x * 0.25)
        dy = rng.uniform(-span_y * 0.25, span_y * 0.25)

        new_mid = midpoint + np.array([dx, dy])
        direction = np.array([np.cos(angle), np.sin(angle)])
        new_start = new_mid - (distance / 2) * direction
        new_target = new_mid + (distance / 2) * direction

        if not arena._check_collision(new_start) and not arena._check_collision(new_target):
            arena.start_pos = new_start
            arena.target_pos = new_target
            return new_start, new_target

    arena.start_pos = start_orig
    arena.target_pos = target_orig
    return start_orig, target_orig


# -- Given / When / Then tests --


class TestRandomizePositions:
    """Tests for randomize_positions function."""

    def test_preserves_distance_between_start_and_target(self):
        """Given original start/target, when randomizing, then distance is preserved."""
        scenario = _default_scenario()
        arena = _create_arena(scenario)
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
        arena = _create_arena(scenario)
        rng = np.random.default_rng(123)

        for _ in range(30):
            new_start, new_target = randomize_positions(arena, scenario, rng)
            assert not arena._check_collision(new_start)
            assert not arena._check_collision(new_target)

    def test_arena_positions_updated(self):
        """Given an arena, when randomizing, then arena.start_pos and arena.target_pos are updated."""
        scenario = _default_scenario()
        arena = _create_arena(scenario)
        rng = np.random.default_rng(7)

        new_start, new_target = randomize_positions(arena, scenario, rng)
        assert np.array_equal(arena.start_pos, new_start)
        assert np.array_equal(arena.target_pos, new_target)

    def test_positions_change_across_calls(self):
        """Given multiple calls, when randomizing with different seeds, then positions differ."""
        scenario = _default_scenario()

        positions = []
        for seed in range(10):
            arena = _create_arena(scenario)
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
        arena = _create_arena(scenario)
        rng = np.random.default_rng(0)

        new_start, new_target = randomize_positions(arena, scenario, rng, max_attempts=5)
        assert np.array_equal(new_start, np.array(scenario["start"]))
        assert np.array_equal(new_target, np.array(scenario["target"]))


class TestTrainReturnsTrajMeta:
    """Tests that train() returns trajectory metadata with start/target positions."""

    def test_train_returns_four_elements(self):
        """Given training, when complete, then returns (optimizer, losses, trajectories, traj_meta)."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.run_and_plot import train

        scenario = _default_scenario()
        result = train("spsa2", scenario, n_iterations=2, seed=0)
        assert len(result) == 4

    def test_traj_meta_matches_trajectories_count(self):
        """Given training, when complete, then traj_meta length equals trajectories length."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.run_and_plot import train

        scenario = _default_scenario()
        _, _, trajectories, traj_meta = train("spsa2", scenario, n_iterations=3, seed=1)
        assert len(traj_meta) == len(trajectories) == 3

    def test_traj_meta_contains_start_target_arrays(self):
        """Given training, when complete, then each traj_meta entry is (start, target) arrays."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.run_and_plot import train

        scenario = _default_scenario()
        _, _, _, traj_meta = train("spsa2", scenario, n_iterations=2, seed=2)
        for start, target in traj_meta:
            assert isinstance(start, np.ndarray)
            assert isinstance(target, np.ndarray)
            assert start.shape == (2,)
            assert target.shape == (2,)
