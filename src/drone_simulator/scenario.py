"""Scenario loading, validation, position randomization, and generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from drone_simulator.core.drone import Drone
from drone_simulator.core.obstacles import parse_obstacle


REQUIRED_SCENARIO_KEYS = {"start", "target", "obstacles"}
OPTIONAL_SCENARIO_DEFAULTS = {
    "speed": 5.0,
    "dt": 0.05,
    "max_duration": 100.0,
}


def load_scenario(config_path: str | Path) -> dict:
    """Load and validate a scenario JSON file."""
    path = Path(config_path)
    if not path.is_absolute():
        # Assume relative to project root when called from examples/
        project_root = Path(__file__).parent.parent.parent
        path = project_root / path

    import json

    with open(path, "r") as f:
        scenario = json.load(f)

    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: dict[str, Any]) -> None:
    """Validate scenario dictionary. Raises ValueError on invalid input."""
    missing = REQUIRED_SCENARIO_KEYS - scenario.keys()
    if missing:
        raise ValueError(f"Missing required scenario keys: {missing}")

    if not isinstance(scenario["obstacles"], list):
        raise ValueError("'obstacles' must be a list")

    for i, obs in enumerate(scenario["obstacles"]):
        if not isinstance(obs, list) or len(obs) < 3:
            raise ValueError(f"Obstacle {i} must be a list of at least 3 numbers, got {obs}")

    for key, default in OPTIONAL_SCENARIO_DEFAULTS.items():
        scenario.setdefault(key, default)


def create_arena(scenario: dict) -> Drone:
    """Create a Drone arena from a validated scenario dict."""
    obstacles = [parse_obstacle(o) for o in scenario["obstacles"]]
    return Drone(
        start_pos=scenario["start"],
        target_pos=scenario["target"],
        obstacles=obstacles,
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
    """Shift start/target while preserving distance and avoiding obstacles."""
    start_orig = np.array(scenario["start"], dtype=float)
    target_orig = np.array(scenario["target"], dtype=float)
    distance = float(np.linalg.norm(target_orig - start_orig))
    midpoint = (start_orig + target_orig) / 2

    all_x = [obs.x for obs in arena.obstacles] + [start_orig[0], target_orig[0]]
    all_y = [obs.y for obs in arena.obstacles] + [start_orig[1], target_orig[1]]
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


def generate_wall_scenario(
    arena_size: float = 30.0,
    arena_height: float = 20.0,
    n_walls: int = 1,
    wall_thickness: float = 1.5,
    gap_size: float = 3.0,
    seed: int = 0,
) -> dict:
    """Generate a scenario with perpendicular walls that force maneuvers.

    The start is at the left edge, target at the right edge.
    Walls are placed vertically (perpendicular to start->target line) with
    a single narrow gap.  Drones cannot fly around the walls because they
    span the full arena height.

    Parameters
    ----------
    arena_size : float
        Distance from start to target along X.
    arena_height : float
        Height of the arena along Y.
    n_walls : int
        Number of walls (1 or 2).
    wall_thickness : float
        Thickness of each wall segment (rectangles).
    gap_size : float
        Size of the gap in each wall.
    seed : int
        Random seed for gap placement.

    Returns
    -------
    dict
        Validated scenario dictionary.
    """
    rng = np.random.default_rng(seed)
    start = [0.0, arena_height / 2]
    target = [arena_size, arena_height / 2]

    obstacles = []
    half_thick = wall_thickness / 2

    center_y = arena_height / 2

    for w in range(n_walls):
        wall_x = arena_size * (w + 1) / (n_walls + 1)
        # Random gap center along Y, but enforce that the gap does NOT
        # cover the central horizontal line (start/target height).
        # This guarantees the straight-line path hits the wall.
        min_gap = gap_size
        max_gap = arena_height - gap_size
        while True:
            gap_center = rng.uniform(min_gap, max_gap)
            if abs(gap_center - center_y) > gap_size / 2 + 0.5:
                break
        gap_bottom = gap_center - gap_size / 2
        gap_top = gap_center + gap_size / 2

        # Bottom segment
        if gap_bottom > 0:
            seg_h = gap_bottom
            seg_y = seg_h / 2
            obstacles.append([
                wall_x, seg_y, wall_thickness, seg_h, "rect"
            ])

        # Top segment
        if gap_top < arena_height:
            seg_h = arena_height - gap_top
            seg_y = gap_top + seg_h / 2
            obstacles.append([
                wall_x, seg_y, wall_thickness, seg_h, "rect"
            ])

    scenario = {
        "start": start,
        "target": target,
        "obstacles": obstacles,
        "speed": 5.0,
        "dt": 0.05,
        "max_duration": 100.0,
        "target_tolerance": 1.0,
    }
    validate_scenario(scenario)
    return scenario


def save_scenario(scenario: dict, path: str | Path) -> None:
    """Save a scenario dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2)
