"""Scenario loading, validation, position randomization, and generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

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

    # Compute obstacle bounding box with a small padding.
    # For grid arenas we want the start to stay outside this box.
    obs_xs = [obs.x for obs in arena.obstacles]
    obs_ys = [obs.y for obs in arena.obstacles]
    pad = max(span_x, span_y) * 0.05
    obs_min_x, obs_max_x = min(obs_xs) - pad, max(obs_xs) + pad
    obs_min_y, obs_max_y = min(obs_ys) - pad, max(obs_ys) + pad

    for _ in range(max_attempts):
        angle = rng.uniform(0, 2 * np.pi)
        dx = rng.uniform(-span_x * 0.25, span_x * 0.25)
        dy = rng.uniform(-span_y * 0.25, span_y * 0.25)

        new_mid = midpoint + np.array([dx, dy])
        direction = np.array([np.cos(angle), np.sin(angle)])
        new_start = new_mid - (distance / 2) * direction
        new_target = new_mid + (distance / 2) * direction

        if arena._check_collision(new_start) or arena._check_collision(new_target):
            continue

        # Reject if the start landed inside the obstacle bounding box.
        # This prevents grid-arena starts from appearing inside the grid.
        if (obs_min_x <= new_start[0] <= obs_max_x and
                obs_min_y <= new_start[1] <= obs_max_y):
            continue

        arena.start_pos = new_start
        arena.target_pos = new_target
        return new_start, new_target

    arena.start_pos = start_orig
    arena.target_pos = target_orig
    return start_orig, target_orig


def _build_crossed_rect_obstacle(
    cx: float, cy: float, size: float, horizontal: bool
) -> list:
    """Build a crossed-rect obstacle: either horizontal or vertical bar."""
    if horizontal:
        return [cx, cy, size * 2.2, size * 0.4, "rect"]
    else:
        return [cx, cy, size * 0.4, size * 2.2, "rect"]


def _build_grid_obstacle(
    shape: str,
    cx: float,
    cy: float,
    size: float,
    is_horizontal: bool = True,
) -> list:
    """Build a single grid obstacle based on shape type."""
    if shape == "circle":
        return [cx, cy, size]
    if shape == "star5":
        return [cx, cy, size, "star5"]
    if shape == "cross":
        return [cx, cy, size, size * 0.3, "cross"]
    if shape == "diamond":
        return [cx, cy, size * 1.6, size * 1.6, "diamond"]
    if shape == "crossed_rect":
        return _build_crossed_rect_obstacle(cx, cy, size, is_horizontal)
    # default rect
    return [cx, cy, size * 1.4, size * 1.2, "rect"]


def generate_grid_scenario(
    shape: Literal["circle", "rect", "diamond", "star5", "cross", "crossed_rect"] = "circle",
    grid_nx: int = 5,
    grid_ny: int = 4,
    spacing: float = 5.0,
    obstacle_size: float = 2.0,
    start_side: Literal["left", "right", "top", "bottom"] = "left",
    seed: int = 0,
) -> dict:
    """Generate a scenario with a dense obstacle grid.

    The target is placed inside the grid; the start is outside.
    The drone must break through the grid to reach the target.

    Parameters
    ----------
    shape : str
        Obstacle shape type.
    grid_nx, grid_ny : int
        Number of obstacles along X and Y.
    spacing : float
        Distance between obstacle centres.
    obstacle_size : float
        Size parameter (radius / arm / half-width).
    start_side : str
        Which side of the grid the start is on.
    seed : int
        Random seed (unused, kept for API consistency).

    Returns
    -------
    dict
        Validated scenario dictionary.
    """
    rng = np.random.default_rng(seed)

    grid_width = (grid_nx - 1) * spacing
    grid_height = (grid_ny - 1) * spacing
    margin = spacing * 1.5

    arena_width = grid_width + 2 * margin
    arena_height = grid_height + 2 * margin

    # Grid top-left corner
    grid_x0 = margin
    grid_y0 = margin

    # Align target and start with the middle row of the grid so the straight
    # line is guaranteed to cross that row of obstacles.
    mid_iy = grid_ny // 2
    target_x = grid_x0 + grid_width / 2
    target_y = grid_y0 + mid_iy * spacing

    if start_side == "left":
        start = [margin * 0.3, target_y]
    elif start_side == "right":
        start = [arena_width - margin * 0.3, target_y]
    elif start_side == "top":
        start = [target_x, arena_height - margin * 0.3]
    else:  # bottom
        start = [target_x, margin * 0.3]

    target = [target_x, target_y]

    obstacles = []
    for iy in range(grid_ny):
        for ix in range(grid_nx):
            cx = grid_x0 + ix * spacing
            cy = grid_y0 + iy * spacing
            is_horizontal = (ix + iy) % 2 == 0
            obstacles.append(_build_grid_obstacle(shape, cx, cy, obstacle_size, is_horizontal))

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

    # Ensure target is not inside any obstacle; if it is, nudge toward start.
    tmp_arena = create_arena(scenario)
    nudge_vec = np.array(start) - np.array(target)
    nudge_vec = nudge_vec / (np.linalg.norm(nudge_vec) + 1e-12)
    while tmp_arena._check_collision(np.array(target)):
        target[0] += nudge_vec[0] * 0.3
        target[1] += nudge_vec[1] * 0.3
        scenario["target"] = target
        tmp_arena = create_arena(scenario)

    return scenario


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
    """
    rng = np.random.default_rng(seed)
    start = [0.0, arena_height / 2]
    target = [arena_size, arena_height / 2]

    obstacles = []

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
