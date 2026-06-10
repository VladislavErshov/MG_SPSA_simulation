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

    zone = scenario.get("start_zone")
    if zone:
        cx, cy = zone["cx"], zone["cy"]
        r_min, r_max = zone["r_min"], zone["r_max"]
        center = np.array([cx, cy], dtype=float)
        for _ in range(max_attempts):
            distance = rng.uniform(r_min, r_max)
            angle = rng.uniform(0, 2 * np.pi)
            new_start = center + distance * np.array([np.cos(angle), np.sin(angle)])
            if arena._check_collision(new_start):
                continue
            arena.start_pos = new_start
            arena.target_pos = center
            return new_start, center
        arena.start_pos = start_orig
        arena.target_pos = target_orig
        return start_orig, target_orig

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
        return [cx, cy, size * 1.5, size * 0.3, "rect"]
    else:
        return [cx, cy, size * 0.3, size * 1.5, "rect"]


def _build_grid_obstacle(
    shape: str,
    cx: float,
    cy: float,
    size: float,
    is_horizontal: bool = True,
) -> list:
    """Build a single grid obstacle based on shape type."""
    if shape == "circle":
        return [cx, cy, size / 1.5]
    if shape == "star5":
        return [cx, cy, size * 0.625, "star5"]
    if shape == "cross":
        return [cx, cy, size * 0.5, size * 0.3, "cross"]
    if shape == "diamond":
        return [cx, cy, size * 1.6, size * 1.6, "diamond"]
    if shape == "crossed_rect":
        return _build_crossed_rect_obstacle(cx, cy, size, is_horizontal)
    # default rect
    return [cx, cy, size * 1, size * 1, "rect"]


def generate_grid_scenario(
    shape: Literal["circle", "rect", "diamond", "star5", "cross", "crossed_rect"] = "circle",
    grid_n: int = 5,
    spacing: float = 5.0,
    obstacle_size: float = 2.0,
    start_side: Literal["left", "right", "top", "bottom", "random"] = "left",
    seed: int = 0,
) -> dict:
    """Generate a scenario with a dense n×n obstacle grid.

    The grid size n must be odd. The target is placed in the centre cell
    and the central obstacle is removed. The start is placed outside the
    grid so that the straight line to the target passes through a row or
    column of obstacles.

    Parameters
    ----------
    shape : str
        Obstacle shape type.
    grid_n : int
        Number of obstacles along each axis (must be odd).
    spacing : float
        Distance between obstacle centres.
    obstacle_size : float
        Size parameter (radius / arm / half-width).
    start_side : str
        Which side of the grid the start is on ("random" picks one).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Validated scenario dictionary.
    """
    if grid_n % 2 == 0:
        raise ValueError("grid_n must be odd so the grid has a unique centre cell")

    rng = np.random.default_rng(seed)

    if start_side == "random":
        start_side = rng.choice(["left", "right", "top", "bottom"]).item()  # type: ignore[assignment]

    grid_width = (grid_n - 1) * spacing
    grid_height = grid_width
    margin = spacing * 1.5

    arena_width = grid_width + 2 * margin
    arena_height = grid_height + 2 * margin

    # Grid top-left corner
    grid_x0 = margin
    grid_y0 = margin

    # Place target in the centre cell and remove that obstacle.
    mid_i = grid_n // 2
    target_x = grid_x0 + mid_i * spacing
    target_y = grid_y0 + mid_i * spacing
    target = [target_x, target_y]

    # Base start (used for distance and span calculations only).
    if start_side == "left":
        base_start = [margin * 0.3, target_y]
    elif start_side == "right":
        base_start = [arena_width - margin * 0.3, target_y]
    elif start_side == "top":
        base_start = [target_x, arena_height - margin * 0.3]
    else:  # bottom
        base_start = [target_x, margin * 0.3]

    obstacles = []
    for iy in range(grid_n):
        for ix in range(grid_n):
            if ix == mid_i and iy == mid_i:
                continue  # centre cell is the target
            cx = grid_x0 + ix * spacing
            cy = grid_y0 + iy * spacing
            is_horizontal = (ix + iy) % 2 == 0
            obstacles.append(_build_grid_obstacle(shape, cx, cy, obstacle_size, is_horizontal))

    # Randomise start position inside a ring strictly around the obstacle grid.
    base_start_arr = np.array(base_start, dtype=float)
    target_arr = np.array(target, dtype=float)

    obs_xs = [o[0] for o in obstacles]
    obs_ys = [o[1] for o in obstacles]
    all_x = obs_xs + [base_start_arr[0], target_arr[0]]
    all_y = obs_ys + [base_start_arr[1], target_arr[1]]
    span_x = max(all_x) - min(all_x)
    span_y = max(all_y) - min(all_y)
    pad = max(span_x, span_y) * 0.05
    obs_min_x, obs_max_x = min(obs_xs) - pad, max(obs_xs) + pad
    obs_min_y, obs_max_y = min(obs_ys) - pad, max(obs_ys) + pad

    r_min = max(
        abs(target_x - obs_min_x),
        abs(target_x - obs_max_x),
        abs(target_y - obs_min_y),
        abs(target_y - obs_max_y),
    )
    start_zone_width = 2.0
    r_max = r_min + start_zone_width

    start_zone = {
        "cx": float(target_x),
        "cy": float(target_y),
        "r_min": float(r_min),
        "r_max": float(r_max),
    }

    tmp_scenario = {
        "start": base_start,
        "target": target,
        "obstacles": obstacles,
        "speed": 5.0,
        "dt": 0.05,
        "max_duration": 100.0,
        "target_tolerance": 1.0,
    }
    validate_scenario(tmp_scenario)
    tmp_arena = create_arena(tmp_scenario)

    start = base_start
    for _ in range(200):
        distance = rng.uniform(r_min, r_max)
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        new_start = target_arr + distance * direction

        if tmp_arena._check_collision(new_start):
            continue
        if (obs_min_x <= new_start[0] <= obs_max_x and
                obs_min_y <= new_start[1] <= obs_max_y):
            continue

        # Ensure the straight line from start to target crosses at least one
        # obstacle. If not, reject this angle and try again.
        n_points = max(100, int(distance / 0.1))
        blocked = False
        for t in np.linspace(0, 1, n_points):
            pt = new_start + t * (target_arr - new_start)
            if tmp_arena._check_collision(pt):
                blocked = True
                break
        if not blocked:
            continue

        start = new_start.tolist()
        break

    scenario = {
        "start": start,
        "target": target,
        "obstacles": obstacles,
        "speed": 5.0,
        "dt": 0.05,
        "max_duration": 100.0,
        "target_tolerance": 1.0,
        "start_zone": start_zone,
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
