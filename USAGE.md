# Usage

## Install

```bash
pip install -e .
```

## Run

```bash
python examples/run_and_plot.py --mode spsa2 --iterations 30
python examples/run_and_plot.py --mode spsa1 --iterations 30 --seed 42
python examples/run_and_plot.py --config configs/simulation/corridor_3.json --mode spsa2 --iterations 30
python examples/run_and_plot.py --config configs/simulation/grid.json --trajectories all --iterations 30
```

Saves results to `results/maneuver_learning_<config_name>.png`.

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/simulation/default.json` | Path to scenario JSON |
| `--mode` | `spsa2` | SPSA mode: `spsa1` (one-measurement) or `spsa2` (centered) |
| `--iterations` | `30` | Number of training iterations |
| `--seed` | `0` | Random seed |
| `--trajectories` | `best` | Plot mode: `best` (baseline + trained) or `all` (all iterations) |

## Scenario Configuration

Scenario parameters are in JSON files under `configs/simulation/`:

- `start` — initial position `[x, y]`
- `target` — target position `[x, y]`
- `obstacles` — list of `[x, y, radius]` (circle), `[x, y, w, h, "rect"]`, or `[x, y, w, h, "diamond"]`
- `speed` — constant flight speed (m/s)
- `dt` — simulation time step (s)
- `max_duration` — episode timeout (s)
- `target_tolerance` — distance to target considered "reached" (m)

Available scenarios:
- `default.json` — scattered circular obstacles
- `single_large.json` — one large circular obstacle
- `grid.json` — grid of circular obstacles
- `grid_stars.json` — grid with star-shaped obstacles
- `rectangles_1.json` — rectangular obstacles
- `ring.json` — ring-shaped obstacle course
- `walls.json` — wall obstacles
- `corridor_1.json` — simple corridor
- `corridor_2.json` — narrow corridor
- `corridor_3.json` — diagonal corridor with funnel

## Architecture

- `Drone` — 2D drone with fixed speed and no inertia.
  - Flies straight toward target until collision.
  - On collision: backtracks `d_back * (1 + 0.5 * n_collisions)` meters, turns by `alpha_evade`, then gradually turns back toward target with rate `omega_turn`.
  - `fly_episode(params)` runs one complete flight and returns `time`, `trajectory`, `n_collisions`, `reached`, `target_pos`.
- `ManeuverOptimizer` — mixed-gradient optimizer for `[d_back, omega_turn, alpha_evade]`.
  - `d_back` and `omega_turn` — exact gradient via central finite difference.
  - `alpha_evade` — SPSA block:
    - `spsa1` — one-measurement (off-center)
    - `spsa2` — two-measurement (centered)
  - Step size: `a / (n + A)^0.602` (standard SPSA schedule).
  - Gradient normalization by component-wise max (preserves relative magnitudes).
  - `get_params()` returns parameters with the best loss from history.
