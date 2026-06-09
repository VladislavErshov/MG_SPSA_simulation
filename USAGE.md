# Usage

## Install

```bash
pip install -e .
```

## Generate Scenarios

### Generate all grid-shape variants

```bash
python examples/generate_random_scenario.py --all-grids --seed 1
```

This creates 6 configs under `configs/simulation/`:
- `grid_circle_random.json`
- `grid_rect_random.json`
- `grid_diamond_random.json`
- `grid_star5_random.json`
- `grid_cross_random.json`
- `grid_crossed_rect_random.json`

### Generate a single wall scenario

```bash
python examples/generate_random_scenario.py --type wall --out configs/simulation/my_wall.json --seed 42
```

### Generate a single grid scenario

```bash
python examples/generate_random_scenario.py --type grid --shape star5 --out configs/simulation/my_stars.json --seed 7
```

## Run Training

### Train on a single scenario

```bash
python examples/run_and_plot.py --config configs/simulation/default.json --mode spsa2 --iterations 30
python examples/run_and_plot.py --mode spsa1 --iterations 30 --seed 42
python examples/run_and_plot.py --config configs/simulation/corridor_3.json --mode spsa2 --iterations 30
```

### Train on all existing configs in `configs/simulation/`

```bash
for cfg in configs/simulation/*.json; do
    echo "=== $(basename $cfg) ==="
    python examples/run_and_plot.py --config "$cfg" --mode spsa2 --iterations 30 --seed 42 --no-display
done
```

### Generate all grids and train on all of them

```bash
python examples/generate_random_scenario.py --all-grids --seed 1
for cfg in configs/simulation/grid_*_random.json; do
    echo "=== $(basename $cfg) ==="
    python examples/run_and_plot.py --config "$cfg" --mode spsa2 --iterations 30 --seed 42 --no-display
done
```

Saves results to `results/maneuver_learning_<config_name>.png`.

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/simulation/default.json` | Path to scenario JSON |
| `--mode` | `spsa2` | SPSA mode: `spsa1` (one-measurement) or `spsa2` (centered) |
| `--iterations` | `30` | Number of training iterations |
| `--seed` | `0` | Random seed |
| `--no-display` | `False` | Save plot without showing window |
| `--n-eval` | `5` | Number of random position samples to average loss over |

## Scenario Configuration

Scenario parameters are in JSON files under `configs/simulation/`:

- `start` — initial position `[x, y]`
- `target` — target position `[x, y]`
- `obstacles` — list of obstacles:
  - `[x, y, radius]` — circle
  - `[x, y, w, h, "rect"]` — rectangle
  - `[x, y, w, h, "diamond"]` — diamond
  - `[x, y, r, "star5"]` — 5-point star
  - `[x, y, arm, t, "cross"]` — cross
- `speed` — constant flight speed (m/s)
- `dt` — simulation time step (s)
- `max_duration` — episode timeout (s)
- `target_tolerance` — distance to target considered "reached" (m)

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
  - Step size: `a / (n + A)` (Robbins–Monro schedule).
  - Perturbation: `c / n^{γ}` where `γ = 1/4` for spsa1 and `γ = 1/6` for spsa2.
  - Gradient normalization by component-wise max.
  - `get_params()` returns the latest parameters.
