# Usage

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python examples/run_and_plot.py --mode spsa1 --iterations 30
python examples/run_and_plot.py --mode spsa2 --iterations 30 --seed 42
python examples/run_and_plot.py --config configs/simulation/grid.json --mode spsa2 --iterations 30
```

Saves results to `results/maneuver_learning.png`.

## Scenario Configuration

Scenario parameters are in JSON files under `configs/simulation/`:

- `start` — initial position `[x, y]`
- `target` — target position `[x, y]`
- `obstacles` — list of `[x, y, radius]` or `[x, y, radius, "star6"]`
- `speed` — constant flight speed (m/s)
- `dt` — simulation time step (s)
- `max_duration` — episode timeout (s)
- `target_tolerance` — distance to target considered "reached" (m)

Available scenarios:
- `configs/simulation/default.json` — scattered circular obstacles
- `configs/simulation/grid.json` — grid of circular obstacles
- `configs/simulation/grid_no_wind.json` — grid without wind
- `configs/simulation/grid_stars.json` — grid of star-shaped obstacles

## Architecture

- `Drone` — 2D drone with fixed speed and no inertia.
  - Flies straight toward target until collision.
  - On collision: backtracks `d_back` meters, turns by `alpha_evade`, then gradually turns back toward target with rate `omega_turn`.
  - `fly_episode(params)` runs one complete flight and returns `time`, `trajectory`, `n_collisions`, `reached`.
- `ManeuverOptimizer` — mixed-gradient optimizer for `[d_back, omega_turn, alpha_evade]`.
  - `d_back` and `omega_turn` — exact gradient via central finite difference.
  - `alpha_evade` — SPSA block:
    - `spsa1` — one-measurement (off-center)
    - `spsa2` — two-measurement (centered)
  - `evaluate(mode, run_fn)` performs one iteration: computes gradient, updates parameters, clips to bounds, logs history.
