# Usage

## Install

```bash
pip install -r requirements.txt
```

## Quick start — single run

```bash
python examples/run_spsa.py
```

## Multiple runs with plot

```bash
python examples/run_and_plot.py --runs 10 --seed 0
```

Saves trajectory image to `results/mixed_optimizer_runs.png`.

## Run tests

```bash
python -m pytest tests/
```

## Configuration

All parameters are in `src/drone_simulator/config.py`.

Key sections:
- `mixed_optimizer`:
  - `a` — step-size amplitude
  - `burn_in` — offset for `alpha_n = a/(n+burn_in)` (0 = pure article theory)
  - `c` — perturbation amplitude
  - `num_perturbations` — N probes per SPSA block
- `wind` — constant external force vector `[vx, vy]` (m/s)

## Architecture

- `MixedOptimizer` — modular mixed-gradient optimizer. Each parameter block can use `exact` gradient or `spsa_off_center` / `spsa_centered`.
- `TargetFollowingSPSA` — drone specialization with 3 blocks:
  1. `speed` — analytical exact gradient
  2. `direction` — one-measurement SPSA (`q=1`, `gamma=1/4`)
  3. `wind_estimate` — one-measurement SPSA (`q=1`, `gamma=1/4`)
- `Drone` — physics with inertia `V_{t+1} = V_t + alpha*(V_cmd - V_t)`, wind drift, and collision handling.
