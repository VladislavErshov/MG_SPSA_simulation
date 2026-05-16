# Usage

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python examples/run_and_plot.py                     # single run, default config
python examples/run_and_plot.py --runs 10 --seed 0  # 10 runs
python examples/run_and_plot.py --config configs/simulation/grid.json --runs 10
```

Saves trajectory image to `results/mixed_optimizer_runs.png`.

## Configuration

All parameters are in JSON files under `configs/`:

- `configs/simulation/*.json` — physics, obstacles, wind, target, simulation duration
- `configs/spsa/default.json` — MixedOptimizer hyper-parameters:
  - `a` — step-size amplitude
  - `burn_in` — offset for `alpha_n = a/(n+burn_in)` (0 = pure article theory)
  - `c` — perturbation amplitude
  - `num_perturbations` — N probes per SPSA block
- `configs/gd/default.json` — unused (reserved for future comparison)

## Architecture

- `MixedOptimizer` — modular mixed-gradient optimizer. Each parameter block can use `exact` gradient or `spsa_off_center` / `spsa_centered`.
- `TargetFollowingSPSA` — drone specialization with 3 blocks:
  1. `speed` — analytical exact gradient
  2. `direction` — one-measurement SPSA (`q=1`, `gamma=1/4`)
  3. `wind_estimate` — analytical exact gradient
- `Drone` — physics with inertia `V_{t+1} = V_t + alpha*(V_cmd - V_t)`, wind drift, and collision handling.
