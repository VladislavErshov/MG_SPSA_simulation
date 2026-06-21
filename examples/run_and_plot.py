"""Episode-based maneuver training for a constant-speed drone.

Usage:
    python examples/run_and_plot.py --mode spsa2 --iterations 30
    python examples/run_and_plot.py --mode spsa1 --iterations 30 --seed 42
    python examples/run_and_plot.py --config configs/simulation/corridor_3.json --mode spsa2 --iterations 30
    python examples/run_and_plot.py --mode spsa2 --iterations 30 --runs 10
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from drone_simulator.core.drone import Drone, ManeuverParams
from drone_simulator.optimizers.spsa import ManeuverOptimizer, ManeuverOptimizerConfig
from drone_simulator.scenario import create_arena, load_scenario, randomize_positions
from drone_simulator.visualization.simulation_viz import plot_results


# ------------------------------------------------------------------
# Episode and loss
# ------------------------------------------------------------------
def run_episode(params: ManeuverParams, arena: Drone) -> dict:
    return arena.fly_episode(params)


def compute_loss(result: dict[str, Any]) -> float:
    traj = result["trajectory"]
    if len(traj) > 1:
        diffs = np.diff(traj, axis=0)
        traj_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        traj_len = 0.0
    loss = result["time"] + 0.01 * traj_len + 5.0 * result["n_collisions"]
    if not result["reached"]:
        dist_to_target = float(np.linalg.norm(traj[-1] - result.get("target_pos", traj[-1])))
        loss += 2.0 * dist_to_target
    return loss


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def _average_loss_over_randomizations(
    params: ManeuverParams,
    arena: Drone,
    scenario: dict,
    rng: np.random.Generator,
    n_samples: int,
) -> float:
    """Evaluate loss averaged over n_samples random start/target positions."""
    if n_samples <= 1:
        result = run_episode(params, arena)
        return compute_loss(result)

    total_loss = 0.0
    for _ in range(n_samples):
        randomize_positions(arena, scenario, rng)
        result = run_episode(params, arena)
        total_loss += compute_loss(result)
    return total_loss / n_samples


def train(
    mode: str,
    scenario: dict[str, Any],
    n_iterations: int,
    seed: int = 0,
    n_eval_samples: int = 5,
) -> tuple[ManeuverOptimizer, list[float], list[float], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    arena = create_arena(scenario)
    config = ManeuverOptimizerConfig()
    optimizer = ManeuverOptimizer(config)

    losses: list[float] = []
    times: list[float] = []
    trajectories: list[np.ndarray] = []
    traj_meta: list[tuple[np.ndarray, np.ndarray]] = []

    def loss_fn(theta: ManeuverParams) -> float:
        result = run_episode(theta, arena)
        return compute_loss(result)

    for i in range(n_iterations):
        randomize_positions(arena, scenario, rng)

        grad = optimizer.evaluate(mode, loss_fn)
        params = optimizer.to_params()

        # Average loss over multiple randomizations for smoother curves
        loss = _average_loss_over_randomizations(
            params, arena, scenario, rng, n_eval_samples
        )
        losses.append(loss)

        # Store trajectory and time from the last randomization (for visualization)
        result = run_episode(params, arena)
        times.append(result["time"])
        trajectories.append(result["trajectory"])
        traj_meta.append((arena.start_pos.copy(), arena.target_pos.copy()))
        print(
            f"Iter {i + 1:3d}: loss={loss:7.2f}  time={result['time']:5.1f}s  "
            f"d_back={params.d_back:.2f}  "
            f"omega={params.omega_turn:.2f}  "
            f"alpha={params.alpha_evade:.3f}  "
            f"grad=[{grad[0]:7.2f} {grad[1]:7.2f} {grad[2]:7.2f}]"
        )

    return optimizer, losses, times, trajectories, traj_meta


# ------------------------------------------------------------------
# Final evaluation helpers
# ------------------------------------------------------------------
def _fly_and_measure(params: ManeuverParams, scenario: dict) -> dict:
    arena = create_arena(scenario)
    result = arena.fly_episode(params)
    result["loss"] = compute_loss(result)
    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Episode-based maneuver training")
    parser.add_argument(
        "--config", type=str, default="configs/simulation/default.json",
        help="Path to scenario JSON (default: configs/simulation/default.json)"
    )
    parser.add_argument(
        "--mode", type=str, default="spsa2", choices=["spsa1", "spsa2"],
        help="SPSA mode: spsa1 (one-measurement) or spsa2 (centered)"
    )
    parser.add_argument(
        "--iterations", type=int, default=30,
        help="Number of training iterations (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not show plot window, only save to file"
    )
    parser.add_argument(
        "--n-eval", type=int, default=5,
        help="Number of random position samples to average loss over (default: 5)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent training runs to average over (default: 1)"
    )
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    scenario = load_scenario(args.config)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}, Iterations: {args.iterations}, Seed: {args.seed}, Runs: {args.runs}")

    all_losses: list[list[float]] = []
    all_times: list[list[float]] = []

    for r in range(args.runs):
        current_seed = args.seed + r
        print(f"\n--- Run {r + 1}/{args.runs} (seed={current_seed}) ---")
        optimizer, losses, times, trajectories, traj_meta = train(
            args.mode, scenario, args.iterations, current_seed, n_eval_samples=args.n_eval
        )
        all_losses.append(losses)
        all_times.append(times)

    # Average curves over runs
    avg_losses = np.mean(all_losses, axis=0).tolist()
    avg_times = np.mean(all_times, axis=0).tolist()

    # Final runs on original config positions (use last run's optimizer)
    baseline_cfg = ManeuverOptimizerConfig()
    baseline_params = ManeuverParams(
        d_back=baseline_cfg.d_back_init,
        omega_turn=baseline_cfg.omega_turn_init,
        alpha_evade=baseline_cfg.alpha_evade_init,
    )
    baseline_result = _fly_and_measure(baseline_params, scenario)
    final_result = _fly_and_measure(optimizer.to_params(), scenario)
    print(
        f"\nFixed policy:  loss={baseline_result['loss']:.2f}  "
        f"time={baseline_result['time']:.1f}s  "
        f"collisions={baseline_result['n_collisions']}"
    )
    print(
        f"Learned policy: loss={final_result['loss']:.2f}  "
        f"time={final_result['time']:.1f}s  "
        f"collisions={final_result['n_collisions']}  "
        f"reached={final_result['reached']}"
    )

    plot_results(
        optimizer, avg_losses, avg_times, trajectories, traj_meta,
        args.mode, scenario, final_result, baseline_result, args.config,
        show_plot=not args.no_display,
    )


if __name__ == "__main__":
    main()
