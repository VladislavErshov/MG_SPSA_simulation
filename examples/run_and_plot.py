"""Episode-based maneuver training for a constant-speed drone.

Usage:
    python examples/run_and_plot.py --mode spsa2 --iterations 30
    python examples/run_and_plot.py --mode spsa1 --iterations 30 --seed 42
    python examples/run_and_plot.py --config configs/simulation/corridor_3.json --mode spsa2 --iterations 30
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from drone_simulator.core.drone import Drone
from drone_simulator.optimizers.spsa import ManeuverOptimizer, ManeuverOptimizerConfig
from drone_simulator.scenario import create_arena, load_scenario, randomize_positions
from drone_simulator.core.obstacles import parse_obstacle
from drone_simulator.visualization.simulation_viz import draw_obstacle


# ------------------------------------------------------------------
# Episode and loss
# ------------------------------------------------------------------
def run_episode(params: dict, arena: Drone) -> dict:
    return arena.fly_episode(params)


def compute_loss(result: dict, max_duration: float = 100.0) -> float:
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
def train(mode: str, scenario: dict, n_iterations: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    arena = create_arena(scenario)
    config = ManeuverOptimizerConfig()
    optimizer = ManeuverOptimizer(config)
    max_dur = scenario.get("max_duration", 100.0)

    losses = []
    trajectories = []
    traj_meta = []  # (start, target) for each trajectory

    def loss_fn(theta_dict: dict) -> float:
        result = run_episode(theta_dict, arena)
        return compute_loss(result, max_dur)

    for i in range(n_iterations):
        randomize_positions(arena, scenario, rng)

        grad = optimizer.evaluate(mode, loss_fn)
        params = optimizer._to_dict(optimizer.theta)
        result = run_episode(params, arena)
        loss = compute_loss(result, max_dur)
        losses.append(loss)
        trajectories.append(result["trajectory"])
        traj_meta.append((arena.start_pos.copy(), arena.target_pos.copy()))
        print(
            f"Iter {i + 1:3d}: loss={loss:7.2f}  "
            f"d_back={params['d_back']:.2f}  "
            f"omega={params['omega_turn']:.2f}  "
            f"alpha={params['alpha_evade']:.3f}  "
            f"grad=[{grad[0]:7.2f} {grad[1]:7.2f} {grad[2]:7.2f}]"
        )

    return optimizer, losses, trajectories, traj_meta


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
def _fly_and_measure(params: dict, scenario: dict) -> dict:
    arena = create_arena(scenario)
    result = arena.fly_episode(params)
    result["loss"] = compute_loss(result, scenario.get("max_duration", 100.0))
    return result


def plot_results(
    optimizer: ManeuverOptimizer,
    losses: list,
    trajectories: list,
    traj_meta: list,
    mode: str,
    scenario: dict,
    final_result: dict,
    baseline_result: dict,
    config_path: str = "default",
    show_plot: bool = True,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.25))
    fig.suptitle(f"Maneuver learning — mode {mode}", fontsize=14, fontweight="bold")

    # --- [0,0] Trajectories: training thin, final thick ----------------
    ax = axes[0, 0]

    # Training trajectories — thin semi-transparent
    cmap = plt.cm.cool
    n_traj = len(trajectories)
    for i, traj in enumerate(trajectories):
        color = cmap(i / max(n_traj - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.5, alpha=0.4, zorder=1)

    # Fixed policy (baseline) — dashed
    ax.plot(
        baseline_result["trajectory"][:, 0],
        baseline_result["trajectory"][:, 1],
        color="grey",
        linewidth=1.5,
        linestyle="--",
        label="Fixed policy",
        zorder=2,
    )

    # Final run — thick line
    ax.plot(
        final_result["trajectory"][:, 0],
        final_result["trajectory"][:, 1],
        color="blue",
        linewidth=3.0,
        label="Learned policy",
        zorder=3,
    )

    # Markers only for final run
    target = np.array(scenario["target"])
    start = np.array(scenario["start"])
    ax.scatter(
        target[0], target[1], color="red", marker="*", s=200,
        label="Target", zorder=4, edgecolors="white", linewidths=0.5,
    )
    ax.scatter(
        start[0], start[1], color="green", marker="s", s=80,
        label="Start", zorder=4, edgecolors="white", linewidths=0.5,
    )

    for obs_data in scenario["obstacles"]:
        draw_obstacle(ax, parse_obstacle(obs_data), alpha=0.15)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectories (training=thin, final=thick)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    pts = [scenario["start"], scenario["target"]]
    for obs_data in scenario["obstacles"]:
        pts.append([obs_data[0], obs_data[1]])
    for res in [baseline_result, final_result]:
        for p in res["trajectory"][::max(1, len(res["trajectory"]) // 30)]:
            pts.append(p.tolist())
    pts = np.array(pts)
    margin = 2.0
    ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)

    # --- [0,1] Parameter convergence ------------------------------
    ax2 = axes[0, 1]
    hist = optimizer.history
    thetas = np.array([h["theta"] for h in hist])
    iters = np.arange(1, len(hist) + 1)

    ax2.plot(iters, thetas[:, 0], label="d_back", linewidth=2)
    ax2.plot(iters, thetas[:, 1], label="omega_turn", linewidth=2)
    ax2.plot(iters, thetas[:, 2], label="alpha_evade", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Parameter value")
    ax2.set_title("Parameter convergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- [1,0] Loss dynamics -------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(losses) + 1), losses, color="darkgreen", linewidth=2)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss dynamics")
    ax3.grid(True, alpha=0.3)

    # --- [1,1] Results summary ---------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.set_title("Final metrics", fontsize=12, fontweight="bold", pad=10)

    params = optimizer._to_dict(optimizer.theta)

    def _fmt(res: dict, label: str) -> str:
        return (
            f"{label}:\n"
            f"  Loss:        {res['loss']:.2f}\n"
            f"  Time:        {res['time']:.1f} s\n"
            f"  Collisions:  {res['n_collisions']}\n"
            f"  Reached:     {'Yes' if res['reached'] else 'No'}"
        )

    scoreboard = (
        f"Mode: {mode}\n"
        f"Training iterations: {len(losses)}\n\n"
        f"Learned parameters:\n"
        f"  d_back     = {params['d_back']:.2f}\n"
        f"  omega_turn = {params['omega_turn']:.2f}\n"
        f"  alpha_evade= {params['alpha_evade']:.3f}\n\n"
        f"{_fmt(baseline_result, 'Fixed policy')}\n\n"
        f"{_fmt(final_result, 'Learned policy')}"
    )
    ax4.text(
        0.5, 0.5, scoreboard,
        transform=ax4.transAxes,
        fontsize=10, fontfamily="monospace",
        verticalalignment="center", horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8),
    )

    plt.tight_layout()
    cfg_name = Path(config_path).stem
    out_path = Path(f"results/maneuver_learning_{cfg_name}.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


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
    args = parser.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    scenario = load_scenario(args.config)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}, Iterations: {args.iterations}, Seed: {args.seed}")

    optimizer, losses, trajectories, traj_meta = train(
        args.mode, scenario, args.iterations, args.seed
    )

    # Final runs on original config positions
    baseline_cfg = ManeuverOptimizerConfig()
    baseline_params = {
        "d_back": baseline_cfg.d_back_init,
        "omega_turn": baseline_cfg.omega_turn_init,
        "alpha_evade": baseline_cfg.alpha_evade_init,
    }
    baseline_result = _fly_and_measure(baseline_params, scenario)
    final_result = _fly_and_measure(optimizer._to_dict(optimizer.theta), scenario)
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
        optimizer, losses, trajectories, traj_meta,
        args.mode, scenario, final_result, baseline_result, args.config,
        show_plot=not args.no_display,
    )


if __name__ == "__main__":
    main()
